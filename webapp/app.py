"""
LiDAR3D AutoLabel — admin portal.
Chạy trên VM (cùng chỗ run.py + venv + model):
  login admin -> upload point cloud -> auto-label (GPU) -> phân công annotator -> đánh giá.

Chạy (systemd service autolabel-web, hoặc tay):
  uvicorn app:app --host 0.0.0.0 --port 8000   (trong thư mục webapp/)
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import subprocess
import threading
import uuid
import zipfile

import requests
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

HOME = os.path.expanduser("~")
RUN_PY = os.path.join(HOME, "auto-label-3d", "run.py")
VENV_PY = os.path.join(HOME, "venv", "bin", "python3")
JOBS_DIR = os.path.join(HOME, "autolabel_jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

# CVAT nội bộ (IP tĩnh của VM — CVAT_HOST routing khớp IP này)
PUBLIC_IP = os.environ.get("PUBLIC_IP", "34.87.125.249")
CVAT_URL = f"http://{PUBLIC_IP}:8080"
CVAT_USER = os.environ.get("CVAT_USER", "admin")
CVAT_PASS = os.environ.get("CVAT_PASS", "Cvat123456")

# Đăng nhập admin của portal — lưu ra file để đổi được lúc chạy + giữ qua restart
ADMIN_FILE = os.path.join(HOME, ".autolabel_admin.json")
_SECRET = secrets.token_bytes(32)


def _load_admin():
    try:
        with open(ADMIN_FILE, encoding="utf-8") as f:
            d = json.load(f)
            return d["user"], d["pass"]
    except Exception:  # noqa: BLE001
        return os.environ.get("ADMIN_USER", "admin"), os.environ.get("ADMIN_PASS", "admin123")


def _save_admin(u, p):
    with open(ADMIN_FILE, "w", encoding="utf-8") as f:
        json.dump({"user": u, "pass": p}, f)
    try:
        os.chmod(ADMIN_FILE, 0o600)
    except OSError:
        pass


def _token_for(u):
    return hmac.new(_SECRET, u.encode(), hashlib.sha256).hexdigest()


ADMIN_USER, ADMIN_PASS = _load_admin()
_TOKEN = _token_for(ADMIN_USER)

app = FastAPI(title="LiDAR3D AutoLabel")
JOBS = {}  # job_id -> {status, log, task_url}


def authed(request: Request) -> bool:
    return request.cookies.get("sess") == _TOKEN


# ---------------------------------------------------------------- pipeline ----
def find_data_dir(root):
    for dirpath, dirnames, _ in os.walk(root):
        if "pointcloud" in dirnames:
            return dirpath
    return None


def run_job(job_id, data_dir, cvat_url, user, password, task_name):
    job = JOBS[job_id]
    logf = os.path.join(JOBS_DIR, job_id, "run.log")
    cmd = [VENV_PY, RUN_PY, "--url", cvat_url, "--user", user,
           "--password", password, "--task-name", task_name, "--data", data_dir]
    try:
        with open(logf, "w") as lf:
            proc = subprocess.Popen(cmd, cwd=os.path.dirname(RUN_PY),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            for line in proc.stdout:
                job["log"] += line
                lf.write(line)
                lf.flush()
                m = re.search(r"https?://\S+/tasks/\d+/jobs/\d+", line)
                if m:
                    job["task_url"] = m.group(0)
            proc.wait()
        job["status"] = "done" if proc.returncode == 0 else "error"
    except Exception as e:  # noqa: BLE001
        job["log"] += f"\n>>> LỖI backend: {e}\n"
        job["status"] = "error"


# ---------------------------------------------------------------- evaluate ----
EVAL_RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_runner.py")


def load_gt_file(path):
    """Nạp GT từ file: JSON trực tiếp, hoặc ZIP export (Datumaro) -> lấy .json lớn nhất."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            jsons = [n for n in z.namelist() if n.endswith(".json")]
            if not jsons:
                raise ValueError("ZIP GT không có file .json")
            jsons.sort(key=lambda n: z.getinfo(n).file_size, reverse=True)
            with z.open(jsons[0]) as f:
                return json.load(f)
    raise ValueError("GT không đọc được (không phải JSON hoặc ZIP)")


def gt_to_pred_format(raw):
    """Chuẩn hoá GT về format tool: [{frame, boxes:[{label,center,size,yaw}]}]."""
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "boxes" in raw[0]:
        return raw
    if isinstance(raw, dict) and "items" in raw:  # Datumaro 3D
        cats = (raw.get("categories", {}) or {}).get("label", {}) or {}
        lmap = {i: l.get("name", str(i)) for i, l in enumerate(cats.get("labels", []))}
        frames = []
        for idx, item in enumerate(raw["items"]):
            boxes = []
            for a in item.get("annotations", []):
                if a.get("type") in ("cuboid_3d", "cuboid"):
                    rot = a.get("rotation", [0, 0, 0])
                    boxes.append({"label": lmap.get(a.get("label_id", -1), str(a.get("label_id"))),
                                  "center": a.get("position", [0, 0, 0]),
                                  "size": a.get("scale", [1, 1, 1]),
                                  "yaw": rot[2] if len(rot) > 2 else 0.0})
            frames.append({"frame": idx, "num_boxes": len(boxes), "boxes": boxes})
        return frames
    raise ValueError("GT không đúng định dạng (cần JSON tool hoặc Datumaro 3D)")


def run_eval_job(job_id, data_dir, gt_path):
    job = JOBS[job_id]
    out_json = os.path.join(JOBS_DIR, job_id, "eval.json")
    cmd = [VENV_PY, EVAL_RUNNER, data_dir, gt_path, out_json]
    try:
        proc = subprocess.Popen(cmd, cwd=os.path.dirname(RUN_PY),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            job["log"] += line
        proc.wait()
        if proc.returncode == 0 and os.path.exists(out_json):
            with open(out_json) as f:
                job["eval"] = json.load(f)
            job["status"] = "done"
        else:
            job["status"] = "error"
    except Exception as e:  # noqa: BLE001
        job["log"] += f"\n>>> LỖI: {e}\n"
        job["status"] = "error"


# ------------------------------------------------------------------- CVAT -----
def cvat_token():
    r = requests.post(f"{CVAT_URL}/api/auth/login",
                      json={"username": CVAT_USER, "password": CVAT_PASS}, timeout=8)
    r.raise_for_status()
    return r.json()["key"]


def cvat_get(path, token, **params):
    r = requests.get(f"{CVAT_URL}{path}", headers={"Authorization": f"Token {token}"},
                     params=params, timeout=10)
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------------ routes ----
@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    if username == ADMIN_USER and password == ADMIN_PASS:
        resp = RedirectResponse("/", status_code=303)
        resp.set_cookie("sess", _TOKEN, httponly=True, samesite="lax")
        return resp
    return RedirectResponse("/login?e=1", status_code=303)


@app.get("/logout")
async def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie("sess")
    return resp


@app.post("/api/change-admin")
def api_change_admin(request: Request, current_password: str = Form(...),
                     new_username: str = Form(...), new_password: str = Form(...)):
    global ADMIN_USER, ADMIN_PASS, _TOKEN
    if not authed(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if current_password != ADMIN_PASS:
        return {"error": "Mật khẩu hiện tại không đúng"}
    nu, np_ = new_username.strip(), new_password
    if not nu or len(np_) < 4:
        return {"error": "Tài khoản trống hoặc mật khẩu < 4 ký tự"}
    ADMIN_USER, ADMIN_PASS = nu, np_
    _save_admin(nu, np_)
    _TOKEN = _token_for(nu)  # token đổi -> phiên cũ hết hạn, phải đăng nhập lại
    return {"ok": True}


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if authed(request):
        return RedirectResponse("/")
    return LOGIN_HTML


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not authed(request):
        return RedirectResponse("/login")
    return APP_HTML


@app.post("/jobs")
def create_job(request: Request, task_name: str = Form(...),
                     cvat_url: str = Form(CVAT_URL), cvat_user: str = Form(CVAT_USER),
                     cvat_pass: str = Form(CVAT_PASS), file: UploadFile = File(...)):
    if not authed(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    job_id = uuid.uuid4().hex[:8]
    jdir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(jdir, exist_ok=True)
    zip_path = os.path.join(jdir, "upload.zip")
    with open(zip_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    extract_dir = os.path.join(jdir, "data")
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
    except zipfile.BadZipFile:
        return JSONResponse({"error": "File không phải .zip hợp lệ"}, status_code=400)
    data_dir = find_data_dir(extract_dir)
    if not data_dir:
        return JSONResponse({"error": "Không tìm thấy 'pointcloud/' trong zip"}, status_code=400)
    JOBS[job_id] = {"status": "running", "log": "", "task_url": None}
    threading.Thread(target=run_job, args=(job_id, data_dir, cvat_url.rstrip("/"),
                     cvat_user, cvat_pass, task_name), daemon=True).start()
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def job_status(request: Request, job_id: str):
    if not authed(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"status": job["status"], "log": job["log"][-6000:], "task_url": job["task_url"]}


@app.get("/api/stats")
def api_stats(request: Request):
    if not authed(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        tok = cvat_token()
        tasks = cvat_get("/api/tasks", tok, page_size=1).get("count", 0)
        jobs = cvat_get("/api/jobs", tok, page_size=1).get("count", 0)
        users = cvat_get("/api/users", tok, page_size=1).get("count", 0)
        return {"ok": True, "tasks": tasks, "jobs": jobs, "users": users}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


@app.get("/api/team")
def api_team(request: Request):
    if not authed(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        tok = cvat_token()
        jobs, page = [], 1
        while True:
            r = cvat_get("/api/jobs", tok, page=page, page_size=100)
            jobs += r.get("results", [])
            if not r.get("next"):
                break
            page += 1
        agg = {}
        for j in jobs:
            a = (j.get("assignee") or {})
            name = a.get("username") or "— chưa giao —"
            d = agg.setdefault(name, {"assigned": 0, "completed": 0, "in_progress": 0, "last": ""})
            d["assigned"] += 1
            st = j.get("state", "")
            if st == "completed":
                d["completed"] += 1
            elif st == "in progress":
                d["in_progress"] += 1
            upd = j.get("updated_date", "") or ""
            if upd > d["last"]:
                d["last"] = upd
        rows = [{"user": k, **v} for k, v in sorted(agg.items())]
        return {"ok": True, "rows": rows}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def list_datasets():
    out = []
    base = os.path.join(HOME, "data_day14")
    if os.path.isdir(os.path.join(base, "pointcloud")):
        out.append({"id": "data_day14", "label": "data_day14 (mẫu)", "dir": base})
    try:
        for d in sorted(os.listdir(JOBS_DIR), reverse=True):
            jd = os.path.join(JOBS_DIR, d, "data")
            dd = find_data_dir(jd) if os.path.isdir(jd) else None
            if dd:
                n = len([f for f in os.listdir(os.path.join(dd, "pointcloud")) if f.endswith(".pcd")])
                out.append({"id": d, "label": f"{d} ({n} frames)", "dir": dd})
    except OSError:
        pass
    return out[:20]


@app.get("/api/datasets")
def api_datasets(request: Request):
    if not authed(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return {"datasets": [{"id": x["id"], "label": x["label"]} for x in list_datasets()]}


@app.post("/api/evaluate")
def api_evaluate(request: Request, gt: UploadFile = File(...),
                 data: UploadFile = File(None), dataset: str = Form(None)):
    if not authed(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    job_id = "ev" + uuid.uuid4().hex[:6]
    jdir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(jdir, exist_ok=True)
    if dataset:  # dùng data có sẵn trên server (không phải upload lại)
        ds = {x["id"]: x["dir"] for x in list_datasets()}
        data_dir = ds.get(dataset)
        if not data_dir:
            return JSONResponse({"error": "Dataset không hợp lệ"}, status_code=400)
    elif data is not None and data.filename:
        dz = os.path.join(jdir, "data.zip")
        with open(dz, "wb") as f:
            shutil.copyfileobj(data.file, f)
        try:
            with zipfile.ZipFile(dz) as z:
                z.extractall(os.path.join(jdir, "data"))
        except zipfile.BadZipFile:
            return JSONResponse({"error": "Data không phải .zip hợp lệ"}, status_code=400)
        data_dir = find_data_dir(os.path.join(jdir, "data"))
        if not data_dir:
            return JSONResponse({"error": "Không thấy pointcloud/ trong data.zip"}, status_code=400)
    else:
        return JSONResponse({"error": "Chọn dataset có sẵn hoặc upload data"}, status_code=400)
    gt_raw = os.path.join(jdir, "gt_raw")
    with open(gt_raw, "wb") as f:
        shutil.copyfileobj(gt.file, f)
    try:
        conv = gt_to_pred_format(load_gt_file(gt_raw))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"GT lỗi: {e}"}, status_code=400)
    gt_path = os.path.join(jdir, "gt.json")
    with open(gt_path, "w") as f:
        json.dump(conv, f)
    JOBS[job_id] = {"status": "running", "log": "", "eval": None}
    threading.Thread(target=run_eval_job, args=(job_id, data_dir, gt_path), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/evaluate/{job_id}")
def api_evaluate_status(request: Request, job_id: str):
    if not authed(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"status": job["status"], "log": job["log"][-4000:], "eval": job.get("eval")}


# ------------------------------------------------- accounts & assignment ------
CVAT_DIR = os.path.join(HOME, "cvat")
_SAFE_USER = re.compile(r"^[A-Za-z0-9_.-]+$")


def create_cvat_users(names, password):
    """Tạo tài khoản annotator trong CVAT qua Django shell (active ngay)."""
    script = (
        "from django.contrib.auth import get_user_model\n"
        "U=get_user_model()\n"
        f"for n in {json.dumps(names)}:\n"
        "    u,c=U.objects.get_or_create(username=n, defaults={'email':n+'@local'})\n"
        f"    u.set_password({json.dumps(password)}); u.is_active=True; u.save()\n"
        "    print(n, 'created' if c else 'exists')\n"
    )
    cmd = ["sudo", "docker", "compose", "exec", "-T", "cvat_server",
           "python3", "/home/django/manage.py", "shell", "-c", script]
    r = subprocess.run(cmd, cwd=CVAT_DIR, capture_output=True, text=True, timeout=150)
    return (r.stdout + r.stderr).strip()


@app.post("/api/accounts")
def api_accounts(request: Request, usernames: str = Form(...),
                 password: str = Form("Annotator123")):
    if not authed(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    names = [x.strip() for x in usernames.replace(",", "\n").splitlines() if x.strip()]
    bad = [n for n in names if not _SAFE_USER.match(n)]
    if bad:
        return JSONResponse({"error": f"Username không hợp lệ (chỉ A-Z,0-9,_,.,-): {bad}"}, status_code=400)
    if not names:
        return JSONResponse({"error": "Chưa nhập tài khoản"}, status_code=400)
    try:
        out = create_cvat_users(names, password)
        return {"ok": True, "result": out}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/assign")
def api_assign(request: Request, task_id: int = Form(...), usernames: str = Form(...)):
    if not authed(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    names = [x.strip() for x in usernames.replace(",", "\n").splitlines() if x.strip()]
    if not names:
        return JSONResponse({"error": "Chưa nhập annotator"}, status_code=400)
    try:
        tok = cvat_token()
        uids = []
        for n in names:
            res = cvat_get("/api/users", tok, search=n).get("results", [])
            m = [u for u in res if u["username"] == n]
            if m:
                uids.append((n, m[0]["id"]))
        if not uids:
            return {"ok": False, "error": "Không thấy tài khoản nào (tạo ở mục ① trước)"}
        jobs = cvat_get("/api/jobs", tok, task_id=task_id).get("results", [])
        if not jobs:
            return {"ok": False, "error": f"Task {task_id} không có job"}
        assigns = []
        for i, j in enumerate(jobs):
            n, uid = uids[i % len(uids)]
            requests.patch(f"{CVAT_URL}/api/jobs/{j['id']}",
                           headers={"Authorization": f"Token {tok}"},
                           json={"assignee": uid}, timeout=10)
            assigns.append({"job": j["id"], "user": n})
        return {"ok": True, "assigns": assigns, "task_url": f"{CVAT_URL}/tasks/{task_id}"}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=500)


# -------------------------------------------------------------------- HTML -----
_STYLE = """
:root{--bg:#070b16;--bg2:#0d1424;--panel:#111a2e;--panel2:#16213a;--line:#233249;
--ink:#e7edf7;--muted:#8394b4;--cyan:#22d3ee;--indigo:#6366f1;--ok:#34d399;--err:#f87171;--warn:#fbbf24}
*{box-sizing:border-box}
body{margin:0;font-family:Inter,system-ui,Segoe UI,sans-serif;background:var(--bg);color:var(--ink)}
a{color:var(--cyan);text-decoration:none}
.grad{background:linear-gradient(120deg,var(--cyan),var(--indigo));-webkit-background-clip:text;background-clip:text;color:transparent}
.btn{border:0;border-radius:10px;padding:11px 18px;font-weight:700;font-size:.95rem;cursor:pointer;
background:linear-gradient(120deg,var(--cyan),var(--indigo));color:#05101c}
.btn:disabled{opacity:.5;cursor:not-allowed}
.btn.ghost{background:#1b2740;color:var(--ink);border:1px solid var(--line)}
input,select{width:100%;padding:11px 13px;border-radius:10px;border:1px solid var(--line);
background:#0b1223;color:var(--ink);font-size:.95rem}
label{display:block;font-size:.8rem;color:var(--muted);margin:14px 0 5px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:22px}
.muted{color:var(--muted)} .mono{font-family:ui-monospace,Menlo,monospace}
"""

LOGIN_HTML = """<!doctype html><html lang=vi><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>LiDAR3D AutoLabel</title>
<link rel=preconnect href=https://fonts.googleapis.com>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel=stylesheet>
<style>__STYLE__
#bg{position:fixed;inset:0;z-index:0}
.wrap{position:relative;z-index:1;min-height:100vh;display:grid;place-items:center;padding:20px}
.lg{width:380px;max-width:100%;background:rgba(17,26,46,.82);backdrop-filter:blur(10px);
border:1px solid var(--line);border-radius:20px;padding:34px}
.logo{font-weight:800;font-size:1.5rem;letter-spacing:-.5px}.dot{color:var(--cyan)}
.tag{color:var(--muted);font-size:.85rem;margin:6px 0 22px}
.err{color:var(--err);font-size:.82rem;margin-top:10px}
</style></head><body>
<canvas id=bg></canvas>
<div class=wrap><form class=lg method=post action=/login>
<div class=logo>LiDAR<span class=dot>3D</span> <span class=grad>AutoLabel</span></div>
<div class=tag>Nền tảng tự động gán nhãn 3D cuboid từ LiDAR · Admin</div>
<label>Tài khoản</label><input name=username value=admin autofocus>
<label>Mật khẩu</label><input name=password type=password placeholder="••••••••">
<div style=margin-top:22px><button class=btn style=width:100%>Đăng nhập</button></div>
<div class=err id=err></div>
</form></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>__CLOUD__
if(location.search.indexOf('e=1')>=0)document.getElementById('err').textContent='Sai tài khoản hoặc mật khẩu';
</script></body></html>""".replace("__STYLE__", _STYLE)

APP_HTML = """<!doctype html><html lang=vi><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>LiDAR3D AutoLabel</title>
<link rel=preconnect href=https://fonts.googleapis.com>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel=stylesheet>
<style>__STYLE__
.app{display:grid;grid-template-columns:230px 1fr;min-height:100vh}
.side{background:var(--bg2);border-right:1px solid var(--line);padding:20px 14px;position:sticky;top:0;height:100vh}
.logo{font-weight:800;font-size:1.15rem;padding:6px 10px 18px}.dot{color:var(--cyan)}
.nav{display:flex;align-items:center;gap:11px;padding:11px 12px;border-radius:11px;color:var(--muted);
cursor:pointer;font-weight:600;font-size:.92rem}
.nav:hover{background:#15203a;color:var(--ink)} .nav.on{background:#17233f;color:var(--ink)}
.nav .ic{width:20px;text-align:center}
.main{padding:26px 30px;max-width:1100px}
.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:22px}
.h1{font-size:1.5rem;font-weight:800;margin:0}
.sec{display:none} .sec.on{display:block}
.hero{position:relative;height:260px;border-radius:18px;overflow:hidden;border:1px solid var(--line);margin-bottom:22px}
.hero canvas{position:absolute;inset:0}
.hero .ov{position:absolute;inset:0;display:flex;flex-direction:column;justify-content:center;padding:30px;
background:linear-gradient(90deg,rgba(7,11,22,.85),rgba(7,11,22,.1))}
.hero h2{margin:0;font-size:1.7rem;font-weight:800;max-width:560px}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:22px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px}
.tile .n{font-size:1.9rem;font-weight:800} .tile .l{color:var(--muted);font-size:.82rem;margin-top:3px}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
#log{background:#060b16;border:1px solid var(--line);border-radius:11px;padding:12px;font-family:ui-monospace,monospace;
font-size:.76rem;white-space:pre-wrap;max-height:300px;overflow:auto;color:#aebdd8}
.badge{display:inline-block;padding:3px 11px;border-radius:999px;font-size:.76rem;font-weight:700}
.b-run{background:#15314f;color:var(--cyan)}.b-ok{background:#064e3b;color:var(--ok)}.b-err{background:#4c1d1d;color:var(--err)}
table{width:100%;border-collapse:collapse;font-size:.88rem}
th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600}
.bar{height:9px;background:#1b2740;border-radius:5px;overflow:hidden}
.bar>i{display:block;height:100%;background:linear-gradient(90deg,var(--cyan),var(--indigo));transition:width .2s}
@keyframes shim{0%{background-position:-220px 0}100%{background-position:220px 0}}
.bar.proc>i{width:100%!important;background:linear-gradient(90deg,#1b2740,var(--cyan),#1b2740);background-size:220px 100%;animation:shim 1.1s infinite linear}
.pstat{font-size:.84rem;color:var(--cyan);margin:10px 0 6px}
.result{display:inline-block;margin-top:12px;padding:10px 16px;border-radius:10px;background:var(--ok);color:#05261c;font-weight:700}
.hidden{display:none}.soon{color:var(--warn);font-size:.82rem}
details summary{cursor:pointer;color:var(--cyan);font-size:.85rem;margin-top:8px}
</style></head><body>
<div class=app>
 <aside class=side>
  <div class=logo>LiDAR<span class=dot>3D</span></div>
  <div class=nav on data-s=dash><span class=ic>📊</span>Dashboard</div>
  <div class=nav data-s=auto><span class=ic>🧊</span>Auto-label</div>
  <div class=nav data-s=assign><span class=ic>👥</span>Phân công</div>
  <div class=nav data-s=team><span class=ic>📋</span>Nhóm</div>
  <div class=nav data-s=eval><span class=ic>🎯</span>Đánh giá</div>
  <div style="position:absolute;bottom:18px"><a href=/logout class=muted>↩ Đăng xuất</a></div>
 </aside>
 <main class=main>
  <div class=top><h1 class=h1 id=title>Dashboard</h1>
   <a href="__CVAT__" target=_blank class="btn ghost" style=padding:8px:14px>Mở CVAT ↗</a></div>

  <section class="sec on" id=dash>
   <div class=hero><canvas id=hc></canvas><div class=ov>
     <h2>Tự động gán nhãn <span class=grad>3D cuboid</span> từ LiDAR point cloud</h2>
     <p class=muted style=max-width:520px>TransFusion trên GPU · chia việc cho annotator · đánh giá chất lượng — tất cả trong một nơi.</p>
   </div></div>
   <div class=tiles id=tiles>
     <div class=tile><div class=n id=t_tasks>—</div><div class=l>Task đã tạo</div></div>
     <div class=tile><div class=n id=t_jobs>—</div><div class=l>Job (phần việc)</div></div>
     <div class=tile><div class=n id=t_users>—</div><div class=l>Tài khoản CVAT</div></div>
     <div class=tile><div class=n id=t_gpu>T4</div><div class=l>GPU inference</div></div>
   </div>
   <div class=card><b>Luồng làm việc</b><p class=muted style=margin:10px:0>
     1️⃣ Upload point cloud → 2️⃣ Auto-label (GPU) → 3️⃣ Phân công annotator → 4️⃣ Đánh giá chất lượng.</p></div>
  </section>

  <section class=sec id=auto>
   <div class=card style=max-width:640px>
    <form id=f>
     <label>Tên task</label><input name=task_name value="Auto Label 3D" required>
     <label>File .zip (chứa <b>pointcloud/*.pcd</b>, kèm context_images/ nếu có)</label>
     <input type=file name=file accept=.zip required>
     <details><summary>Tùy chọn CVAT (mặc định: server này)</summary>
       <label>CVAT URL</label><input name=cvat_url>
       <div class=row2><div><label>User</label><input name=cvat_user value=admin></div>
       <div><label>Password</label><input name=cvat_pass type=password value=Cvat123456></div></div>
     </details>
     <div style=margin-top:18px><button class=btn id=btn>Bắt đầu auto-label</button></div>
    </form>
   </div>
   <div class="card hidden" id=prog style=max-width:640px;margin-top:14px>
    <div>Trạng thái: <span id=badge class="badge b-run">Đang chạy…</span></div>
    <div class=pstat id=pstat></div>
    <div class=bar id=pbarwrap><i id=pbar style=width:0%></i></div>
    <a id=rlink class="result hidden" target=_blank>Mở task trong CVAT →</a>
    <label>Nhật ký</label><div id=log></div>
   </div>
  </section>

  <section class=sec id=assign>
   <div class=card style=max-width:680px>
    <b>① Tạo tài khoản annotator</b>
    <p class=muted style=font-size:.84rem>Tạo sẵn tài khoản trong CVAT để annotator đăng nhập (http://IP:8080) làm việc được giao.</p>
    <label>Username (mỗi dòng 1, hoặc cách dấu phẩy)</label>
    <textarea id=ac_users rows=4 style="width:100%;border-radius:10px;border:1px solid var(--line);background:#0b1223;color:var(--ink);padding:11px">annotator1
annotator2</textarea>
    <label>Mật khẩu mặc định (chung)</label><input id=ac_pass value="Annotator123">
    <div style=margin-top:12px><button class=btn id=ac_btn>Tạo tài khoản</button></div>
    <pre id=ac_result class=mono style="margin-top:10px;color:#aebdd8;font-size:.78rem;white-space:pre-wrap"></pre>
   </div>
   <div class=card style=max-width:680px;margin-top:14px>
    <b>② Chia việc cho annotator</b>
    <p class=muted style=font-size:.84rem>Nhập Task ID (task đã auto-label) + danh sách annotator → mỗi job gán 1 người luân phiên.</p>
    <label>Task ID</label><input id=as_task placeholder="vd: 15">
    <label>Annotator (mỗi dòng 1 username đã tạo)</label>
    <textarea id=as_users rows=4 style="width:100%;border-radius:10px;border:1px solid var(--line);background:#0b1223;color:var(--ink);padding:11px">annotator1
annotator2</textarea>
    <div style=margin-top:12px><button class=btn id=as_btn>Phân công</button></div>
    <pre id=as_result class=mono style="margin-top:10px;color:#aebdd8;font-size:.78rem;white-space:pre-wrap"></pre>
   </div>
  </section>

  <section class=sec id=team>
   <div class=card>
    <div style=display:flex;justify-content:space-between;align-items:center>
      <b>Trạng thái nhóm</b><button class="btn ghost" id=team_refresh style=padding:7px:13px>↻ Làm mới</button></div>
    <table style=margin-top:12px><thead><tr><th>Tài khoản</th><th>Được giao</th><th>Đang làm</th><th>Xong</th><th>Tiến độ</th><th>Hoạt động cuối</th></tr></thead>
    <tbody id=team_body><tr><td colspan=6 class=muted>Bấm "Làm mới" để tải (cần VM đang chạy)…</td></tr></tbody></table>
   </div>
  </section>

  <section class=sec id=eval>
   <div class=card style=max-width:680px>
    <b>Đánh giá chất lượng TOOL</b> <span class=muted>(so tool vs ground truth)</span>
    <p class=muted style=font-size:.86rem>Đẩy <b>2 luồng</b>: data để tool tự gán + ground truth (đáp án). Tool chạy inference rồi chấm: % Giữ/Sửa/Xóa/Bổ sung, 3D IoU, heading, theo class.</p>
    <label>1️⃣ Data point cloud — chọn data <b>có sẵn trên server</b> (nhanh) hoặc upload mới</label>
    <select id=ev_dataset></select>
    <div id=ev_dataup style="margin-top:8px;display:none"><input type=file id=ev_data accept=.zip></div>
    <label>2️⃣ Ground truth — <b>Datumaro 3D</b> (.json hoặc .zip export từ CVAT), hoặc JSON của tool</label>
    <input type=file id=ev_gt accept=".json,.zip">
    <div style=margin-top:16px><button class=btn id=ev_btn>Chạy đánh giá</button></div>
   </div>
   <div class="card hidden" id=ev_prog style=max-width:680px;margin-top:14px>
    <div>Trạng thái: <span id=ev_badge class="badge b-run">Đang chạy…</span></div>
    <div class=pstat id=ev_pstat></div>
    <div class=bar id=ev_pbarwrap><i id=ev_pbar style=width:0%></i></div>
    <div id=ev_result style=margin-top:14px></div>
    <details style=margin-top:10px><summary>Nhật ký</summary>
      <div id=ev_log class=mono style="background:#060b16;border:1px solid var(--line);border-radius:9px;padding:10px;font-size:.74rem;white-space:pre-wrap;max-height:220px;overflow:auto;color:#aebdd8;margin-top:8px"></div>
    </details>
   </div>
  </section>
 </main>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const CVAT="__CVAT__";
// nav
document.querySelectorAll('.nav').forEach(n=>n.onclick=()=>{
  document.querySelectorAll('.nav').forEach(x=>x.classList.remove('on'));n.classList.add('on');
  const s=n.dataset.s;document.querySelectorAll('.sec').forEach(x=>x.classList.toggle('on',x.id===s));
  document.getElementById('title').textContent=n.textContent.trim();
  if(s==='team')loadTeam();
});
// default cvat url = this host
document.querySelector('input[name=cvat_url]').value=location.protocol+'//'+location.hostname+':8080';
// stats
fetch('/api/stats').then(r=>r.json()).then(d=>{if(d.ok){t_tasks.textContent=d.tasks;t_jobs.textContent=d.jobs;t_users.textContent=d.users;}});
// ===== helpers tien trinh =====
function uploadXHR(url,fd,onprog){return new Promise((res,rej)=>{
  const x=new XMLHttpRequest();x.open('POST',url);
  x.upload.onprogress=e=>{if(e.lengthComputable)onprog(Math.round(e.loaded/e.total*100));};
  x.onload=()=>{try{res(JSON.parse(x.responseText));}catch(err){rej(err);}};
  x.onerror=()=>rej(new Error('net'));x.send(fd);});}
function setBar(pre,pct,text,proc){
  const s=document.getElementById(pre+'stat'),w=document.getElementById(pre+'barwrap'),b=document.getElementById(pre+'bar');
  if(s)s.textContent=text||'';
  if(proc){w.classList.add('proc');}else{w.classList.remove('proc');b.style.width=(pct||0)+'%';}}
function lastLog(log){const ls=(log||'').split('\\n').filter(l=>l.trim()&&!/Warning|custom_|backward|meshgrid|_VF|FutureWarning|Loading param/.test(l));return ls.length?ls[ls.length-1].trim():'';}
// ===== auto-label =====
const f=document.getElementById('f'),btn=document.getElementById('btn');
f.onsubmit=e=>{e.preventDefault();btn.disabled=true;btn.textContent='Đang chạy…';
  document.getElementById('prog').classList.remove('hidden');document.getElementById('rlink').classList.add('hidden');
  setBar('p',0,'Chuẩn bị tải lên…',false);
  uploadXHR('/jobs',new FormData(f),p=>setBar('p',p,'📤 Đang tải lên… '+p+'%',false))
   .then(d=>{if(d.error){alert(d.error);btn.disabled=false;btn.textContent='Bắt đầu auto-label';return;}
     setBar('p',100,'⚙️ Đang xử lý trên GPU…',true);poll(d.job_id);})
   .catch(()=>{alert('Lỗi mạng khi tải lên');btn.disabled=false;btn.textContent='Bắt đầu auto-label';});};
function fixU(u){try{return u.replace('localhost',location.hostname)}catch(e){return u}}
async function poll(id){const j=await(await fetch('/jobs/'+id)).json();
  log.textContent=j.log||'(đang khởi động…)';log.scrollTop=log.scrollHeight;
  if(j.task_url){rlink.href=fixU(j.task_url);rlink.classList.remove('hidden');}
  if(j.status==='running'){setBar('p',100,'⚙️ '+(lastLog(j.log)||'Đang xử lý trên GPU…'),true);setTimeout(()=>poll(id),2000);return;}
  btn.disabled=false;btn.textContent='Bắt đầu auto-label';
  if(j.status==='done'){badge.className='badge b-ok';badge.textContent='Hoàn tất ✓';setBar('p',100,'✓ Xong',false);}
  else{badge.className='badge b-err';badge.textContent='Lỗi ✕';setBar('p',100,'✕ Lỗi — xem nhật ký',false);}}
// team
async function loadTeam(){const b=document.getElementById('team_body');b.innerHTML='<tr><td colspan=6 class=muted>Đang tải…</td></tr>';
  const d=await(await fetch('/api/team')).json();
  if(!d.ok){b.innerHTML='<tr><td colspan=6 class=muted>Chưa tải được (VM/CVAT chưa sẵn sàng): '+(d.error||'')+'</td></tr>';return;}
  if(!d.rows.length){b.innerHTML='<tr><td colspan=6 class=muted>Chưa có job nào.</td></tr>';return;}
  b.innerHTML=d.rows.map(r=>{const p=r.assigned?Math.round(r.completed/r.assigned*100):0;
    return `<tr><td>${r.user}</td><td>${r.assigned}</td><td>${r.in_progress}</td><td>${r.completed}</td>
    <td><div class=bar><i style=width:${p}%></i></div><span class=muted style=font-size:.75rem>${p}%</span></td>
    <td class=muted>${(r.last||'').slice(0,16).replace('T',' ')}</td></tr>`}).join('');}
document.getElementById('team_refresh').onclick=loadTeam;
// tao tai khoan annotator
document.getElementById('ac_btn').onclick=async()=>{
  const b=document.getElementById('ac_btn'), out=document.getElementById('ac_result');
  const fd=new FormData();fd.append('usernames',document.getElementById('ac_users').value);fd.append('password',document.getElementById('ac_pass').value);
  b.disabled=true;b.textContent='Đang tạo…';out.textContent='';
  const j=await(await fetch('/api/accounts',{method:'POST',body:fd})).json();
  b.disabled=false;b.textContent='Tạo tài khoản';
  out.textContent=j.error?('✕ Lỗi: '+j.error):('✓ Xong:\\n'+(j.result||''));
};
// phan cong job
document.getElementById('as_btn').onclick=async()=>{
  const b=document.getElementById('as_btn'), out=document.getElementById('as_result');
  const fd=new FormData();fd.append('task_id',document.getElementById('as_task').value);fd.append('usernames',document.getElementById('as_users').value);
  b.disabled=true;b.textContent='Đang phân công…';out.textContent='';
  const j=await(await fetch('/api/assign',{method:'POST',body:fd})).json();
  b.disabled=false;b.textContent='Phân công';
  if(j.error||!j.ok){out.textContent='✕ Lỗi: '+(j.error||'');return;}
  out.textContent='✓ Đã giao '+j.assigns.length+' job:\\n'+j.assigns.map(a=>'  job '+a.job+' → '+a.user).join('\\n');
};
// ===== Đánh giá =====
const evBtn=document.getElementById('ev_btn'), evSel=document.getElementById('ev_dataset');
function updDataup(){document.getElementById('ev_dataup').style.display=evSel.value?'none':'block';}
fetch('/api/datasets').then(r=>r.json()).then(d=>{
  const ds=d.datasets||[];
  evSel.innerHTML=ds.map(x=>`<option value="${x.id}">${x.label}</option>`).join('')+'<option value="">— Upload data mới —</option>';
  updDataup();
});
evSel.onchange=updDataup;
evBtn.onclick=()=>{
  const g=document.getElementById('ev_gt').files[0];
  if(!g){alert('Chọn file Ground truth');return;}
  const fd=new FormData();fd.append('gt',g);
  if(evSel.value){fd.append('dataset',evSel.value);}
  else{const d=document.getElementById('ev_data').files[0];if(!d){alert('Chọn data có sẵn hoặc upload data');return;}fd.append('data',d);}
  evBtn.disabled=true;evBtn.textContent='Đang chạy…';
  document.getElementById('ev_prog').classList.remove('hidden');document.getElementById('ev_result').innerHTML='';
  setBar('ev_p',0,'Chuẩn bị tải lên…',false);
  uploadXHR('/api/evaluate',fd,p=>setBar('ev_p',p,'📤 Đang tải GT lên… '+p+'%',false))
   .then(j=>{if(j.error){alert(j.error);evBtn.disabled=false;evBtn.textContent='Chạy đánh giá';return;}
     setBar('ev_p',100,'⚙️ Đang chấm điểm (inference + so sánh)…',true);pollEval(j.job_id);})
   .catch(()=>{alert('Lỗi mạng');evBtn.disabled=false;evBtn.textContent='Chạy đánh giá';});
};
async function pollEval(id){
  const j=await(await fetch('/api/evaluate/'+id)).json();
  document.getElementById('ev_log').textContent=j.log||'(đang khởi động…)';
  if(j.status==='running'){setBar('ev_p',100,'⚙️ '+(lastLog(j.log)||'Đang chấm điểm…'),true);setTimeout(()=>pollEval(id),2000);return;}
  const b=document.getElementById('ev_badge');evBtn.disabled=false;evBtn.textContent='Chạy đánh giá';
  if(j.status==='done'&&j.eval){b.className='badge b-ok';b.textContent='Hoàn tất ✓';setBar('ev_p',100,'✓ Xong',false);renderEval(j.eval);}
  else{b.className='badge b-err';b.textContent='Lỗi ✕';setBar('ev_p',100,'✕ Lỗi — xem nhật ký',false);}
}
function evbar(label,val,total,color){const p=total?Math.round(val/total*100):0;
  return `<div style=margin:7px:0><div style="display:flex;justify-content:space-between;font-size:.8rem"><span>${label}</span><span class=muted>${val} (${p}%)</span></div><div class=bar><i style="width:${p}%;background:${color}"></i></div></div>`;}
function renderEval(m){
  const tot=(m.accept+m.edit+m.reject+m.miss)||1;
  const acc=m.total_gt?Math.round((m.accept+m.edit)/m.total_gt*100):0;
  let h=`<div class=tiles>
   <div class=tile><div class=n>${(m.mean_iou*100).toFixed(0)}%</div><div class=l>IoU trung bình</div></div>
   <div class=tile><div class=n>${acc}%</div><div class=l>Bắt đúng GT</div></div>
   <div class=tile><div class=n>${m.total_pred}</div><div class=l>Box tool gán</div></div>
   <div class=tile><div class=n>${(m.mean_heading_error_deg||0).toFixed(1)}°</div><div class=l>Lỗi hướng TB</div></div>
  </div>
  <div class=card><b>Chất lượng pre-label (so với GT)</b>
   ${evbar('✅ Giữ nguyên (đúng, IoU≥0.7)',m.accept,tot,'#34d399')}
   ${evbar('✏️ Phải sửa (lệch, 0.25≤IoU<0.7)',m.edit,tot,'#fbbf24')}
   ${evbar('❌ Thừa/sai (xóa)',m.reject,tot,'#f87171')}
   ${evbar('➕ Bỏ sót (bổ sung)',m.miss,tot,'#60a5fa')}
  </div>`;
  const cls=Object.entries(m.per_class||{}).filter(([k,v])=>v.total_gt||v.total_pred);
  if(cls.length){h+=`<div class=card style=margin-top:12px><b>Theo class</b><table style=margin-top:8px><thead><tr><th>Class</th><th>GT</th><th>Giữ</th><th>Sửa</th><th>Sót</th><th>IoU</th></tr></thead><tbody>`;
   for(const [k,v] of cls){h+=`<tr><td>${k}</td><td>${v.total_gt}</td><td>${v.accept}</td><td>${v.edit}</td><td>${v.miss}</td><td>${(v.mean_iou*100).toFixed(0)}%</td></tr>`;}
   h+='</tbody></table></div>';}
  document.getElementById('ev_result').innerHTML=h;
}
__CLOUD__
</script></body></html>""".replace("__STYLE__", _STYLE).replace("__CVAT__", CVAT_URL)


# three.js: điểm LiDAR giả + box wireframe, camera orbit — dùng cho cả login & hero
_CLOUD_JS = """
(function(){
 const cv=document.getElementById('hc')||document.getElementById('bg'); if(!cv||!window.THREE)return;
 const hero=cv.id==='hc';
 const W=()=>cv.clientWidth||window.innerWidth, H=()=>cv.clientHeight||window.innerHeight;
 const rn=new THREE.WebGLRenderer({canvas:cv,antialias:true,alpha:true});
 rn.setPixelRatio(Math.min(devicePixelRatio,2)); rn.setSize(W(),H(),false);
 const sc=new THREE.Scene();
 const cam=new THREE.PerspectiveCamera(60,W()/H(),.1,1000); cam.position.set(0,14,26);
 // điểm: mặt đất + cụm vật thể
 const N=5000,pos=new Float32Array(N*3),col=new Float32Array(N*3);
 for(let i=0;i<N;i++){let x,z;const r=Math.random();
   if(r<.75){const a=Math.random()*6.28,d=Math.random()*34;x=Math.cos(a)*d;z=Math.sin(a)*d;}
   else{const c=Math.floor(Math.random()*6),cx=[-14,10,-6,16,0,-18][c],cz=[-10,8,14,-14,-20,2][c];
        x=cx+(Math.random()-.5)*5;z=cz+(Math.random()-.5)*5;}
   const y=(r<.75)?Math.random()*.4:Math.random()*4;
   pos[i*3]=x;pos[i*3+1]=y;pos[i*3+2]=z;
   const t=Math.min(1,y/4);col[i*3]=.1+t*.9;col[i*3+1]=.8;col[i*3+2]=1-t*.3;}
 const g=new THREE.BufferGeometry();
 g.setAttribute('position',new THREE.BufferAttribute(pos,3));
 g.setAttribute('color',new THREE.BufferAttribute(col,3));
 sc.add(new THREE.Points(g,new THREE.PointsMaterial({size:.14,vertexColors:true,transparent:true,opacity:.95})));
 // box wireframe (cuboid phát hiện)
 const boxes=[[-14,-10,4,2,3.5],[10,8,4.6,2,2],[-6,14,4,1.8,3],[16,-14,4.2,2,2],[0,-20,4,2,3],[-18,2,3.6,1.8,2.4]];
 boxes.forEach(b=>{const m=new THREE.Mesh(new THREE.BoxGeometry(b[2],b[4],b[3]),0);
   const e=new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry),
     new THREE.LineBasicMaterial({color:0x22d3ee}));
   e.position.set(b[0],b[4]/2,b[1]);sc.add(e);});
 let a=0;(function loop(){a+=.0016;cam.position.x=Math.sin(a)*28;cam.position.z=Math.cos(a)*28;
   cam.lookAt(0,2,0);rn.render(sc,cam);requestAnimationFrame(loop);})();
 addEventListener('resize',()=>{rn.setSize(W(),H(),false);cam.aspect=W()/H();cam.updateProjectionMatrix();});
})();
"""

LOGIN_HTML = LOGIN_HTML.replace("__CLOUD__", _CLOUD_JS)
APP_HTML = APP_HTML.replace("__CLOUD__", _CLOUD_JS)

# === Giao diện mới (Perception Lab) nạp từ file nếu có — ghi đè bản cũ ở trên ===
_HERE = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(_HERE, "login.html"), encoding="utf-8") as _f:
        LOGIN_HTML = _f.read()
    with open(os.path.join(_HERE, "app.html"), encoding="utf-8") as _f:
        APP_HTML = _f.read()
except OSError:
    pass
