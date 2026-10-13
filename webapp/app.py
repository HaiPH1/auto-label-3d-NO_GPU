"""
LiDAR3D AutoLabel — admin portal.
Chạy trên VM (cùng chỗ run.py + venv + model):
  login admin -> upload point cloud -> auto-label (GPU) -> phân công annotator -> đánh giá.

Chạy (systemd service autolabel-web, hoặc tay):
  uvicorn app:app --host 0.0.0.0 --port 8000   (trong thư mục webapp/)
"""
import hashlib
import hmac
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

# Đăng nhập admin của portal
ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("ADMIN_PASS", "admin123")
_SECRET = secrets.token_bytes(32)
_TOKEN = hmac.new(_SECRET, ADMIN_USER.encode(), hashlib.sha256).hexdigest()

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
    cmd = [VENV_PY, RUN_PY, "--url", cvat_url, "--user", user,
           "--password", password, "--task-name", task_name, "--data", data_dir]
    try:
        proc = subprocess.Popen(cmd, cwd=os.path.dirname(RUN_PY),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            job["log"] += line
            m = re.search(r"https?://\S+/tasks/\d+/jobs/\d+", line)
            if m:
                job["task_url"] = m.group(0)
        proc.wait()
        job["status"] = "done" if proc.returncode == 0 else "error"
    except Exception as e:  # noqa: BLE001
        job["log"] += f"\n>>> LỖI backend: {e}\n"
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
async def create_job(request: Request, task_name: str = Form(...),
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
async def api_stats(request: Request):
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
async def api_team(request: Request):
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
.bar{height:7px;background:#1b2740;border-radius:5px;overflow:hidden}
.bar>i{display:block;height:100%;background:linear-gradient(90deg,var(--cyan),var(--indigo))}
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
    <a id=rlink class="result hidden" target=_blank>Mở task trong CVAT →</a>
    <label>Nhật ký</label><div id=log></div>
   </div>
  </section>

  <section class=sec id=assign>
   <div class=card style=max-width:640px>
    <b>Chia việc cho annotator</b>
    <p class=muted style=font-size:.86rem>Nhập ID task + danh sách tài khoản → mỗi job gán cho 1 người (luân phiên).</p>
    <label>Task ID</label><input id=as_task placeholder="vd: 9">
    <label>Danh sách tài khoản (mỗi dòng 1 username CVAT)</label>
    <textarea id=as_users rows=5 style="width:100%;border-radius:10px;border:1px solid var(--line);background:#0b1223;color:var(--ink);padding:11px" placeholder="annotator1&#10;annotator2"></textarea>
    <div style=margin-top:16px><button class="btn" id=as_btn>Phân công</button></div>
    <p class=soon id=as_note style=margin-top:12px>⏳ Backend phân công sẽ bật khi VM chạy (gọi CVAT job assignment API).</p>
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
   <div class=card style=max-width:640px>
    <b>Đánh giá chất lượng (Ground Truth)</b>
    <p class=muted style=font-size:.86rem>Upload tập nhãn chuẩn → so với pre-label: 3D IoU, Accept/Edit/Reject/Miss, heading error, theo class.</p>
    <label>File ground truth (.json)</label><input type=file accept=.json>
    <div style=margin-top:16px><button class=btn>Chạy đánh giá</button></div>
    <p class=soon style=margin-top:12px>⏳ Tái dùng <span class=mono>run.py --evaluate</span> + vẽ chart — bật khi VM chạy.</p>
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
// upload
const f=document.getElementById('f'),btn=document.getElementById('btn');
f.onsubmit=async e=>{e.preventDefault();btn.disabled=true;btn.textContent='Đang tải lên…';
  const r=await fetch('/jobs',{method:'POST',body:new FormData(f)});const d=await r.json();
  if(d.error){alert(d.error);btn.disabled=false;btn.textContent='Bắt đầu auto-label';return;}
  document.getElementById('prog').classList.remove('hidden');poll(d.job_id);};
function fixU(u){try{return u.replace('localhost',location.hostname)}catch(e){return u}}
async function poll(id){const j=await(await fetch('/jobs/'+id)).json();
  log.textContent=j.log||'(đang khởi động…)';log.scrollTop=log.scrollHeight;
  if(j.task_url){rlink.href=fixU(j.task_url);rlink.classList.remove('hidden');}
  if(j.status==='running'){setTimeout(()=>poll(id),2000);return;}
  if(j.status==='done'){badge.className='badge b-ok';badge.textContent='Hoàn tất ✓';btn.disabled=false;btn.textContent='Bắt đầu auto-label';}
  else{badge.className='badge b-err';badge.textContent='Lỗi ✕';btn.disabled=false;btn.textContent='Bắt đầu auto-label';}}
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
document.getElementById('as_btn').onclick=()=>alert('Backend phân công sẽ bật ở bước sau (khi VM chạy).');
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
