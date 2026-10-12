"""
Web portal cho Auto Label 3D.
Chạy trên VM (cùng chỗ run.py + venv + model). Người dùng:
  upload zip point cloud -> chọn CVAT (server này / CVAT khác) -> auto-label -> link CVAT.

Chạy:
  source ~/venv/bin/activate
  pip install fastapi "uvicorn[standard]" python-multipart
  uvicorn app:app --host 0.0.0.0 --port 8000      (chạy trong thư mục webapp/)
"""
import os
import re
import shutil
import threading
import uuid
import zipfile

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import subprocess

HOME = os.path.expanduser("~")
RUN_PY = os.path.join(HOME, "auto-label-3d", "run.py")
VENV_PY = os.path.join(HOME, "venv", "bin", "python3")
JOBS_DIR = os.path.join(HOME, "autolabel_jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

DEFAULT_CVAT_URL = "http://localhost:8080"
DEFAULT_CVAT_USER = "admin"
DEFAULT_CVAT_PASS = "Cvat123456"

app = FastAPI(title="Auto Label 3D")
JOBS = {}  # job_id -> {status, log, task_url}


def find_data_dir(root):
    """Tìm thư mục chứa pointcloud/ trong cây thư mục đã giải nén."""
    for dirpath, dirnames, _ in os.walk(root):
        if "pointcloud" in dirnames:
            return dirpath
    return None


def run_job(job_id, data_dir, cvat_url, user, password, task_name):
    job = JOBS[job_id]
    cmd = [VENV_PY, RUN_PY, "--url", cvat_url, "--user", user,
           "--password", password, "--task-name", task_name, "--data", data_dir]
    try:
        proc = subprocess.Popen(
            cmd, cwd=os.path.dirname(RUN_PY),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
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


@app.post("/jobs")
async def create_job(
    task_name: str = Form(...),
    cvat_url: str = Form(DEFAULT_CVAT_URL),
    cvat_user: str = Form(DEFAULT_CVAT_USER),
    cvat_pass: str = Form(DEFAULT_CVAT_PASS),
    file: UploadFile = File(...),
):
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
        return JSONResponse(
            {"error": "Không tìm thấy thư mục 'pointcloud/' trong zip. "
                      "Zip phải chứa pointcloud/*.pcd (và context_images/ tùy chọn)."},
            status_code=400)

    JOBS[job_id] = {"status": "running", "log": "", "task_url": None}
    threading.Thread(
        target=run_job,
        args=(job_id, data_dir, cvat_url.rstrip("/"), cvat_user, cvat_pass, task_name),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"status": job["status"], "log": job["log"][-6000:], "task_url": job["task_url"]}


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


HTML_PAGE = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Auto Label 3D</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --ink:#e2e8f0; --muted:#94a3b8;
          --accent:#38bdf8; --ok:#34d399; --err:#f87171; --border:#334155; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,Segoe UI,Roboto,sans-serif;
         background:var(--bg); color:var(--ink); padding:24px 16px; }
  .wrap { max-width:720px; margin:0 auto; }
  h1 { font-size:1.4rem; margin:0 0 4px; }
  .sub { color:var(--muted); margin:0 0 20px; font-size:.9rem; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:14px;
          padding:20px; margin-bottom:16px; }
  label { display:block; font-size:.85rem; color:var(--muted); margin:12px 0 4px; }
  input[type=text], input[type=password], input[type=file] {
    width:100%; padding:10px 12px; border-radius:9px; border:1px solid var(--border);
    background:#0f172a; color:var(--ink); font-size:.95rem; }
  .row2 { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  details { margin-top:8px; }
  summary { cursor:pointer; color:var(--accent); font-size:.85rem; }
  button { margin-top:18px; width:100%; padding:12px; border:0; border-radius:10px;
           background:var(--accent); color:#06202b; font-weight:700; font-size:1rem; cursor:pointer; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .hint { font-size:.78rem; color:var(--muted); margin-top:6px; }
  #log { background:#0b1220; border:1px solid var(--border); border-radius:9px; padding:12px;
         font-family:ui-monospace,Menlo,monospace; font-size:.78rem; white-space:pre-wrap;
         max-height:340px; overflow:auto; color:#cbd5e1; }
  .badge { display:inline-block; padding:3px 10px; border-radius:999px; font-size:.78rem; font-weight:700; }
  .b-run{ background:#1e3a5f; color:var(--accent);} .b-ok{ background:#064e3b; color:var(--ok);}
  .b-err{ background:#4c1d1d; color:var(--err);}
  a.result { display:inline-block; margin-top:12px; padding:10px 16px; border-radius:9px;
             background:var(--ok); color:#042f23; font-weight:700; text-decoration:none; }
  .hidden { display:none; }
</style>
</head>
<body>
<div class="wrap">
  <h1>🧊 Auto Label 3D</h1>
  <p class="sub">Upload point cloud → tự động gán nhãn 3D cuboid (TransFusion / GPU) → đẩy vào CVAT.</p>

  <div class="card" id="formCard">
    <form id="f">
      <label>Tên task</label>
      <input type="text" name="task_name" value="Auto Label 3D" required>

      <label>File dữ liệu (.zip chứa <code>pointcloud/*.pcd</code>, kèm <code>context_images/</code> nếu có)</label>
      <input type="file" name="file" accept=".zip" required>

      <details>
        <summary>Tùy chọn CVAT (mặc định: CVAT trên server này)</summary>
        <label>CVAT URL</label>
        <input type="text" name="cvat_url" value="http://localhost:8080">
        <div class="row2">
          <div><label>User</label><input type="text" name="cvat_user" value="admin"></div>
          <div><label>Password</label><input type="password" name="cvat_pass" value="Cvat123456"></div>
        </div>
        <p class="hint">Để nguyên = đẩy vào CVAT ngay trên server này. Muốn đẩy sang CVAT khác thì sửa URL/tài khoản.</p>
      </details>

      <button type="submit" id="btn">Bắt đầu auto-label</button>
    </form>
  </div>

  <div class="card hidden" id="progCard">
    <div>Trạng thái: <span id="badge" class="badge b-run">Đang chạy…</span></div>
    <a id="resultLink" class="result hidden" target="_blank">Mở task trong CVAT →</a>
    <label style="margin-top:14px">Nhật ký</label>
    <div id="log"></div>
    <button id="again" class="hidden" style="background:#334155;color:#e2e8f0">Làm cái mới</button>
  </div>
</div>

<script>
const f = document.getElementById('f');
const btn = document.getElementById('btn');
const progCard = document.getElementById('progCard');
const formCard = document.getElementById('formCard');
const logEl = document.getElementById('log');
const badge = document.getElementById('badge');
const resultLink = document.getElementById('resultLink');
const again = document.getElementById('again');

function fixUrl(u){ try { return u.replace('localhost', location.hostname); } catch(e){ return u; } }

f.addEventListener('submit', async (e) => {
  e.preventDefault();
  btn.disabled = true; btn.textContent = 'Đang tải lên…';
  const res = await fetch('/jobs', { method:'POST', body:new FormData(f) });
  const data = await res.json();
  if (data.error) { alert(data.error); btn.disabled=false; btn.textContent='Bắt đầu auto-label'; return; }
  formCard.classList.add('hidden'); progCard.classList.remove('hidden');
  poll(data.job_id);
});

async function poll(id){
  const r = await fetch('/jobs/'+id); const j = await r.json();
  logEl.textContent = j.log || '(đang khởi động…)'; logEl.scrollTop = logEl.scrollHeight;
  if (j.task_url){ resultLink.href = fixUrl(j.task_url); resultLink.classList.remove('hidden'); }
  if (j.status === 'running'){ setTimeout(()=>poll(id), 2000); return; }
  if (j.status === 'done'){ badge.className='badge b-ok'; badge.textContent='Hoàn tất ✓'; }
  else { badge.className='badge b-err'; badge.textContent='Lỗi ✕'; }
  again.classList.remove('hidden');
}
again.addEventListener('click', ()=>location.reload());
</script>
</body>
</html>"""
