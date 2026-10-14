import os
import random
import requests

# Cấu hình qua biến môi trường (không hardcode secret trong repo)
BASE = os.environ.get("CVAT_URL", "http://localhost:8080")
CVAT_USER = os.environ.get("CVAT_USER", "admin")
CVAT_PASS = os.environ.get("CVAT_PASS", "")
s = requests.Session()

r = s.post(BASE + "/api/auth/login", json={"username": CVAT_USER, "password": CVAT_PASS})
print("login:", r.status_code)
tok = r.json()["key"]
H = {"Authorization": "Token " + tok}

uname = "annot_%d" % random.randint(100, 999)
r = s.post(BASE + "/api/auth/register", json={
    "username": uname, "email": uname + "@test.com",
    "password1": "Annot12345x", "password2": "Annot12345x",
    "first_name": "Annot", "last_name": "Test"})
print("register %s:" % uname, r.status_code, r.text[:160])

r = s.get(BASE + "/api/users", headers=H, params={"search": uname})
users = r.json().get("results", [])
print("lookup user:", r.status_code, [(u["id"], u["username"]) for u in users])

r = s.get(BASE + "/api/jobs", headers=H, params={"task_id": 15})
jobs = r.json().get("results", [])
print("jobs task15:", r.status_code, [(j["id"], j.get("assignee")) for j in jobs][:3])

if jobs and users:
    uid, jid = users[0]["id"], jobs[0]["id"]
    r = s.patch(BASE + "/api/jobs/%d" % jid, headers=H, json={"assignee": uid})
    a = r.json().get("assignee")
    print("assign job %d -> %s:" % (jid, uname), r.status_code,
          a.get("username") if isinstance(a, dict) else a)
