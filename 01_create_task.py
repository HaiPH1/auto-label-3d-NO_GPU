"""
Bước 1: Tạo CVAT 3D task + upload data (PCD + camera images).

Cách dùng:
    python3 01_create_task.py \
        --url http://localhost:8080 \
        --user admin --password <password> \
        --task-name "My 3D Task" \
        --data ./data

Cấu trúc thư mục data:
    data/
    ├── pointcloud/
    │   ├── 0.pcd
    │   ├── 1.pcd
    │   └── ...
    └── context_images/       (tùy chọn)
        ├── 0/
        │   ├── CAM_FRONT.jpg
        │   ├── CAM_BACK.jpg
        │   └── ...
        └── 1/
            └── ...

Output: config.json (chứa job_id, token, label_map để bước 3 dùng)
"""

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import zipfile

import requests

LABELS_10CLASS = [
    {"name": "car", "type": "cuboid"},
    {"name": "truck", "type": "cuboid"},
    {"name": "bus", "type": "cuboid"},
    {"name": "trailer", "type": "cuboid"},
    {"name": "construction_vehicle", "type": "cuboid"},
    {"name": "pedestrian", "type": "cuboid"},
    {"name": "motorcycle", "type": "cuboid"},
    {"name": "bicycle", "type": "cuboid"},
    {"name": "traffic_cone", "type": "cuboid"},
    {"name": "barrier", "type": "cuboid"},
]


def login(cvat_url, username, password):
    resp = requests.post(f"{cvat_url}/api/auth/login",
                         json={"username": username, "password": password})
    if resp.status_code != 200:
        print(f"Lỗi đăng nhập: {resp.status_code} {resp.text[:200]}")
        sys.exit(1)
    token = resp.json().get("key")
    print(f"Đăng nhập thành công: {username}")
    return token


def create_task(cvat_url, token, task_name):
    headers = {"Authorization": f"Token {token}", "Content-Type": "application/json"}
    resp = requests.post(f"{cvat_url}/api/tasks", headers=headers, json={
        "name": task_name,
        "labels": [{"name": lb["name"]} for lb in LABELS_10CLASS],
    })
    if resp.status_code not in (200, 201):
        print(f"Lỗi tạo task: {resp.status_code} {resp.text[:300]}")
        sys.exit(1)
    task = resp.json()
    print(f"Tạo task thành công: ID={task['id']}, name={task['name']}")
    return task


def build_zip(data_dir, zip_path):
    pcd_dir = os.path.join(data_dir, "pointcloud")
    ctx_dir = os.path.join(data_dir, "context_images")

    if not os.path.isdir(pcd_dir):
        print(f"Không tìm thấy thư mục pointcloud/ trong {data_dir}")
        sys.exit(1)

    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))
    if not pcd_files:
        print("Không tìm thấy file .pcd nào")
        sys.exit(1)

    print(f"Tìm thấy {len(pcd_files)} file PCD")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for pcd in pcd_files:
            name = os.path.basename(pcd)
            zf.write(pcd, f"pointcloud/{name}")

        if os.path.isdir(ctx_dir):
            for pcd in pcd_files:
                stem = os.path.splitext(os.path.basename(pcd))[0]
                img_dir = os.path.join(ctx_dir, stem)
                if not os.path.isdir(img_dir):
                    continue
                for img in sorted(os.listdir(img_dir)):
                    if img.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(img_dir, img)
                        zf.write(img_path, f"related_images/{stem}_pcd/{img}")
            print(f"Đã thêm camera images vào zip")
        else:
            print("Không có context_images/ — task sẽ chỉ có point cloud")

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Tạo zip: {zip_path} ({size_mb:.1f} MB)")


def upload_data(cvat_url, token, task_id, zip_path):
    headers = {"Authorization": f"Token {token}"}

    print("Đang upload dữ liệu lên CVAT (có thể mất vài phút)...")
    with open(zip_path, "rb") as f:
        resp = requests.post(
            f"{cvat_url}/api/tasks/{task_id}/data",
            headers=headers,
            files={"client_files[0]": (os.path.basename(zip_path), f, "application/zip")},
            data={
                "image_quality": 70,
                "use_zip_chunks": "true",
            }
        )
    if resp.status_code not in (200, 201, 202):
        print(f"Lỗi upload: {resp.status_code} {resp.text[:300]}")
        sys.exit(1)
    print("Upload thành công!")


def get_job_and_labels(cvat_url, token, task_id):
    import time
    headers = {"Authorization": f"Token {token}"}

    print("Đang chờ CVAT xử lý data...")
    for attempt in range(30):
        resp = requests.get(f"{cvat_url}/api/jobs?task_id={task_id}", headers=headers)
        jobs = resp.json().get("results", [])
        if jobs:
            break
        time.sleep(2)
    if not jobs:
        print("Không tìm thấy job sau 60s. Kiểm tra CVAT logs.")
        sys.exit(1)
    job_id = jobs[0]["id"]

    resp = requests.get(f"{cvat_url}/api/labels?job_id={job_id}", headers=headers)
    labels = resp.json().get("results", [])
    label_map = {lb["name"]: lb["id"] for lb in labels}

    print(f"Job ID: {job_id}")
    print(f"Label map: {label_map}")
    return job_id, label_map


def get_frame_info(cvat_url, token, task_id):
    headers = {"Authorization": f"Token {token}"}
    resp = requests.get(f"{cvat_url}/api/tasks/{task_id}/data/meta", headers=headers)
    if resp.status_code != 200:
        return 0, 0
    meta = resp.json()
    frames = meta.get("frames", [])
    if not frames:
        return 0, 0
    first_name = frames[0].get("name", "")
    stem = os.path.splitext(os.path.basename(first_name))[0]
    try:
        first_frame_num = int(stem)
    except ValueError:
        first_frame_num = 0
    return len(frames), first_frame_num


def main():
    parser = argparse.ArgumentParser(description="Tạo CVAT 3D task + upload data")
    parser.add_argument("--url", required=True, help="CVAT URL (vd: http://localhost:8080)")
    parser.add_argument("--user", required=True, help="CVAT username")
    parser.add_argument("--password", required=True, help="CVAT password")
    parser.add_argument("--task-name", required=True, help="Tên task trên CVAT")
    parser.add_argument("--data", required=True, help="Thư mục chứa pointcloud/ và context_images/")
    parser.add_argument("--output", default="config.json", help="File config output (default: config.json)")
    args = parser.parse_args()

    cvat_url = args.url.rstrip("/")

    token = login(cvat_url, args.user, args.password)
    task = create_task(cvat_url, token, args.task_name)
    task_id = task["id"]

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "data.zip")
        build_zip(args.data, zip_path)
        upload_data(cvat_url, token, task_id, zip_path)

    job_id, label_map = get_job_and_labels(cvat_url, token, task_id)
    n_frames, first_frame = get_frame_info(cvat_url, token, task_id)

    config = {
        "cvat_url": cvat_url,
        "token": token,
        "task_id": task_id,
        "job_id": job_id,
        "label_map": label_map,
        "frame_offset": first_frame,
        "n_frames": n_frames,
        "pcd_dir": os.path.abspath(os.path.join(args.data, "pointcloud")),
    }

    with open(args.output, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\nConfig đã lưu: {args.output}")
    print(f"Task URL: {cvat_url}/tasks/{task_id}/jobs/{job_id}")
    print(f"\nBước tiếp: chạy Colab inference → tải predictions.json → chạy:")
    print(f"  python3 03_upload_predictions.py --config {args.output} --predictions predictions.json")


if __name__ == "__main__":
    main()
