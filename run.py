"""
Pipeline 1 lệnh: tạo task CVAT + inference + upload predictions.

Cách dùng:
    # Có GPU trên máy (RTX 3090, 4090, 5060 Ti...):
    python3 run.py \
        --url http://localhost:8080 \
        --user admin --password <password> \
        --task-name "My 3D Task" \
        --data ./data

    # Không có GPU — chỉ tạo task + upload (inference chạy trên Colab riêng):
    python3 run.py \
        --url http://localhost:8080 \
        --user admin --password <password> \
        --task-name "My 3D Task" \
        --data ./data \
        --predictions transfusion_predictions.json

    # Chỉ inference (đã có task CVAT):
    python3 run.py --config config.json

    # Chỉ upload (đã có config + predictions):
    python3 run.py --config config.json --predictions predictions.json --skip-inference

Cấu trúc thư mục data:
    data/
    ├── pointcloud/
    │   ├── 0.pcd
    │   └── ...
    └── context_images/       (tùy chọn)
        ├── 0/
        │   ├── CAM_FRONT.jpg
        │   └── ...
        └── ...
"""

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
from scipy.optimize import linear_sum_assignment

import requests

try:
    import yaml
except ImportError:
    yaml = None

# ============================================================
# Config
# ============================================================

LABELS_10CLASS = [
    {"name": "car"}, {"name": "truck"}, {"name": "bus"},
    {"name": "trailer"}, {"name": "construction_vehicle"},
    {"name": "pedestrian"}, {"name": "motorcycle"}, {"name": "bicycle"},
    {"name": "traffic_cone"}, {"name": "barrier"},
]

THRESH_MAP = {
    'truck': 0.20, 'bus': 0.25, 'motorcycle': 0.25, 'car': 0.28,
    'trailer': 0.25, 'pedestrian': 0.30, 'traffic_cone': 0.25, 'barrier': 0.25,
    'construction_vehicle': 0.25, 'bicycle': 0.25,
}

HEADING_CLASSES = {'car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'motorcycle'}

# Ngưỡng IoU "giữ nguyên (không cần sửa)" THEO CLASS — 3D IoU gắt với vật nhỏ:
# xe lớn giữ 0.7; người/cone/2-bánh nới 0.5 để phản ánh đúng công sửa thật.
ACCEPT_IOU_BY_CLASS = {
    'car': 0.7, 'truck': 0.7, 'bus': 0.7, 'trailer': 0.7, 'construction_vehicle': 0.7,
    'pedestrian': 0.5, 'bicycle': 0.5, 'motorcycle': 0.5, 'traffic_cone': 0.5, 'barrier': 0.5,
    '_default': 0.7,
}


def _accept_thr(cls, iou_accept):
    """Ngưỡng accept cho 1 class: dict -> tra theo class (fallback _default); số -> dùng chung."""
    if isinstance(iou_accept, dict):
        return iou_accept.get(cls, iou_accept.get('_default', 0.7))
    return iou_accept

CLASS_COLORS = {
    'car': (0, 255, 0),
    'truck': (0, 165, 255),
    'bus': (255, 0, 255),
    'trailer': (180, 130, 70),
    'construction_vehicle': (0, 215, 255),
    'pedestrian': (0, 255, 255),
    'motorcycle': (255, 255, 0),
    'bicycle': (255, 50, 50),
    'traffic_cone': (0, 0, 255),
    'barrier': (128, 128, 255),
}

OPENPCDET_DIR = os.path.expanduser("~/OpenPCDet")
TRANSFUSION_CFG = "tools/cfgs/nuscenes_models/transfusion_lidar.yaml"
CHECKPOINT_URL = "https://drive.google.com/uc?id=1cuZ2qdDnxSwTCsiXWwbqCGF-uoazTXbz"
CHECKPOINT_DIR = os.path.expanduser("~/checkpoints")
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "transfusion_lidar.pth")
SCORE_THRESH = 0.20


# ============================================================
# Step 1: Create CVAT task + upload data
# ============================================================

def login(cvat_url, username, password):
    resp = requests.post(f"{cvat_url}/api/auth/login",
                         json={"username": username, "password": password})
    if resp.status_code != 200:
        print(f"[ERROR] Đăng nhập thất bại: {resp.status_code} {resp.text[:200]}")
        sys.exit(1)
    token = resp.json().get("key")
    print(f"  Đăng nhập thành công: {username}")
    return token


def create_task(cvat_url, token, task_name):
    headers = {"Authorization": f"Token {token}", "Content-Type": "application/json"}
    resp = requests.post(f"{cvat_url}/api/tasks", headers=headers, json={
        "name": task_name,
        "labels": [{"name": lb["name"]} for lb in LABELS_10CLASS],
    })
    if resp.status_code not in (200, 201):
        print(f"[ERROR] Tạo task thất bại: {resp.status_code} {resp.text[:300]}")
        sys.exit(1)
    task = resp.json()
    print(f"  Tạo task: ID={task['id']}, name={task['name']}")
    return task


def build_zip(data_dir, zip_path):
    pcd_dir = os.path.join(data_dir, "pointcloud")
    ctx_dir = os.path.join(data_dir, "context_images")

    if not os.path.isdir(pcd_dir):
        print(f"[ERROR] Không tìm thấy pointcloud/ trong {data_dir}")
        sys.exit(1)

    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))
    if not pcd_files:
        print("[ERROR] Không tìm thấy file .pcd nào")
        sys.exit(1)

    print(f"  Tìm thấy {len(pcd_files)} file PCD")

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
            print(f"  Đã thêm camera images")

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"  Zip: {size_mb:.1f} MB")


def upload_data(cvat_url, token, task_id, zip_path):
    headers = {"Authorization": f"Token {token}"}
    print("  Đang upload lên CVAT...")
    with open(zip_path, "rb") as f:
        resp = requests.post(
            f"{cvat_url}/api/tasks/{task_id}/data",
            headers=headers,
            files={"client_files[0]": (os.path.basename(zip_path), f, "application/zip")},
            data={"image_quality": 70, "use_zip_chunks": "true"},
        )
    if resp.status_code not in (200, 201, 202):
        print(f"[ERROR] Upload thất bại: {resp.status_code} {resp.text[:300]}")
        sys.exit(1)
    print("  Upload thành công!")


def get_job_and_labels(cvat_url, token, task_id):
    import time
    headers = {"Authorization": f"Token {token}"}
    print("  Đang chờ CVAT xử lý data...")
    for attempt in range(30):
        resp = requests.get(f"{cvat_url}/api/jobs?task_id={task_id}", headers=headers)
        jobs = resp.json().get("results", [])
        if jobs:
            break
        time.sleep(2)
    if not jobs:
        print("[ERROR] Không tìm thấy job sau 60s. Kiểm tra CVAT logs.")
        sys.exit(1)
    job_id = jobs[0]["id"]
    resp = requests.get(f"{cvat_url}/api/labels?job_id={job_id}", headers=headers)
    labels = resp.json().get("results", [])
    label_map = {lb["name"]: lb["id"] for lb in labels}
    print(f"  Job ID: {job_id}, Labels: {len(label_map)}")
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
    stem = os.path.splitext(os.path.basename(frames[0].get("name", "")))[0]
    try:
        first_frame_num = int(stem)
    except ValueError:
        first_frame_num = 0
    return len(frames), first_frame_num


def step1_create_task(args):
    print("\n[STEP 1] Tạo task CVAT + upload data")
    print("=" * 50)

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

    config_path = args.config or "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  Config: {config_path}")
    return config, config_path


# ============================================================
# Step 2: Inference (local GPU)
# ============================================================

def detect_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        name = torch.cuda.get_device_name(0)
        arch = torch.cuda.get_device_capability(0)
        return True, f"{name} (sm_{arch[0]}{arch[1]})"
    except ImportError:
        return False, "PyTorch not installed"


def check_openpcdet():
    if not os.path.isdir(OPENPCDET_DIR):
        return False, f"OpenPCDet not found at {OPENPCDET_DIR}"
    try:
        import pcdet  # noqa: F401
        return True, f"pcdet {pcdet.__version__}"
    except ImportError:
        return False, "pcdet not importable"


def download_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        return
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"  Đang tải checkpoint TransFusion-L...")
    try:
        import gdown
        gdown.download(CHECKPOINT_URL, CHECKPOINT_FILE, quiet=False)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
        gdown.download(CHECKPOINT_URL, CHECKPOINT_FILE, quiet=False)


def load_pcd(path):
    """Đọc PCD binary/ascii. Suy số trường từ FIELDS (hỗ trợ 4 trường x,y,z,intensity/rgb
    hoặc 5 trường x,y,z,intensity,timestamp — nuScenes 10-sweep). Giả định mỗi trường 4 byte."""
    with open(path, 'rb') as f:
        n_points = 0
        n_fields = 4
        data_type = 'binary'
        while True:
            line = f.readline().decode('ascii', 'ignore').strip()
            if line.startswith('FIELDS'):
                n_fields = max(1, len(line.split()) - 1)
            if line.startswith('POINTS'):
                n_points = int(line.split()[1])
            if line.startswith('DATA'):
                if 'ascii' in line.lower():
                    data_type = 'ascii'
                break

        if data_type == 'binary':
            data = f.read(n_points * n_fields * 4)
            pts = np.frombuffer(data, dtype=np.float32).reshape(-1, n_fields)
        else:
            f.seek(0)
            lines = f.readlines()
            data_start = 0
            for idx, l in enumerate(lines):
                if l.strip().startswith(b'DATA'):
                    data_start = idx + 1
                    break
            pts = np.loadtxt(lines[data_start:], dtype=np.float32)
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
    return pts


def run_inference(pcd_dir, output_path):
    import torch

    for attr in ['int', 'float', 'long']:
        if not hasattr(np, attr):
            setattr(np, attr, int if attr != 'float' else float)
    if not hasattr(np, 'bool'):
        np.bool = np.bool_

    sys.path.insert(0, OPENPCDET_DIR)
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets.dataset import DatasetTemplate
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils
    from pathlib import Path

    cfg_file = os.path.join(OPENPCDET_DIR, TRANSFUSION_CFG)
    # Config OpenPCDet dùng _BASE_CONFIG_ là đường dẫn tương đối với tools/ -> chdir khi nạp
    _old_cwd = os.getcwd()
    os.chdir(os.path.join(OPENPCDET_DIR, "tools"))
    try:
        cfg_from_yaml_file(cfg_file, cfg)
    finally:
        os.chdir(_old_cwd)
    class_names = cfg.CLASS_NAMES

    class PCDDataset(DatasetTemplate):
        def __init__(self, dataset_cfg, class_names_, pcd_dir_):
            super().__init__(dataset_cfg=dataset_cfg, class_names=class_names_,
                             training=False, root_path=Path('.'), logger=None)
            def sort_key(f):
                base = os.path.splitext(os.path.basename(f))[0]
                return int(base) if base.isdigit() else base
            self.pcd_files = sorted(glob.glob(os.path.join(pcd_dir_, '*.pcd')), key=sort_key)

        def __len__(self):
            return len(self.pcd_files)

        def __getitem__(self, index):
            pcd_path = self.pcd_files[index]
            base_name = os.path.splitext(os.path.basename(pcd_path))[0]
            frame_id = int(base_name) if base_name.isdigit() else index
            points = load_pcd(pcd_path).astype(np.float32)
            # Model cần 5 trường (x,y,z,intensity,timestamp). PCD 5 trường (nuScenes 10-sweep)
            # dùng thẳng; PCD 4 trường (data_day14) đệm timestamp=0.
            if points.shape[1] >= 5:
                points = points[:, :5]
            else:
                pad = np.zeros((points.shape[0], 5 - points.shape[1]), dtype=np.float32)
                points = np.hstack([points, pad])
            return self.prepare_data(data_dict={'points': points, 'frame_id': frame_id})

    dataset = PCDDataset(cfg.DATA_CONFIG, class_names, pcd_dir)
    logger = common_utils.create_logger()
    model = build_network(model_cfg=cfg.MODEL, num_class=len(class_names), dataset=dataset)
    model.load_params_from_file(filename=CHECKPOINT_FILE, logger=logger, to_cpu=False)
    model.cuda().eval()

    all_results = []
    print(f"  Inference trên {len(dataset)} frames...")

    with torch.no_grad():
        for i in range(len(dataset)):
            data_dict = dataset[i]
            frame_id = data_dict['frame_id']
            batch_dict = dataset.collate_batch([data_dict])
            load_data_to_gpu(batch_dict)
            pred_dicts, _ = model.forward(batch_dict)

            boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            scores = pred_dicts[0]['pred_scores'].cpu().numpy()
            labels = pred_dicts[0]['pred_labels'].cpu().numpy()

            frame_boxes = []
            for b, s, l in zip(boxes, scores, labels):
                if float(s) < SCORE_THRESH:
                    continue
                frame_boxes.append({
                    'label': class_names[l - 1],
                    'score': round(float(s), 3),
                    'center': [round(float(b[0]), 3), round(float(b[1]), 3), round(float(b[2]), 3)],
                    'size': [round(float(b[3]), 3), round(float(b[4]), 3), round(float(b[5]), 3)],
                    'yaw': round(float(b[6]), 3),
                })
            all_results.append({'frame': frame_id, 'num_boxes': len(frame_boxes), 'boxes': frame_boxes})
            print(f"    Frame {frame_id}: {len(frame_boxes)} boxes")

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Predictions: {output_path} ({sum(r['num_boxes'] for r in all_results)} boxes)")
    return all_results


def step2_inference(pcd_dir, predictions_path):
    print("\n[STEP 2] Inference (TransFusion-L)")
    print("=" * 50)

    gpu_ok, gpu_info = detect_gpu()
    if not gpu_ok:
        print(f"  GPU: {gpu_info}")
        print("  => Không có GPU. Chạy inference trên Colab:")
        print("     1. Mở 02_inference_colab.ipynb trên Colab")
        print("     2. Upload PCD files + chạy notebook")
        print("     3. Tải predictions.json về")
        print(f"     4. Chạy lại: python3 run.py --config config.json --predictions <file> --skip-inference")
        return None

    print(f"  GPU: {gpu_info}")

    pcdet_ok, pcdet_info = check_openpcdet()
    if not pcdet_ok:
        print(f"  OpenPCDet: {pcdet_info}")
        print("  => Chạy setup trước: ./setup_rtx5060ti.sh")
        return None

    print(f"  OpenPCDet: {pcdet_info}")

    download_checkpoint()

    return run_inference(pcd_dir, predictions_path)


# ============================================================
# Step 3: Heading correction + upload to CVAT
# ============================================================

def normalize_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def points_in_box(points, cx, cy, cz, l, w, h, yaw):
    cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    dz = points[:, 2] - cz
    local_x = dx * cos_y - dy * sin_y
    local_y = dx * sin_y + dy * cos_y
    mask = (np.abs(local_x) < l / 2) & (np.abs(local_y) < w / 2) & (np.abs(dz) < h / 2)
    return local_x[mask], local_y[mask]


def correct_heading(preds, pcd_dir):
    frames = sorted(preds, key=lambda f: f['frame'])
    all_boxes = {}
    for fd in frames:
        fid = fd['frame']
        all_boxes[fid] = []
        for i, b in enumerate(fd['boxes']):
            all_boxes[fid].append({
                'idx': i, 'label': b['label'], 'score': b['score'],
                'cx': b['center'][0], 'cy': b['center'][1], 'cz': b['center'][2],
                'l': b['size'][0], 'w': b['size'][1], 'h': b['size'][2],
                'yaw': b['yaw'], 'flipped': False, 'confident': False,
            })

    frame_ids = sorted(all_boxes.keys())

    # Phase 1: Velocity-based
    for fi in range(len(frame_ids) - 1):
        f1, f2 = frame_ids[fi], frame_ids[fi + 1]
        boxes1 = [b for b in all_boxes[f1] if b['label'] in HEADING_CLASSES]
        boxes2 = [b for b in all_boxes[f2] if b['label'] in HEADING_CLASSES]
        if not boxes1 or not boxes2:
            continue
        cost = np.zeros((len(boxes1), len(boxes2)))
        for i, b1 in enumerate(boxes1):
            for j, b2 in enumerate(boxes2):
                if b1['label'] != b2['label']:
                    cost[i, j] = 1e6
                else:
                    cost[i, j] = math.hypot(b1['cx'] - b2['cx'], b1['cy'] - b2['cy'])
        row_ind, col_ind = linear_sum_assignment(cost)
        displacements = []
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 5.0:
                displacements.append((boxes2[c]['cx'] - boxes1[r]['cx'],
                                      boxes2[c]['cy'] - boxes1[r]['cy']))
        if len(displacements) < 3:
            continue
        ego_dx = np.median([d[0] for d in displacements])
        ego_dy = np.median([d[1] for d in displacements])
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= 5.0:
                continue
            b1 = boxes1[r]
            vx = (boxes2[c]['cx'] - b1['cx']) - ego_dx
            vy = (boxes2[c]['cy'] - b1['cy']) - ego_dy
            speed = math.hypot(vx, vy)
            if speed < 1.5:
                continue
            heading_x = math.cos(b1['yaw'])
            heading_y = math.sin(b1['yaw'])
            cos_angle = (heading_x * vx + heading_y * vy) / speed
            if cos_angle < -0.5:
                b1['yaw'] = normalize_angle(b1['yaw'] + math.pi)
                b1['flipped'] = True
                b1['confident'] = True

    flipped_p1 = sum(1 for fid in frame_ids for b in all_boxes[fid] if b['flipped'])
    print(f"    Phase 1 (velocity): flipped {flipped_p1}")

    # Phase 2: Point-cloud density
    flipped_p2 = 0
    for fid in frame_ids:
        pcd_path = os.path.join(pcd_dir, f'{fid}.pcd')
        if not os.path.exists(pcd_path):
            continue
        try:
            pts = load_pcd(pcd_path)[:, :3]
        except Exception:
            continue
        for b in all_boxes[fid]:
            if b['label'] not in HEADING_CLASSES or b['confident']:
                continue
            if b['score'] < THRESH_MAP.get(b['label'], 0.28):
                continue
            local_x, _ = points_in_box(pts, b['cx'], b['cy'], b['cz'],
                                        b['l'], b['w'], b['h'], b['yaw'])
            n_pts = len(local_x)
            if n_pts < 10:
                continue
            front_count = int(np.sum(local_x > 0))
            back_count = int(np.sum(local_x < 0))
            dist = math.hypot(b['cx'], b['cy'])
            if dist < 1.0:
                continue
            heading_x = math.cos(b['yaw'])
            heading_y = math.sin(b['yaw'])
            dot_front = (-b['cx'] / dist) * heading_x + (-b['cy'] / dist) * heading_y
            if abs(dot_front) < 0.5:
                continue
            if dot_front > 0 and back_count > front_count * 1.5 and n_pts > 20:
                b['yaw'] = normalize_angle(b['yaw'] + math.pi)
                b['flipped'] = True
                flipped_p2 += 1
            elif dot_front < 0 and front_count > back_count * 1.5 and n_pts > 20:
                b['yaw'] = normalize_angle(b['yaw'] + math.pi)
                b['flipped'] = True
                flipped_p2 += 1
    print(f"    Phase 2 (point cloud): flipped {flipped_p2}")

    # Phase 3: Lane consistency
    flipped_p3 = 0
    for fid in frame_ids:
        vehicles = [b for b in all_boxes[fid]
                     if b['label'] in HEADING_CLASSES
                     and b['score'] >= THRESH_MAP.get(b['label'], 0.28)]
        for b in vehicles:
            aligned = 0
            opposed = 0
            for n in vehicles:
                if n is b:
                    continue
                dist = math.hypot(n['cx'] - b['cx'], n['cy'] - b['cy'])
                if dist > 20:
                    continue
                diff = abs(normalize_angle(b['yaw'] - n['yaw']))
                if diff < math.pi / 3:
                    aligned += 1
                elif diff > 2 * math.pi / 3:
                    opposed += 1
            if opposed >= 2 and opposed > aligned * 2:
                b['yaw'] = normalize_angle(b['yaw'] + math.pi)
                b['flipped'] = True
                flipped_p3 += 1
    print(f"    Phase 3 (lane consistency): flipped {flipped_p3}")

    for fd in preds:
        fid = fd['frame']
        for i, b in enumerate(fd['boxes']):
            b['yaw'] = all_boxes[fid][i]['yaw']
    return preds


def track_objects(preds, max_dist=5.0):
    """Match boxes across frames, assign track_id to each box."""
    frames = sorted(preds, key=lambda f: f['frame'])
    if not frames:
        return preds

    next_id = 0
    for b in frames[0]['boxes']:
        b['track_id'] = next_id
        next_id += 1

    for fi in range(1, len(frames)):
        prev_boxes = frames[fi - 1]['boxes']
        curr_boxes = frames[fi]['boxes']

        if not prev_boxes or not curr_boxes:
            for b in curr_boxes:
                b['track_id'] = next_id
                next_id += 1
            continue

        cost = np.zeros((len(prev_boxes), len(curr_boxes)))
        for i, pb in enumerate(prev_boxes):
            for j, cb in enumerate(curr_boxes):
                if pb['label'] != cb['label']:
                    cost[i, j] = 1e6
                else:
                    cost[i, j] = math.hypot(
                        pb['center'][0] - cb['center'][0],
                        pb['center'][1] - cb['center'][1],
                    )

        row_ind, col_ind = linear_sum_assignment(cost)
        matched = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < max_dist:
                curr_boxes[c]['track_id'] = prev_boxes[r]['track_id']
                matched.add(c)

        for j, cb in enumerate(curr_boxes):
            if j not in matched:
                cb['track_id'] = next_id
                next_id += 1

    n_tracks = len({b['track_id'] for f in frames for b in f['boxes']})
    n_multi = len({b['track_id'] for f in frames for b in f['boxes']
                   if sum(1 for f2 in frames for b2 in f2['boxes']
                          if b2.get('track_id') == b['track_id']) > 1})
    print(f"  Tracking: {n_tracks} tracks ({n_multi} multi-frame)")
    return preds


# ============================================================
# Calibration pipeline
# ============================================================

def load_calibration(calib_path):
    """Read calib.json, return parsed calibration data."""
    if not calib_path or not os.path.exists(calib_path):
        return None
    with open(calib_path) as f:
        calib = json.load(f)
    print(f"  Calibration: {calib_path}")
    for cam_name in calib.get('cameras', {}):
        print(f"    Camera: {cam_name}")
    return calib


def box_corners_3d(cx, cy, cz, l, w, h, yaw):
    """8 corners of a 3D box in LiDAR frame."""
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    half_l, half_w, half_h = l / 2, w / 2, h / 2
    corners_local = np.array([
        [ half_l,  half_w, -half_h],
        [ half_l, -half_w, -half_h],
        [-half_l, -half_w, -half_h],
        [-half_l,  half_w, -half_h],
        [ half_l,  half_w,  half_h],
        [ half_l, -half_w,  half_h],
        [-half_l, -half_w,  half_h],
        [-half_l,  half_w,  half_h],
    ])
    rot = np.array([[cos_y, -sin_y, 0],
                     [sin_y,  cos_y, 0],
                     [0,      0,     1]])
    corners = (rot @ corners_local.T).T + np.array([cx, cy, cz])
    return corners


def project_to_camera(corners_3d, calib, camera_name):
    """Project 3D corners (N×3) to 2D pixel coords for a camera.
    Uses cv2.projectPoints for accurate distortion handling.
    Returns (N×2) pixel coords and visibility mask.
    """
    import cv2

    cam = calib['cameras'][camera_name]
    K = np.array(cam['K'], dtype=np.float64)
    ego2cam = np.array(cam['ego2cam'], dtype=np.float64)
    dist_coeffs = np.array(cam.get('dist', [0, 0, 0, 0, 0]), dtype=np.float64)
    img_w, img_h = cam.get('image_size', [1920, 1536])

    lidar_s2e = np.array(calib.get('lidar', {}).get('sensor2ego', np.eye(4).tolist()), dtype=np.float64)

    T = ego2cam @ lidar_s2e
    R = T[:3, :3]
    t = T[:3, 3]
    rvec, _ = cv2.Rodrigues(R)

    pts_h = np.hstack([corners_3d, np.ones((len(corners_3d), 1))])
    pts_cam = (T @ pts_h.T).T[:, :3]
    visible = pts_cam[:, 2] > 0.1

    px, _ = cv2.projectPoints(corners_3d.astype(np.float64), rvec, t, K, dist_coeffs)
    px = px.reshape(-1, 2)

    in_frame = visible & (px[:, 0] >= 0) & (px[:, 0] < img_w) & \
               (px[:, 1] >= 0) & (px[:, 1] < img_h)

    return px, in_frame


def project_predictions(preds, calib):
    """Add 2D projections to each box for all cameras."""
    if calib is None:
        return preds
    cameras = list(calib.get('cameras', {}).keys())
    if not cameras:
        return preds
    total_projected = 0
    for frame_data in preds:
        for b in frame_data['boxes']:
            corners = box_corners_3d(
                b['center'][0], b['center'][1], b['center'][2],
                b['size'][0], b['size'][1], b['size'][2], b['yaw'],
            )
            b['projections'] = {}
            for cam_name in cameras:
                px, vis = project_to_camera(corners, calib, cam_name)
                if np.any(vis):
                    b['projections'][cam_name] = {
                        'corners_2d': px.round(1).tolist(),
                        'visible': vis.tolist(),
                    }
                    total_projected += 1
    print(f"  Projection: {total_projected} box-camera pairs")
    return preds


# ============================================================
# Camera overlay (P2) — edge subdivision pipeline
# ============================================================

CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]
FRONT_FACE = [(0, 1), (1, 5), (5, 4), (4, 0)]
EDGE_STEPS = 16
NEAR_Z = 0.1


def _radial_fold_r2(dist):
    """Max r² where radial distortion stays monotonic."""
    d = [float(v) for v in list(dist)[:5]] + [0.0] * 5
    c = [1.0, 3.0 * d[0], 5.0 * d[1], 7.0 * d[4]]
    deg = max(i for i in range(4) if c[i] != 0.0)
    if deg == 0:
        return math.inf

    def g(s):
        return c[0] + s * (c[1] + s * (c[2] + s * c[3]))

    bound = 1.0 + max(abs(c[i]) for i in range(deg)) / abs(c[deg])
    crit = []
    qa, qb, qc = 3.0 * c[3], 2.0 * c[2], c[1]
    if qa != 0.0:
        disc = qb * qb - 4.0 * qa * qc
        if disc >= 0.0:
            sq = math.sqrt(disc)
            crit = [(-qb - sq) / (2.0 * qa), (-qb + sq) / (2.0 * qa)]
    elif qb != 0.0:
        crit = [-qc / qb]
    knots = sorted([0.0, bound] + [s for s in crit if 0.0 < s < bound])
    for lo, hi in zip(knots, knots[1:]):
        if g(hi) > 0.0:
            continue
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if mid <= lo or mid >= hi:
                break
            if g(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return lo
    return math.inf


def _distort_to_pixel(xn, yn, K, dist):
    """Brown-Conrady: normalized camera coords -> pixel coords."""
    d = np.zeros(5)
    d[:min(5, len(dist))] = dist[:5]
    k1, k2, p1, p2, k3 = d
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    r2 = xn * xn + yn * yn
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    return np.stack([fx * xd + cx, fy * yd + cy], axis=1)


def _clip_edge_view(a, b, max_r2=math.inf):
    """Visible segment [t0,t1] of edge a->b in camera frame (Z>NEAR_Z, inside max_r2 cone)."""
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    bx, by, bz = float(b[0]), float(b[1]), float(b[2])
    if az <= NEAR_Z and bz <= NEAR_Z:
        return None
    t0, t1 = 0.0, 1.0
    if az <= NEAR_Z or bz <= NEAR_Z:
        t = (NEAR_Z - az) / (bz - az)
        if az > NEAR_Z:
            t1 = t
        else:
            t0 = t
    if math.isinf(max_r2):
        return t0, t1
    dx, dy, dz = bx - ax, by - ay, bz - az
    qa = dx * dx + dy * dy - max_r2 * dz * dz
    qb = 2.0 * (ax * dx + ay * dy - max_r2 * az * dz)
    qc = ax * ax + ay * ay - max_r2 * az * az
    roots = []
    if qa != 0.0:
        disc = qb * qb - 4.0 * qa * qc
        if disc > 0.0:
            sq = math.sqrt(disc)
            h = -0.5 * (qb + sq) if qb >= 0.0 else -0.5 * (qb - sq)
            roots = [h / qa, qc / h]
    elif qb != 0.0:
        roots = [-qc / qb]
    knots = sorted([t0, t1] + [r for r in roots if t0 < r < t1])
    inside = [(lo, hi) for lo, hi in zip(knots, knots[1:])
              if (qa * (0.5 * (lo + hi)) + qb) * (0.5 * (lo + hi)) + qc <= 0.0]
    if not inside or inside[-1][1] <= inside[0][0]:
        return None
    return inside[0][0], inside[-1][1]


def _edge_polyline(a, b, K, dist, max_r2):
    """Project visible part of edge a-b (camera frame) as polyline with EDGE_STEPS subdivision."""
    span = _clip_edge_view(a, b, max_r2)
    if span is None:
        return None
    t0, t1 = span
    a, b = np.asarray(a, float), np.asarray(b, float)
    ts = np.array([t0 + (t1 - t0) * k / EDGE_STEPS for k in range(EDGE_STEPS + 1)])
    p = a + ts[:, None] * (b - a)
    return _distort_to_pixel(p[:, 0] / p[:, 2], p[:, 1] / p[:, 2], K, dist)


def _color_for_track(track_id):
    """Deterministic color from track_id."""
    h = (int(track_id) * 2654435761) % 360 if track_id is not None else 0
    c, x = 255, int(255 * (1 - abs((h / 60) % 2 - 1)))
    table = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)]
    return table[int(h // 60) % 6]


def find_camera_image(ctx_dir, frame_stem, camera_name, camera_index=1):
    """Find the camera image file for a frame. Khớp thư mục theo tên; nếu không có thì
    khớp theo SỐ frame (vd frame_0 ↔ frame_000 ↔ 0) cho đồng bộ naming."""
    frame_dir = os.path.join(ctx_dir, frame_stem)
    if not os.path.isdir(frame_dir):
        m = re.search(r'(\d+)\s*$', frame_stem)
        if m and os.path.isdir(ctx_dir):
            num = int(m.group(1))
            for d in sorted(os.listdir(ctx_dir)):
                dm = re.search(r'(\d+)\s*$', d)
                if dm and int(dm.group(1)) == num and os.path.isdir(os.path.join(ctx_dir, d)):
                    frame_dir = os.path.join(ctx_dir, d)
                    break
    if not os.path.isdir(frame_dir):
        return None

    for fname in sorted(os.listdir(frame_dir)):
        if camera_name.lower() in fname.lower():
            return os.path.join(frame_dir, fname)

    images = sorted([f for f in os.listdir(frame_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if camera_index < len(images):
        return os.path.join(frame_dir, images[camera_index])
    return None


def generate_overlays(preds, calib, ctx_dir, output_dir, camera_index=1):
    """Draw 3D boxes on camera images using edge subdivision for distortion accuracy."""
    import cv2

    if not calib or not ctx_dir or not os.path.isdir(ctx_dir):
        print("  [WARN] Thiếu calibration hoặc context_images — bỏ qua overlay")
        return []

    cameras = list(calib.get('cameras', {}).keys())
    if not cameras:
        print("  [WARN] Không có camera trong calibration")
        return []

    os.makedirs(output_dir, exist_ok=True)
    saved = []

    for cam_name in cameras:
        cam = calib['cameras'][cam_name]
        K = np.array(cam['K'], dtype=np.float64)
        dist = np.array(cam.get('dist', [0, 0, 0, 0, 0]), dtype=np.float64)
        ego2cam = np.array(cam['ego2cam'], dtype=np.float64)
        img_w, img_h = cam.get('image_size', [1920, 1536])
        lidar_s2e = np.array(
            calib.get('lidar', {}).get('sensor2ego', np.eye(4).tolist()),
            dtype=np.float64,
        )
        T = ego2cam @ lidar_s2e
        max_r2 = _radial_fold_r2(dist)

        for frame_data in sorted(preds, key=lambda f: f['frame']):
            fid = frame_data['frame']
            frame_stem = f"frame_{fid}"

            img_path = find_camera_image(ctx_dir, frame_stem, cam_name, camera_index)
            if img_path is None:
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            overlay = img.copy()
            drawn = 0

            for b in frame_data['boxes']:
                if b['score'] < THRESH_MAP.get(b['label'], 0.28):
                    continue

                corners = box_corners_3d(
                    b['center'][0], b['center'][1], b['center'][2],
                    b['size'][0], b['size'][1], b['size'][2], b['yaw'],
                )
                pts_h = np.hstack([corners, np.ones((8, 1))])
                corners_cam = (T @ pts_h.T).T[:, :3]

                front = corners_cam[:, 2] > NEAR_Z
                if not np.any(front):
                    continue

                color = CLASS_COLORS.get(b['label'], (200, 200, 200))
                tid = b.get('track_id')
                if tid is not None:
                    color = _color_for_track(tid)

                all_drawn_pts = []
                for i, j in CUBOID_EDGES:
                    pts = _edge_polyline(corners_cam[i], corners_cam[j], K, dist, max_r2)
                    if pts is None:
                        continue
                    is_front = (i, j) in FRONT_FACE or (j, i) in FRONT_FACE
                    width = 3 if is_front else 2
                    ipts = np.clip(pts, -1e4, max(img_w, img_h) + 1e4).astype(np.int32)
                    for k in range(len(ipts) - 1):
                        cv2.line(img, tuple(ipts[k]), tuple(ipts[k + 1]), color, width,
                                 cv2.LINE_AA)
                    all_drawn_pts.append(pts)

                # front face shading (alpha=60) for heading
                front_rings = [_edge_polyline(corners_cam[i], corners_cam[j], K, dist, max_r2)
                               for i, j in FRONT_FACE]
                front_ok = np.array([front[idx] for idx in [0, 1, 5, 4]])
                if front_ok.all() and all(r is not None for r in front_rings):
                    poly_pts = np.concatenate([r[:-1] for r in front_rings])
                    poly_pts = np.clip(poly_pts, -1e4, max(img_w, img_h) + 1e4).astype(np.int32)
                    cv2.fillPoly(overlay, [poly_pts], color, cv2.LINE_AA)

                if all_drawn_pts:
                    drawn += 1
                    all_pts = np.concatenate(all_drawn_pts)
                    in_img = ((all_pts[:, 0] >= 0) & (all_pts[:, 0] < img_w) &
                              (all_pts[:, 1] >= 0) & (all_pts[:, 1] < img_h))
                    if np.any(in_img):
                        vis = all_pts[in_img]
                        tx = int(np.clip(vis[:, 0].min(), 0, img_w - 1))
                        ty = max(20, int(np.clip(vis[:, 1].min(), 0, img_h - 1)) - 10)
                        depth = float(corners_cam[:, 2].mean())
                        label_text = f"{b['label']} {depth:.0f}m"
                        if tid is not None:
                            label_text = f"#{tid} {label_text}"
                        cv2.putText(img, label_text, (tx, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            # blend front face shading
            cv2.addWeighted(overlay, 0.23, img, 0.77, 0, img)

            if drawn > 0:
                out_name = f"frame_{fid}_{cam_name}.jpg"
                out_path = os.path.join(output_dir, out_name)
                cv2.imwrite(out_path, img)
                saved.append(out_path)
                print(f"    {out_name}: {drawn} boxes")

    print(f"  Overlay: {len(saved)} images saved to {output_dir}/")
    return saved


# ============================================================
# Evaluation metrics (P2)
# ============================================================

def _box_bev_corners(cx, cy, l, w, yaw):
    """4 BEV corners of a rotated box (counterclockwise)."""
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    hl, hw = l / 2, w / 2
    dx = [hl, -hl, -hl, hl]
    dy = [hw, hw, -hw, -hw]
    return [(cx + dx[i] * cos_y - dy[i] * sin_y,
             cy + dx[i] * sin_y + dy[i] * cos_y) for i in range(4)]


def _polygon_clip(subject, clip):
    """Sutherland-Hodgman convex polygon clipping."""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def line_isect(p1, p2, a, b):
        n1 = (a[0] - p1[0]) * (b[1] - a[1]) - (a[1] - p1[1]) * (b[0] - a[0])
        n2 = (p2[0] - p1[0]) * (b[1] - a[1]) - (p2[1] - p1[1]) * (b[0] - a[0])
        if abs(n2) < 1e-12:
            return p1
        t = n1 / n2
        return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))

    output = list(subject)
    for i in range(len(clip)):
        if not output:
            break
        a, b = clip[i - 1], clip[i]
        inp = list(output)
        output = []
        for j in range(len(inp)):
            p, q = inp[j - 1], inp[j]
            if inside(q, a, b):
                if not inside(p, a, b):
                    output.append(line_isect(p, q, a, b))
                output.append(q)
            elif inside(p, a, b):
                output.append(line_isect(p, q, a, b))
    return output


def _polygon_area(poly):
    """Shoelace formula."""
    n = len(poly)
    if n < 3:
        return 0.0
    area = sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
               for i in range(n))
    return abs(area) / 2.0


def iou_3d(pred, gt):
    """3D IoU between two boxes (dict with center, size, yaw)."""
    p_corners = _box_bev_corners(pred['center'][0], pred['center'][1],
                                 pred['size'][0], pred['size'][1], pred['yaw'])
    g_corners = _box_bev_corners(gt['center'][0], gt['center'][1],
                                 gt['size'][0], gt['size'][1], gt['yaw'])

    inter_poly = _polygon_clip(p_corners, g_corners)
    bev_inter = _polygon_area(inter_poly)
    if bev_inter < 1e-6:
        return 0.0

    p_z_lo = pred['center'][2] - pred['size'][2] / 2
    p_z_hi = pred['center'][2] + pred['size'][2] / 2
    g_z_lo = gt['center'][2] - gt['size'][2] / 2
    g_z_hi = gt['center'][2] + gt['size'][2] / 2
    z_overlap = max(0.0, min(p_z_hi, g_z_hi) - max(p_z_lo, g_z_lo))

    inter_3d = bev_inter * z_overlap
    vol_p = pred['size'][0] * pred['size'][1] * pred['size'][2]
    vol_g = gt['size'][0] * gt['size'][1] * gt['size'][2]
    union = vol_p + vol_g - inter_3d
    return inter_3d / union if union > 1e-6 else 0.0


def download_cvat_annotations(config):
    """Download GT annotations from CVAT job, convert to predictions format."""
    headers = {'Authorization': f"Token {config['token']}"}
    cvat_url = config['cvat_url']
    job_id = config['job_id']
    frame_offset = config.get('frame_offset', 0)
    label_map_inv = {v: k for k, v in config['label_map'].items()}

    resp = requests.get(f'{cvat_url}/api/jobs/{job_id}/annotations', headers=headers)
    if resp.status_code != 200:
        print(f"[ERROR] Cannot download annotations: {resp.status_code}")
        return None

    data = resp.json()
    frame_boxes = {}

    for shape in data.get('shapes', []):
        if shape['type'] != 'cuboid':
            continue
        frame = shape['frame'] + frame_offset
        pts = shape['points']
        box = {
            'label': label_map_inv.get(shape['label_id'], 'unknown'),
            'score': 1.0,
            'center': [pts[0], pts[1], pts[2] + 0.15],
            'size': [pts[6], pts[7], pts[8]],
            'yaw': pts[5],
        }
        frame_boxes.setdefault(frame, []).append(box)

    for track in data.get('tracks', []):
        label_name = label_map_inv.get(track['label_id'], 'unknown')
        for shape in track.get('shapes', []):
            if shape.get('outside', False):
                continue
            frame = shape['frame'] + frame_offset
            pts = shape['points']
            box = {
                'label': label_name, 'score': 1.0,
                'center': [pts[0], pts[1], pts[2] + 0.15],
                'size': [pts[6], pts[7], pts[8]],
                'yaw': pts[5],
            }
            frame_boxes.setdefault(frame, []).append(box)

    result = []
    for frame in sorted(frame_boxes):
        result.append({
            'frame': frame,
            'num_boxes': len(frame_boxes[frame]),
            'boxes': frame_boxes[frame],
        })
    print(f"  Downloaded GT: {sum(len(f['boxes']) for f in result)} boxes / {len(result)} frames")
    return result


def evaluate_predictions(preds, gt, iou_accept=0.7, iou_edit=0.25):
    """Evaluate predictions against ground truth.

    iou_accept MẶC ĐỊNH 0.7 NGHIÊM (trung thực, không nới cho đẹp). Có thể truyền
    số khác, hoặc dict theo class CHỈ KHI đã chốt chuẩn nghiệm thu với khách.
    Returns accept/edit/reject/miss, IoU stats, heading, per-class.
    """
    pred_by_frame = {f['frame']: f['boxes'] for f in preds}
    gt_by_frame = {f['frame']: f['boxes'] for f in gt}
    all_frames = sorted(set(list(pred_by_frame.keys()) + list(gt_by_frame.keys())))

    stats = {
        'total_pred': 0, 'total_gt': 0,
        'accept': 0, 'edit': 0, 'reject': 0, 'miss': 0,
        'matched_ious': [], 'heading_errors': [],
        'per_class': {},
    }

    for frame in all_frames:
        p_boxes = pred_by_frame.get(frame, [])
        g_boxes = gt_by_frame.get(frame, [])
        classes = set([b['label'] for b in p_boxes] + [b['label'] for b in g_boxes])

        for cls in classes:
            p_cls = [b for b in p_boxes if b['label'] == cls]
            g_cls = [b for b in g_boxes if b['label'] == cls]

            if cls not in stats['per_class']:
                stats['per_class'][cls] = {
                    'total_pred': 0, 'total_gt': 0,
                    'accept': 0, 'edit': 0, 'reject': 0, 'miss': 0,
                    'matched_ious': [], 'heading_errors': [],
                }
            cs = stats['per_class'][cls]
            acc_thr = _accept_thr(cls, iou_accept)
            stats['total_pred'] += len(p_cls)
            stats['total_gt'] += len(g_cls)
            cs['total_pred'] += len(p_cls)
            cs['total_gt'] += len(g_cls)

            if not p_cls and not g_cls:
                continue
            if not p_cls:
                stats['miss'] += len(g_cls)
                cs['miss'] += len(g_cls)
                continue
            if not g_cls:
                stats['reject'] += len(p_cls)
                cs['reject'] += len(p_cls)
                continue

            iou_matrix = np.zeros((len(p_cls), len(g_cls)))
            for i, pb in enumerate(p_cls):
                for j, gb in enumerate(g_cls):
                    iou_matrix[i, j] = iou_3d(pb, gb)

            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_p, matched_g = set(), set()

            for r, c in zip(row_ind, col_ind):
                iou_val = iou_matrix[r, c]
                if iou_val < 0.01:
                    continue
                matched_p.add(r)
                matched_g.add(c)
                stats['matched_ious'].append(float(iou_val))
                cs['matched_ious'].append(float(iou_val))

                h_err = abs(normalize_angle(p_cls[r]['yaw'] - g_cls[c]['yaw']))
                stats['heading_errors'].append(float(h_err))
                cs['heading_errors'].append(float(h_err))

                if iou_val >= acc_thr:
                    stats['accept'] += 1
                    cs['accept'] += 1
                elif iou_val >= iou_edit:
                    stats['edit'] += 1
                    cs['edit'] += 1
                else:
                    stats['reject'] += 1
                    cs['reject'] += 1

            for i in range(len(p_cls)):
                if i not in matched_p:
                    stats['reject'] += 1
                    cs['reject'] += 1
            for j in range(len(g_cls)):
                if j not in matched_g:
                    stats['miss'] += 1
                    cs['miss'] += 1

    return stats


def _match_pairs(preds, gt, iou_min=0.1):
    """Ghép pred↔GT cùng class theo IoU (Hungarian). Trả list (cls, pred_box, gt_box, iou)."""
    pred_by = {f['frame']: f['boxes'] for f in preds}
    gt_by = {f['frame']: f['boxes'] for f in gt}
    pairs = []
    for frame in sorted(set(list(pred_by.keys()) + list(gt_by.keys()))):
        pb = pred_by.get(frame, [])
        gb = gt_by.get(frame, [])
        for cls in set([b['label'] for b in pb] + [b['label'] for b in gb]):
            p = [b for b in pb if b['label'] == cls]
            g = [b for b in gb if b['label'] == cls]
            if not p or not g:
                continue
            m = np.zeros((len(p), len(g)))
            for i, pbx in enumerate(p):
                for j, gbx in enumerate(g):
                    m[i, j] = iou_3d(pbx, gbx)
            ri, ci = linear_sum_assignment(-m)
            for r, c in zip(ri, ci):
                if m[r, c] >= iou_min:
                    pairs.append((cls, p[r], g[c], float(m[r, c])))
    return pairs


def compute_calibration(preds, gt, iou_min=0.25, min_n=5):
    """Suy hiệu chỉnh hệ thống theo class từ cặp pred↔GT đã match:
    scale size [sx,sy,sz] (median gt/pred), offset z (median), offset yaw (median,
    gập về ±90° để bỏ nhập nhằng 180°). Chỉ nhận class có >= min_n mẫu."""
    acc = {}
    for cls, P, G, _iou in _match_pairs(preds, gt, iou_min):
        d = acc.setdefault(cls, {'sx': [], 'sy': [], 'sz': [], 'dz': [], 'dyaw': []})
        for k, key in enumerate(('sx', 'sy', 'sz')):
            if P['size'][k] > 1e-3:
                d[key].append(G['size'][k] / P['size'][k])
        d['dz'].append(G['center'][2] - P['center'][2])
        diff = normalize_angle(G['yaw'] - P['yaw'])
        if diff > math.pi / 2:
            diff -= math.pi
        elif diff < -math.pi / 2:
            diff += math.pi
        d['dyaw'].append(diff)
    calib = {}
    for cls, d in acc.items():
        if len(d['dz']) < min_n:
            continue
        calib[cls] = {
            'scale': [float(np.median(d['sx'])), float(np.median(d['sy'])), float(np.median(d['sz']))],
            'dz': float(np.median(d['dz'])),
            'dyaw': float(np.median(d['dyaw'])),
            'n': len(d['dz']),
        }
    return calib


def apply_calibration(preds, calib):
    """Áp hiệu chỉnh per-class vào preds (trả bản mới, không sửa gốc)."""
    import copy
    out = copy.deepcopy(preds)
    for f in out:
        for b in f.get('boxes', []):
            c = calib.get(b['label'])
            if not c:
                continue
            s = c.get('scale', [1, 1, 1])
            b['size'] = [b['size'][0] * s[0], b['size'][1] * s[1], b['size'][2] * s[2]]
            b['center'] = [b['center'][0], b['center'][1], b['center'][2] + c.get('dz', 0.0)]
            b['yaw'] = normalize_angle(b['yaw'] + c.get('dyaw', 0.0))
    return out


def bias_report(preds, gt):
    """In bảng lệch hệ thống theo class (giúp biết 'edit' do size/z/yaw/vị trí)."""
    pairs = _match_pairs(preds, gt, iou_min=0.25)
    if not pairs:
        print("  (không có cặp match để phân tích bias)")
        return
    by = {}
    for cls, P, G, iou in pairs:
        d = by.setdefault(cls, {'dx': [], 'dy': [], 'dz': [], 'rl': [], 'rw': [], 'rh': [], 'dyaw': []})
        d['dx'].append(G['center'][0] - P['center'][0])
        d['dy'].append(G['center'][1] - P['center'][1])
        d['dz'].append(G['center'][2] - P['center'][2])
        for k, key in zip(range(3), ('rl', 'rw', 'rh')):
            if P['size'][k] > 1e-3:
                d[key].append(G['size'][k] / P['size'][k])
        diff = normalize_angle(G['yaw'] - P['yaw'])
        if diff > math.pi / 2:
            diff -= math.pi
        elif diff < -math.pi / 2:
            diff += math.pi
        d['dyaw'].append(abs(diff))
    print(f"\n  --- Bias theo class (median, n cặp) ---")
    print(f"  {'class':<14}{'n':>4}  {'Δx':>6}{'Δy':>6}{'Δz':>6}  {'l×':>6}{'w×':>6}{'h×':>6}  {'|Δyaw|°':>8}")
    for cls, d in sorted(by.items()):
        md = lambda a: float(np.median(a)) if a else 0.0
        print(f"  {cls:<14}{len(d['dz']):>4}  {md(d['dx']):>6.2f}{md(d['dy']):>6.2f}{md(d['dz']):>6.2f}  "
              f"{md(d['rl']):>6.2f}{md(d['rw']):>6.2f}{md(d['rh']):>6.2f}  {math.degrees(md(d['dyaw'])):>8.1f}")


def print_evaluation_report(stats, iou_accept=0.7, iou_edit=0.25):
    """Print formatted evaluation report."""
    total_gt = stats['total_gt']
    total_pred = stats['total_pred']
    acc_txt = "theo class" if isinstance(iou_accept, dict) else str(iou_accept)

    print("\n" + "=" * 65)
    print("  EVALUATION REPORT — Annotation Efficiency")
    print("=" * 65)

    print(f"\n  Predictions: {total_pred} boxes")
    print(f"  Ground Truth: {total_gt} boxes")
    print(f"  IoU thresholds: Accept >= {acc_txt}, Edit >= {iou_edit}")

    denom = max(total_gt, 1)
    a_pct = stats['accept'] / denom * 100
    e_pct = stats['edit'] / denom * 100
    m_pct = stats['miss'] / denom * 100
    r_pct = stats['reject'] / max(total_pred, 1) * 100

    print(f"\n  --- Annotation Efficiency (vs GT) ---")
    print(f"  Accept (IoU >= {acc_txt}):    {stats['accept']:>4d} / {total_gt}  = {a_pct:5.1f}%  -- giu nguyen")
    print(f"  Edit (IoU < accept):  {stats['edit']:>4d} / {total_gt}  = {e_pct:5.1f}%  -- can sua")
    print(f"  Miss (FN):              {stats['miss']:>4d} / {total_gt}  = {m_pct:5.1f}%  -- bi sot")
    print(f"  Reject (FP):            {stats['reject']:>4d} / {total_pred}  = {r_pct:5.1f}%  -- can xoa")

    if stats['matched_ious']:
        ious = np.array(stats['matched_ious'])
        print(f"\n  --- 3D IoU ---")
        print(f"  Mean IoU:   {ious.mean():.3f}")
        print(f"  Median IoU: {np.median(ious):.3f}")
        print(f"  IoU > 0.5:  {int((ious > 0.5).sum()):>4d} / {len(ious)}  = {(ious > 0.5).mean() * 100:.1f}%")
        print(f"  IoU > 0.7:  {int((ious > 0.7).sum()):>4d} / {len(ious)}  = {(ious > 0.7).mean() * 100:.1f}%")

    if stats['heading_errors']:
        herr = np.degrees(stats['heading_errors'])
        print(f"\n  --- Heading Error ---")
        print(f"  Mean:   {herr.mean():.1f} deg")
        print(f"  Median: {np.median(herr):.1f} deg")
        print(f"  < 10 deg:  {int((herr < 10).sum()):>4d} / {len(herr)}  = {(herr < 10).mean() * 100:.1f}%")
        print(f"  > 90 deg:  {int((herr > 90).sum()):>4d} / {len(herr)}  = {(herr > 90).mean() * 100:.1f}%  (dao dau)")

    if stats['per_class']:
        print(f"\n  --- Per-Class Breakdown ---")
        hdr = f"  {'Class':<20} {'GT':>4} {'Pred':>4} {'Accept':>7} {'Edit':>6} {'Miss':>6} {'Reject':>7} {'mIoU':>6} {'Heading':>8}"
        print(hdr)
        print(f"  {'-' * (len(hdr) - 2)}")
        for cls in sorted(stats['per_class'].keys()):
            cs = stats['per_class'][cls]
            d = max(cs['total_gt'], 1)
            pd_n = max(cs['total_pred'], 1)
            miou = np.mean(cs['matched_ious']) if cs['matched_ious'] else 0
            herr_mean = np.degrees(np.mean(cs['heading_errors'])) if cs['heading_errors'] else 0
            print(f"  {cls:<20} {cs['total_gt']:>4} {cs['total_pred']:>4}"
                  f" {cs['accept'] / d * 100:>6.1f}%"
                  f" {cs['edit'] / d * 100:>5.1f}%"
                  f" {cs['miss'] / d * 100:>5.1f}%"
                  f" {cs['reject'] / pd_n * 100:>6.1f}%"
                  f" {miou:>5.3f}"
                  f" {herr_mean:>6.1f} deg")

    print("\n" + "=" * 65)


def save_evaluation_json(stats, output_path, model_name="unknown"):
    """Save evaluation results as JSON for multi-model comparison."""
    report = {
        'model': model_name,
        'total_pred': stats['total_pred'],
        'total_gt': stats['total_gt'],
        'accept': stats['accept'],
        'edit': stats['edit'],
        'reject': stats['reject'],
        'miss': stats['miss'],
        'mean_iou': float(np.mean(stats['matched_ious'])) if stats['matched_ious'] else 0,
        'median_iou': float(np.median(stats['matched_ious'])) if stats['matched_ious'] else 0,
        'mean_heading_error_deg': float(np.degrees(np.mean(stats['heading_errors']))) if stats['heading_errors'] else 0,
        'per_class': {},
    }
    for cls, cs in stats['per_class'].items():
        report['per_class'][cls] = {
            'total_gt': cs['total_gt'], 'total_pred': cs['total_pred'],
            'accept': cs['accept'], 'edit': cs['edit'],
            'reject': cs['reject'], 'miss': cs['miss'],
            'mean_iou': float(np.mean(cs['matched_ious'])) if cs['matched_ious'] else 0,
            'mean_heading_error_deg': float(np.degrees(np.mean(cs['heading_errors']))) if cs['heading_errors'] else 0,
        }
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Evaluation saved: {output_path}")
    return report


def step_evaluate(config, preds, gt_source):
    """Run evaluation: compare predictions vs ground truth."""
    print("\n[EVALUATE] So sanh predictions vs ground truth")
    print("=" * 50)

    if gt_source == "cvat":
        print("  Downloading GT from CVAT...")
        gt = download_cvat_annotations(config)
        if gt is None:
            print("[ERROR] Khong tai duoc GT tu CVAT")
            return None
    else:
        if not os.path.exists(gt_source):
            print(f"[ERROR] File GT khong ton tai: {gt_source}")
            return None
        with open(gt_source) as f:
            gt = json.load(f)
        print(f"  GT file: {gt_source} ({sum(len(r['boxes']) for r in gt)} boxes)")

    stats = evaluate_predictions(preds, gt)
    print_evaluation_report(stats)
    return stats


# ============================================================
# Project config (YAML)
# ============================================================

def load_project_config(yaml_path):
    """Read project YAML config, return dict."""
    if yaml is None:
        print("[ERROR] PyYAML cần thiết: pip install pyyaml")
        sys.exit(1)
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    print(f"  Project config: {yaml_path}")
    print(f"    Project: {cfg.get('project_name', '(unnamed)')}")
    return cfg


def apply_project_config(cfg, args):
    """Override CLI args with project config values."""
    cvat = cfg.get('cvat', {})
    if not args.url and cvat.get('url'):
        args.url = cvat['url']
    if not args.user and cvat.get('user'):
        args.user = cvat['user']
    if not args.password and cvat.get('password'):
        args.password = cvat['password']

    data = cfg.get('data', {})
    if not args.data and data.get('path'):
        args.data = data['path']

    thresholds = cfg.get('thresholds', {})
    if thresholds:
        THRESH_MAP.update(thresholds)

    model = cfg.get('model', {})
    global SCORE_THRESH
    if model.get('score_threshold'):
        SCORE_THRESH = model['score_threshold']

    return cfg.get('calibration', None), cfg.get('tracking', {})


def upload_to_cvat(config, preds):
    headers = {
        'Authorization': f"Token {config['token']}",
        'Content-Type': 'application/json',
    }
    cvat_url = config['cvat_url']
    job_id = config['job_id']
    label_map = config['label_map']
    frame_offset = config.get('frame_offset', 0)

    print(f"  Xóa annotation cũ (Job {job_id})...")
    requests.put(f'{cvat_url}/api/jobs/{job_id}/annotations', headers=headers,
                 json={'shapes': [], 'tracks': [], 'tags': []})

    has_tracks = any(
        'track_id' in b for f in preds for b in f['boxes']
    )

    if has_tracks:
        tracks_dict = {}
        for frame_data in preds:
            frame_idx = frame_data['frame']
            cvat_frame = frame_idx - frame_offset
            for b in frame_data['boxes']:
                cname = b['label']
                if b['score'] < THRESH_MAP.get(cname, 0.28):
                    continue
                label_id = label_map.get(cname)
                if label_id is None:
                    continue
                tid = b['track_id']
                x, y, z_raw = b['center']
                l, w, h = b['size']
                z = z_raw - 0.15
                shape = {
                    'type': 'cuboid',
                    'frame': cvat_frame,
                    'points': [
                        round(x, 3), round(y, 3), round(z, 3),
                        0.0, 0.0, round(normalize_angle(b['yaw']), 4),
                        round(l, 3), round(w, 3), round(h, 3),
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    ],
                    'occluded': False,
                    'outside': False,
                    'attributes': [],
                }
                if tid not in tracks_dict:
                    tracks_dict[tid] = {
                        'label_id': label_id,
                        'frame': cvat_frame,
                        'group': 0,
                        'shapes': [],
                        'attributes': [],
                        'source': 'auto',
                    }
                tracks_dict[tid]['shapes'].append(shape)
                if cvat_frame < tracks_dict[tid]['frame']:
                    tracks_dict[tid]['frame'] = cvat_frame

        tracks = list(tracks_dict.values())
        total_shapes = sum(len(t['shapes']) for t in tracks)
        print(f"  Upload {len(tracks)} tracks ({total_shapes} cuboids)...")

        batch_size = 20
        for i in range(0, len(tracks), batch_size):
            batch = tracks[i:i + batch_size]
            resp = requests.patch(
                f'{cvat_url}/api/jobs/{job_id}/annotations?action=create',
                headers=headers,
                json={'tracks': batch},
            )
            status = "OK" if resp.status_code in (200, 201) else f"Error {resp.status_code}"
            print(f"    Batch {i // batch_size + 1}/{(len(tracks) - 1) // batch_size + 1} — {status}")
    else:
        shapes = []
        for frame_data in preds:
            frame_idx = frame_data['frame']
            cvat_frame = frame_idx - frame_offset
            for b in frame_data['boxes']:
                cname = b['label']
                if b['score'] < THRESH_MAP.get(cname, 0.28):
                    continue
                label_id = label_map.get(cname)
                if label_id is None:
                    continue
                x, y, z_raw = b['center']
                l, w, h = b['size']
                z = z_raw - 0.15
                shapes.append({
                    'type': 'cuboid',
                    'frame': cvat_frame,
                    'label_id': label_id,
                    'points': [
                        round(x, 3), round(y, 3), round(z, 3),
                        0.0, 0.0, round(normalize_angle(b['yaw']), 4),
                        round(l, 3), round(w, 3), round(h, 3),
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    ],
                    'occluded': False,
                    'outside': False,
                    'source': 'auto',
                })

        print(f"  Upload {len(shapes)} cuboids...")
        batch_size = 50
        for i in range(0, len(shapes), batch_size):
            batch = shapes[i:i + batch_size]
            resp = requests.patch(
                f'{cvat_url}/api/jobs/{job_id}/annotations?action=create',
                headers=headers,
                json={'shapes': batch},
            )
            status = "OK" if resp.status_code in (200, 201) else f"Error {resp.status_code}"
            print(f"    Batch {i // batch_size + 1}/{(len(shapes) - 1) // batch_size + 1} — {status}")

    task_url = f"{cvat_url}/tasks/{config['task_id']}/jobs/{job_id}"
    print(f"\n  CVAT: {task_url}")
    return task_url


def step3_upload(config, preds, calib_path=None, tracking_cfg=None,
                  overlay=False, overlay_output="overlays", ctx_dir=None, camera_index=0):
    print("\n[STEP 3] Heading correction + tracking + upload CVAT")
    print("=" * 50)

    pcd_dir = config.get('pcd_dir', '')
    print("  Heading correction...")
    preds = correct_heading(preds, pcd_dir)

    if tracking_cfg is None:
        tracking_cfg = {}
    if tracking_cfg.get('enabled', True):
        max_dist = tracking_cfg.get('max_distance', 5.0)
        preds = track_objects(preds, max_dist=max_dist)

    # Hiệu chỉnh hệ thống box theo GT (file box_calib.json) — bám GT hơn, giảm "cần sửa"
    box_calib_path = os.environ.get("BOX_CALIB", os.path.expanduser("~/box_calib.json"))
    if os.path.exists(box_calib_path):
        try:
            with open(box_calib_path) as _bf:
                _bc = json.load(_bf)
            if _bc:
                preds = apply_calibration(preds, _bc)
                print(f"  Box calibration: áp {len(_bc)} class ({box_calib_path})")
        except Exception as e:  # noqa: BLE001 — lỗi calib không chặn upload
            print(f"  [WARN] Bỏ qua box calibration: {e}")

    calib = None
    if calib_path:
        try:
            calib = load_calibration(calib_path)
            if calib:
                preds = project_predictions(preds, calib)
        except Exception as e:  # noqa: BLE001 — overlay/projection không được chặn upload
            print(f"  [WARN] Bỏ qua chiếu 3D->2D (lỗi calib): {e}")
            calib = None

    if overlay and calib:
        try:
            print("\n  Camera overlay...")
            generate_overlays(preds, calib, ctx_dir, overlay_output, camera_index)
        except Exception as e:  # noqa: BLE001 — lỗi overlay không được chặn upload
            print(f"  [WARN] Bỏ qua vẽ overlay: {e}")

    return upload_to_cvat(config, preds)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Auto Label 3D — pipeline 1 lệnh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", help="CVAT URL (vd: http://localhost:8080)")
    parser.add_argument("--user", help="CVAT username")
    parser.add_argument("--password", help="CVAT password")
    parser.add_argument("--task-name", help="Tên task CVAT")
    parser.add_argument("--data", help="Thư mục chứa pointcloud/ + context_images/")
    parser.add_argument("--config", help="File config.json (bỏ qua step 1 nếu có)")
    parser.add_argument("--predictions", help="File predictions.json (bỏ qua inference)")
    parser.add_argument("--skip-inference", action="store_true", help="Bỏ qua inference, chỉ upload")
    parser.add_argument("--output-predictions", default="predictions.json", help="Output predictions path")
    parser.add_argument("--project", help="File project config YAML")
    parser.add_argument("--calibration", help="File calib.json")
    parser.add_argument("--no-tracking", action="store_true", help="Tắt tracking")
    parser.add_argument("--overlay", action="store_true", help="Vẽ box 3D lên ảnh camera")
    parser.add_argument("--overlay-output", default="overlays", help="Thư mục output overlay (default: overlays)")
    parser.add_argument("--camera-index", type=int, default=1, help="Index ảnh camera (default: 1 = CAM_P_F)")
    parser.add_argument("--evaluate", action="store_true", help="Đánh giá predictions vs ground truth")
    parser.add_argument("--ground-truth", help="File GT JSON hoặc 'cvat' để tải từ CVAT")
    parser.add_argument("--model-name", default="transfusion", help="Tên model (cho report)")
    parser.add_argument("--eval-output", help="Lưu kết quả evaluation ra JSON")
    args = parser.parse_args()

    print("=" * 50)
    print("  Auto Label 3D Pipeline")
    print("=" * 50)

    calib_path = args.calibration
    tracking_cfg = {'enabled': not args.no_tracking}

    if args.project:
        print(f"\n[CONFIG] Loading project config...")
        proj_cfg = load_project_config(args.project)
        calib_from_cfg, tracking_from_cfg = apply_project_config(proj_cfg, args)
        if not calib_path and calib_from_cfg:
            calib_path = calib_from_cfg
        if tracking_from_cfg:
            tracking_cfg.update(tracking_from_cfg)

    # Step 1: Create task or load config
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = json.load(f)
        print(f"\n[STEP 1] Dùng config có sẵn: {args.config}")
        config_path = args.config
    else:
        if not all([args.url, args.user, args.password, args.task_name, args.data]):
            parser.error("Cần --url, --user, --password, --task-name, --data (hoặc --config)")
        config, config_path = step1_create_task(args)

    pcd_dir = config.get('pcd_dir', '')

    # Step 2: Inference
    if args.predictions and os.path.exists(args.predictions):
        print(f"\n[STEP 2] Dùng predictions có sẵn: {args.predictions}")
        with open(args.predictions) as f:
            preds = json.load(f)
    elif args.skip_inference:
        print("\n[STEP 2] Bỏ qua inference (--skip-inference)")
        print("  Chạy lại với --predictions <file> khi có kết quả từ Colab")
        return
    else:
        preds = step2_inference(pcd_dir, args.output_predictions)
        if preds is None:
            print(f"\n  Khi có predictions, chạy:")
            print(f"  python3 run.py --config {config_path} --predictions <file>")
            return

    # Evaluate mode
    if args.evaluate:
        if not args.ground_truth:
            print("[ERROR] Can --ground-truth <file.json> hoac --ground-truth cvat")
            sys.exit(1)
        eval_stats = step_evaluate(config, preds, args.ground_truth)
        if eval_stats and args.eval_output:
            save_evaluation_json(eval_stats, args.eval_output, model_name=args.model_name)
        return

    # Step 3: Upload
    total_boxes = sum(len(f['boxes']) for f in preds)
    total_frames = len(preds)
    print(f"\n  Input: {total_boxes} boxes / {total_frames} frames")

    ctx_dir = None
    if args.data:
        ctx_dir = os.path.join(args.data, "context_images")

    task_url = step3_upload(
        config, preds, calib_path=calib_path, tracking_cfg=tracking_cfg,
        overlay=args.overlay, overlay_output=args.overlay_output,
        ctx_dir=ctx_dir, camera_index=args.camera_index,
    )

    print("\n" + "=" * 50)
    print("  HOÀN TẤT!")
    print(f"  Mở CVAT: {task_url}")
    print("=" * 50)


if __name__ == "__main__":
    main()
