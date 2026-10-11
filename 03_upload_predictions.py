"""
Bước 3: Upload predictions lên CVAT với heading correction.

Cách dùng:
    python3 03_upload_predictions.py \
        --config config.json \
        --predictions transfusion_predictions.json

config.json được tạo tự động bởi 01_create_task.py.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

import requests

THRESH_MAP = {
    'truck': 0.20, 'bus': 0.25, 'motorcycle': 0.25, 'car': 0.28,
    'trailer': 0.25, 'pedestrian': 0.30, 'traffic_cone': 0.25, 'barrier': 0.25
}

HEADING_CLASSES = {'car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'motorcycle'}


def normalize_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def load_pcd(path):
    with open(path, 'rb') as f:
        n_points = 0
        while True:
            line = f.readline().decode('ascii', 'ignore').strip()
            if line.startswith('POINTS'):
                n_points = int(line.split()[1])
            if line == 'DATA binary':
                break
        data = f.read(n_points * 16)
    pts = np.frombuffer(data, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]


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
                'yaw': b['yaw'], 'flipped': False, 'confident': False
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
    print(f"  Phase 1 (velocity): flipped {flipped_p1}")

    # Phase 2: Point-cloud density
    flipped_p2 = 0
    for fid in frame_ids:
        pcd_path = os.path.join(pcd_dir, f'{fid}.pcd')
        if not os.path.exists(pcd_path):
            continue
        try:
            pts = load_pcd(pcd_path)
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
            front_count = np.sum(local_x > 0)
            back_count = np.sum(local_x < 0)
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
    print(f"  Phase 2 (point cloud): flipped {flipped_p2}")

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
    print(f"  Phase 3 (lane consistency): flipped {flipped_p3}")

    for fd in preds:
        fid = fd['frame']
        for i, b in enumerate(fd['boxes']):
            b['yaw'] = all_boxes[fid][i]['yaw']
    return preds


def upload(config, preds):
    headers = {
        'Authorization': f"Token {config['token']}",
        'Content-Type': 'application/json'
    }
    cvat_url = config['cvat_url']
    job_id = config['job_id']
    label_map = config['label_map']
    frame_offset = config.get('frame_offset', 0)

    print(f"\nXóa annotation cũ trên Job {job_id}...")
    requests.put(f'{cvat_url}/api/jobs/{job_id}/annotations', headers=headers,
                 json={'shapes': [], 'tracks': [], 'tags': []})

    shapes = []
    for frame_data in preds:
        frame_idx = frame_data['frame']
        cvat_frame = frame_idx - frame_offset
        count = 0
        for b in frame_data['boxes']:
            cname = b['label']
            if b['score'] < THRESH_MAP.get(cname, 0.28):
                continue
            label_id = label_map.get(cname)
            if label_id is None:
                continue

            x, y, z_raw = b['center']
            l, w, h = b['size']
            yaw = b['yaw']
            z = z_raw - 0.15

            points = [
                round(x, 3), round(y, 3), round(z, 3),
                0.0, 0.0, round(normalize_angle(yaw), 4),
                round(l, 3), round(w, 3), round(h, 3),
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]
            shapes.append({
                'type': 'cuboid',
                'frame': cvat_frame,
                'label_id': label_id,
                'points': points,
                'occluded': False,
                'outside': False,
                'source': 'auto'
            })
            count += 1
        print(f"  Frame {frame_idx} -> CVAT frame {cvat_frame}: {count} cuboids")

    print(f"\nTổng: {len(shapes)} cuboids, đang upload...")

    batch_size = 50
    for i in range(0, len(shapes), batch_size):
        batch = shapes[i:i + batch_size]
        resp = requests.patch(
            f'{cvat_url}/api/jobs/{job_id}/annotations?action=create',
            headers=headers,
            json={'shapes': batch}
        )
        status = "OK" if resp.status_code in (200, 201) else f"Lỗi {resp.status_code}"
        print(f"  Batch {i // batch_size + 1} ({len(batch)} shapes) - {status}")

    print(f"\nHoàn tất! Mở CVAT: {cvat_url}/tasks/{config['task_id']}/jobs/{job_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload predictions lên CVAT")
    parser.add_argument("--config", required=True, help="File config.json từ bước 1")
    parser.add_argument("--predictions", required=True, help="File transfusion_predictions.json từ Colab")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    with open(args.predictions) as f:
        preds = json.load(f)

    pcd_dir = config.get('pcd_dir', '')

    print("Heading correction...")
    preds = correct_heading(preds, pcd_dir)

    upload(config, preds)


if __name__ == "__main__":
    main()
