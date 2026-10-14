"""
Adapter: nuScenes THÔ (v1.0-*/ + samples/) -> format nội bộ của tool.
Dùng nuscenes-devkit. Convention GT đã verify (yaw=quaternion_yaw, size=wlh[[1,0,2]]).

    python3 -I nuscenes_ingest.py list    <dataroot>
    python3 -I nuscenes_ingest.py convert <dataroot> <scene_name> <out_dir>

convert -> out_dir/{pointcloud/frame_N.pcd, context_images/frame_N/CAM_*.jpg, calib.json, gt.json}
Chỉ xử lý scene có đủ file (bỏ qua scene chỉ có metadata).
"""
import json
import math
import os
import shutil
import sys

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw

CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
IMG_SIZE = [1600, 900]

CAT2DET = {
    "vehicle.car": "car", "vehicle.truck": "truck",
    "vehicle.bus.rigid": "bus", "vehicle.bus.bendy": "bus",
    "vehicle.trailer": "trailer", "vehicle.construction": "construction_vehicle",
    "human.pedestrian.adult": "pedestrian", "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "vehicle.motorcycle": "motorcycle", "vehicle.bicycle": "bicycle",
    "movable_object.trafficcone": "traffic_cone", "movable_object.barrier": "barrier",
}


def find_version(dataroot):
    for v in ("v1.0-mini", "v1.0-trainval", "v1.0-test"):
        if os.path.isdir(os.path.join(dataroot, v)):
            return v
    raise SystemExit(f"[ingest] không thấy thư mục v1.0-* trong {dataroot}")


def quat_to_R(q):
    w, x, y, z = q
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def mat4(R, t):
    return [[R[0][0], R[0][1], R[0][2], t[0]], [R[1][0], R[1][1], R[1][2], t[1]],
            [R[2][0], R[2][1], R[2][2], t[2]], [0.0, 0.0, 0.0, 1.0]]


def inv_rigid(R, t):
    Rt = [[R[j][i] for j in range(3)] for i in range(3)]
    ti = [-(Rt[i][0] * t[0] + Rt[i][1] * t[1] + Rt[i][2] * t[2]) for i in range(3)]
    return mat4(Rt, ti)


def write_binary_pcd(path, pts):
    n = pts.shape[0]
    header = ("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n"
              "TYPE F F F F\nCOUNT 1 1 1 1\n"
              f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(pts.astype(np.float32).tobytes())


def scene_has_files(nusc, scene):
    sd = nusc.get("sample_data", nusc.get("sample", scene["first_sample_token"])["data"]["LIDAR_TOP"])
    return os.path.exists(os.path.join(nusc.dataroot, sd["filename"]))


def build_calib(nusc, sample):
    calib = {"pcd_frame": "lidar", "lidar": {}, "cameras": {}}
    lc = nusc.get("calibrated_sensor",
                  nusc.get("sample_data", sample["data"]["LIDAR_TOP"])["calibrated_sensor_token"])
    calib["lidar"]["sensor2ego"] = mat4(quat_to_R(lc["rotation"]), lc["translation"])
    for cam in CAMS:
        if cam not in sample["data"]:
            continue
        cc = nusc.get("calibrated_sensor",
                      nusc.get("sample_data", sample["data"][cam])["calibrated_sensor_token"])
        calib["cameras"][cam] = {
            "K": [[float(v) for v in row] for row in cc["camera_intrinsic"]],
            "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
            "ego2cam": inv_rigid(quat_to_R(cc["rotation"]), cc["translation"]),
            "image_size": IMG_SIZE,
        }
    return calib


def convert(nusc, scene, out_dir):
    pcd_dir = os.path.join(out_dir, "pointcloud")
    os.makedirs(pcd_dir, exist_ok=True)
    token = scene["first_sample_token"]
    first_sample = nusc.get("sample", token)
    with open(os.path.join(out_dir, "calib.json"), "w") as f:
        json.dump(build_calib(nusc, first_sample), f)

    gt = []
    frame = 0
    while token:
        sample = nusc.get("sample", token)
        lidar_path, boxes, _ = nusc.get_sample_data(sample["data"]["LIDAR_TOP"])
        raw = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
        write_binary_pcd(os.path.join(pcd_dir, f"frame_{frame}.pcd"), raw[:, :4])

        cdir = os.path.join(out_dir, "context_images", f"frame_{frame}")
        os.makedirs(cdir, exist_ok=True)
        for cam in CAMS:
            if cam not in sample["data"]:
                continue
            src = os.path.join(nusc.dataroot, nusc.get("sample_data", sample["data"][cam])["filename"])
            if os.path.exists(src):
                shutil.copy(src, os.path.join(cdir, cam + ".jpg"))

        fbx = []
        for b in boxes:
            label = CAT2DET.get(b.name)
            if label is None:
                continue
            ann = nusc.get("sample_annotation", b.token)
            if ann["num_lidar_pts"] < 1 or math.hypot(float(b.center[0]), float(b.center[1])) > 50.0:
                continue
            w, l, h = float(b.wlh[0]), float(b.wlh[1]), float(b.wlh[2])
            fbx.append({"label": label,
                        "center": [float(b.center[0]), float(b.center[1]), float(b.center[2])],
                        "size": [l, w, h], "yaw": float(quaternion_yaw(b.orientation))})
        gt.append({"frame": frame, "boxes": fbx})
        token = sample["next"]
        frame += 1

    with open(os.path.join(out_dir, "gt.json"), "w") as f:
        json.dump(gt, f)
    print(f"[ingest] {scene['name']}: {frame} frames, {sum(len(r['boxes']) for r in gt)} GT boxes")


def main():
    cmd = sys.argv[1]
    dataroot = sys.argv[2]
    nusc = NuScenes(version=find_version(dataroot), dataroot=dataroot, verbose=False)
    if cmd == "list":
        for s in nusc.scene:
            if scene_has_files(nusc, s):
                print(s["name"])
    elif cmd == "convert":
        scene_name, out_dir = sys.argv[3], sys.argv[4]
        scene = next((s for s in nusc.scene if s["name"] == scene_name), None)
        if scene is None:
            raise SystemExit(f"[ingest] không thấy scene {scene_name}")
        convert(nusc, scene, out_dir)
    else:
        raise SystemExit("usage: nuscenes_ingest.py list|convert ...")


if __name__ == "__main__":
    main()
