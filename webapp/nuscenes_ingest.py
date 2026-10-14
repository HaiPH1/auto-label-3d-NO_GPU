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
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

NSWEEPS = 10  # model nuScenes được train trên 10 sweep gộp (MAX_SWEEPS=10)

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


def write_binary_pcd(path, pts, fields):
    n, nf = pts.shape[0], len(fields)
    header = (f"# .PCD v0.7\nVERSION 0.7\nFIELDS {' '.join(fields)}\n"
              f"SIZE {' '.join(['4'] * nf)}\nTYPE {' '.join(['F'] * nf)}\n"
              f"COUNT {' '.join(['1'] * nf)}\n"
              f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(pts.astype(np.float32).tobytes())


def load_multisweep(nusc, sample, nsweeps=NSWEEPS, min_dist=1.0):
    """Gộp tối đa `nsweeps` quét LiDAR về frame keyframe (bù chuyển động ego), trả
    Nx5 [x,y,z,intensity,time]. Bỏ qua sweep thiếu file -> data đầy đủ được 10 sweep,
    data chỉ có keyframe vẫn chạy (1 sweep)."""
    ref_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ref_cs = nusc.get("calibrated_sensor", ref_sd["calibrated_sensor_token"])
    ref_pose = nusc.get("ego_pose", ref_sd["ego_pose_token"])
    ref_time = 1e-6 * ref_sd["timestamp"]
    ref_from_car = transform_matrix(ref_cs["translation"], Quaternion(ref_cs["rotation"]), inverse=True)
    car_from_global = transform_matrix(ref_pose["translation"], Quaternion(ref_pose["rotation"]), inverse=True)

    chunks = []
    sd = ref_sd
    for _ in range(nsweeps):
        path = os.path.join(nusc.dataroot, sd["filename"])
        if os.path.exists(path):
            pc = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :4].T  # 4xn
            keep = np.sqrt(pc[0] ** 2 + pc[1] ** 2) > min_dist
            pc = pc[:, keep]
            cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            pose = nusc.get("ego_pose", sd["ego_pose_token"])
            global_from_car = transform_matrix(pose["translation"], Quaternion(pose["rotation"]), inverse=False)
            car_from_cur = transform_matrix(cs["translation"], Quaternion(cs["rotation"]), inverse=False)
            trans = ref_from_car @ car_from_global @ global_from_car @ car_from_cur
            xyz = (trans @ np.vstack([pc[:3, :], np.ones(pc.shape[1])]))[:3, :]
            tlag = ref_time - 1e-6 * sd["timestamp"]
            chunks.append(np.vstack([xyz, pc[3, :], np.full(pc.shape[1], tlag)]))  # 5xn
        if sd["prev"] == "":
            break
        sd = nusc.get("sample_data", sd["prev"])
    return np.concatenate(chunks, axis=1).T.astype(np.float32)


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
        _, boxes, _ = nusc.get_sample_data(sample["data"]["LIDAR_TOP"])
        pts5 = load_multisweep(nusc, sample)  # Nx5: x,y,z,intensity,time (10 sweep)
        # zero-pad để sort theo chữ cái == thứ tự số (khớp eval + CVAT frame order)
        write_binary_pcd(os.path.join(pcd_dir, f"frame_{frame:03d}.pcd"), pts5,
                         ["x", "y", "z", "intensity", "timestamp"])

        cdir = os.path.join(out_dir, "context_images", f"frame_{frame:03d}")
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
