# KIẾN TRÚC TOOL TỰ ĐỘNG GÁN NHÃN 3D CUBOID

## Mục tiêu

Xây dựng tool nội bộ cho công ty, tự động sinh nhãn 3D cuboid từ dữ liệu LiDAR point cloud, đẩy lên CVAT và đo lường hiệu quả so với gán tay. Tool hỗ trợ nhiều loại sensor/calibration và nhiều model detection.

---

## 1. Tổng quan kiến trúc

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│  Point Cloud (.pcd)  +  Camera Images  +  Calibration (calib.json) │
└──────────┬──────────────────┬──────────────────┬────────────────┘
           │                  │                  │
           ▼                  │                  │
┌─────────────────────┐       │                  │
│  1. INFERENCE       │       │                  │
│  Multi-Model Engine │       │                  │
│  ─────────────────  │       │                  │
│  • TransFusion-L    │       │                  │
│  • CenterPoint      │       │                  │
│  • PointPillars      │       │                  │
│  → raw predictions   │       │                  │
└──────────┬──────────┘       │                  │
           │                  │                  │
           ▼                  │                  │
┌─────────────────────┐       │                  │
│  2. POST-PROCESSING │       │                  │
│  ─────────────────  │       │                  │
│  • NMS + Threshold   │       │                  │
│  • Heading Correct.  │◄─────┘ (PCD data)       │
│    (3 pha)           │                          │
│  • Ground Alignment  │                          │
│  → refined cuboids   │                          │
└──────────┬──────────┘                          │
           │                                      │
     ┌─────┼──────────────┐                       │
     │     │              │                       │
     ▼     ▼              ▼                       │
┌────────┐ ┌───────────┐ ┌────────────────┐       │
│3. CVAT │ │4. OVERLAY │ │5. EVALUATION   │       │
│ EXPORT │ │  PROJECTION│ │  & BENCHMARK   │       │
│────────│ │───────────│ │────────────────│       │
│Upload  │ │Chiếu 3D   │◄────────────────┘       │
│REST API│ │lên ảnh    │ │3D IoU vs GT    │
│shapes/ │ │camera qua │ │% Giữ/Sửa/Xóa  │
│tracks  │ │calibration│ │Latency, VRAM   │
└────────┘ └───────────┘ └────────────────┘
```

---

## 2. Các module chi tiết

### 2.1. Calibration Pipeline (`calibration/`)

Lõi của hệ thống — quyết định chất lượng overlay projection và khả năng hỗ trợ nhiều loại sensor.

**Input:** File `calib.json` theo chuẩn:
```json
{
  "pcd_frame": "lidar",
  "lidar": {
    "sensor2ego": [[4x4 matrix]]
  },
  "cameras": {
    "CAM_FRONT": {
      "K": [[3x3 intrinsics]],
      "dist": [k1, k2, p1, p2, k3],
      "ego2cam": [[4x4 extrinsics]],
      "image_size": [W, H]
    }
  }
}
```

**Hỗ trợ nguồn calibration:**
- nuScenes (trích xuất từ `nuscenes-devkit` qua sample token)
- Custom sensor (nhập tay intrinsics/extrinsics)
- Tham chiếu từ dự án Day14 (`cvat3d_fusion.py` — đã có sẵn bộ đọc calib, projection với lens distortion Brown-Conrady)

### 2.2. Multi-Model Inference Engine (`inference/`)

| Model | mAP (nuScenes) | Tốc độ | Khi nào dùng |
|-------|----------------|--------|-------------|
| PointPillars | ~40% | ~60 FPS | Prototype nhanh, GPU yếu |
| CenterPoint | ~56% | ~20 FPS | Baseline chuẩn công nghiệp |
| TransFusion-L | ~65% | ~8 FPS | Production, cần độ chính xác cao |

**Output chuẩn hóa** (`predictions.json`):
```json
[
  {
    "frame": 20,
    "boxes": [
      {
        "label": "car",
        "score": 0.65,
        "center": [x, y, z],
        "size": [length, width, height],
        "yaw": 1.57
      }
    ]
  }
]
```

### 2.3. Post-Processing (`postprocess/`)

#### Heading Correction — 3 pha

Giải quyết lỗi **đảo đầu 180°** do BEV đối xứng. Không có module này, ~25-35% xe bị quay ngược đầu.

| Pha | Phương pháp | Điều kiện flip |
|-----|------------|---------------|
| 1. Velocity-based | So khớp box giữa 2 frame liên tiếp (Hungarian matching), tính vận tốc tương đối sau khi trừ ego-motion | `cos(heading, velocity) < -0.5` và `speed > 1.5 m/s` |
| 2. Point-cloud density | Đếm điểm nửa trước/sau box, đối chiếu hướng nhìn từ LiDAR | `back_count > front_count × 1.5` khi mặt trước hướng về LiDAR |
| 3. Lane consistency | So heading với các xe lân cận trong bán kính 20m | `opposed ≥ 2` và `opposed > aligned × 2` |

#### Ground Alignment

Hiệu chỉnh cao độ tâm: `z_adj = z - 0.15m` để đáy box chạm mặt đường.

### 2.4. CVAT Export (`export/`)

**Coordinate mapping (đã kiểm chứng):**
```
dx = length    (front arrow CVAT = trục +X)
dy = width
dz = height
rz = yaw       (KHÔNG cần trừ π/2)
```

**CVAT points array (16 phần tử):**
```
[x, y, z_adj, 0, 0, rz, dx, dy, dz, 0, 0, 0, 0, 0, 0, 0]
```

**Upload:** REST API `PATCH /api/jobs/{id}/annotations?action=create`, batch 50 shapes/request.

### 2.5. Camera Overlay Projection (`overlay/`)

Chiếu 3D cuboid lên ảnh camera thật trong CVAT — tham khảo kiến trúc từ dự án Day14:

- **Offline:** Script Python (`cvat3d_fusion.py`) — render overlay image với wireframe + heading direction + depth label
- **Online (trong CVAT):** JavaScript plugin inject vào `cvat_ui` container qua nginx, chiếu cuboid real-time trên panel camera khi annotator làm việc

Yêu cầu: **calibration data** (calib.json) — không có calibration thì không chiếu được.

### 2.6. Evaluation & Benchmark (`evaluation/`)

**Metric chính — Annotation Efficiency:**

| Metric | Định nghĩa | Ý nghĩa |
|--------|-----------|---------|
| % Giữ (Accept) | `IoU_3D ≥ 0.7` | Box đúng, annotator không cần sửa |
| % Sửa (Edit) | `0.25 ≤ IoU_3D < 0.7` | Box gần đúng, cần chỉnh kích thước/yaw |
| % Xóa (Reject) | False Positive — box nhiễu | Annotator phải xóa |
| % Bổ sung (Miss) | False Negative — vật bị sót | Annotator phải vẽ mới |

**Metric phụ:**
- 3D IoU so với Ground Truth (nhãn đã nghiệm thu)
- Latency: ms/frame, FPS
- VRAM peak usage
- So sánh đa model trên cùng dataset

---

## 3. Cấu trúc thư mục đề xuất

```
auto-label-3d-NO_GPU/
├── configs/
│   └── calib.json              # Calibration cho từng dự án/sensor
├── inference/
│   ├── run_transfusion.py
│   ├── run_centerpoint.py
│   └── run_pointpillars.py
├── postprocess/
│   ├── heading_correction.py   # 3-phase heading fix
│   ├── ground_alignment.py
│   └── nms_filter.py
├── export/
│   ├── cvat_upload.py          # REST API upload (shapes/tracks)
│   └── cvat_format.py          # Coordinate mapping
├── overlay/
│   ├── projection.py           # Offline render (PIL/OpenCV)
│   └── cvat_plugin/            # Online JS plugin cho CVAT UI
│       ├── overlay.js
│       ├── overlay.py           # Install/uninstall plugin
│       └── nginx.conf
├── evaluation/
│   ├── metrics.py              # 3D IoU, Accept/Edit/Reject rate
│   └── benchmark.py            # Multi-model comparison
├── data/
│   ├── pointcloud/
│   └── context_images/
└── README.md
```

---

## 4. Model Detection — 3 tầng

### Tầng 1 — Core (ưu tiên làm trước)

| Model | mAP (nuScenes paper) | Đặc điểm | Trạng thái |
|-------|---------------------|----------|-----------|
| **TransFusion-L** | 65.52% (CVPR 2022) | LiDAR-only, chính xác nhất, Sparse Conv + Transformer | **Đã chạy được** trên Colab T4 |

> Lưu ý: mAP 65.52% là con số từ paper trên nuScenes test set. Trên dữ liệu thực tế của công ty sẽ khác tùy domain — cần module Evaluation để đo con số thật.

### Tầng 2 — Baseline so sánh

| Model | mAP (nuScenes paper) | Đặc điểm | Trạng thái |
|-------|---------------------|----------|-----------|
| **CenterPoint** | 56.4% (CVPR 2021) | Chuẩn công nghiệp, cân bằng tốc độ/chính xác | Chưa tích hợp |
| **PointPillars** | 40.1% (CVPR 2019) | Nhanh nhất (~60 FPS), nhẹ, chấp nhận annotator sửa nhiều hơn | Chưa tích hợp |

Cả 3 model tầng 1+2 đều chạy trên **OpenPCDet** — cùng framework, cùng pipeline, chỉ đổi config + checkpoint. Colab notebook hiện tại chỉ cần sửa file config + weight path là chuyển model.

### Tầng 3 — Mở rộng (khi có nhu cầu)

| Model | mAP (nuScenes paper) | Đặc điểm | Khi nào cần |
|-------|---------------------|----------|------------|
| **BEVFusion** | 70.2% (ICRA 2023) | LiDAR + Camera fusion, tận dụng 6 ảnh camera | Khi có calibration chuẩn và cần độ chính xác cao nhất |
| **VoxelNeXt** | 60.5% (CVPR 2023) | Fully sparse, không cần NMS, deploy gọn | Khi cần tối ưu inference pipeline |

---

## 5. Tiến độ hiện tại

### Đã hoàn thành

| Module | File hiện có | Trạng thái |
|--------|-------------|-----------|
| Inference (TransFusion-L) | `02_inference_colab.ipynb` | ✅ Chạy được trên Colab T4, output `predictions.json` |
| Heading Correction 3 pha | `cvat_auto_label.py`, `cvat_local_upload.py` | ✅ Velocity + Point-cloud density + Lane consistency |
| Ground Alignment | Tích hợp trong upload scripts | ✅ `z_adj = z - 0.15m` |
| CVAT Export (remote) | `cvat_auto_label.py` | ✅ Upload Job 4073, 145 cuboids |
| CVAT Export (local) | `cvat_local_upload.py` | ✅ Upload Job 49, 145 cuboids |
| Camera Projection (basic) | `verify_camera_projection.py` | ⚠️ Chỉ CAM_FRONT, calibration xấp xỉ |

### Chưa làm

| Module | Mô tả | Ưu tiên |
|--------|-------|---------|
| Calibration pipeline | Đọc calib từ nhiều nguồn (nuScenes, custom) | P0 |
| Camera Overlay (chuẩn) | Projection đúng với lens distortion, tham khảo Day14 `cvat3d_fusion.py` | P2 |
| CVAT Overlay Plugin | JS plugin chiếu box lên ảnh camera real-time trong CVAT, tham khảo Day14 `scripts/cvat-overlay/` | P2 |
| Evaluation metrics | 3D IoU, % Giữ/Sửa/Xóa | P2 |
| Multi-model (CenterPoint, PointPillars) | Thêm config + checkpoint vào Colab notebook | P3 |
| Benchmark visualization | Biểu đồ so sánh đa model | P3 |

---

## 6. Vai trò của Colab trong pipeline

Colab **thay thế cho bước Inference** khi không có GPU server.

```
Pipeline đầy đủ (có server GPU):
  Server GPU → inference → predictions.json → heading correction → CVAT upload

Pipeline fallback (không có server):
  Colab T4/A100 → inference → predictions.json → (tải về local) → heading correction → CVAT upload
```

| | Server GPU | Google Colab |
|---|-----------|-------------|
| **Khi nào dùng** | Có server với GPU (sm_75 → sm_120, bao gồm RTX 50xx) | Không có server GPU |
| **GPU** | Tùy server (RTX 3090, A100...) | T4 (free) hoặc A100 (Pro) |
| **Ưu điểm** | Nhanh, batch lớn, tự chủ | Zero setup, luôn hoạt động, miễn phí |
| **Nhược điểm** | Cần setup CUDA/PyTorch/SpConv | Giới hạn thời gian, phải tải file thủ công |
| **Output** | `predictions.json` (cùng format) | `predictions.json` (cùng format) |

Các bước sau inference (heading correction, ground alignment, CVAT upload) chạy trên máy local — **không cần GPU**, chỉ cần Python + numpy + scipy.

File Colab hiện tại: `02_inference_colab.ipynb` — dùng micromamba + Python 3.10 + PyTorch cu121 + spconv-cu120 trên T4.

---

## 7. Tham chiếu code hiện có

| File | Vai trò | Tái sử dụng vào module |
|------|---------|----------------------|
| `cvat_auto_label.py` | Upload remote CVAT + heading correction 3 pha | → `export/` + `postprocess/` |
| `cvat_local_upload.py` | Upload local CVAT | → `export/` |
| `verify_camera_projection.py` | Chiếu 3D lên CAM_FRONT (nuScenes approx) | → `overlay/` |
| `02_inference_colab.ipynb` | Inference trên Colab T4 GPU | → `inference/` (fallback) |
| Day14 `cvat3d_fusion.py` | Projection chuẩn với calibration + lens distortion | → `overlay/` (tham khảo chính) |
| Day14 `scripts/cvat-overlay/` | JS plugin chiếu box real-time trong CVAT UI | → `overlay/cvat_plugin/` |

---

## 8. Setup GPU server — RTX 5060 Ti (Blackwell sm_120)

**Đã test thành công: 2026-10-07** trên RTX 5060 Ti 16GB, Ubuntu 24.04, Driver 580, CUDA 13.0.

GPU Blackwell (RTX 50xx) cần setup đặc biệt vì PyTorch stable chưa có sm_120.

### Yêu cầu

| Thành phần | Version | Lý do |
|---|---|---|
| PyTorch | **nightly cu128** | Bản stable chưa có sm_120 |
| spconv | **spconv-cu126** | spconv-cu128 chưa có trên PyPI, cu126 forward compatible |
| kornia | **0.7.3** | Version mới lỗi TorchScript với OpenPCDet |
| OpenPCDet | build từ source | Cần bypass CUDA version check (system CUDA 13.x vs PyTorch cu128) |

### Cách chạy

```bash
# Trên server RTX 5060 Ti:
chmod +x setup_rtx5060ti.sh
./setup_rtx5060ti.sh

# Sau khi setup xong:
source ~/transfusion_env/bin/activate
cd ~/OpenPCDet
python3 tools/demo.py --cfg_file <config> --ckpt <checkpoint> --data_path <pcd_folder>
```

### Lưu ý khi build OpenPCDet

1. `requirements.txt` sẽ đè PyTorch nightly bằng bản stable → **phải cài lại nightly sau đó**
2. System CUDA 13.x ≠ PyTorch cu128 → cần **monkey-patch `_check_cuda_version`**
3. Argo2 dataset dùng `@torch.jit.script` không tương thích PyTorch nightly → **comment dòng đó** (không ảnh hưởng TransFusion/nuScenes)

### GPU tương thích

| Kiến trúc | sm | GPU | Trạng thái |
|---|---|---|---|
| Turing | sm_75 | T4, RTX 2080 | PyTorch stable OK |
| Ampere | sm_80/86 | A100, RTX 3090 | PyTorch stable OK |
| Ada Lovelace | sm_89 | RTX 4090, RTX 4070 | PyTorch stable OK |
| **Blackwell** | **sm_120** | **RTX 5060 Ti, 5070, 5090** | **PyTorch nightly OK** (dùng `setup_rtx5060ti.sh`) |
