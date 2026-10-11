# Auto Label 3D — Tool tự động gán nhãn 3D Cuboid

Tự động sinh nhãn 3D cuboid từ LiDAR point cloud và đẩy lên CVAT.

## Yêu cầu

- Python 3.8+
- Thư viện: `pip install numpy scipy requests`
- CVAT đang chạy (local hoặc remote)

## Chuẩn bị dữ liệu

Tạo thư mục `data/` với cấu trúc sau:

```
data/
├── pointcloud/           # BẮT BUỘC
│   ├── 0.pcd
│   ├── 1.pcd
│   └── ...
└── context_images/       # TÙY CHỌN (ảnh camera tham chiếu)
    ├── 0/
    │   ├── CAM_FRONT.jpg
    │   ├── CAM_BACK.jpg
    │   └── ...
    └── 1/
        └── ...
```

Tên file PCD là số thứ tự frame. Thư mục camera images trùng tên với PCD (không có đuôi .pcd).

---

## Cách 1: Pipeline 1 lệnh (khuyến nghị)

### Có GPU trên máy

```bash
python3 run.py \
    --url http://localhost:8080 \
    --user admin --password <password> \
    --task-name "My 3D Task" \
    --data ./data
```

Tự động: tạo task CVAT → inference trên GPU → heading correction → upload cuboids.

### Không có GPU

```bash
# Bước 1+3: tạo task + upload (inference từ Colab)
python3 run.py \
    --url http://localhost:8080 \
    --user admin --password <password> \
    --task-name "My 3D Task" \
    --data ./data \
    --predictions transfusion_predictions.json
```

File `transfusion_predictions.json` lấy từ Colab (xem `02_inference_colab.ipynb`).

### Đã có config từ lần trước

```bash
# Chỉ upload predictions mới
python3 run.py --config config.json --predictions predictions.json --skip-inference
```

---

## Cách 2: Chạy từng bước riêng

### Bước 1: Tạo task CVAT + upload data

```bash
python3 01_create_task.py \
    --url http://localhost:8080 \
    --user admin --password <password> \
    --task-name "My 3D Project" \
    --data ./data
```

Output: `config.json` — chứa token, job_id, label_map. **Không cần sửa gì thêm.**

### Bước 2: Chạy inference

**Có GPU:** Script tự chạy trong `run.py`.

**Không có GPU:** Mở `02_inference_colab.ipynb` trên Google Colab:
1. Runtime → Change runtime type → **T4 GPU**
2. Upload các file PCD
3. Chạy toàn bộ notebook
4. Tải `transfusion_predictions.json` về

### Bước 3: Upload predictions lên CVAT

```bash
python3 03_upload_predictions.py \
    --config config.json \
    --predictions transfusion_predictions.json
```

---

## Overlay camera — box 3D lên ảnh thật

Cần `calib.json` (intrinsics K + extrinsics ego2cam + distortion). Có 2 cách hiển thị:

### A. Offline — xuất ảnh JPG có box (để lưu/gửi báo cáo)

```bash
python3 run.py --config config.json --predictions transfusion_predictions.json \
    --calibration /path/to/calib.json \
    --overlay --overlay-output overlays_v2 \
    --data /path/to/data
```

Box được render vào file `overlays_v2/frame_X_<CAM>.jpg`. Lệnh này cũng upload cuboids lên CVAT.

### B. Plugin CVAT — box real-time trên ảnh camera trong CVAT (lúc annotate)

```bash
# Bật plugin
bash scripts/cvat-overlay/overlay.sh up \
    --calib /path/to/calib.json \
    --pcd-dir /path/to/pointcloud

# Kiểm tra trạng thái
bash scripts/cvat-overlay/overlay.sh status

# Tắt plugin
bash scripts/cvat-overlay/overlay.sh down
```

Sau khi `up` → vào CVAT bấm **Ctrl+Shift+R** để nạp plugin. Box sẽ chiếu lên panel context image real-time.

| Cách | Box hiện ở đâu | Dùng khi |
|---|---|---|
| **A. Offline** | File `overlays_v2/*.jpg` riêng | Lưu / gửi báo cáo, không cần CVAT |
| **B. Plugin** | Panel ảnh camera trong CVAT | Xem trực tiếp lúc annotate |

Mapping camera mặc định: `image_1` → `CAM_P_F` (sửa trong `scripts/cvat-overlay/overlay-config.py` nếu khác).

### Ví dụ data_day14 (đầy đủ)

```bash
cd /home/hai/task3/auto-label-3d-NO_GPU

# Offline + upload CVAT
python3 run.py --config config.json --predictions transfusion_predictions.json \
    --calibration /home/hai/task3/data_day14/calib.json \
    --overlay --overlay-output overlays_v2 \
    --data /home/hai/task3/data_day14

# Plugin CVAT
bash scripts/cvat-overlay/overlay.sh up \
    --calib /home/hai/task3/data_day14/calib.json \
    --pcd-dir /home/hai/task3/data_day14/pointcloud

# Tắt plugin khi xong
bash scripts/cvat-overlay/overlay.sh down
```

---

## Evaluation — đo chất lượng pre-label

So predictions với ground truth (nhãn đã sửa tay) để tính % Giữ / Sửa / Xóa / Bổ sung.

```bash
# GT từ file JSON
python3 run.py --config config.json --predictions transfusion_predictions.json \
    --evaluate --ground-truth gt.json --model-name transfusion

# GT tải từ CVAT (sau khi annotator sửa)
python3 run.py --config config.json --predictions transfusion_predictions.json \
    --evaluate --ground-truth cvat --eval-output eval_transfusion.json
```

Metrics: 3D IoU, Accept (IoU≥0.7) / Edit (0.25≤IoU<0.7) / Reject (FP) / Miss (FN), heading error, breakdown theo class.

---

## Setup GPU server (RTX 50xx / Blackwell)

GPU dòng RTX 50xx cần setup đặc biệt:

```bash
chmod +x ../setup_rtx5060ti.sh
../setup_rtx5060ti.sh
```

Chi tiết xem mục 8 trong `Huong_phat_trien.md`.

---

## Pipeline

```
                    run.py (1 lệnh)
                    ════════════════
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
│ STEP 1      │    │ STEP 2       │    │ STEP 3           │
│ Tạo task    │───▶│ Inference    │───▶│ Heading correct  │
│ Upload PCD  │    │ TransFusion  │    │ Ground alignment │
│             │    │              │    │ Upload CVAT      │
└─────────────┘    └──────────────┘    └──────────────────┘
      │              GPU local              │
      │              hoặc Colab             │
      ▼                  ▼                  ▼
  config.json     predictions.json     CVAT (cuboids)
```

## Files

| File | Mô tả |
|---|---|
| `run.py` | Pipeline 1 lệnh (gộp 3 bước) |
| `01_create_task.py` | Tạo task CVAT + upload data |
| `02_inference_colab.ipynb` | Inference trên Google Colab (fallback khi không có GPU) |
| `03_upload_predictions.py` | Upload predictions lên CVAT |
| `../setup_rtx5060ti.sh` | Setup GPU server RTX 50xx (Blackwell) — nằm ở `task3/` |
| `Huong_phat_trien.md` | Kiến trúc & roadmap phát triển |
