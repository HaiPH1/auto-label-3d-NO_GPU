# Deploy ALL-IN-ONE: CVAT + inference GPU trên cùng 1 VM Google Cloud

**Kiến trúc:** một VM GPU chạy cả CVAT (self-host, full 3D) lẫn inference TransFusion.
Dán nhãn xong có thể export task ra file backup để mang sang CVAT khác.

```
GPU VM (Google Cloud)
┌──────────────────────────────────────────┐
│  CVAT self-host (http://<IP>:8080)         │ ← nhóm truy cập qua internet
│  run.py: tạo task + inference GPU + upload │
└──────────────────────────────────────────┘
        │ export (04)              ▲ import (05)
        ▼                          │
   task_backup.zip  ───────────▶  CVAT khác (bất kỳ)
   (ảnh + box)
```

## Chuẩn bị (1 lần)
1. Bật **Billing** trên GCP.
2. Xin **quota GPU**: IAM & Admin → Quotas → `GPUs (all regions)` → Request → `1`.
3. `gcloud auth login` + `gcloud config set project <MÃ_DỰ_ÁN>`.

## Dùng
```bash
nano deploy/config.env          # đổi GCP_PROJECT, CVAT_PASS, DATA_DIR
bash deploy/cloud_setup.sh      # 1 lần: tạo VM + cài CVAT + model (~10-20')
bash deploy/autolabel.sh        # mỗi lần: đẩy data → infer → upload vào CVAT trên VM
```
Xong mở CVAT ở `http://<IP-VM>:8080` để annotate.

## Export sang CVAT khác (ảnh + box)
```bash
# Trên VM (hoặc máy nào có config.json + cvat-sdk):
pip install cvat-sdk
python3 04_export_task.py --config config.json --user admin --password <pass> --output task_backup.zip

# Đẩy backup vào CVAT khác:
python3 05_import_task.py --url http://cvat-khac:8080 --user admin --password <pass> --backup task_backup.zip
```
> Dùng **CVAT backup** nên file zip gồm cả dữ liệu lẫn annotation → CVAT khác restore ra task y hệt.

## Chi phí (QUAN TRỌNG)
All-in-one nghĩa là **tắt VM thì CVAT cũng tắt**. Khi annotate phải để VM chạy (tốn tiền GPU).
Khi nghỉ hẳn thì tắt:
```bash
gcloud compute instances stop autolabel-gpu --zone=us-central1-a
```
Muốn tiết kiệm lâu dài (annotate nhiều) → tách CVAT sang VM rẻ riêng (nói tôi làm bản đó).

## File
| File | Chạy ở đâu | Vai trò |
|---|---|---|
| `config.env` | — | Cấu hình |
| `cloud_setup.sh` | máy bạn | 1 lần: tạo VM + CVAT + inference env |
| `provision.sh` | VM | driver + Docker/CVAT + OpenPCDet + checkpoint |
| `autolabel.sh` | máy bạn | mỗi lần: đẩy data → run.py (create+infer+upload) |
| `../04_export_task.py` | bất kỳ | export task CVAT → backup .zip |
| `../05_import_task.py` | bất kỳ | import backup .zip → CVAT khác |
