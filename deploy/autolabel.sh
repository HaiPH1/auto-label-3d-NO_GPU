#!/bin/bash
# ==============================================================================
# AUTOLABEL (chạy MỖI LẦN, từ máy bạn) — ALL-IN-ONE
#   Đẩy data lên VM → chạy run.py NGAY TRÊN VM (tạo task + inference GPU +
#   upload cuboid) vào CVAT trên chính VM đó.
#
# Cách dùng:  bash deploy/autolabel.sh
# ==============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"

log() { echo -e "\n\033[1;36m==> $*\033[0m"; }
die() { echo -e "\033[1;31mLỖI: $*\033[0m" >&2; exit 1; }

G=(--zone="$GCP_ZONE" --project="$GCP_PROJECT")
[ -d "$DATA_DIR" ] || die "Không thấy thư mục data: $DATA_DIR"

# Bật VM nếu đang tắt
STATUS="$(gcloud compute instances describe "$VM_NAME" "${G[@]}" --format='get(status)' 2>/dev/null || echo NOTFOUND)"
[ "$STATUS" = "NOTFOUND" ] && die "Chưa có VM. Chạy: bash deploy/cloud_setup.sh"
if [ "$STATUS" != "RUNNING" ]; then
  log "VM đang '$STATUS' — bật lên..."
  gcloud compute instances start "$VM_NAME" "${G[@]}"
  for i in $(seq 1 30); do
    gcloud compute ssh "$VM_NAME" "${G[@]}" --command="echo ok" >/dev/null 2>&1 && break; sleep 10
  done
fi

EXTERNAL_IP="$(gcloud compute instances describe "$VM_NAME" "${G[@]}" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"

# ① Đẩy data lên VM
log "[1/2] Đẩy data lên VM..."
gcloud compute ssh "$VM_NAME" "${G[@]}" --command="rm -rf ~/data && mkdir -p ~/data"
gcloud compute scp --recurse "$DATA_DIR"/* "$VM_NAME":~/data/ "${G[@]}"

# ② Chạy toàn bộ pipeline trên VM (tới CVAT localhost của VM)
log "[2/2] Chạy pipeline trên VM (tạo task + inference GPU + upload)..."
gcloud compute ssh "$VM_NAME" "${G[@]}" --command="\
  source ~/venv/bin/activate && cd ~/auto-label-3d && \
  python3 run.py --url http://localhost:8080 \
    --user '$CVAT_USER' --password '$CVAT_PASS' \
    --task-name '$TASK_NAME' --data ~/data"

log "XONG. Mở CVAT: http://$EXTERNAL_IP:8080"
echo "  ⚠️ Tắt VM cho đỡ tốn tiền:  gcloud compute instances stop $VM_NAME --zone=$GCP_ZONE"
