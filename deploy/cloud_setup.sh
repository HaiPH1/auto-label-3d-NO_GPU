#!/bin/bash
# ==============================================================================
# CLOUD SETUP (chạy 1 LẦN, từ máy bạn / Cloud Shell)
#   Tạo VM GPU + mở cổng 8080 + cài CVAT + môi trường inference.
# ==============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$HERE")"
source "$HERE/config.env"

log() { echo -e "\n\033[1;36m==> $*\033[0m"; }
die() { echo -e "\033[1;31mLỖI: $*\033[0m" >&2; exit 1; }
command -v gcloud >/dev/null || die "Chưa cài gcloud. Dùng Google Cloud Shell."

G=(--zone="$GCP_ZONE" --project="$GCP_PROJECT")

# ---- 1. Tạo VM + firewall ----------------------------------------------------
log "[1/4] Tạo VM GPU '$VM_NAME'..."
SPOT_FLAGS=()
[ "$USE_SPOT" = "true" ] && SPOT_FLAGS=(--provisioning-model=SPOT --instance-termination-action=STOP)

if gcloud compute instances describe "$VM_NAME" "${G[@]}" >/dev/null 2>&1; then
  log "VM đã tồn tại — bỏ qua tạo."
else
  gcloud compute instances create "$VM_NAME" "${G[@]}" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
    --maintenance-policy=TERMINATE \
    --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
    --boot-disk-size="$DISK_SIZE" \
    --metadata="install-nvidia-driver=True" \
    --tags=cvat-server \
    "${SPOT_FLAGS[@]}"
fi

gcloud compute firewall-rules create allow-cvat-8080 \
  --project="$GCP_PROJECT" --allow=tcp:8080 --target-tags=cvat-server \
  --source-ranges=0.0.0.0/0 >/dev/null 2>&1 || true

EXTERNAL_IP="$(gcloud compute instances describe "$VM_NAME" "${G[@]}" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"
log "VM IP: $EXTERNAL_IP"

# ---- 2. Chờ SSH --------------------------------------------------------------
log "[2/4] Chờ VM sẵn sàng..."
for i in $(seq 1 30); do
  gcloud compute ssh "$VM_NAME" "${G[@]}" --command="echo ok" >/dev/null 2>&1 && break
  echo "  ...thử lại ($i/30)"; sleep 10
done

# ---- 3. Copy code ------------------------------------------------------------
log "[3/4] Copy code lên VM..."
gcloud compute ssh "$VM_NAME" "${G[@]}" --command="mkdir -p ~/auto-label-3d"
gcloud compute scp "$PROJECT_DIR"/*.py "$VM_NAME":~/auto-label-3d/ "${G[@]}"
gcloud compute scp --recurse "$PROJECT_DIR/scripts" "$PROJECT_DIR/webapp" "$HERE" "$VM_NAME":~/auto-label-3d/ "${G[@]}"

# ---- 4. Provision ------------------------------------------------------------
log "[4/4] Cài CVAT + inference (10-20 phút)..."
gcloud compute ssh "$VM_NAME" "${G[@]}" --command="\
  CVAT_USER='$CVAT_USER' CVAT_PASS='$CVAT_PASS' CVAT_EMAIL='$CVAT_EMAIL' \
  CVAT_HOST='$EXTERNAL_IP' bash ~/auto-label-3d/deploy/provision.sh"

log "SETUP XONG!"
echo "  CVAT:  http://$EXTERNAL_IP:8080  (user: $CVAT_USER)"
echo "  Dán nhãn:  bash deploy/autolabel.sh"
echo "  ⚠️ Tắt VM khi xong: gcloud compute instances stop $VM_NAME --zone=$GCP_ZONE"
