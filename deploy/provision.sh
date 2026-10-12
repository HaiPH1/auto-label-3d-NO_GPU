#!/bin/bash
# ==============================================================================
# PROVISION — chạy TRÊN VM (do cloud_setup.sh gọi). ALL-IN-ONE:
#   [1] GPU/driver   [2] Docker + CVAT self-host   [3] inference env + checkpoint
#
# Biến môi trường nhận từ cloud_setup.sh:
#   CVAT_USER, CVAT_PASS, CVAT_EMAIL, CVAT_HOST (IP public của VM)
# ==============================================================================
set -euo pipefail

CVAT_USER="${CVAT_USER:-admin}"
CVAT_PASS="${CVAT_PASS:-admin}"
CVAT_EMAIL="${CVAT_EMAIL:-admin@example.com}"
CVAT_HOST="${CVAT_HOST:-localhost}"

VENV_DIR="$HOME/venv"
OPENPCDET_DIR="$HOME/OpenPCDet"
CKPT_DIR="$HOME/checkpoints"
CKPT_FILE="$CKPT_DIR/transfusion_lidar.pth"
CKPT_GDRIVE_ID="1cuZ2qdDnxSwTCsiXWwbqCGF-uoazTXbz"

log() { echo -e "\n\033[1;32m### $*\033[0m"; }

# ---- 1. GPU / driver ---------------------------------------------------------
log "[1/3] Kiểm tra GPU..."
if ! nvidia-smi >/dev/null 2>&1; then
  sudo /opt/deeplearning/install-driver.sh || true
  nvidia-smi >/dev/null 2>&1 || { echo "!! GPU chưa dùng được"; exit 1; }
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- 2. Docker + CVAT --------------------------------------------------------
log "[2/3] Cài Docker + CVAT (self-host, full 3D)..."
if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y docker.io docker-compose-v2
  sudo usermod -aG docker "$USER" || true
fi
if [ ! -d "$HOME/cvat" ]; then
  git clone https://github.com/cvat-ai/cvat.git "$HOME/cvat"
fi
cd "$HOME/cvat"
export CVAT_HOST="$CVAT_HOST"
sudo -E docker compose up -d
echo "Chờ CVAT khởi động..."; sleep 45

# Tạo superuser không tương tác
sudo docker compose exec -T cvat_server python3 /home/django/manage.py shell -c "\
from django.contrib.auth import get_user_model; U=get_user_model(); \
u,_=U.objects.get_or_create(username='$CVAT_USER', defaults={'email':'$CVAT_EMAIL'}); \
u.is_superuser=True; u.is_staff=True; u.set_password('$CVAT_PASS'); u.save(); \
print('CVAT user ready:', u.username)" \
  || echo "!! Tạo user CVAT lỗi — tạo tay: cd ~/cvat && sudo docker compose exec cvat_server python3 ~/manage.py createsuperuser"

# ---- 3. Inference env + checkpoint ------------------------------------------
log "[3/3] Cài PyTorch cu124 + spconv + OpenPCDet + checkpoint..."
sudo apt-get install -y git build-essential cmake libboost-all-dev python3-venv >/dev/null 2>&1 || true
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install -U pip wheel setuptools
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install spconv-cu120 gdown pyyaml numba scipy opencv-python requests open3d kornia cvat-sdk

if [ ! -d "$OPENPCDET_DIR" ]; then
  git clone https://github.com/open-mmlab/OpenPCDet.git "$OPENPCDET_DIR"
fi
cd "$OPENPCDET_DIR"
pip install -r requirements.txt
# --no-build-isolation: build dùng torch trong venv (tránh ModuleNotFoundError: torch)
pip install -e . --no-build-isolation

mkdir -p "$CKPT_DIR"
[ -f "$CKPT_FILE" ] || gdown "$CKPT_GDRIVE_ID" -O "$CKPT_FILE"

python3 -c "import torch, pcdet; print('torch', torch.__version__, '| GPU', torch.cuda.get_device_name(0), '| pcdet', pcdet.__version__)"
log "PROVISION HOÀN TẤT."
