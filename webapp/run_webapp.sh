#!/bin/bash
# Khởi động web portal trên VM (chạy nền). Chạy: bash run_webapp.sh
set -e
source "$HOME/venv/bin/activate"
pip install -q fastapi "uvicorn[standard]" python-multipart requests openpyxl
cd "$HOME/auto-label-3d/webapp"
# tắt tiến trình cũ nếu có
pkill -f "uvicorn app:app" 2>/dev/null || true
sleep 1
nohup uvicorn app:app --host 0.0.0.0 --port 8000 > "$HOME/webapp.log" 2>&1 &
sleep 2
echo "=== Web portal dang chay (cong 8000). Log: ~/webapp.log ==="
tail -n 5 "$HOME/webapp.log"
