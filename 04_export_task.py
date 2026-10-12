"""
Bước 4: Export một task CVAT ra file BACKUP (.zip).
File backup gồm CẢ dữ liệu (point cloud / ảnh) LẪN annotation (box đã làm),
restore được sang một CVAT khác bằng 05_import_task.py.

Cài 1 lần:  pip install cvat-sdk

Cách dùng:
    # Lấy url + task_id từ config.json
    python3 04_export_task.py --config config.json \
        --user admin --password <pass> --output task_backup.zip

    # Hoặc chỉ định tay
    python3 04_export_task.py --url http://1.2.3.4:8080 --task-id 40 \
        --user admin --password <pass> --output task_backup.zip
"""
import argparse
import json
import os
import sys
from urllib.parse import urlparse


def split_host(url):
    if "://" not in url:
        url = "http://" + url
    u = urlparse(url)
    port = u.port or (443 if u.scheme == "https" else 80)
    return f"{u.scheme}://{u.hostname}", port


def main():
    ap = argparse.ArgumentParser(description="Export CVAT task -> backup .zip (ảnh + annotation)")
    ap.add_argument("--config", help="config.json để lấy cvat_url + task_id")
    ap.add_argument("--url", help="CVAT URL (vd http://1.2.3.4:8080)")
    ap.add_argument("--task-id", type=int, help="ID task cần export")
    ap.add_argument("--user", required=True, help="CVAT username")
    ap.add_argument("--password", required=True, help="CVAT password")
    ap.add_argument("--output", default="task_backup.zip", help="File backup output")
    args = ap.parse_args()

    url, task_id = args.url, args.task_id
    if args.config and os.path.exists(args.config):
        cfg = json.load(open(args.config))
        url = url or cfg.get("cvat_url")
        task_id = task_id or cfg.get("task_id")

    if not url or not task_id:
        ap.error("Cần --url + --task-id (hoặc --config có sẵn hai giá trị đó)")

    try:
        from cvat_sdk import make_client
    except ImportError:
        sys.exit("Thiếu thư viện. Chạy: pip install cvat-sdk")

    host, port = split_host(url)
    print(f"Export task {task_id} từ {host}:{port} ...")
    with make_client(host=host, port=port, credentials=(args.user, args.password)) as client:
        task = client.tasks.retrieve(task_id)
        task.download_backup(args.output)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Đã lưu backup: {args.output} ({size_mb:.1f} MB)")
    print(f"\nĐẩy sang CVAT khác:")
    print(f"  python3 05_import_task.py --url <CVAT_KHAC> --user .. --password .. --backup {args.output}")


if __name__ == "__main__":
    main()
