"""
Bước 5: Import file BACKUP (.zip) từ 04_export_task.py vào MỘT CVAT KHÁC.
Tạo lại task đầy đủ: dữ liệu (point cloud / ảnh) + annotation (box đã làm).

Cài 1 lần:  pip install cvat-sdk

Cách dùng:
    python3 05_import_task.py \
        --url http://cvat-khac:8080 \
        --user admin --password <pass> \
        --backup task_backup.zip
"""
import argparse
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
    ap = argparse.ArgumentParser(description="Import backup .zip -> tạo task trên CVAT khác")
    ap.add_argument("--url", required=True, help="CVAT đích (vd http://cvat-khac:8080)")
    ap.add_argument("--user", required=True, help="CVAT username")
    ap.add_argument("--password", required=True, help="CVAT password")
    ap.add_argument("--backup", required=True, help="File backup .zip từ 04_export_task.py")
    args = ap.parse_args()

    if not os.path.exists(args.backup):
        sys.exit(f"Không thấy file backup: {args.backup}")

    try:
        from cvat_sdk import make_client
    except ImportError:
        sys.exit("Thiếu thư viện. Chạy: pip install cvat-sdk")

    host, port = split_host(args.url)
    print(f"Import {args.backup} vào {host}:{port} ...")
    with make_client(host=host, port=port, credentials=(args.user, args.password)) as client:
        task = client.tasks.create_from_backup(args.backup)

    base = args.url.rstrip("/")
    print(f"Tạo task thành công: ID={task.id}")
    print(f"Mở: {base}/tasks/{task.id}")


if __name__ == "__main__":
    main()
