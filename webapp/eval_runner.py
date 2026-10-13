"""
Chạy đánh giá TOOL: inference trên data → so với ground truth → xuất metrics JSON.
Không tạo task CVAT (chỉ chấm điểm).

    python3 eval_runner.py <data_dir> <gt_file.json> <out_eval.json>
"""
import json
import os
import sys

RUN_DIR = os.path.expanduser("~/auto-label-3d")
sys.path.insert(0, RUN_DIR)


def main():
    data_dir, gt_file, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    import run  # noqa: E402

    pcd_dir = os.path.join(data_dir, "pointcloud")
    if not os.path.isdir(pcd_dir):
        print(f"[ERROR] Không thấy {pcd_dir}")
        sys.exit(1)

    print("[1/4] Inference bằng tool (TransFusion)...")
    run.download_checkpoint()
    preds = run.run_inference(pcd_dir, os.path.join(os.path.dirname(out_json), "eval_preds.json"))

    print("[2/4] Hậu xử lý như lúc đẩy CVAT (heading correction + tracking)...")
    try:
        preds = run.correct_heading(preds, pcd_dir)
        preds = run.track_objects(preds, max_dist=5.0)
    except Exception as e:  # noqa: BLE001
        print(f"  (bỏ qua hậu xử lý: {e})")
    print(f"  Output cuối: {sum(len(r.get('boxes', [])) for r in preds)} boxes")

    print("[3/4] Nạp ground truth...")
    with open(gt_file) as f:
        gt = json.load(f)
    print(f"  GT: {sum(len(r.get('boxes', [])) for r in gt)} boxes / {len(gt)} frames")

    print("[4/4] So sánh output tool vs GT...")
    stats = run.evaluate_predictions(preds, gt)
    run.save_evaluation_json(stats, out_json, model_name="tool")
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
