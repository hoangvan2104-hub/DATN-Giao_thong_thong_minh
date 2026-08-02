"""Kiểm tra nhanh kết quả đọc biển số trên 1 video — in ra track nào đã thử, đọc được gì.
Dùng khi cần xác nhận bằng mắt PlateReader có hoạt động thật không (không chỉ chạy không lỗi).

Đọc `plate_reader.history`/`ever_attempted` (state vĩnh viễn, không bị dọn khi track bị xoá)
thay vì `_plates`/`_attempts` (state throttle, bị dọn theo vòng đời track) — nếu không sẽ đếm
THIẾU các track đã rời khung hình trước khi video kết thúc (đã từng gây báo sai "0 track đọc
được" trong khi thực ra có đọc thành công, chỉ là track đó bị dọn state trước khi video xong).

Usage:
    python eval/check_plate_ocr.py vid_test --max-frames 300
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.config_schema import load_config
from src.engine.pipeline import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("--model", default="models/pretrained/yolo11s.pt")
parser.add_argument("--max-frames", type=int, default=300)
args = parser.parse_args()

video_path = str(next(Path("data/input").glob(f"{args.name}.*")))
config = load_config(f"config/videos/{args.name}.json")

pipeline = Pipeline(config=config, model_path=args.model)
stats = pipeline.run(video_path=video_path, output_path=None, show=False, max_frames=args.max_frames)
print(json.dumps(stats, ensure_ascii=False, indent=2))

pr = pipeline.plate_reader
print("\n--- Plate reader state (cumulative, không bị dọn theo vòng đời track) ---")
print("Track đã từng thử:", sorted(pr.ever_attempted))
print("Đọc thành công:", pr.history)
print(f"Tổng: {len(pr.ever_attempted)} track đã thử, {len(pr.history)} track đọc được biển số")
