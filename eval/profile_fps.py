"""Đo FPS + phân tích thời gian theo từng giai đoạn (detect/track, OCR biển số, mũ bảo hiểm,
render, ...) bằng cProfile trên 1 video thật — dùng TRƯỚC KHI tối ưu để biết chính xác điểm
nghẽn thật sự (đo trực tiếp, không đoán — đúng bài học đã rút ra nhiều lần trong dự án, xem
CLAUDE.md/docs/nhat-ky-ky-thuat.md). KHÔNG ghi log/evidence (không truyền log_name) để không làm
rác data/logs, data/evidence.

Dùng:
    python eval/profile_fps.py <ten_video> [--max-frames N]
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import resolve_and_validate
from src.engine.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--model", default="models/pretrained/yolo11s.pt")
    parser.add_argument("--top", type=int, default=25, help="So dong top ham ton thoi gian nhat de in ra")
    args = parser.parse_args()

    video_path, config = resolve_and_validate(args.name)
    pipeline = Pipeline(config=config, model_path=args.model)

    profiler = cProfile.Profile()
    t0 = time.time()
    profiler.enable()
    result = pipeline.run(str(video_path), output_path=None, show=False, max_frames=args.max_frames)
    profiler.disable()
    elapsed = time.time() - t0

    print(f"\n=== {args.name}: {result['frames']} khung hinh, {elapsed:.2f}s, "
          f"{result['frames'] / elapsed:.2f} FPS xu ly (nguon: {result['fps_source']} FPS) ===\n")

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(args.top)

    print("\n--- Sap theo TOTAL TIME (khong tinh ham con goi) ---\n")
    stats.sort_stats("tottime")
    stats.print_stats(args.top)


if __name__ == "__main__":
    main()
