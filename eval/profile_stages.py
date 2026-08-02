"""Đo thời gian THẬT theo từng giai đoạn (decode, detect+track, OCR biển số, mũ bảo hiểm,
render+ghi) bằng cách bọc (monkeypatch) đúng các hàm Pipeline gọi, có `torch.cuda.synchronize()`
quanh mỗi đoạn đo — tránh nhầm lẫn thời gian do CUDA chạy bất đồng bộ (1 lệnh Python có vẻ "chậm"
chỉ vì nó tình cờ là chỗ GPU đồng bộ hoá, không phải bản thân nó chậm — cProfile đo cumulative
KHÔNG phân biệt được điều này). KHÔNG sửa file nguồn nào — chỉ patch tạm trong tiến trình script
này. KHÔNG ghi log/evidence.

Dùng:
    python eval/profile_stages.py <ten_video> [--max-frames N]
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import resolve_and_validate
from src.engine.pipeline import Pipeline
from src.engine.tracker import VehicleTracker
from src.engine.plate_ocr import PlateReader
from src.engine.rules.no_helmet import NoHelmetRule
from src.engine import pipeline as pipeline_mod

_times: dict[str, float] = defaultdict(float)
_counts: dict[str, int] = defaultdict(int)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(name: str, fn):
    def wrapper(*args, **kwargs):
        _sync()
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        _sync()
        _times[name] += time.perf_counter() - t0
        _counts[name] += 1
        return result
    return wrapper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--model", default="models/pretrained/yolo11s.pt")
    args = parser.parse_args()

    video_path, config = resolve_and_validate(args.name)
    pipeline = Pipeline(config=config, model_path=args.model)

    pipeline.tracker.update = _timed("1_detect_track", pipeline.tracker.update)
    if pipeline.plate_reader is not None:
        pipeline.plate_reader.read = _timed("2_plate_ocr", pipeline.plate_reader.read)
    if pipeline.helmet_rule is not None:
        pipeline.helmet_rule.update = _timed("3_helmet", pipeline.helmet_rule.update)
    pipeline_mod.render_frame = _timed("4_render", pipeline_mod.render_frame)
    pipeline.traffic_lights.update = _timed("5_traffic_light", pipeline.traffic_lights.update)
    if pipeline.congestion is not None:
        pipeline.congestion.update = _timed("6_congestion", pipeline.congestion.update)
    for rule_name, rule in pipeline.rules.items():
        rule.update = _timed(f"7_rule_{rule_name}", rule.update)
    pipeline.history.update = _timed("8_history_update", pipeline.history.update)
    pipeline.stats.register_tracks = _timed("9_stats_register", pipeline.stats.register_tracks)
    cv2.VideoCapture.read = _timed("0_decode", cv2.VideoCapture.read)
    cv2.bitwise_and = _timed("a_roi_mask", cv2.bitwise_and)
    cv2.resize = _timed("b_frame_resize", cv2.resize)

    t_start = time.perf_counter()
    result = pipeline.run(str(video_path), output_path=None, show=False, max_frames=args.max_frames)
    total = time.perf_counter() - t_start

    print(f"\n=== {args.name}: {result['frames']} khung hinh, {total:.2f}s tong, "
          f"{result['frames'] / total:.2f} FPS (nguon: {result['fps_source']} FPS) ===\n")
    accounted = 0.0
    for name in sorted(_times):
        t = _times[name]
        accounted += t
        pct = t / total * 100
        print(f"  {name:20s} {t:7.2f}s  ({pct:5.1f}%)  {_counts[name]:5d} calls  "
              f"{t / max(_counts[name],1)*1000:.2f} ms/call")
    other = total - accounted
    print(f"  {'khac (decode/ROI/rules/ghi file/...)':20s} {other:7.2f}s  ({other/total*100:5.1f}%)")


if __name__ == "__main__":
    main()
