"""Công cụ dev: quét toàn video, đếm track ID đã cấp + số lần 2 track khác ID chồng lấp
cao (nghi ngờ 1 xe bị tách 2 box) — dùng để đánh giá nhanh chất lượng tracking khi đổi
model/tham số, không phải rule đánh giá chính thức cho báo cáo (mAP/MOTA/IDF1 làm riêng).

Ví dụ:
    python eval/check_track_overlap.py --video data/input/vid_test.mp4 --config config/videos/vid_test.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.config_schema import load_config
from src.engine.pipeline import _build_roi_mask
from src.engine.tracker import VehicleTracker
from src.engine.zones import ZoneMap


def iou(b1: tuple, b2: tuple) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Đếm số lần track chồng lấp cao (nghi ngờ trùng box)")
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="models/pretrained/yolo11s.pt")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    config = load_config(args.config)
    zone_map = ZoneMap(config)
    system = config.get("system", {})
    tracker = VehicleTracker(
        model_path=args.model,
        classes=config.get("classes"),
        tracker_params=system.get("tracker_params"),
        **system.get("detection_params", {}),
    )

    cap = cv2.VideoCapture(args.video)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    detection_zone = zone_map.detection_zone()
    roi_mask = _build_roi_mask((height, width), detection_zone.polygon) if detection_zone else None

    overlap_events = 0
    seen_ids: set[int] = set()
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tracks = tracker.update(frame, roi_mask=roi_mask)
        for t in tracks:
            seen_ids.add(t.track_id)
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                v = iou(tracks[i].bbox, tracks[j].bbox)
                if v > args.iou_threshold:
                    overlap_events += 1
                    print(f"frame {frame_idx}: IoU={v:.2f} giữa #{tracks[i].track_id} và #{tracks[j].track_id}")
        frame_idx += 1
    cap.release()

    print(f"\nTổng số frame: {frame_idx}")
    print(f"Tổng số track ID khác nhau đã thấy: {len(seen_ids)}")
    print(f"Số lần 2 track chồng lấp cao (IoU>{args.iou_threshold}): {overlap_events}")


if __name__ == "__main__":
    main()
