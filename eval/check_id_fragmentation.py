"""Công cụ dev: phát hiện nghi ngờ "1 xe bị gán nhiều ID" (track gãy) — khác với
check_track_overlap.py (vốn phát hiện 2 ID cùng lúc chồng lên nhau). Ở đây tìm các cặp
track A kết thúc rồi track B xuất hiện ngay sau đó (vài frame), tại vị trí rất gần điểm A
vừa kết thúc — dấu hiệu cùng 1 xe bị "quên" giữa chừng rồi cấp ID mới, không phải xe rời
khung hình thật.

Ví dụ:
    python eval/check_id_fragmentation.py --video data/input/vid_test.mp4 --config config/videos/vid_test.json
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.config_schema import load_config
from src.engine.pipeline import _build_roi_mask
from src.engine.tracker import VehicleTracker
from src.engine.zones import ZoneMap


def main() -> None:
    parser = argparse.ArgumentParser(description="Phát hiện nghi ngờ 1 xe bị tách nhiều ID")
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="models/pretrained/yolo11s.pt")
    parser.add_argument("--max-frame-gap", type=int, default=10, help="Số frame tối đa giữa lúc ID cũ mất và ID mới xuất hiện")
    parser.add_argument("--max-distance", type=float, default=50.0, help="Khoảng cách tối đa (px) giữa điểm cuối ID cũ và điểm đầu ID mới")
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

    first_seen: dict[int, tuple[int, tuple[float, float]]] = {}
    last_seen: dict[int, tuple[int, tuple[float, float]]] = {}
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tracks = tracker.update(frame, roi_mask=roi_mask)
        for t in tracks:
            point = zone_map.anchor_point(t.bbox)
            if t.track_id not in first_seen:
                first_seen[t.track_id] = (frame_idx, point)
            last_seen[t.track_id] = (frame_idx, point)
        frame_idx += 1
    cap.release()

    ids = sorted(first_seen, key=lambda tid: first_seen[tid][0])
    suspects = []
    for old_id in ids:
        old_end_frame, old_end_point = last_seen[old_id]
        for new_id in ids:
            if new_id == old_id:
                continue
            new_start_frame, new_start_point = first_seen[new_id]
            gap = new_start_frame - old_end_frame
            if gap <= 0 or gap > args.max_frame_gap:
                continue
            dist = ((new_start_point[0] - old_end_point[0]) ** 2 + (new_start_point[1] - old_end_point[1]) ** 2) ** 0.5
            if dist <= args.max_distance:
                suspects.append((old_id, new_id, gap, dist, old_end_frame))

    print(f"Tổng số track ID: {len(ids)}")
    print(f"Số cặp nghi ngờ cùng 1 xe bị tách ID (gap<={args.max_frame_gap} frame, distance<={args.max_distance}px):")
    for old_id, new_id, gap, dist, end_frame in suspects:
        print(f"  #{old_id} (mất ở frame {end_frame}) -> #{new_id} (gap={gap} frame, distance={dist:.1f}px)")
    print(f"TONG: {len(suspects)}")


if __name__ == "__main__":
    main()
