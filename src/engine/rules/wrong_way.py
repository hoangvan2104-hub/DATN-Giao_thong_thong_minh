"""Rule: đi ngược chiều.

Fix bug hệ thống cũ: so vector chuyển động của xe với TOÀN BỘ vectors trong config, không
lọc theo lane, gây false positive/negative khi có nhiều lane hướng khác nhau. Giờ dùng
ZoneMap.vector_for_lane() để lấy đúng vector áp dụng cho lane của xe (xem zones.py).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.engine.tracker import Track
from src.engine.zones import ZoneMap
from src.utils.geometry import cosine_similarity, point_in_polygon


@dataclass
class WrongWayViolation:
    track_id: int
    frame_idx: int


class WrongWayRule:
    def __init__(self, zone_map: ZoneMap, config: dict, min_frames: int = 8):
        self.zone_map = zone_map
        wwc = config.get("system", {}).get("wrong_way_config", {})
        self.zone_id = wwc.get("zone_id")
        self.min_move_distance = wwc.get("min_move_distance", 30)
        self.cos_threshold = wwc.get("cos_threshold", -0.5)
        # min_frames: số frame tối thiểu kể từ mốc trước khi xét hướng — chỉ đủ khoảng cách
        # (min_move_distance) là chưa đủ tin cậy, vì track mới vào vùng biên có thể bị nhiễu
        # vị trí bbox (rung nhẹ) tạo ra "hướng di chuyển" giả trong vài frame đầu dù xe gần
        # như đứng yên. Yêu cầu thêm đủ thời gian giúp hướng đo được phản ánh chuyển động thật.
        self.min_frames = min_frames
        self._start_points: dict[int, tuple[float, float]] = {}
        self._start_frames: dict[int, int] = {}
        self._reported: set[int] = set()

    def update(self, tracks: list[Track], frame_idx: int) -> list[WrongWayViolation]:
        violations: list[WrongWayViolation] = []
        zone = self.zone_map.zones.get(self.zone_id) if self.zone_id else None
        if zone is None:
            return violations

        for t in tracks:
            if t.track_id in self._reported:
                continue
            point = self.zone_map.anchor_point(t.bbox)

            if not point_in_polygon(point, zone.polygon):
                self._start_points.pop(t.track_id, None)  # ra khỏi vùng kiểm tra, reset mốc
                self._start_frames.pop(t.track_id, None)
                continue

            start = self._start_points.get(t.track_id)
            if start is None:
                self._start_points[t.track_id] = point
                self._start_frames[t.track_id] = frame_idx
                continue

            if frame_idx - self._start_frames[t.track_id] < self.min_frames:
                continue  # chưa đủ thời gian, tránh nhiễu vị trí bbox gây hướng giả

            dx, dy = point[0] - start[0], point[1] - start[1]
            distance = (dx**2 + dy**2) ** 0.5
            if distance < self.min_move_distance:
                continue

            lane = self.zone_map.find_lane(t.bbox)
            if lane is None:
                # Xe không nằm trong lane nào đã định nghĩa (vd hướng tiếp cận khác của ngã
                # tư chưa vẽ lane) -> không đủ thông tin biết hướng đúng thật sự, bỏ qua thay
                # vì đoán bằng vector mặc định (dễ false positive).
                self._start_points[t.track_id] = point
                self._start_frames[t.track_id] = frame_idx
                continue

            flow_vector = self.zone_map.vector_for_lane(lane.id)
            if flow_vector is None:
                self._start_points[t.track_id] = point
                self._start_frames[t.track_id] = frame_idx
                continue

            flow_dir = (
                flow_vector.end[0] - flow_vector.start[0],
                flow_vector.end[1] - flow_vector.start[1],
            )
            cos = cosine_similarity((dx, dy), flow_dir)
            if cos < self.cos_threshold:
                self._reported.add(t.track_id)
                violations.append(WrongWayViolation(track_id=t.track_id, frame_idx=frame_idx))
            else:
                # di chuyển đúng chiều -> dời mốc, tiếp tục theo dõi đoạn tiếp theo
                self._start_points[t.track_id] = point
                self._start_frames[t.track_id] = frame_idx

        return violations

    def remove(self, track_ids: set[int]) -> None:
        for tid in track_ids:
            self._start_points.pop(tid, None)
            self._start_frames.pop(tid, None)
            self._reported.discard(tid)
