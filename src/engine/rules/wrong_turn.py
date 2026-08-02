"""Rule: sai hướng rẽ.

Xe được coi là "xuất phát" từ 1 lane khi còn nằm trong zone type=road role=analysis (vùng
R1 — xem CLAUDE.md). Hướng đi thực tế xác định bằng vạch end_direction nào bị cắt qua, rồi
đối chiếu với hướng cho phép/cấm của lane xuất phát.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.engine.tracker import Track
from src.engine.zones import ZoneMap
from src.utils.geometry import point_in_polygon, polyline_crossed


@dataclass
class WrongTurnViolation:
    track_id: int
    lane_id: str
    direction: str
    frame_idx: int


class WrongTurnRule:
    def __init__(self, zone_map: ZoneMap):
        self.zone_map = zone_map
        self._origin_lane: dict[int, str] = {}
        self._prev_points: dict[int, tuple[float, float]] = {}
        self._reported: set[int] = set()

    def update(self, tracks: list[Track], frame_idx: int) -> list[WrongTurnViolation]:
        violations: list[WrongTurnViolation] = []
        analysis_zone = self.zone_map.analysis_zone()

        for t in tracks:
            point = self.zone_map.anchor_point(t.bbox)
            prev = self._prev_points.get(t.track_id)
            self._prev_points[t.track_id] = point

            # Còn trong vùng xuất phát -> ghi nhớ/cập nhật lane xuất phát gần nhất.
            if analysis_zone is not None and point_in_polygon(point, analysis_zone.polygon):
                lane = self.zone_map.find_lane(t.bbox)
                if lane is not None:
                    self._origin_lane[t.track_id] = lane.id

            if prev is None or t.track_id in self._reported:
                continue

            origin_lane_id = self._origin_lane.get(t.track_id)
            if origin_lane_id is None:
                continue
            lane = self.zone_map.zones.get(origin_lane_id)
            if lane is None:
                continue

            for line in self.zone_map.lines_by_type("end_direction"):
                if not line.direction or not polyline_crossed(prev, point, line.points):
                    continue
                if not lane.is_direction_allowed(line.direction):
                    self._reported.add(t.track_id)
                    violations.append(WrongTurnViolation(
                        track_id=t.track_id,
                        lane_id=origin_lane_id,
                        direction=line.direction,
                        frame_idx=frame_idx,
                    ))
                break  # đã xác định được hướng thoát của xe frame này, không cần xét vạch khác

        return violations

    def remove(self, track_ids: set[int]) -> None:
        for tid in track_ids:
            self._origin_lane.pop(tid, None)
            self._prev_points.pop(tid, None)
            self._reported.discard(tid)
