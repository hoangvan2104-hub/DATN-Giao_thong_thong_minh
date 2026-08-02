"""Rule: sai làn theo loại phương tiện.

Không tính vi phạm ngay khi xe vừa vào vùng lane cấm (dễ false positive nếu xe chỉ cắt
ngang để đổi làn) — dùng ngưỡng số frame HOẶC quãng đường di chuyển liên tục trong lane cấm
trước khi xác nhận vi phạm (xem docs/brainstorm-notes.md).

`min_track_age`: track VỪA SINH có bbox/class kém ổn định nhất (đã gặp nhiều lần trong dự án —
vd bbox_smoothing_alpha, min_new_track_conf) — cụ thể đã đo được thật: model nhận nhầm 1 xe máy
thành "bicycle" (không được phép ở lane chỉ cho car/motorcycle/bus) đúng 3 khung hình đầu track
mới xuất hiện, đủ để nhánh `min_distance` (quãng đường, không cần đủ `min_frames_in_lane`) xác
nhận NHẦM vi phạm chỉ sau 3 khung — né được ngưỡng "chống nhiễu" `min_frames_in_lane` vì đó chỉ
đếm số khung CÙNG lane cấm, không biết gì về tuổi track. Chặn thêm bằng tuổi track (giống
`min_frames` đã dùng cho `WrongWayRule`) — không đổi 2 ngưỡng frame/distance hiện có, chỉ thêm 1
điều kiện tiên quyết trước khi bắt đầu đếm dwell.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.engine.tracker import Track
from src.engine.zones import ZoneMap


@dataclass
class WrongLaneViolation:
    track_id: int
    lane_id: str
    frame_idx: int


class WrongLaneRule:
    def __init__(
        self, zone_map: ZoneMap, min_frames_in_lane: int = 15, min_distance: float = 20.0,
        min_track_age: int = 8,
    ):
        self.zone_map = zone_map
        self.min_frames_in_lane = min_frames_in_lane
        self.min_distance = min_distance
        self.min_track_age = min_track_age
        self._first_seen_frame: dict[int, int] = {}
        self._dwell: dict[int, dict] = {}
        self._reported: set[int] = set()

    def update(self, tracks: list[Track], frame_idx: int) -> list[WrongLaneViolation]:
        violations: list[WrongLaneViolation] = []

        for t in tracks:
            self._first_seen_frame.setdefault(t.track_id, frame_idx)
            if t.track_id in self._reported:
                continue
            lane = self.zone_map.find_lane(t.bbox)
            if lane is None or lane.is_class_allowed(t.cls_name):
                self._dwell.pop(t.track_id, None)
                continue

            if frame_idx - self._first_seen_frame[t.track_id] < self.min_track_age:
                continue  # track vừa sinh, bbox/class chưa ổn định — xem docstring lớp

            point = self.zone_map.anchor_point(t.bbox)
            state = self._dwell.get(t.track_id)
            if state is None or state["lane_id"] != lane.id:
                self._dwell[t.track_id] = {"lane_id": lane.id, "frames": 1, "start_point": point}
                continue

            state["frames"] += 1
            dx = point[0] - state["start_point"][0]
            dy = point[1] - state["start_point"][1]
            distance = (dx**2 + dy**2) ** 0.5

            if state["frames"] >= self.min_frames_in_lane or distance >= self.min_distance:
                self._reported.add(t.track_id)
                violations.append(WrongLaneViolation(track_id=t.track_id, lane_id=lane.id, frame_idx=frame_idx))

        return violations

    def remove(self, track_ids: set[int]) -> None:
        for tid in track_ids:
            self._first_seen_frame.pop(tid, None)
            self._dwell.pop(tid, None)
            self._reported.discard(tid)
