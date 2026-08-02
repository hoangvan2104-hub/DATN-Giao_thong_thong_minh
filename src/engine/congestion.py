"""Cảnh báo ùn tắc — đếm số xe trong 1 vùng theo thời gian, đơn giản (đếm ngưỡng, không dự
đoán/ML). Khác các rule vi phạm (không gắn với 1 track cụ thể) — là trạng thái CHUNG của cả
vùng, phơi ra qua StatsTracker để hiển thị (bảng thống kê / web UI sau này), không phải
violation_events gắn theo track_id.
"""
from __future__ import annotations

from src.engine.tracker import Track
from src.engine.zones import ZoneMap
from src.utils.geometry import point_in_polygon


class CongestionMonitor:
    def __init__(
        self,
        zone_map: ZoneMap,
        zone_id: str | None = None,
        vehicle_threshold: int = 8,
        sustain_seconds: float = 5.0,
        fps: float = 25.0,
    ):
        self.zone_map = zone_map
        self.zone_id = zone_id
        self.vehicle_threshold = vehicle_threshold
        # sustain_seconds: số xe phải VƯỢT ngưỡng LIÊN TỤC trong khoảng thời gian này mới tính
        # là ùn tắc thật (tránh báo sai vì 1 nhóm xe đông nhất thời lúc dừng đèn đỏ rồi đi hết
        # ngay sau đó — ùn tắc thật phải kéo dài).
        self.sustain_frames = max(1, round(sustain_seconds * fps))
        self._high_count_streak = 0
        self.vehicle_count: int = 0
        self.is_congested: bool = False

    def update(self, tracks: list[Track]) -> None:
        zone = self.zone_map.zones.get(self.zone_id) if self.zone_id else None
        if zone is None:
            self.vehicle_count = len(tracks)
        else:
            self.vehicle_count = sum(
                1 for t in tracks if point_in_polygon(self.zone_map.anchor_point(t.bbox), zone.polygon)
            )

        if self.vehicle_count >= self.vehicle_threshold:
            self._high_count_streak += 1
        else:
            self._high_count_streak = 0

        self.is_congested = self._high_count_streak >= self.sustain_frames
