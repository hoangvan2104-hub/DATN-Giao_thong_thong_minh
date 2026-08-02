"""Tổng hợp số liệu đang chạy (đếm xe theo loại, đếm vi phạm, log gần nhất).

Dữ liệu có cấu trúc thuần — KHÔNG biết gì về hiển thị (đúng nguyên tắc tách Engine/Render,
xem CLAUDE.md). `src/render/layout.py` đọc object này để vẽ, nhưng bản thân class này không
import cv2/vẽ gì cả — để sau này Web UI cũng đọc được cùng 1 nguồn dữ liệu qua JSON.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from src.engine.tracker import Track
from src.engine.zones import ZoneMap
from src.utils.geometry import point_in_polygon


@dataclass
class ViolationLogEntry:
    frame_idx: int
    track_id: int
    violation_type: str
    lane_id: str | None = None


class StatsTracker:
    def __init__(self, zone_map: ZoneMap | None = None, max_log: int = 50):
        self.class_counts: dict[str, int] = {}
        self.violation_counts: dict[str, int] = {}
        self.violation_log: deque[ViolationLogEntry] = deque(maxlen=max_log)
        self._counted_track_ids: set[int] = set()
        # zone_map: dùng để đếm xe theo vùng nhận diện/vùng xuất phát/làn đường (yêu cầu bảng
        # thống kê tổng quan). Optional (None = bỏ qua, các số liệu vùng luôn = 0/rỗng) để không
        # phá các nơi khởi tạo StatsTracker() cũ (vd eval/*.py) không truyền zone_map.
        self.zone_map = zone_map
        # Tổng xe đã TỪNG xuất hiện trong từng làn (type=lane) — 1 xe có thể được tính cho NHIỀU
        # làn khác nhau nếu đi qua nhiều làn trong đời track (đúng nghĩa "đã xuất hiện trên làn
        # X", không loại trừ lẫn nhau giữa các làn).
        self.lane_counts: dict[str, int] = {}
        self._counted_lane_pairs: set[tuple[int, str]] = set()
        # Vùng nhận diện (role="detection", vd R0): tổng xe đã từng xuất hiện. Không cần vùng
        # riêng để "hiện tại đang có bao nhiêu" vì ROI mask đã giới hạn detect trong vùng này —
        # mọi track sống đều đang ở trong vùng, dùng chung total_vehicles cho nghĩa "hiện tại".
        self._detection_zone_total: int = 0
        # Vùng xuất phát (role="analysis", vd R1): tổng xe XUẤT PHÁT từ vùng này (vị trí lúc
        # track được thấy LẦN ĐẦU nằm trong vùng) + số xe ĐANG hiện diện trong vùng ở khung hình
        # hiện tại (giá trị "sống", chỉ có ý nghĩa lúc đang xử lý trực tiếp — xem docstring
        # analysis_zone_current).
        self.analysis_zone_total: int = 0
        self.analysis_zone_current: int = 0
        # Tính năng ùn tắc có đang BẬT hay không (pipeline set 1 lần khi khởi tạo
        # CongestionMonitor) — dùng để quyết định CÓ HIỆN mục "GIAO THONG" trên bảng thống kê
        # hay không. KHÔNG được dùng congestion_vehicle_count>0 để quyết định hiện/ẩn (0 là số
        # hợp lệ — xe hiện tại vắng trong vùng — nhưng 0 lại "falsy" trong Python, khiến mục
        # ẩn/hiện chập chờn theo từng khung hình dù tính năng vẫn đang bật — đã gặp phản hồi
        # thật, trông như 1 bug hiển thị).
        self.congestion_enabled: bool = False
        # congestion_vehicle_count/is_congested phản ánh khung hình GẦN NHẤT (dùng để hiển thị
        # trực tiếp lúc đang chạy), còn peak_vehicle_count/was_ever_congested tích luỹ suốt
        # video (dùng để tổng kết cuối cùng, xem src/engine/congestion.py).
        self.congestion_vehicle_count: int = 0
        self.is_congested: bool = False
        self.peak_vehicle_count: int = 0
        self.was_ever_congested: bool = False
        # Trạng thái đèn tín hiệu hiện tại ({id: "red"|"yellow"|"green"|"unknown"}) — hiển thị
        # trực tiếp trên bảng thống kê, xem src/render/layout.py.
        self.traffic_light_states: dict[str, str] = {}

    def register_tracks(self, tracks: list[Track]) -> None:
        """Đếm 1 xe đúng 1 lần duy nhất (lần đầu tiên track xuất hiện), theo loại phương tiện +
        theo vùng nhận diện/vùng xuất phát/làn đường (nếu có `zone_map`, xem __init__)."""
        analysis_zone = self.zone_map.analysis_zone() if self.zone_map else None
        detection_zone = self.zone_map.detection_zone() if self.zone_map else None
        lanes = self.zone_map.zones_by_type("lane") if self.zone_map else []

        self.analysis_zone_current = 0
        for t in tracks:
            is_new = t.track_id not in self._counted_track_ids
            if is_new:
                self._counted_track_ids.add(t.track_id)
                self.class_counts[t.cls_name] = self.class_counts.get(t.cls_name, 0) + 1

            if self.zone_map is None:
                continue
            point = self.zone_map.anchor_point(t.bbox)

            if is_new and (detection_zone is None or point_in_polygon(point, detection_zone.polygon)):
                self._detection_zone_total += 1

            if analysis_zone is not None:
                inside_analysis = point_in_polygon(point, analysis_zone.polygon)
                if inside_analysis:
                    self.analysis_zone_current += 1
                if is_new and inside_analysis:
                    self.analysis_zone_total += 1

            for lane in lanes:
                if point_in_polygon(point, lane.polygon):
                    key = (t.track_id, lane.id)
                    if key not in self._counted_lane_pairs:
                        self._counted_lane_pairs.add(key)
                        self.lane_counts[lane.id] = self.lane_counts.get(lane.id, 0) + 1

    def register_violation(
        self, frame_idx: int, track_id: int, violation_type: str, lane_id: str | None = None
    ) -> None:
        self.violation_counts[violation_type] = self.violation_counts.get(violation_type, 0) + 1
        self.violation_log.appendleft(
            ViolationLogEntry(frame_idx=frame_idx, track_id=track_id, violation_type=violation_type, lane_id=lane_id)
        )

    def unregister_violation(self, violation_type: str) -> None:
        """Trừ 1 vi phạm khỏi số đếm khi bị RÚT LẠI (đèn đỏ xác nhận rẽ đúng hướng miễn trừ, mũ
        bảo hiểm đảo phiếu — xem pipeline.py) — trước đây KHÔNG có hàm này, chỉ dict cục bộ trong
        `Pipeline.run()` (dùng cho kết quả trả về CLI) được trừ, khiến `self.violation_counts`
        (nguồn cho `traffic_statistics.csv` VÀ số liệu live Web UI) bị đếm THỪA các vi phạm đã rút
        lại — phát hiện khi dựng lại Web UI, xem CLAUDE.md. `max(0, ...)` phòng lệch số nếu gọi
        sai thứ tự (không nên xảy ra, nhưng tránh số âm vô nghĩa nếu có)."""
        current = self.violation_counts.get(violation_type, 0)
        self.violation_counts[violation_type] = max(0, current - 1)

    @property
    def total_vehicles(self) -> int:
        return len(self._counted_track_ids)

    @property
    def total_violations(self) -> int:
        return sum(self.violation_counts.values())

    @property
    def has_detection_zone(self) -> bool:
        return bool(self.zone_map and self.zone_map.detection_zone() is not None)

    @property
    def has_analysis_zone(self) -> bool:
        return bool(self.zone_map and self.zone_map.analysis_zone() is not None)

    @property
    def detection_zone_total(self) -> int:
        """Không khai báo vùng nhận diện (role="detection") trong config = coi cả khung hình là
        vùng nhận diện, bằng đúng `total_vehicles`."""
        return self._detection_zone_total if self.has_detection_zone else self.total_vehicles
