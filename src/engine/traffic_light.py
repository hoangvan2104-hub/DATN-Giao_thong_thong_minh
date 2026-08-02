"""Phát hiện trạng thái đèn tín hiệu — chế độ "detect" (đọc màu HSV thật qua traffic_light_zone)
hoặc "manual" (mô phỏng chu kỳ cố định, dùng khi đèn không nhìn rõ trong khung hình)."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.engine.zones import ZoneMap

STATE_UNKNOWN = "unknown"


@dataclass
class TrafficLightConfig:
    id: str
    mode: str  # "detect" | "manual"
    traffic_light_zone_id: str | None
    lane_ids: list[str]
    # Hướng cụ thể đèn này áp dụng trong lane (rỗng = áp dụng mọi hướng của lane) — hỗ trợ
    # nhiều đèn/lane theo hướng (vd đèn rẽ trái riêng, đèn đi thẳng riêng trong cùng 1 lane).
    directions: list[str]
    hsv_ranges: dict = field(default_factory=dict)
    min_pixels: int = 50
    manual_states: list[str] = field(default_factory=lambda: ["red", "green"])
    manual_durations: dict = field(default_factory=dict)
    manual_start_offset: float = 0.0
    # Vẽ icon mô phỏng màu đèn hiện tại trực tiếp lên frame tại đúng vị trí traffic_light_zone
    # — hữu ích khi đèn dùng mode=manual (không có đèn thật để nhìn trong khung hình) hoặc khi
    # muốn xác nhận nhanh bằng mắt hệ thống đang đọc đúng màu (mode=detect). Tắt mặc định.
    show_icon: bool = False
    icon_radius: int = 10


class TrafficLightEngine:
    def __init__(self, config: dict, zone_map: ZoneMap):
        self.zone_map = zone_map
        self.lights: dict[str, TrafficLightConfig] = {}
        self.lane_to_lights: dict[str, list[str]] = {}

        for tl in config.get("traffic_lights", []):
            detect = tl.get("detect", {})
            manual = tl.get("manual", {})
            apply_to = tl.get("apply_to", {})
            display = tl.get("display", {})
            lane_ids = apply_to.get("lane_ids", [])
            cfg = TrafficLightConfig(
                id=tl["id"],
                mode=tl.get("mode", "manual"),
                traffic_light_zone_id=tl.get("traffic_light_zone_id"),
                lane_ids=lane_ids,
                directions=apply_to.get("directions", []),
                hsv_ranges=detect.get("hsv_ranges", {}),
                min_pixels=detect.get("min_pixels", 50),
                manual_states=manual.get("states", ["red", "green"]),
                manual_durations=manual.get("durations", {}),
                manual_start_offset=manual.get("start_offset", 0.0),
                show_icon=display.get("show_icon", False),
                icon_radius=display.get("radius", 10),
            )
            self.lights[cfg.id] = cfg
            for lid in lane_ids:
                self.lane_to_lights.setdefault(lid, []).append(cfg.id)

        self._states: dict[str, str] = {tid: STATE_UNKNOWN for tid in self.lights}

    def update(self, frame: np.ndarray, timestamp_sec: float) -> None:
        """Cập nhật trạng thái mọi đèn theo frame hiện tại — gọi 1 lần mỗi frame."""
        for tl in self.lights.values():
            if tl.mode == "detect":
                self._states[tl.id] = self._detect_color(frame, tl)
            else:
                self._states[tl.id] = self._manual_state(tl, timestamp_sec)

    @property
    def states(self) -> dict[str, str]:
        """Trạng thái hiện tại của mọi đèn ({id: "red"|"yellow"|"green"|"unknown"}) — dùng để
        hiển thị (bảng thống kê/icon mô phỏng), không phải để xét vi phạm (rules dùng
        `state_for_lane()` để lọc đúng theo lane/hướng)."""
        return dict(self._states)

    def _detect_color(self, frame: np.ndarray, tl: TrafficLightConfig) -> str:
        if tl.traffic_light_zone_id is None:
            return STATE_UNKNOWN
        zone = self.zone_map.zones.get(tl.traffic_light_zone_id)
        if zone is None or len(zone.polygon) < 3:
            return STATE_UNKNOWN

        xs = [p[0] for p in zone.polygon]
        ys = [p[1] for p in zone.polygon]
        x1, x2 = max(int(min(xs)), 0), min(int(max(xs)) + 1, frame.shape[1])
        y1, y2 = max(int(min(ys)), 0), min(int(max(ys)) + 1, frame.shape[0])
        if x2 <= x1 or y2 <= y1:
            return STATE_UNKNOWN

        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        counts = {state: self._count_pixels_in_ranges(hsv, ranges) for state, ranges in tl.hsv_ranges.items()}

        # Uu tien "red" khi du nguong pixel, KE CA khi "yellow" dem duoc nhieu pixel hon — pha
        # do+vang (den giao thong VN thuong bat do+vang cung luc vai giay truoc khi chuyen xanh)
        # ve mat luat van la CAM DI, giong het "red" thuan tuy. Neu chon theo count cao nhat nhu
        # truoc, giai doan nay se bi doc thanh "yellow" (do bulb do van sang nhung it pixel hon
        # bulb vang dang len) khien RedLightRule (chi bat vi pham luc state=="red") bo sot vi pham
        # that trong pha nay — da xac nhan bang du lieu HSV thuc te tren vid_test.mp4 (xem nhat ky
        # ky thuat). Cung dung logic uu tien "red" nhu state_for_lane() da lam khi gop nhieu den.
        if counts.get("red", 0) >= tl.min_pixels:
            return "red"

        best_state, best_count = STATE_UNKNOWN, 0
        for state, count in counts.items():
            if count > best_count:
                best_state, best_count = state, count
        if best_count < tl.min_pixels:
            return STATE_UNKNOWN
        return best_state

    @staticmethod
    def _count_pixels_in_ranges(hsv: np.ndarray, ranges: list) -> int:
        # ranges: list phẳng các cặp [lower, upper] — màu đỏ cần 2 dải do quấn quanh hue 0/179.
        mask_total: np.ndarray | None = None
        for i in range(0, len(ranges) - 1, 2):
            lower = np.array(ranges[i], dtype=np.uint8)
            upper = np.array(ranges[i + 1], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            mask_total = mask if mask_total is None else cv2.bitwise_or(mask_total, mask)
        return int(cv2.countNonZero(mask_total)) if mask_total is not None else 0

    @staticmethod
    def _manual_state(tl: TrafficLightConfig, timestamp_sec: float) -> str:
        if not tl.manual_states or not tl.manual_durations:
            return STATE_UNKNOWN
        cycle = sum(tl.manual_durations.get(s, 0) for s in tl.manual_states)
        if cycle <= 0:
            return STATE_UNKNOWN
        t = (timestamp_sec + tl.manual_start_offset) % cycle
        acc = 0.0
        for s in tl.manual_states:
            acc += tl.manual_durations.get(s, 0)
            if t < acc:
                return s
        return tl.manual_states[-1]

    def _light_applies_to_direction(self, light_id: str, direction: str) -> bool:
        """1 đèn áp dụng cho `direction` nếu `directions` khai báo RỖNG (áp dụng mọi hướng của
        lane — xem docstring `TrafficLightConfig.directions`) HOẶC `direction` nằm trong danh
        sách đó."""
        directions = self.lights[light_id].directions
        return not directions or direction in directions

    def direction_has_light(self, lane_id: str, direction: str) -> bool:
        """Có ÍT NHẤT 1 đèn (đã gán cho lane này) thực sự áp dụng cho `direction` hay không —
        câu hỏi CẤU TRÚC dựa vào `apply_to.directions` trong config, KHÔNG phụ thuộc màu đèn
        hiện tại. Dùng để tự động suy ra 1 hướng có bị đèn quản lý hay không MÀ KHÔNG cần khai
        báo tay `red_light_exceptions` riêng — nếu user đã khai báo `apply_to.directions` (vd
        chỉ thẳng+trái), hướng còn lại (vd rẽ phải) tự động được coi là không bị quản lý."""
        light_ids = self.lane_to_lights.get(lane_id, [])
        return any(self._light_applies_to_direction(tid, direction) for tid in light_ids)

    def state_for_lane(self, lane_id: str, direction: str | None = None) -> str:
        """Trạng thái đèn áp dụng cho 1 lane, có thể lọc theo hướng cụ thể (nếu lane có nhiều
        đèn theo hướng). Kết hợp nhiều đèn khớp: ưu tiên đỏ (an toàn — chỉ coi là được đi khi
        TẤT CẢ đèn liên quan đều xanh)."""
        light_ids = self.lane_to_lights.get(lane_id, [])
        if not light_ids:
            return STATE_UNKNOWN

        if direction is not None:
            relevant = [tid for tid in light_ids if self._light_applies_to_direction(tid, direction)]
            if not relevant:
                # KHÔNG fallback về toàn bộ đèn của lane nữa — nếu mọi đèn đều khai báo
                # directions KHÔNG RỖNG và không đèn nào quản `direction` này, nghĩa là hướng
                # đó không chịu sự điều khiển của đèn nào (vd đèn chỉ quản thẳng+trái, rẽ phải
                # tự do) — không thể "vượt đèn đỏ" theo 1 hướng không có đèn quản lý. Fallback cũ
                # (dùng lại TOÀN BỘ đèn của lane) từng gây báo sai vi phạm cho đúng trường hợp
                # này.
                return STATE_UNKNOWN
        else:
            relevant = light_ids

        states = [self._states.get(tid, STATE_UNKNOWN) for tid in relevant]
        if "red" in states:
            return "red"
        if "yellow" in states:
            return "yellow"
        if states and all(s == "green" for s in states):
            return "green"
        return STATE_UNKNOWN
