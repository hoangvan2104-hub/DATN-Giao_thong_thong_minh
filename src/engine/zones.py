"""Nạp zones/lines/vectors từ config và tra cứu (dùng cho rule vi phạm sau này)."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.utils.geometry import bbox_anchor_point, point_in_polygon


@dataclass
class Zone:
    id: str
    type: str
    name: str
    polygon: list[tuple[float, float]]
    # Chỉ dùng cho type="lane". Cả 2 đều dạng {mode: "allow"|"deny", values: [...] | "all"}
    # — xem CONFIG_GUIDE.md để hiểu cơ chế + ví dụ. Không khai báo trong config (hoặc khai
    # báo thiếu) thì mặc định "allow" + "all" (không giới hạn gì).
    direction_mode: str = "allow"  # "allow" | "deny"
    directions: list[str] | str = "all"
    class_mode: str = "allow"  # "allow" | "deny"
    classes: list[str] | str = "all"
    # Chỉ dùng cho type="road": "detection" = giới hạn vùng nhận diện (không set = cả khung
    # hình). "analysis" = vùng xuất phát của xe — xe được coi là "xuất phát" khi còn trong
    # vùng này, trước khi cắt qua 1 vạch end_direction để xác định hướng đi thực tế (dùng
    # cho rule sai hướng rẽ, Phase 2).
    role: str | None = None
    # Chỉ dùng cho type="lane": ngoại lệ khi xét vi phạm vượt đèn đỏ.
    # allow_directions_on_red: các hướng được phép đi tiếp khi đèn đỏ (vd rẽ phải khi đỏ nếu
    # luật cho phép) — chỉ áp dụng khi lane CHỈ cho đúng 1 hướng trùng khớp (không đủ thông
    # tin xác định hướng thực tế nếu lane cho nhiều hướng, xem rules/red_light.py).
    # exempt_classes: các lớp phương tiện được đi tiếp khi đèn đỏ (vd xe máy theo luật riêng).
    red_light_exceptions: dict = field(default_factory=dict)

    def is_direction_allowed(self, direction: str) -> bool:
        if self.directions == "all":
            return self.direction_mode == "allow"
        if self.direction_mode == "allow":
            return direction in self.directions
        return direction not in self.directions

    def is_class_allowed(self, cls_name: str) -> bool:
        if self.classes == "all":
            return self.class_mode == "allow"
        if self.class_mode == "allow":
            return cls_name in self.classes
        return cls_name not in self.classes


@dataclass
class Line:
    id: str
    type: str
    name: str
    points: list[tuple[float, float]]
    lane_id: str | None = None
    direction: str | None = None


@dataclass
class Vector:
    id: str
    name: str
    start: tuple[float, float]
    end: tuple[float, float]
    # None = vector mặc định dùng chung cho lane nào không có vector riêng. Có giá trị =
    # chỉ áp dụng cho đúng lane đó — fix bug hệ thống cũ (so toàn bộ vectors không lọc theo
    # lane, gây false positive/negative khi nhiều lane có hướng lưu thông khác nhau).
    lane_id: str | None = None


class ZoneMap:
    """Tra cứu hình học đã nạp từ 1 file config — dùng chung cho render và rule engines."""

    def __init__(self, config: dict):
        self.zones: dict[str, Zone] = {}
        self.lines: dict[str, Line] = {}
        self.vectors: dict[str, Vector] = {}

        # Điểm neo dùng thống nhất cho MỌI nơi tra cứu hình học (zone matching, cắt vạch,
        # quỹ đạo...) — đọc từ system.tracking_point, mặc định "center" (tâm bbox) nếu không
        # khai báo trong config — trung tính, hợp lý cho đa số góc camera khi user chưa kịp
        # tinh chỉnh. Không dùng cứng 1 giá trị ở từng nơi nữa để tránh lệch điểm neo giữa các
        # rule (đã gặp vấn đề khi điểm neo không phù hợp góc quay fisheye).
        tp = config.get("system", {}).get("tracking_point", {})
        self.anchor: str = tp.get("anchor", "center")
        self.anchor_offset: tuple[float, float] = tuple(tp.get("offset", [0, 0]))

        for z in config.get("zones", []):
            rules = z.get("rules", {})
            direction_cfg = rules.get("direction_mode", {})
            class_cfg = rules.get("distribution_classes", {})
            self.zones[z["id"]] = Zone(
                id=z["id"],
                type=z["type"],
                name=z.get("name", z["id"]),
                polygon=[tuple(p) for p in z.get("polygon", [])],
                direction_mode=direction_cfg.get("mode", "allow"),
                directions=direction_cfg.get("directions", "all"),
                class_mode=class_cfg.get("mode", "allow"),
                classes=class_cfg.get("classes", "all"),
                role=z.get("role"),
                red_light_exceptions=rules.get("red_light_exceptions", {}),
            )

        for l in config.get("lines", []):
            self.lines[l["id"]] = Line(
                id=l["id"],
                type=l["type"],
                name=l.get("name", l["id"]),
                points=[tuple(p) for p in l.get("points", [])],
                lane_id=l.get("lane_id"),
                direction=l.get("direction"),
            )

        for v in config.get("vectors", []):
            self.vectors[v["id"]] = Vector(
                id=v["id"],
                name=v.get("name", v["id"]),
                start=tuple(v["start"]),
                end=tuple(v["end"]),
                lane_id=v.get("lane_id"),
            )

    def zones_by_type(self, zone_type: str) -> list[Zone]:
        return [z for z in self.zones.values() if z.type == zone_type]

    def detection_zone(self) -> Zone | None:
        """Zone type=road có role=detection, dùng để giới hạn vùng nhận diện.
        Không có thì nhận diện toàn bộ khung hình (trả về None)."""
        for zone in self.zones_by_type("road"):
            if zone.role == "detection":
                return zone
        return None

    def analysis_zone(self) -> Zone | None:
        """Zone type=road có role=analysis, dùng làm phạm vi áp dụng luật vi phạm (Phase 2)."""
        for zone in self.zones_by_type("road"):
            if zone.role == "analysis":
                return zone
        return None

    def lines_by_type(self, line_type: str) -> list[Line]:
        return [l for l in self.lines.values() if l.type == line_type]

    def anchor_point(self, bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        """Điểm neo thống nhất của bbox theo cấu hình `system.tracking_point`."""
        return bbox_anchor_point(bbox, self.anchor, self.anchor_offset)

    def find_lane(self, bbox: tuple[float, float, float, float]) -> Zone | None:
        """Trả về lane đang chứa điểm neo của bbox, hoặc None nếu không nằm trong lane nào."""
        point = self.anchor_point(bbox)
        for zone in self.zones_by_type("lane"):
            if point_in_polygon(point, zone.polygon):
                return zone
        return None

    def stop_line_for_lane(self, lane_id: str) -> Line | None:
        for line in self.lines_by_type("stop_line"):
            if line.lane_id == lane_id:
                return line
        return None

    def vector_for_lane(self, lane_id: str | None) -> Vector | None:
        """Vector hướng lưu thông đúng áp dụng cho 1 lane — ưu tiên vector gán riêng cho
        lane đó, không có thì dùng vector mặc định (lane_id=None). Tránh so vượt lane."""
        for v in self.vectors.values():
            if v.lane_id == lane_id:
                return v
        for v in self.vectors.values():
            if v.lane_id is None:
                return v
        return None

    def rescale(self, factor: float) -> None:
        """Co giãn toàn bộ toạ độ hình học (zones/lines/vectors/anchor_offset) theo 1 hệ số cố
        định — dùng khi pipeline giảm kích thước khung hình xử lý (vd nguồn 4K quá nặng, xem
        Pipeline.run) mà KHÔNG cần vẽ lại config, vì mọi toạ độ trong config tuyến tính theo độ
        phân giải gốc. Gọi 1 LẦN DUY NHẤT trước khi vòng lặp frame bắt đầu."""
        if factor == 1.0:
            return
        for zone in self.zones.values():
            zone.polygon = [(x * factor, y * factor) for x, y in zone.polygon]
        for line in self.lines.values():
            line.points = [(x * factor, y * factor) for x, y in line.points]
        for vector in self.vectors.values():
            vector.start = (vector.start[0] * factor, vector.start[1] * factor)
            vector.end = (vector.end[0] * factor, vector.end[1] * factor)
        self.anchor_offset = (self.anchor_offset[0] * factor, self.anchor_offset[1] * factor)
