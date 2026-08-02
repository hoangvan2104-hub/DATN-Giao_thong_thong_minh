"""Hình học dùng chung: point-in-polygon, vector/hướng, giao cắt đoạn thẳng."""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

Point = tuple[float, float]


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    if len(polygon) < 3:
        return False
    pts = np.array(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False) >= 0


def bbox_anchor_point(
    bbox: tuple[float, float, float, float],
    anchor: str = "bottom_center",
    offset: tuple[float, float] = (0.0, 0.0),
) -> Point:
    x1, y1, x2, y2 = bbox
    if anchor == "bottom_center":
        point = ((x1 + x2) / 2, y2)
    elif anchor == "center":
        point = ((x1 + x2) / 2, (y1 + y2) / 2)
    elif anchor == "right_center":
        point = (x2, (y1 + y2) / 2)
    else:
        raise ValueError(f"Anchor không hợp lệ: {anchor}")
    return (point[0] + offset[0], point[1] + offset[1])


def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    a = np.asarray(v1, dtype=np.float32)
    b = np.asarray(v2, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _ccw(a: Point, b: Point, c: Point) -> float:
    return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, p: Point) -> bool:
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """Đoạn thẳng p1-p2 có cắt đoạn thẳng p3-p4 không (kể cả trường hợp thẳng hàng chạm biên)."""
    d1, d2 = _ccw(p3, p4, p1), _ccw(p3, p4, p2)
    d3, d4 = _ccw(p1, p2, p3), _ccw(p1, p2, p4)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 != 0 and d2 != 0 and d3 != 0 and d4 != 0:
        return True
    if d1 == 0 and _on_segment(p3, p4, p1):
        return True
    if d2 == 0 and _on_segment(p3, p4, p2):
        return True
    if d3 == 0 and _on_segment(p1, p2, p3):
        return True
    if d4 == 0 and _on_segment(p1, p2, p4):
        return True
    return False


def polyline_crossed(prev_point: Point, curr_point: Point, polyline: Sequence[Point]) -> bool:
    """Đoạn di chuyển prev->curr có cắt qua polyline (thẳng hoặc gấp khúc, >=2 điểm) không."""
    for i in range(len(polyline) - 1):
        if segments_intersect(prev_point, curr_point, polyline[i], polyline[i + 1]):
            return True
    return False
