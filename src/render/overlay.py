"""Vẽ overlay lên khung hình video: zones mờ, vạch, bbox + ID track, cảnh báo vi phạm.

Nguyên tắc (xem CLAUDE.md): chỉ vẽ những gì gắn liền với video (hình học + đối tượng).
Bảng thống kê KHÔNG vẽ ở đây — xem src/render/layout.py.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from src.engine.traffic_light import TrafficLightEngine
from src.engine.tracker import Track
from src.engine.zones import ZoneMap
from src.utils.geometry import bbox_anchor_point

ZONE_COLORS = {
    "road": (96, 235, 9),
    "lane": (2, 155, 250),
    "wrong_way_zone": (50, 248, 255),
    "traffic_light_zone": (255, 255, 0),
}
LINE_COLORS = {
    "stop_line": (0, 165, 255),
    "end_direction": (255, 0, 255),
}
TL_ICON_COLORS = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
    "unknown": (128, 128, 128),
}
TRACK_COLOR = (0, 200, 0)
ANCHOR_COLOR = (0, 0, 255)

# Rút gọn tên lớp khi hiển thị trên nhãn bbox — "motorcycle" quá dài, dễ tràn dòng cùng #ID/conf.
DISPLAY_CLASS_NAMES = {
    "motorcycle": "moto",
}

# Màu box theo loại vi phạm (tham khảo 1 project Colab cũ của user) — không vi phạm = xanh lá.
VIOLATION_COLORS = {
    "red_light": (0, 0, 255),      # đỏ
    "wrong_lane": (0, 255, 255),   # vàng
    "wrong_way": (255, 0, 255),    # tím/hồng
    "wrong_turn": (0, 140, 255),   # cam
    "no_helmet": (255, 191, 0),    # xanh dương (deep sky blue)
}
# Không dùng dấu tiếng Việt — font mặc định OpenCV (Hershey) không vẽ được ký tự Unicode có dấu.
VIOLATION_LABELS = {
    "red_light": "VUOT DEN DO",
    "wrong_lane": "SAI LAN",
    "wrong_way": "NGUOC CHIEU",
    "wrong_turn": "SAI HUONG RE",
    "no_helmet": "KHONG MU BAO HIEM",
}
# Số frame sau khi 1 vi phạm MỚI xảy ra để nhấp nháy cảnh báo (viền dày + nhãn chữ) — thu hút
# chú ý đúng lúc vi phạm vừa xảy ra. Sau đó vẫn giữ màu vi phạm (persistent) nhưng thôi nhấp
# nháy, viền mảnh lại như bình thường.
FLASH_FRAMES = 20
# Nhấp nháy KHÔNG dùng màu đệm (trắng) — chỉ luân phiên qua chính các màu vi phạm mà xe đó đã
# phạm (1 loại thì đứng yên màu đó, nhiều loại thì đan xen qua lại). BLINK_INTERVAL = nhịp đan
# xen NHANH dùng trong cửa sổ nhấp nháy (~4 lần đổi/giây ở 25 FPS, thu hút chú ý); CYCLE_FRAMES
# = nhịp đan xen CHẬM dùng sau khi hết nhấp nháy (~0.8s/loại, đỡ chói mắt nhưng vẫn không loại
# nào bị 1 màu cố định che khuất vĩnh viễn).
BLINK_INTERVAL = 6
CYCLE_FRAMES = 20


class TrajectoryHistory:
    """Lưu vệt di chuyển (điểm neo) của từng track qua các frame — dùng để vẽ quỹ đạo.
    Chỉ phục vụ hiển thị (render), không phải dữ liệu dùng cho rule vi phạm."""

    def __init__(self, max_len: int = 30, anchor: str = "bottom_center", offset: tuple[float, float] = (0.0, 0.0)):
        self.max_len = max_len
        self.anchor = anchor
        self.offset = offset
        self._history: dict[int, deque[tuple[int, int]]] = {}

    def update(self, tracks: list[Track]) -> None:
        for t in tracks:
            x, y = bbox_anchor_point(t.bbox, self.anchor, self.offset)
            point = (int(x), int(y))
            if t.track_id not in self._history:
                self._history[t.track_id] = deque(maxlen=self.max_len)
            self._history[t.track_id].append(point)

    def remove(self, track_ids: set[int]) -> None:
        for tid in track_ids:
            self._history.pop(tid, None)

    def get(self, track_id: int) -> list[tuple[int, int]]:
        return list(self._history.get(track_id, []))


def draw_zones(frame: np.ndarray, zone_map: ZoneMap, alpha: float = 0.15) -> np.ndarray:
    overlay = frame.copy()
    for zone in zone_map.zones.values():
        if len(zone.polygon) < 3:
            continue
        pts = np.array(zone.polygon, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], ZONE_COLORS.get(zone.type, (200, 200, 200)))
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
    for zone in zone_map.zones.values():
        if len(zone.polygon) < 3:
            continue
        pts = np.array(zone.polygon, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    return frame


def draw_traffic_light_icons(frame: np.ndarray, zone_map: ZoneMap, traffic_lights: TrafficLightEngine) -> np.ndarray:
    """Vẽ icon tròn màu đèn hiện tại tại đúng vị trí traffic_light_zone — chỉ vẽ đèn có
    `display.show_icon=true` trong config (mặc định tắt). Hữu ích cho đèn `mode=manual` (không
    có đèn thật để nhìn trong khung hình) hoặc để xác nhận nhanh bằng mắt đèn `mode=detect`
    đang đọc đúng màu."""
    states = traffic_lights.states
    for tl in traffic_lights.lights.values():
        if not tl.show_icon or tl.traffic_light_zone_id is None:
            continue
        zone = zone_map.zones.get(tl.traffic_light_zone_id)
        if zone is None or len(zone.polygon) < 3:
            continue
        cx = int(sum(p[0] for p in zone.polygon) / len(zone.polygon))
        cy = int(sum(p[1] for p in zone.polygon) / len(zone.polygon))
        color = TL_ICON_COLORS.get(states.get(tl.id, "unknown"), TL_ICON_COLORS["unknown"])
        cv2.circle(frame, (cx, cy), tl.icon_radius, color, -1)
        cv2.circle(frame, (cx, cy), tl.icon_radius, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_lines(frame: np.ndarray, zone_map: ZoneMap) -> np.ndarray:
    for line in zone_map.lines.values():
        if len(line.points) < 2:
            continue
        pts = np.array(line.points, dtype=np.int32)
        color = LINE_COLORS.get(line.type, (255, 255, 255))
        cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
    return frame


def _track_violation_state(events: dict[str, int], current_frame: int) -> tuple[str | None, bool]:
    """Từ {loại_vi_phạm: frame_xảy_ra} của 1 track, trả về (loại đang hiển thị MÀU + NHÃN ngay
    lúc này, có đang trong cửa sổ nhấp nháy hay không). `events` là dict thường của Python (giữ
    thứ tự chèn từ 3.7+) nên thứ tự đan xen = đúng thứ tự các loại vi phạm xảy ra lần đầu.

    Không dùng màu đệm (trắng/khác) để nhấp nháy — "nhấp nháy" chính là đan xen NHANH
    (BLINK_INTERVAL) qua các màu vi phạm thật của xe đó; 1 loại thì đứng yên (không có gì để
    đan xen). Hết cửa sổ nhấp nháy (không loại nào vừa xảy ra gần đây) thì đan xen CHẬM lại
    (CYCLE_FRAMES) để đỡ chói mắt nhưng vẫn không loại nào bị màu khác che khuất vĩnh viễn.
    Nhãn cảnh báo LUÔN lấy đúng theo `active_type` trả về (không lấy danh sách riêng) để không
    bao giờ lệch với màu box đang hiển thị."""
    if not events:
        return None, False
    vtypes = list(events.keys())
    is_flashing = any(current_frame - fr < FLASH_FRAMES for fr in events.values())
    interval = BLINK_INTERVAL if is_flashing else CYCLE_FRAMES
    active_type = vtypes[(current_frame // interval) % len(vtypes)]
    return active_type, is_flashing


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    violation_events: dict[int, dict[str, int]] | None = None,
    current_frame: int = 0,
    plate_numbers: dict[int, str] | None = None,
) -> np.ndarray:
    """violation_events: {track_id: {loại_vi_phạm: frame_xảy_ra_gần_nhất}} — 1 track có thể
    có nhiều loại vi phạm cùng lúc (vd vừa vượt đèn đỏ vừa ngược chiều), màu bbox đan xen luân
    phiên qua từng loại thay vì chỉ hiện 1 màu cố định."""
    violation_events = violation_events or {}
    plate_numbers = plate_numbers or {}
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)
        active_type, is_flashing = _track_violation_state(violation_events.get(t.track_id, {}), current_frame)
        color = VIOLATION_COLORS.get(active_type, TRACK_COLOR)
        thickness = 4 if is_flashing else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Biển số xuống dòng RIÊNG (thay vì nối cùng dòng với #ID/loại/conf) — ghép chung 1
        # dòng dễ tràn ra ngoài khung hình, không đọc được hết (đã gặp phản hồi thật).
        cls_label = DISPLAY_CLASS_NAMES.get(t.cls_name, t.cls_name)
        label = f"#{t.track_id} {cls_label} {t.conf:.2f}"
        plate = plate_numbers.get(t.track_id)
        lines = [label, f"[{plate}]"] if plate else [label]
        y_cursor = y1
        for line in reversed(lines):
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_top = max(y_cursor - th - 6, 0)
            cv2.rectangle(frame, (x1, y_top), (x1 + tw + 4, y_cursor), color, -1)
            cv2.putText(
                frame, line, (x1 + 2, max(y_cursor - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
            y_cursor = y_top

        if is_flashing:
            warn = VIOLATION_LABELS.get(active_type, "VI PHAM")
            (ww, wh), _ = cv2.getTextSize(warn, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            wy = min(y2 + wh + 10, frame.shape[0] - 4)
            cv2.rectangle(frame, (x1, wy - wh - 6), (x1 + ww + 8, wy + 4), color, -1)
            cv2.putText(
                frame, warn, (x1 + 4, wy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )
    return frame


EVIDENCE_PLATE_INSET_WIDTH = 220  # px — chiều rộng cố định của ảnh cận cảnh biển số ghép vào evidence


def draw_evidence_box(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    track_id: int,
    violation_type: str,
    plate_crop: np.ndarray | None = None,
    plate_text: str | None = None,
) -> np.ndarray:
    """Vẽ ảnh minh chứng vi phạm (evidence) — khác `draw_tracks()` (dùng cho preview/video output,
    vẽ MỌI track + zones/lines/vectors): ở đây `frame` truyền vào phải là khung hình SẠCH (chưa
    qua render_frame()), chỉ vẽ ĐÚNG 1 bbox bao trọn xe vi phạm + nhãn ID — không ghi loại xe/
    conf/zones/track khác để ảnh evidence tập trung hoàn toàn vào đối tượng vi phạm. Mutates
    `frame` in place, trả về `frame` luôn cho tiện gọi 1 dòng.

    `plate_crop`/`plate_text`: nếu track này đã đọc được biển số (xem `PlateReader.plate_crops`),
    ghép thêm 1 ảnh cận cảnh biển số + chữ số đọc được vào góc dưới-phải, phóng to theo chiều rộng
    cố định `EVIDENCE_PLATE_INSET_WIDTH` (crop biển số gốc thường rất nhỏ, vài chục px, cần phóng
    to mới đọc được bằng mắt trên ảnh evidence)."""
    color = VIOLATION_COLORS.get(violation_type, TRACK_COLOR)
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    label = f"ID {track_id}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    y_top = max(y1 - th - 12, 0)
    cv2.rectangle(frame, (x1, y_top), (x1 + tw + 10, y1), color, -1)
    cv2.putText(
        frame, label, (x1 + 5, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )

    if plate_crop is not None and plate_crop.size > 0:
        fh, fw = frame.shape[:2]
        ch, cw = plate_crop.shape[:2]
        scale = EVIDENCE_PLATE_INSET_WIDTH / cw
        inset_w, inset_h = EVIDENCE_PLATE_INSET_WIDTH, max(1, round(ch * scale))
        inset = cv2.resize(plate_crop, (inset_w, inset_h), interpolation=cv2.INTER_CUBIC)

        label_h = 26
        pad = 6
        box_w, box_h = inset_w + pad * 2, inset_h + label_h + pad * 2
        ix1 = max(0, fw - box_w - 12)
        iy1 = max(0, fh - box_h - 12)
        ix2, iy2 = min(fw, ix1 + box_w), min(fh, iy1 + box_h)

        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (30, 30, 30), -1)
        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (255, 255, 255), 2)
        img_y1 = iy1 + label_h + pad
        img_x1 = ix1 + pad
        avail_h, avail_w = iy2 - img_y1, ix2 - img_x1
        if avail_h > 0 and avail_w > 0:
            paste = inset[: max(0, avail_h), : max(0, avail_w)]
            ph, pw = paste.shape[:2]
            frame[img_y1:img_y1 + ph, img_x1:img_x1 + pw] = paste
        cv2.putText(
            frame, plate_text or "?", (ix1 + pad, iy1 + label_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

    return frame


def draw_trajectories(
    frame: np.ndarray,
    tracks: list[Track],
    history: TrajectoryHistory,
    violation_events: dict[int, dict[str, int]] | None = None,
    current_frame: int = 0,
) -> np.ndarray:
    """Vẽ vệt di chuyển (đường nối các điểm neo cũ) + điểm neo hiện tại của từng track."""
    violation_events = violation_events or {}
    for t in tracks:
        points = history.get(t.track_id)
        active_type, _ = _track_violation_state(violation_events.get(t.track_id, {}), current_frame)
        color = VIOLATION_COLORS.get(active_type, TRACK_COLOR)
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], color, 2, cv2.LINE_AA)
        if points:
            cv2.circle(frame, points[-1], 4, ANCHOR_COLOR, -1)
    return frame


PLATE_BOX_COLOR = (255, 255, 255)
HELMET_BOX_COLORS = {"helmet": (0, 255, 0), "no_helmet": (0, 0, 255)}
# Cả 2 model biển số/mũ bảo hiểm đều THROTTLE (không chạy mọi khung hình) rồi cache — box tỉ lệ đã
# lưu có thể là của vài/nhiều khung hình trước, trong lúc đó xe có thể tiến/lùi khác hẳn khoảng
# cách tới camera làm bbox đổi kích thước nhiều (phối cảnh) khiến vị trí tỉ lệ không còn đúng (đã
# gặp thật: khung biển số trôi xuống dưới yên xe khi xe tới gần camera hơn hẳn lúc đọc được biển
# số — xem docs/nhat-ky-ky-thuat.md). Nếu kích thước bbox HIỆN TẠI lệch quá xa so với lúc chụp
# (theo 1 trong 2 chiều rộng/cao) thì ẨN khung thay vì vẽ sai vị trí — thà không hiện còn hơn hiện
# sai.
BOX_DRIFT_TOLERANCE = 1.6
# CHỈ so kích thước tổng thể (BOX_DRIFT_TOLERANCE) là CHƯA ĐỦ — đã đo trực tiếp 1 ca thật lệch rõ
# dù cả 2 trục riêng lẻ vẫn trong ngưỡng (rộng co lại 0.94x, CAO giãn ra 1.33x — bbox đổi từ hình
# chữ nhật NẰM NGANG sang gần vuông khi xe tới gần camera, dù "kích thước tổng" không đổi nhiều).
# Vấn đề thật là bbox đổi HÌNH DẠNG (tỉ lệ rộng/cao), không chỉ đổi kích thước — 1 box tỉ lệ tính
# cho hình chữ nhật dẹt sẽ rơi sai chỗ khi áp lên hình gần vuông dù tổng diện tích tương đương.
# Thêm kiểm tra riêng: tỉ lệ giữa ratio_w/ratio_h (đo mức "méo" so với lúc chụp) phải đủ gần 1.
ASPECT_DRIFT_TOLERANCE = 1.3


def _reproject_relative_box(
    bbox: tuple[float, float, float, float], relative_box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Quy đổi 1 box TỈ LỆ (0-1, gốc lúc detect thành công) thành toạ độ tuyệt đối trong `bbox`
    THAM CHIẾU (có thể là bbox xe hiện tại, khác hẳn bbox lúc detect) — dùng chung cho cả khung
    biển số (PlateReader.plate_relative_box) lẫn khung mũ bảo hiểm (NoHelmetRule.head_relative_box,
    xem `draw_helmet_boxes` tự tính vùng mở rộng trước khi gọi hàm này)."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    rx1, ry1, rx2, ry2 = relative_box
    return (x1 + rx1 * w, y1 + ry1 * h, x1 + rx2 * w, y1 + ry2 * h)


def _size_drifted_too_much(current: tuple[float, float], captured: tuple[float, float]) -> bool:
    cw, ch = current
    ow, oh = captured
    if ow <= 0 or oh <= 0 or cw <= 0 or ch <= 0:
        return True
    ratio_w, ratio_h = cw / ow, ch / oh
    if not (1 / BOX_DRIFT_TOLERANCE <= ratio_w <= BOX_DRIFT_TOLERANCE
            and 1 / BOX_DRIFT_TOLERANCE <= ratio_h <= BOX_DRIFT_TOLERANCE):
        return True
    aspect_shift = ratio_w / ratio_h
    return not (1 / ASPECT_DRIFT_TOLERANCE <= aspect_shift <= ASPECT_DRIFT_TOLERANCE)


def draw_plate_boxes(
    frame: np.ndarray,
    tracks: list[Track],
    plate_relative_box: dict[int, tuple[float, float, float, float]],
    plate_capture_size: dict[int, tuple[float, float]],
) -> np.ndarray:
    """Vẽ khung quanh vùng biển số đã đọc được — tái chiếu tỉ lệ đã lưu lúc OCR thành công
    (`PlateReader.plate_relative_box`) lên bbox xe HIỆN TẠI mỗi khung hình, để khung luôn bám
    đúng vị trí xe đang di chuyển mà không cần chạy lại detector biển số mỗi khung hình (vốn chỉ
    chạy throttle, xem plate_ocr.py). Bỏ qua nếu bbox xe hiện tại đã đổi kích thước quá nhiều so
    với lúc chụp (xem `BOX_DRIFT_TOLERANCE`)."""
    for t in tracks:
        rel = plate_relative_box.get(t.track_id)
        if rel is None:
            continue
        x1, y1, x2, y2 = t.bbox
        captured = plate_capture_size.get(t.track_id)
        if captured is not None and _size_drifted_too_much((x2 - x1, y2 - y1), captured):
            continue
        bx1, by1, bx2, by2 = map(int, _reproject_relative_box(t.bbox, rel))
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), PLATE_BOX_COLOR, 2)
    return frame


def draw_helmet_boxes(
    frame: np.ndarray,
    tracks: list[Track],
    head_relative_box: dict[int, tuple[float, float, float, float]],
    head_label: dict[int, str],
    head_capture_size: dict[int, tuple[float, float]],
    head_extend_ratio: float,
) -> np.ndarray:
    """Vẽ khung quanh vùng đầu/mũ bảo hiểm — tái chiếu tỉ lệ đã lưu lúc kiểm tra gần nhất
    (`NoHelmetRule.head_relative_box`) lên vùng MỞ RỘNG lên trên tính từ bbox xe HIỆN TẠI (cùng
    công thức `head_extend_ratio` đã dùng lúc crop để detect — xem `NoHelmetRule._crop`), màu
    theo nhãn phân loại gần nhất (xanh = có mũ, đỏ = không mũ). Bỏ qua nếu bbox xe hiện tại đã
    đổi kích thước quá nhiều so với lúc chụp (xem `BOX_DRIFT_TOLERANCE`)."""
    for t in tracks:
        rel = head_relative_box.get(t.track_id)
        if rel is None:
            continue
        x1, y1, x2, y2 = t.bbox
        captured = head_capture_size.get(t.track_id)
        if captured is not None and _size_drifted_too_much((x2 - x1, y2 - y1), captured):
            continue
        extended_y1 = y1 - (y2 - y1) * head_extend_ratio
        bx1, by1, bx2, by2 = map(int, _reproject_relative_box((x1, extended_y1, x2, y2), rel))
        color = HELMET_BOX_COLORS.get(head_label.get(t.track_id), HELMET_BOX_COLORS["helmet"])
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
    return frame


def render_frame(
    frame: np.ndarray,
    zone_map: ZoneMap,
    tracks: list[Track],
    history: TrajectoryHistory | None = None,
    violation_events: dict[int, dict[str, int]] | None = None,
    current_frame: int = 0,
    plate_numbers: dict[int, str] | None = None,
    traffic_lights: TrafficLightEngine | None = None,
    show_zones: bool = True,
    show_lines: bool = True,
    show_trajectories: bool = True,
    show_vehicle_box: bool = True,
    show_plate_box: bool = False,
    show_helmet_box: bool = False,
    plate_relative_box: dict[int, tuple[float, float, float, float]] | None = None,
    plate_capture_size: dict[int, tuple[float, float]] | None = None,
    head_relative_box: dict[int, tuple[float, float, float, float]] | None = None,
    head_label: dict[int, str] | None = None,
    head_capture_size: dict[int, tuple[float, float]] | None = None,
    head_extend_ratio: float = 1.5,
) -> np.ndarray:
    """4 cờ `show_zones/show_lines/show_trajectories/show_vehicle_box` (mặc định True — không đổi
    hành vi khi caller không truyền) cho phép tắt bớt layer overlay theo `system.display_params`
    trong config, vd chỉ muốn giữ lại bbox phương tiện (tắt 3 cờ đầu) để dễ nhìn hơn khi không cần
    xem zones/lines/quỹ đạo, hoặc tắt luôn cả bbox+ID/nhãn vi phạm (`show_vehicle_box=False`) khi
    chỉ cần xem khung biển số/mũ bảo hiểm mà không muốn bbox xe che khuất.

    2 cờ `show_plate_box`/`show_helmet_box` mặc định FALSE (khác các cờ trên) — chỉ có ý nghĩa khi
    tính năng đọc biển số/mũ bảo hiểm đang BẬT (`system.features.detect_plate`/`detect_helmet`),
    và cần đúng 2 dict `plate_relative_box`/`head_relative_box` từ `PlateReader`/`NoHelmetRule`
    tương ứng mới vẽ được gì (không có thì coi như tắt, không lỗi)."""
    if show_zones:
        draw_zones(frame, zone_map)
    if show_lines:
        draw_lines(frame, zone_map)
    if traffic_lights is not None:
        draw_traffic_light_icons(frame, zone_map, traffic_lights)
    if show_trajectories and history is not None:
        draw_trajectories(frame, tracks, history, violation_events, current_frame)
    if show_vehicle_box:
        draw_tracks(frame, tracks, violation_events, current_frame, plate_numbers)
    # Vẽ SAU draw_tracks (không phải trước) — 2 khung mới đè lên trên nhãn ID/loại xe nếu có
    # trùng vị trí (vd khung mũ bảo hiểm nằm cao hơn cả bbox xe, dễ chồng lên nhãn), đảm bảo
    # luôn nhìn thấy rõ đúng tính năng vừa bật thay vì bị nhãn cũ che mất.
    if show_plate_box and plate_relative_box:
        draw_plate_boxes(frame, tracks, plate_relative_box, plate_capture_size or {})
    if show_helmet_box and head_relative_box:
        draw_helmet_boxes(frame, tracks, head_relative_box, head_label or {}, head_capture_size or {}, head_extend_ratio)
    return frame
