"""Ghép layout 3 vùng cho preview cục bộ: video trái | bảng thống kê phải-trên |
log vi phạm phải-dưới. Đây là bản nháp trực tiếp cho layout web UI sau này (xem
docs/brainstorm-notes.md) — chỉ dùng cho chế độ xem trực tiếp (--show), KHÔNG áp dụng cho
video output lưu ra file (video lưu ra chỉ có bbox/ID/nhãn vi phạm, đúng yêu cầu rubric).
"""
from __future__ import annotations

import cv2
import numpy as np

from src.engine.stats import StatsTracker

PANEL_WIDTH = 340
BG_COLOR = (35, 35, 35)
TEXT_COLOR = (230, 230, 230)
LABEL_COLOR = (0, 200, 255)
DIVIDER_COLOR = (70, 70, 70)

# Không dùng dấu tiếng Việt — font Hershey của OpenCV không vẽ được Unicode có dấu.
VIOLATION_LABELS_VI = {
    "red_light": "Vuot den do",
    "wrong_lane": "Sai lan",
    "wrong_way": "Nguoc chieu",
    "wrong_turn": "Sai huong re",
    "no_helmet": "Khong mu bao hiem",
}
CONGESTED_COLOR = (0, 0, 255)
CLEAR_COLOR = (0, 200, 0)

# Màu badge trạng thái đèn tín hiệu theo yêu cầu user (hex RGB gốc -> BGR cho OpenCV):
# đỏ #ff3131/#7a0000, xanh #457a00/#233d00, vàng #ffd21f/#7a4900.
TL_BADGE_COLORS = {
    "red": {"bg": (49, 49, 255), "text": (0, 0, 122)},
    "green": {"bg": (0, 122, 69), "text": (0, 61, 35)},
    "yellow": {"bg": (31, 210, 255), "text": (0, 73, 122)},
    "unknown": {"bg": (90, 90, 90), "text": (200, 200, 200)},
}


def _text(panel: np.ndarray, text: str, x: int, y: int, scale: float = 0.55, color=TEXT_COLOR, thickness: int = 1) -> None:
    cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_tl_badges(panel: np.ndarray, x: int, y: int, states: dict[str, str], max_width: int) -> int:
    """Vẽ badge trạng thái từng đèn (vd TL1 đỏ, TL2 xanh...) nối hàng ngang, tự xuống hàng nếu
    tràn `max_width`. Trả về y sau khi vẽ xong."""
    cur_x, cur_y, row_h = x, y, 0
    for light_id in sorted(states):
        label = light_id.replace("_", "")
        colors = TL_BADGE_COLORS.get(states[light_id], TL_BADGE_COLORS["unknown"])
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        badge_w, badge_h = tw + 20, th + 16
        if cur_x + badge_w > x + max_width:
            cur_x = x
            cur_y += badge_h + 8
        cv2.rectangle(panel, (cur_x, cur_y), (cur_x + badge_w, cur_y + badge_h), colors["bg"], -1)
        cv2.putText(
            panel, label, (cur_x + 10, cur_y + badge_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors["text"], 2, cv2.LINE_AA,
        )
        cur_x += badge_w + 10
        row_h = badge_h
    return cur_y + row_h


def build_stats_panel(width: int, stats: StatsTracker, fps: float, max_height: int = 2000) -> np.ndarray:
    """Vẽ trên canvas TẠM đủ lớn (`max_height`) rồi cắt đúng phần nội dung thật đã vẽ — panel
    trước đó có chiều cao CỐ ĐỊNH (`h // 2`) trong khi số mục hiển thị (đèn tín hiệu/ùn tắc có
    hay không) thay đổi theo khung hình, gây tràn/mất nội dung phía dưới không đều giữa các
    khung hình (cảm giác như "đan xen" — đã gặp phản hồi thật). Trả về panel đúng chiều cao nội
    dung thật, không cố định."""
    panel = np.full((max_height, width, 3), BG_COLOR, dtype=np.uint8)
    y = 30
    _text(panel, "THONG KE", 15, y, 0.7, LABEL_COLOR, 2)
    y += 34
    cv2.line(panel, (15, y - 22), (width - 15, y - 22), DIVIDER_COLOR, 1)
    _text(panel, f"FPS xu ly: {fps:.1f}", 15, y)
    y += 30
    _text(panel, f"Tong so xe: {stats.total_vehicles}", 15, y)
    y += 26
    for cls_name in sorted(stats.class_counts):
        _text(panel, f"  - {cls_name}: {stats.class_counts[cls_name]}", 15, y, 0.5)
        y += 22

    if stats.has_detection_zone or stats.has_analysis_zone or stats.lane_counts:
        y += 16
        _text(panel, "VUNG / LAN DUONG", 15, y, 0.7, LABEL_COLOR, 2)
        y += 30
        cv2.line(panel, (15, y - 22), (width - 15, y - 22), DIVIDER_COLOR, 1)
        if stats.has_detection_zone:
            _text(panel, f"Vung nhan dien: {stats.detection_zone_total}", 15, y, 0.5)
            y += 22
        if stats.has_analysis_zone:
            _text(panel, f"Xuat phat tu vung bat dau: {stats.analysis_zone_total}", 15, y, 0.5)
            y += 22
            _text(panel, f"Dang o vung bat dau: {stats.analysis_zone_current}", 15, y, 0.5)
            y += 22
        for lane_id in sorted(stats.lane_counts):
            _text(panel, f"  - lan {lane_id}: {stats.lane_counts[lane_id]}", 15, y, 0.5)
            y += 22

    if stats.traffic_light_states:
        y += 16
        _text(panel, "DEN TIN HIEU", 15, y, 0.7, LABEL_COLOR, 2)
        y += 30
        cv2.line(panel, (15, y - 22), (width - 15, y - 22), DIVIDER_COLOR, 1)
        y = _draw_tl_badges(panel, 15, y - 14, stats.traffic_light_states, width - 30) + 10

    if stats.congestion_enabled:
        y += 16
        _text(panel, "GIAO THONG", 15, y, 0.7, LABEL_COLOR, 2)
        y += 30
        cv2.line(panel, (15, y - 22), (width - 15, y - 22), DIVIDER_COLOR, 1)
        _text(panel, f"So xe trong vung: {stats.congestion_vehicle_count}", 15, y)
        y += 26
        status = "UN TAC" if stats.is_congested else "Thong thoang"
        color = CONGESTED_COLOR if stats.is_congested else CLEAR_COLOR
        _text(panel, status, 15, y, 0.6, color, 2)
        y += 26

    y += 16
    _text(panel, "VI PHAM", 15, y, 0.7, LABEL_COLOR, 2)
    y += 30
    cv2.line(panel, (15, y - 22), (width - 15, y - 22), DIVIDER_COLOR, 1)
    _text(panel, f"Tong: {stats.total_violations}", 15, y)
    y += 26
    for vtype, count in stats.violation_counts.items():
        label = VIOLATION_LABELS_VI.get(vtype, vtype)
        _text(panel, f"  - {label}: {count}", 15, y, 0.5)
        y += 22

    return panel[: min(y + 10, max_height)]


def build_log_panel(width: int, height: int, stats: StatsTracker) -> np.ndarray:
    panel = np.full((height, width, 3), BG_COLOR, dtype=np.uint8)
    _text(panel, "LOG VI PHAM GAN NHAT", 15, 30, 0.65, LABEL_COLOR, 2)
    cv2.line(panel, (15, 42), (width - 15, 42), DIVIDER_COLOR, 1)

    y = 68
    row_h = 22
    max_rows = max((height - y) // row_h, 0)
    if not stats.violation_log:
        _text(panel, "(chua co vi pham nao)", 15, y, 0.48, (150, 150, 150))
    for entry in list(stats.violation_log)[:max_rows]:
        label = VIOLATION_LABELS_VI.get(entry.violation_type, entry.violation_type)
        _text(panel, f"#{entry.track_id} - {label} (frame {entry.frame_idx})", 15, y, 0.46)
        y += row_h

    return panel


def _get_screen_size() -> tuple[int, int]:
    """Trả về (width, height) màn hình thật đang dùng — hệ thống tái sử dụng cho nhiều video
    kích thước/tỉ lệ khác nhau (ngang, dọc, 4K, độ phân giải thấp...) nên KHÔNG hardcode giới
    hạn hiển thị theo 1 video cụ thể nào, phải hỏi màn hình thật. Fallback an toàn nếu không
    lấy được (không phải Windows, hoặc môi trường không có màn hình)."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


def _fit_frame(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Co khung hình vừa khít (max_w, max_h), GIỮ NGUYÊN tỉ lệ khung hình gốc (không bóp méo)
    — hoạt động đúng với mọi độ phân giải/tỉ lệ (ngang, dọc, vuông...). Chỉ co nhỏ lại, không
    phóng to (video đã nhỏ hơn màn hình thì giữ nguyên)."""
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def compose_layout(frame: np.ndarray, stats: StatsTracker, fps: float) -> np.ndarray:
    """Ghép video (trái) + bảng thống kê (phải-trên) + log vi phạm (phải-dưới).

    Video được co vừa màn hình thật trước khi ghép (giữ nguyên tỉ lệ khung hình) — tránh tràn
    màn hình với video độ phân giải cao/tỉ lệ bất kỳ (đã gặp phản hồi thật: video 4K hiển thị
    tràn, trông như bị phóng to). Không ảnh hưởng độ phân giải xử lý/lưu file, chỉ ảnh hưởng
    khung xem trực tiếp (--show)."""
    screen_w, screen_h = _get_screen_size()
    margin = 80  # chừa chỗ thanh tiêu đề cửa sổ/taskbar
    max_video_w = max(screen_w - PANEL_WIDTH - margin, 320)
    max_video_h = max(screen_h - margin, 240)
    frame = _fit_frame(frame, max_video_w, max_video_h)

    h = frame.shape[0]
    # Chiều cao bảng thống kê theo ĐÚNG nội dung thật (số mục thay đổi theo khung hình — có/không
    # đèn tín hiệu, có/không ùn tắc) thay vì chia cố định 50/50 — tránh tràn/mất nội dung phía
    # dưới không đều giữa các khung hình. Dành tối thiểu 1 khoảng cho log panel phía dưới.
    min_log_h = 120
    stats_panel = build_stats_panel(PANEL_WIDTH, stats, fps, max_height=max(h - min_log_h, 1))
    stats_h = stats_panel.shape[0]
    log_h = h - stats_h
    log_panel = build_log_panel(PANEL_WIDTH, log_h, stats)
    right_col = np.vstack([stats_panel, log_panel])
    return np.hstack([frame, right_col])
