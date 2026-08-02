"""Orchestrator: Tracker -> ZoneMap -> TrafficLight -> Rules -> Render.

Mỗi rule vi phạm bật/tắt độc lập qua config `system.features` (tiết kiệm tài nguyên khi
không cần). Khi tracker xoá 1 track, mọi rule + history đều được dọn state tương ứng qua
`tracker.last_removed_ids` — tránh rò rỉ bộ nhớ như hệ thống cũ (xem CLAUDE.md).
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from src.engine.congestion import CongestionMonitor
from src.engine.evidence import EvidenceWriter
from src.engine.logger import ViolationLogger
from src.engine.plate_ocr import PlateReader
from src.engine.rules.no_helmet import NoHelmetRule
from src.engine.rules.red_light import RedLightRule
from src.engine.rules.wrong_lane import WrongLaneRule
from src.engine.rules.wrong_turn import WrongTurnRule
from src.engine.rules.wrong_way import WrongWayRule
from src.engine.stats import StatsTracker
from src.engine.traffic_light import TrafficLightEngine
from src.engine.tracker import VehicleTracker
from src.engine.zones import ZoneMap
from src.render.layout import compose_layout
from src.render.overlay import TrajectoryHistory, draw_evidence_box, render_frame


# Hệ thống xử lý video giao thông Việt Nam — ngày giờ vi phạm (evidence/log) PHẢI theo múi giờ
# Việt Nam rõ ràng, không phụ thuộc múi giờ hệ điều hành máy chạy server (có thể lệch nếu máy đặt
# múi giờ khác, đặc biệt khi deploy lên server ở nơi khác).
VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def _build_roi_mask(shape: tuple[int, int], polygon: list[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


class Pipeline:
    def __init__(self, config: dict, model_path: str = "yolo11s.pt", device: str | None = None):
        self.config = config
        self.zone_map = ZoneMap(config)
        classes = config.get("classes") or None
        system = config.get("system", {})
        self.tracker = VehicleTracker(
            model_path=model_path,
            classes=classes,
            device=device,
            tracker_params=system.get("tracker_params"),
            **system.get("detection_params", {}),
        )
        self.history = TrajectoryHistory(anchor=self.zone_map.anchor, offset=self.zone_map.anchor_offset)
        self.traffic_lights = TrafficLightEngine(config, self.zone_map)
        # Tích luỹ suốt vòng đời track (không reset mỗi frame) — vi phạm chỉ xảy ra đúng 1
        # frame khi xe cắt qua vạch, nhưng cần GIỮ màu cảnh báo lâu hơn để người xem kịp thấy
        # (1 frame ở 25-45 FPS là 1 nháy sáng dưới 1/25 giây, gần như không thể nhận ra).
        # {track_id: {loại_vi_phạm: frame_xảy_ra_gần_nhất}} — 1 track có thể tích luỹ NHIỀU
        # loại vi phạm khác nhau, overlay tự đan xen màu + nhấp nháy lại khi có vi phạm mới.
        self.violation_events: dict[int, dict[str, int]] = {}
        # Số liệu đếm xe/vi phạm — dữ liệu có cấu trúc, chỉ dùng để hiển thị (chế độ --show
        # hoặc web UI sau này), KHÔNG bake vào video output lưu ra file.
        self.stats = StatsTracker(zone_map=self.zone_map)

        features = system.get("features", {})
        self.rules: dict[str, object] = {}
        # Đèn đỏ: tách riêng khỏi self.rules (cùng lý do plate_reader/helmet_rule bên dưới) — có
        # thể RÚT LẠI vi phạm (làn nhiều hướng, xác nhận muộn xe rẽ đúng hướng được miễn trừ qua
        # vạch end_direction — xem rules/red_light.py), nên interface update() trả về 2 giá trị
        # (vi phạm mới, track_id được rút lại) giống NoHelmetRule, khác các rule hình học thuần
        # còn lại trong self.rules chỉ trả về 1 danh sách vi phạm.
        self.red_light_rule: RedLightRule | None = None
        if features.get("detect_red_light", True):
            self.red_light_rule = RedLightRule(self.zone_map, self.traffic_lights)
        if features.get("detect_wrong_way", True):
            self.rules["wrong_way"] = WrongWayRule(self.zone_map, config)
        if features.get("detect_wrong_turn", True):
            self.rules["wrong_turn"] = WrongTurnRule(self.zone_map)
        if features.get("detect_wrong_lane", True):
            self.rules["wrong_lane"] = WrongLaneRule(self.zone_map)

        # Biển số: tắt mặc định (model YOLOv5 nặng, load chậm) — bật qua
        # system.features.detect_plate khi cần. Tham số throttle/max_attempts đọc từ
        # system.plate_params nếu có, không thì dùng mặc định của PlateReader.
        self.plate_reader: PlateReader | None = None
        if features.get("detect_plate", False):
            self.plate_reader = PlateReader(device=device, **system.get("plate_params", {}))
        self.plate_numbers: dict[int, str] = {}

        # Mũ bảo hiểm: cùng lý do tách riêng khỏi self.rules như plate_reader (cần đọc pixel
        # frame, không chỉ hình học bbox). Tắt mặc định — bật qua system.features.detect_helmet.
        self.helmet_rule: NoHelmetRule | None = None
        if features.get("detect_helmet", False):
            self.helmet_rule = NoHelmetRule(device=device, **system.get("helmet_params", {}))

        # Ùn tắc: không phải violation gắn theo track, mà trạng thái CHUNG của 1 vùng — chỉ
        # đếm ngưỡng, không dự đoán/ML (đơn giản theo đúng yêu cầu). Cần biết fps nguồn để tính
        # sustain_frames nên khởi tạo thật trong run() (xem _build_congestion_monitor), ở đây
        # chỉ giữ tham số thô.
        self._congestion_params: dict | None = system.get("congestion_config", {}) if features.get("detect_congestion", False) else None
        self.congestion: CongestionMonitor | None = None

        # Giới hạn cạnh dài nhất khung hình xử lý (vd nguồn 4K quá nặng cho detect/track/render
        # — bản thân detect vốn đã resize xuống detection_params.imgsz nên giữ nguyên độ phân
        # giải gốc cho các bước sau không có lợi gì về chất lượng, chỉ tốn chi phí). None/không
        # khai báo = không resize (giữ nguyên hành vi cũ). Áp dụng 1 lần khi biết được kích
        # thước khung hình thật trong run() — xem _apply_max_dimension.
        self.max_dimension: int | None = system.get("max_dimension")
        self.scale: float = 1.0

        # Bật/tắt từng layer overlay (zones/lines/quỹ đạo/bbox phương tiện) độc lập nhau — xem
        # overlay.py::render_frame. Mặc định True cả 4, khớp hành vi cũ khi config không khai báo
        # `display_params`. 2 cờ khung biển số/mũ bảo hiểm mặc định False (khác 4 cờ trên) — chỉ
        # có ý nghĩa khi feature tương ứng đang bật.
        display_params = system.get("display_params", {})
        self.show_zones: bool = display_params.get("show_zones", True)
        self.show_lines: bool = display_params.get("show_lines", True)
        self.show_trajectories: bool = display_params.get("show_trajectories", True)
        self.show_vehicle_box: bool = display_params.get("show_vehicle_box", True)
        self.show_plate_box: bool = display_params.get("show_plate_box", False)
        self.show_helmet_box: bool = display_params.get("show_helmet_box", False)

    def _apply_max_dimension(self, native_width: int, native_height: int) -> tuple[int, int]:
        """Tính hệ số co giãn (nếu cạnh dài nhất vượt `max_dimension`) và áp dụng NGAY cho
        zone_map/history (toạ độ hình học) — chỉ chạy 1 lần đầu run(), trước vòng lặp frame.
        Trả về (width, height) đã co giãn để dùng cho ROI mask/VideoWriter/resize mỗi frame."""
        if not self.max_dimension or max(native_width, native_height) <= self.max_dimension:
            return native_width, native_height
        self.scale = self.max_dimension / max(native_width, native_height)
        self.zone_map.rescale(self.scale)
        self.history.offset = self.zone_map.anchor_offset
        return round(native_width * self.scale), round(native_height * self.scale)

    def _resolve_base_datetime(self) -> datetime:
        """Mốc ngày giờ THẬT ứng với frame đầu tiên (frame_idx=0) của video — cộng thêm số giây
        đã trôi qua trong video tới lúc vi phạm để ra ngày giờ vi phạm thật (xem nơi gọi trong
        run()). Ưu tiên `meta.recording_started_at` (ISO 8601, vd "2026-07-18T08:30:00") nếu user
        khai báo trong config — chỉ dùng cho MỤC ĐÍCH LỊCH SỬ/hồ sơ khi biết chắc thời điểm camera
        thật sự bắt đầu quay (vd nhập lại từ metadata gốc), không có ý nghĩa realtime. Không khai
        báo (trường hợp phổ biến — vd video demo/test) thì lấy đúng THỜI ĐIỂM BẮT ĐẦU XỬ LÝ (giờ
        hệ thống lúc gọi `run()`) làm mốc — theo yêu cầu user (2026-07-26): mtime file video từng
        dùng làm dự phòng trước đây phản ánh lúc file được ghi/copy vào máy (vd video demo nằm sẵn
        trong repo từ đầu dự án), KHÔNG liên quan gì đến lúc thật sự chạy xử lý, gây hiểu nhầm
        ngày vi phạm là ngày cũ dù vừa chạy hôm nay. LUÔN gắn múi giờ Việt Nam (`VN_TZ`) rõ ràng —
        `recording_started_at` không có offset trong chuỗi ISO thì coi NHƯ đã là giờ Việt Nam
        (không phải giờ UTC/giờ hệ thống)."""
        recorded_at = self.config.get("meta", {}).get("recording_started_at")
        if recorded_at:
            dt = datetime.fromisoformat(recorded_at)
            return dt if dt.tzinfo else dt.replace(tzinfo=VN_TZ)
        return datetime.now(VN_TZ)

    def run(
        self, video_path: str, output_path: str | None = None, show: bool = False,
        max_frames: int | None = None, log_name: str | None = None,
        frame_callback: Callable[[np.ndarray, int, list], None] | None = None,
        stop_flag: threading.Event | None = None,
    ) -> dict:
        """`log_name`: nếu có, bật ghi log lịch sử vi phạm (data/logs/<log_name>/events.csv/json,
        vehicle_summary.csv, traffic_statistics.csv, traffic_flow.csv) + ảnh minh chứng
        (data/evidence/<log_name>/) — xem `src/engine/logger.py`/`evidence.py`. Không truyền
        (mặc định) = không ghi gì, dùng cho các lần chạy debug/eval nhanh (`eval/check_*.py`) để
        không làm rác `data/logs/`.

        `frame_callback`: gọi lại MỖI khung hình với `(frame_đã_render, frame_idx, tracks)` —
        khung hình đã vẽ đầy đủ overlay (giống hệt frame ghi ra `output_path`/hiện trong
        `--show`) + danh sách `Track` hiện tại, dùng để Web UI (src/web/) lấy khung hình sống
        stream ra trình duyệt + xây bảng xe/thống kê trực tiếp trong lúc đang xử lý mà KHÔNG cần
        đọc lại file output hay sửa engine để "biết về" web — điểm mở rộng thuần opt-in, im lặng
        bỏ qua nếu không truyền, không đổi hành vi của bất kỳ caller nào khác (main.py, eval/*).
        Callback chạy ĐỒNG BỘ trong vòng lặp xử lý — phải nhanh (vd chỉ đẩy vào 1 hàng đợi/biến
        rồi return ngay), không được block hay chạy tính toán nặng.

        `stop_flag`: cờ dừng bên NGOÀI (opt-in, mặc định None = bỏ qua, không đổi hành vi caller
        khác) — kiểm tra mỗi khung hình, dừng sớm nếu `.is_set()`. Cần cho nguồn KHÔNG có điểm kết
        thúc tự nhiên như file video (vd webcam sống — `video_path` là số nguyên index thiết bị,
        `cv2.VideoCapture` chấp nhận cả file path lẫn device index, không cần đổi gì thêm ở đây),
        nơi vòng lặp `cap.read()` sẽ không bao giờ tự dừng qua `if not ok: break`."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Không mở được video: {video_path}")

        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width, height = self._apply_max_dimension(width, height)

        if self._congestion_params is not None:
            self.congestion = CongestionMonitor(self.zone_map, fps=fps_src, **self._congestion_params)
            self.stats.congestion_enabled = True

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps_src, (width, height))

        evidence: EvidenceWriter | None = None
        violation_logger: ViolationLogger | None = None
        base_datetime: datetime | None = None
        if log_name:
            # meta.video_id bắt buộc (validate_config đã chặn config thiếu field này) — ghép vào
            # MỌI track_id xuất ra file (events/vehicle_summary/evidence) để duy nhất tuyệt đối
            # giữa các video, xem config_schema.py/docs/nhat-ky-ky-thuat.md.
            video_id = self.config["meta"]["video_id"]
            # EvidenceWriter tự tạo sẵn thư mục cho CẢ 5 loại vi phạm hệ thống hỗ trợ (xem
            # evidence.ALL_VIOLATION_TYPES) — không phụ thuộc video này có bật rule đó hay không.
            evidence = EvidenceWriter(log_name, video_id)
            flow_bucket_seconds = self.config.get("system", {}).get("log_params", {}).get("flow_bucket_seconds", 60)
            violation_logger = ViolationLogger(log_name, video_id, flow_bucket_seconds=flow_bucket_seconds)
            base_datetime = self._resolve_base_datetime()

        detection_zone = self.zone_map.detection_zone()
        roi_mask = _build_roi_mask((height, width), detection_zone.polygon) if detection_zone else None

        violation_counts: dict[str, int] = {name: 0 for name in self.rules}
        if self.red_light_rule is not None:
            violation_counts["red_light"] = 0
        if self.helmet_rule is not None:
            violation_counts["no_helmet"] = 0
        frame_idx = 0
        t_start = time.time()
        try:
            while True:
                ok, native_frame = cap.read()
                if not ok:
                    break
                if self.scale != 1.0:
                    # LƯU Ý: đã thử INTER_LINEAR (nhanh hơn ~4-5 lần so với INTER_AREA khi đo cô
                    # lập — xem docs/nhat-ky-ky-thuat.md mục tối ưu FPS) — đo trên dữ liệu thật
                    # phát hiện lệch nhẹ 1 xe trong tổng số xe đếm được (không lệch số vi phạm)
                    # trên camera.mp4, cùng LOẠI rủi ro "thay đổi nhỏ ở input detector chính gây
                    # lệch dây chuyền" đã xác nhận rõ ràng hơn với FP16 tracker (xem tracker.py).
                    # Giữ INTER_AREA (chậm hơn nhưng đã kiểm chứng khớp TUYỆT ĐỐI baseline đã
                    # dùng suốt dự án) để ưu tiên đúng khớp số liệu hơn tốc độ, nhất quán với lựa
                    # chọn tương tự ở tracker.py.
                    frame = cv2.resize(native_frame, (width, height), interpolation=cv2.INTER_AREA)
                else:
                    frame = native_frame

                # Tính 1 LẦN DUY NHẤT, dùng lại xuyên suốt vòng lặp (trước đó tính lặp lại
                # `frame_idx / fps_src` ở 3-4 chỗ khác nhau cùng 1 khung hình — cùng công thức,
                # gộp lại cho gọn, không đổi giá trị/hành vi).
                timestamp_sec = frame_idx / fps_src

                tracks = self.tracker.update(frame, roi_mask=roi_mask)
                self.history.update(tracks)
                self.stats.register_tracks(tracks)
                if violation_logger is not None:
                    for t in tracks:
                        lane = self.zone_map.find_lane(t.bbox)
                        violation_logger.register_vehicle(
                            t.track_id, t.cls_name, frame_idx, timestamp_sec,
                            lane_id=lane.id if lane else None,
                        )

                # Vi phạm MỚI xác nhận trong khung hình này — dùng để ghi evidence/log TRƯỚC khi
                # render_frame() vẽ overlay preview (xem lý do ở khối xử lý bên dưới).
                new_events: list[tuple[int, str, str | None]] = []

                if self.congestion is not None:
                    self.congestion.update(tracks)
                    self.stats.congestion_vehicle_count = self.congestion.vehicle_count
                    self.stats.is_congested = self.congestion.is_congested
                    self.stats.peak_vehicle_count = max(self.stats.peak_vehicle_count, self.congestion.vehicle_count)
                    self.stats.was_ever_congested = self.stats.was_ever_congested or self.congestion.is_congested

                if violation_logger is not None:
                    violation_logger.register_flow_frame(
                        timestamp_sec, time.time(),
                        self.congestion.is_congested if self.congestion is not None else False,
                    )

                if self.plate_reader is not None:
                    for t in tracks:
                        # Crop từ native_frame (độ phân giải gốc, chưa hạ) để giữ đủ chi tiết
                        # biển số — bbox track đang ở toạ độ khung hình ĐÃ hạ (self.scale) nên
                        # phải nhân ngược lại đúng hệ số trước khi crop từ native_frame.
                        crop_bbox = tuple(v / self.scale for v in t.bbox) if self.scale != 1.0 else t.bbox
                        plate = self.plate_reader.read(native_frame, t.track_id, crop_bbox, frame_idx)
                        if plate:
                            self.plate_numbers[t.track_id] = plate

                self.traffic_lights.update(frame, timestamp_sec=timestamp_sec)
                self.stats.traffic_light_states = self.traffic_lights.states
                if violation_logger is not None:
                    violation_logger.register_traffic_light_frame(timestamp_sec, self.traffic_lights.states)

                for name, rule in self.rules.items():
                    for violation in rule.update(tracks, frame_idx):
                        lane_id = getattr(violation, "lane_id", None)
                        self.violation_events.setdefault(violation.track_id, {})[name] = frame_idx
                        violation_counts[name] += 1
                        self.stats.register_violation(frame_idx, violation.track_id, name, lane_id)
                        new_events.append((violation.track_id, name, lane_id))

                if self.red_light_rule is not None:
                    new_violations, retracted_ids = self.red_light_rule.update(tracks, frame_idx)
                    for violation in new_violations:
                        self.violation_events.setdefault(violation.track_id, {})["red_light"] = frame_idx
                        violation_counts["red_light"] += 1
                        self.stats.register_violation(frame_idx, violation.track_id, "red_light", violation.lane_id)
                        new_events.append((violation.track_id, "red_light", violation.lane_id))
                    _RED_LIGHT_RETRACT_REASONS = {
                        "allow_on_red": "he_thong_xac_nhan_re_dung_huong_duoc_mien_tru_den_do",
                        "not_governed": "he_thong_xac_nhan_huong_re_khong_bi_den_nay_quan_ly",
                        "class_exempt": "he_thong_xac_nhan_loai_xe_nay_duoc_mien_tru_theo_huong_da_di",
                    }
                    for tid, retract_reason_key in retracted_ids:
                        # Rút lại vi phạm đèn đỏ ghi nhận tạm trên làn nhiều hướng, sau đó xác
                        # nhận xe rẽ đúng hướng được miễn trừ HOẶC hướng đó không bị đèn nào quản
                        # lý (xem rules/red_light.py) — cùng nguyên tắc audit trail như rút lại
                        # no_helmet bên dưới (giữ evidence/log, chỉ đánh dấu retracted).
                        self.violation_events.get(tid, {}).pop("red_light", None)
                        violation_counts["red_light"] -= 1
                        self.stats.unregister_violation("red_light")
                        if violation_logger is not None:
                            violation_logger.log_retraction(
                                tid, "red_light", reason=_RED_LIGHT_RETRACT_REASONS[retract_reason_key],
                            )

                if self.helmet_rule is not None:
                    new_violations, retracted_ids = self.helmet_rule.update(
                        tracks, frame_idx, frame, native_frame, self.scale
                    )
                    for violation in new_violations:
                        self.violation_events.setdefault(violation.track_id, {})["no_helmet"] = frame_idx
                        violation_counts["no_helmet"] += 1
                        self.stats.register_violation(frame_idx, violation.track_id, "no_helmet")
                        new_events.append((violation.track_id, "no_helmet", None))
                    for tid in retracted_ids:
                        # Rút lại vi phạm đã báo sai (xem NoHelmetRule) — dọn màu/nhãn cảnh báo
                        # trên overlay và trừ lại số đếm, không xoá dòng đã ghi trong log lịch
                        # sử (violation_log/violation_logger — giữ làm dấu vết đã từng xảy ra +
                        # đã được sửa).
                        self.violation_events.get(tid, {}).pop("no_helmet", None)
                        violation_counts["no_helmet"] -= 1
                        self.stats.unregister_violation("no_helmet")
                        if violation_logger is not None:
                            violation_logger.log_retraction(
                                tid, "no_helmet", reason="he_thong_tu_dao_nguoc_phieu_bau_sau_khi_co_them_bang_chung",
                            )

                removed_ids = self.tracker.last_removed_ids
                self.history.remove(removed_ids)
                for rule in self.rules.values():
                    rule.remove(removed_ids)
                if self.red_light_rule is not None:
                    self.red_light_rule.remove(removed_ids)
                if self.plate_reader is not None:
                    self.plate_reader.remove(removed_ids)
                if self.helmet_rule is not None:
                    self.helmet_rule.remove(removed_ids)
                for tid in removed_ids:
                    self.violation_events.pop(tid, None)
                    self.plate_numbers.pop(tid, None)

                if new_events and (evidence is not None or violation_logger is not None):
                    # PHẢI chụp evidence TRƯỚC khi render_frame() vẽ overlay bên dưới — khi
                    # scale=1.0, `frame` và `native_frame` là CÙNG 1 object (xem gán ở đầu vòng
                    # lặp), render_frame() mutate `frame` in-place sẽ làm bẩn luôn `native_frame`
                    # nếu chụp SAU. Ảnh evidence vẽ trên bản sao riêng của native_frame (độ phân
                    # giải gốc, không phải `frame` đã hạ), chỉ 1 bbox cho đúng xe vi phạm + ID —
                    # không lẫn zones/lines/track khác như khung hình preview.
                    track_by_id = {t.track_id: t for t in tracks}
                    plate_history = self.plate_reader.history if self.plate_reader is not None else {}
                    violation_dt = base_datetime + timedelta(seconds=timestamp_sec)
                    for track_id, vtype, lane_id in new_events:
                        evidence_path = None
                        t = track_by_id.get(track_id)
                        if evidence is not None and t is not None:
                            crop_bbox = tuple(v / self.scale for v in t.bbox) if self.scale != 1.0 else t.bbox
                            plate_crop = None
                            plate_text = plate_history.get(track_id)
                            if self.plate_reader is not None:
                                plate_crop = self.plate_reader.plate_crops.get(track_id)
                            evidence_frame = draw_evidence_box(
                                native_frame.copy(), crop_bbox, track_id, vtype,
                                plate_crop=plate_crop, plate_text=plate_text,
                            )
                            evidence_path = evidence.capture(track_id, violation_dt, vtype, evidence_frame)
                        if violation_logger is not None:
                            violation_logger.log_event(
                                frame_idx, timestamp_sec, violation_dt.isoformat(), track_id,
                                t.cls_name if t is not None else "unknown", vtype,
                                lane_id=lane_id, plate_number=plate_history.get(track_id), evidence_path=evidence_path,
                            )

                render_frame(
                    frame, self.zone_map, tracks, history=self.history,
                    violation_events=self.violation_events,
                    current_frame=frame_idx,
                    plate_numbers=self.plate_numbers,
                    traffic_lights=self.traffic_lights,
                    show_zones=self.show_zones,
                    show_lines=self.show_lines,
                    show_trajectories=self.show_trajectories,
                    show_vehicle_box=self.show_vehicle_box,
                    show_plate_box=self.show_plate_box,
                    show_helmet_box=self.show_helmet_box,
                    plate_relative_box=self.plate_reader.plate_relative_box if self.plate_reader is not None else None,
                    plate_capture_size=self.plate_reader.plate_capture_size if self.plate_reader is not None else None,
                    head_relative_box=self.helmet_rule.head_relative_box if self.helmet_rule is not None else None,
                    head_label=self.helmet_rule.head_label if self.helmet_rule is not None else None,
                    head_capture_size=self.helmet_rule.head_capture_size if self.helmet_rule is not None else None,
                    head_extend_ratio=self.helmet_rule.head_extend_ratio if self.helmet_rule is not None else 1.5,
                )

                if frame_callback is not None:
                    frame_callback(frame, frame_idx, tracks)

                if writer is not None:
                    writer.write(frame)
                if show:
                    current_fps = frame_idx / (time.time() - t_start) if frame_idx > 0 else 0.0
                    display_frame = compose_layout(frame, self.stats, current_fps)
                    cv2.imshow("preview", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_idx += 1
                if max_frames and frame_idx >= max_frames:
                    break
                if stop_flag is not None and stop_flag.is_set():
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()
            if evidence is not None:
                evidence.close()
            if violation_logger is not None:
                plate_history = self.plate_reader.history if self.plate_reader is not None else {}
                violation_logger.finalize(self.stats, plate_history, fps_src, base_datetime)

        elapsed = time.time() - t_start
        result = {
            "frames": frame_idx,
            "elapsed_sec": round(elapsed, 2),
            "fps_processed": round(frame_idx / elapsed, 2) if elapsed > 0 else 0.0,
            "fps_source": round(fps_src, 2),
            "violations": violation_counts,
            "vehicle_counts": {"total": self.stats.total_vehicles, **self.stats.class_counts},
        }
        if self.congestion is not None:
            result["congestion"] = {
                "peak_vehicle_count": self.stats.peak_vehicle_count,
                "was_ever_congested": self.stats.was_ever_congested,
            }
        return result
