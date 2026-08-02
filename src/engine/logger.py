"""Ghi log lịch sử vi phạm ra CSV/JSON — data/logs/<video_name>/. Dữ liệu thuần (không phụ thuộc
cv2/hiển thị, cùng nguyên tắc tách Engine/Render như stats.py) để Web UI sau này đọc trực tiếp
file JSON mà không cần chạy lại pipeline.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from src.engine.evidence import ALL_VIOLATION_TYPES
from src.engine.stats import StatsTracker

# 5 loại phương tiện chuẩn dùng xuyên suốt dự án (khớp classes khai báo trong config các video,
# xem CLAUDE.md) — dùng làm cột cố định cho traffic_flow.csv thay vì suy ra động từ 1 video cụ
# thể, để mọi video ghi CÙNG 1 bộ cột (dễ gộp lại ở export_report_data.py).
FLOW_VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck", "bicycle"]

# Lý do khi CON NGƯỜI xem xét lại 1 cặp (xe, loại vi phạm) sau khi xem lại kết quả đã lưu — khác
# `EventRecord.review_reason` (lý do HỆ THỐNG tự rút lại lúc đang xử lý, xem RedLightRule/
# NoHelmetRule). Web UI hiển thị đúng 4 lựa chọn này (xem state.py::apply_manual_review,
# docs/nhat-ky-ky-thuat.md mục 28) — đặt ở đây (không phải state.py) vì đây là phần định nghĩa
# SCHEMA log, state.py chỉ gọi lại.
MANUAL_REVIEW_VERDICTS = {
    "vi_pham_that": "Vi phạm thật",
    "uu_tien_bo_qua": "Vi phạm thật nhưng được ưu tiên/bất khả kháng, bỏ qua",
    "loi_nhan_dien": "Không vi phạm do lỗi nhận diện hệ thống",
    "bo_sung_thu_cong": "Bổ sung vi phạm bị bỏ sót",
}
# 2 verdict khiến cờ vehicle_summary.csv KHÔNG còn tính là vi phạm đang hiệu lực (-1) — 2 verdict
# còn lại ("vi_pham_that"/"bo_sung_thu_cong") giữ/đặt cờ = 1.
_MANUAL_VERDICTS_AS_RETRACTED = {"uu_tien_bo_qua", "loi_nhan_dien"}


def resolve_violation_flag(
    auto_present: bool, auto_retracted: bool, auto_reason: str | None, manual_review: dict | None,
) -> tuple[int, str]:
    """Tính cờ (1/0/-1) + nội dung cột `review_note` cho 1 cặp (xe, loại vi phạm) — CON NGƯỜI có
    tiếng nói CUỐI CÙNG: nếu có `manual_review` (dict `{"verdict":..., "note":...}`), ghi đè hoàn
    toàn trạng thái tự động (`auto_present`/`auto_retracted`/`auto_reason` — từ hệ thống lúc xử
    lý). Dùng chung cho `_write_vehicle_summary()` (lúc Pipeline vừa chạy xong, `manual_review`
    luôn `None` vì chưa ai kịp xem xét) và `state.py::apply_manual_review()` (lúc tái tạo
    vehicle_summary.csv sau khi có xem xét thủ công)."""
    if manual_review is not None:
        verdict = manual_review["verdict"]
        note = (manual_review.get("note") or "").strip()
        label = MANUAL_REVIEW_VERDICTS.get(verdict, verdict)
        text = f"{label}: {note}" if note else label
        flag = -1 if verdict in _MANUAL_VERDICTS_AS_RETRACTED else 1
        return flag, text
    if not auto_present:
        return 0, ""
    if auto_retracted:
        return -1, auto_reason or ""
    return 1, ""


def format_hms(seconds: float) -> str:
    """Định dạng số giây thành chuỗi HH:MM:SS (giờ 2 chữ số dù video dài hơn 24h — không dùng
    MM:SS vì video có thể dài hơn 60 phút, dùng dạng đủ 3 phần để không mơ hồ khi đọc)."""
    total = max(0, round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class EventRecord:
    frame_idx: int
    video_time: str
    violation_datetime: str
    track_id: int
    vehicle_class: str
    violation_type: str
    lane_id: str | None
    plate_number: str | None
    evidence_path: str | None
    # Ghi ra events.csv/json kèm rõ ràng (xem _write_events) — Web UI (panel "Log vi phạm" lúc xem
    # lại) dùng cờ này để hiện dòng đã rút lại dạng gạch/tô xám thay vì ẩn hẳn, theo yêu cầu user.
    retracted: bool = False
    # Lý do rút lại tự động của HỆ THỐNG (mũ bảo hiểm: đa số phiếu đảo chiều; đèn đỏ: xác nhận rẽ
    # đúng hướng được luật cho phép) — điền vào cột "review_note" của cả `events.csv` lẫn
    # `vehicle_summary.csv` cùng với kết quả xem xét thủ công của CON NGƯỜI sau này (xem
    # `state.py::apply_manual_review`, mục 28 trong docs/nhat-ky-ky-thuat.md) — 2 nguồn khác nhau
    # nhưng gộp chung 1 cột hiển thị.
    review_reason: str | None = None


@dataclass
class _FlowBucket:
    class_counts: dict[str, int] = field(default_factory=dict)
    frame_count: int = 0
    wall_start: float | None = None
    wall_end: float | None = None
    congested: bool = False


class ViolationLogger:
    def __init__(
        self, video_name: str, video_id: str, base_dir: str | Path = "data/logs", flow_bucket_seconds: int = 60,
    ):
        self.video_name = video_name
        # video_id: mã ngắn từ meta.video_id (bắt buộc, xem config_schema.py) — mọi track_id GHI
        # RA FILE (events.csv/json, vehicle_summary.csv) đều ghép tiền tố này (vd "VD1_57") để
        # duy nhất tuyệt đối giữa các video, không chỉ trong phạm vi 1 video — xem
        # docs/nhat-ky-ky-thuat.md. CHỈ áp dụng lúc XUẤT (_write_*), mọi dict nội bộ bên dưới vẫn
        # dùng track_id thô làm khoá (không đổi gì, tránh rủi ro đụng vào logic tracker/matching).
        self.video_id = video_id
        self.dir = Path(base_dir) / video_name
        self.events: list[EventRecord] = []
        self._vehicle_class: dict[int, str] = {}
        self._vehicle_first_frame: dict[int, int] = {}
        self._vehicle_last_frame: dict[int, int] = {}
        self._vehicle_first_ts: dict[int, float] = {}
        self._vehicle_last_ts: dict[int, float] = {}
        # Làn xe đang ở lúc track được thấy LẦN ĐẦU — chỉ ghi 1 lần lúc đó, không cập nhật lại
        # dù xe đi qua nhiều làn sau này (khác `stats.lane_counts`, vốn đếm MỌI làn xe từng ghé
        # qua — 2 mục đích khác nhau, không hợp nhất). None nếu xe không nằm trong lane nào lúc
        # xuất hiện (vd chưa vào tới vùng có vẽ lane).
        self._vehicle_entry_lane: dict[int, str] = {}
        # Log "nhịp đập lưu lượng" theo khung thời gian cố định — xem _write_traffic_flow().
        self.flow_bucket_seconds = flow_bucket_seconds
        self._flow_buckets: dict[int, _FlowBucket] = {}
        # Lịch sử trạng thái đèn tín hiệu theo thời gian — chỉ ghi 1 dòng MỖI KHI 1 đèn ĐỔI trạng
        # thái (không phải mỗi frame, giữ file nhỏ) — dùng để Web UI tra lại "đèn nào đang màu gì"
        # lúc TUA LẠI video đã lưu (trước đây chỉ có state HIỆN TẠI qua StatsTracker, không có
        # lịch sử nên panel ẩn hẳn lúc xem lại — xem register_traffic_light_frame()).
        self._tl_last_states: dict[str, str] = {}
        self._tl_transitions: list[dict] = []

    def log_event(
        self, frame_idx: int, timestamp_sec: float, violation_datetime: str, track_id: int, vehicle_class: str,
        violation_type: str, lane_id: str | None = None, plate_number: str | None = None,
        evidence_path: str | None = None,
    ) -> None:
        """`timestamp_sec`: giây kể từ đầu video — CHỈ dùng để tính `video_time` (HH:MM:SS) lưu
        lại, bản thân số giây thô không được lưu (Web UI không cần vì đã có `frame_idx` đủ để tự
        tính lại số giây khi cần so sánh/sắp xếp — xem `docs/nhat-ky-ky-thuat.md`).
        `violation_datetime`: ngày giờ THẬT của vi phạm (ISO 8601), tính từ
        `meta.recording_started_at` trong config (nếu có) hoặc thời gian sửa đổi file video —
        xem `Pipeline._resolve_base_datetime()`. `vehicle_class`: loại phương tiện (car/
        motorcycle/...) — lấy trực tiếp từ `Track.cls_name` lúc vi phạm xảy ra, phục vụ đúng yêu
        cầu báo cáo "bảng thống kê vi phạm theo loại xe" mà không cần JOIN qua vehicle_summary.csv."""
        self.events.append(EventRecord(
            frame_idx=frame_idx, video_time=format_hms(timestamp_sec), violation_datetime=violation_datetime,
            track_id=track_id, vehicle_class=vehicle_class, violation_type=violation_type, lane_id=lane_id,
            plate_number=plate_number, evidence_path=evidence_path,
        ))

    def log_retraction(self, track_id: int, violation_type: str, reason: str) -> None:
        """Đánh dấu event GẦN NHẤT cùng track_id+violation_type là đã bị rút lại TỰ ĐỘNG bởi HỆ
        THỐNG (khác xem xét THỦ CÔNG của con người, xem `state.py::apply_manual_review`) — KHÔNG
        xoá dòng đã ghi (giữ làm dấu vết đã từng xảy ra + đã được sửa, đúng nguyên tắc audit
        trail). `reason`: mô tả ngắn gọn lý do rút lại, hiện ra trong cột "review_note" của
        vehicle_summary.csv (xem `resolve_violation_flag`/`_write_vehicle_summary`)."""
        for e in reversed(self.events):
            if e.track_id == track_id and e.violation_type == violation_type and not e.retracted:
                e.retracted = True
                e.review_reason = reason
                return

    def register_vehicle(
        self, track_id: int, cls_name: str, frame_idx: int, timestamp_sec: float, lane_id: str | None = None,
    ) -> None:
        if track_id not in self._vehicle_first_frame:
            self._vehicle_first_frame[track_id] = frame_idx
            self._vehicle_first_ts[track_id] = timestamp_sec
            if lane_id:
                self._vehicle_entry_lane[track_id] = lane_id
            # Đếm xe vào ĐÚNG bucket lúc track được thấy LẦN ĐẦU (tái dùng logic "lần đầu" đã có
            # sẵn ở trên, không cần thêm 1 set theo dõi riêng cho traffic_flow.csv).
            bucket = self._flow_buckets.setdefault(self._bucket_index(timestamp_sec), _FlowBucket())
            bucket.class_counts[cls_name] = bucket.class_counts.get(cls_name, 0) + 1
        self._vehicle_class[track_id] = cls_name
        self._vehicle_last_frame[track_id] = frame_idx
        self._vehicle_last_ts[track_id] = timestamp_sec

    def register_flow_frame(self, timestamp_sec: float, wall_time: float, is_congested: bool) -> None:
        """Gọi 1 LẦN mỗi khung hình (không phải mỗi track) — cập nhật số khung hình xử lý +
        trạng thái ùn tắc cho đúng bucket thời gian hiện tại, dùng để tính `avg_fps`/
        `congestion_state` khi ghi `traffic_flow.csv`."""
        bucket = self._flow_buckets.setdefault(self._bucket_index(timestamp_sec), _FlowBucket())
        bucket.frame_count += 1
        if bucket.wall_start is None:
            bucket.wall_start = wall_time
        bucket.wall_end = wall_time
        bucket.congested = bucket.congested or is_congested

    def _bucket_index(self, timestamp_sec: float) -> int:
        return int(timestamp_sec // self.flow_bucket_seconds)

    def register_traffic_light_frame(self, timestamp_sec: float, states: dict[str, str]) -> None:
        """Gọi 1 lần mỗi khung hình (giống `register_flow_frame`) — chỉ THỰC SỰ ghi thêm 1 dòng
        vào lịch sử khi 1 đèn vừa đổi màu so với lần ghi trước, không phải mọi frame (đèn thường
        giữ nguyên màu hàng chục/trăm frame liên tục, ghi mỗi frame sẽ phí dung lượng vô ích)."""
        for light_id, state in states.items():
            if self._tl_last_states.get(light_id) == state:
                continue
            self._tl_last_states[light_id] = state
            self._tl_transitions.append({
                "light_id": light_id, "state": state, "at_sec": round(timestamp_sec, 2),
            })

    def finalize(
        self, stats: StatsTracker, plate_history: dict[int, str], fps_source: float, base_datetime: datetime,
    ) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._write_events()
        self._write_vehicle_summary(plate_history, base_datetime)
        self._write_traffic_statistics(stats, fps_source)
        self._write_traffic_flow(base_datetime)
        self._write_traffic_light_states()

    def _write_traffic_light_states(self) -> None:
        """`traffic_light_states.json` — danh sách các lần đổi màu (không phải mọi frame, xem
        `register_traffic_light_frame()`) — Web UI tra "đèn nào đang màu gì tại giây X" bằng cách
        tìm lần đổi màu GẦN NHẤT trước X cho từng đèn. Rỗng nếu video không có đèn tín hiệu nào."""
        (self.dir / "traffic_light_states.json").write_text(
            json.dumps(self._tl_transitions, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def _write_events(self) -> None:
        # GIỮ LẠI dòng đã bị rút lại (retracted) trong file xuất ra — kèm rõ 2 cột "retracted"/
        # "review_note" thể hiện trạng thái + lý do, KHÔNG còn ẩn đi như thiết kế trước (xem mục
        # 21 trong docs/nhat-ky-ky-thuat.md). Đổi theo yêu cầu user: Web UI (panel "Log vi phạm"
        # lúc xem lại) cần hiện được cả trường hợp hệ thống ghi nhận tạm rồi tự sửa (vd rẽ phải khi
        # đèn đỏ được xác nhận muộn là hợp lệ) — dòng đó vẫn hiện, chỉ gạch/tô xám, không biến mất
        # hoàn toàn khỏi dữ liệu như trước. Ảnh evidence trên đĩa (`data/evidence/`) vốn đã không
        # bị xoá dù rút lại — nay dữ liệu log cũng nhất quán, không còn khác nhau giữa 2 nơi.
        # Thứ tự cột CỐ Ý khác thứ tự field trong dataclass (dataclass ưu tiên nhóm theo mục đích
        # tạo/tính toán, thứ tự cột CSV/JSON ưu tiên theo mức độ người đọc quan tâm trước/sau).
        # track_id XUẤT RA ghép tiền tố video_id (vd "VD1_57") — duy nhất tuyệt đối giữa các video,
        # khác track_id THÔ dùng làm khoá nội bộ (self.events/dict) ở mọi nơi khác trong file này.
        rows = [{
            "track_id": f"{self.video_id}_{e.track_id}", "vehicle_class": e.vehicle_class,
            "violation_datetime": e.violation_datetime, "violation_type": e.violation_type,
            "lane_id": e.lane_id, "plate_number": e.plate_number,
            "frame_idx": e.frame_idx, "video_time": e.video_time, "evidence_path": e.evidence_path,
            "retracted": e.retracted, "review_note": e.review_reason or "",
        } for e in self.events]

        (self.dir / "events.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        with (self.dir / "events.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "track_id", "vehicle_class", "violation_datetime", "violation_type", "lane_id", "plate_number",
                "frame_idx", "video_time", "evidence_path", "retracted", "review_note",
            ])
            for r in rows:
                writer.writerow([
                    r["track_id"], r["vehicle_class"], r["violation_datetime"], r["violation_type"],
                    r["lane_id"] or "", r["plate_number"] or "", r["frame_idx"], r["video_time"],
                    r["evidence_path"] or "", r["retracted"], r["review_note"],
                ])

    def _write_vehicle_summary(self, plate_history: dict[int, str], base_datetime: datetime) -> None:
        # Trạng thái CUỐI CÙNG cho mỗi cặp (track_id, loại vi phạm) — không phải "có từng xảy ra".
        # Quan trọng vì 1 vi phạm có thể bị RÚT LẠI rồi XÁC NHẬN LẠI sau đó (vd no_helmet cho phép
        # tái xác nhận sau khi rút lại — xem docstring NoHelmetRule) — dùng bản ghi GẦN NHẤT mới
        # phản ánh đúng thực tế. self.events là list theo đúng thứ tự thời gian ghi nhận (append
        # dần trong vòng lặp xử lý), nên duyệt hết 1 lượt, dict tự động giữ lại bản ghi CUỐI CÙNG
        # của mỗi cặp (ghi đè bản trước nếu có).
        last_event_by_pair: dict[tuple[int, str], EventRecord] = {}
        for e in self.events:
            last_event_by_pair[(e.track_id, e.violation_type)] = e

        with (self.dir / "vehicle_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 5 cột riêng theo từng loại thay vì 1 cột chuỗi gộp — mở Excel SUM/PivotTable được
            # ngay. Giá trị: 1 = đang là vi phạm; 0 = chưa từng xảy ra; -1 = TỪNG bị ghi nhận
            # nhưng đã được RÚT LẠI (xem cột review_note giải thích lý do) — không dùng SUM(flags)
            # trực tiếp cho total_violations vì -1 sẽ bị trừ nhầm, xem bên dưới. Thứ tự lấy từ
            # evidence.ALL_VIOLATION_TYPES (nguồn chân lý duy nhất của 5 loại vi phạm hệ thống hỗ
            # trợ — xem evidence.py). 4 cột frame/thời điểm đặt CUỐI (ít quan trọng hơn khi đọc
            # nhanh so với xe nào/vi phạm gì).
            writer.writerow([
                "track_id", "class", "entry_lane", "plate_number",
                *ALL_VIOLATION_TYPES, "total_violations", "review_note",
                "first_frame", "last_frame", "first_seen_datetime", "last_seen_datetime",
            ])
            for tid in sorted(self._vehicle_class):
                first_dt = base_datetime + timedelta(seconds=self._vehicle_first_ts[tid])
                last_dt = base_datetime + timedelta(seconds=self._vehicle_last_ts[tid])

                # manual_review luôn None ở đây — Pipeline vừa chạy xong, chưa ai kịp xem xét thủ
                # công (việc đó xảy ra SAU, qua state.py::apply_manual_review tái tạo lại file
                # này). Dùng chung resolve_violation_flag() để công thức tính cờ nhất quán tuyệt
                # đối giữa 2 thời điểm ghi (lúc chạy xong vs lúc tái tạo sau xem xét).
                flags: list[int] = []
                reason_parts: list[str] = []
                for vtype in ALL_VIOLATION_TYPES:
                    e = last_event_by_pair.get((tid, vtype))
                    flag, note = resolve_violation_flag(
                        auto_present=e is not None,
                        auto_retracted=e.retracted if e is not None else False,
                        auto_reason=e.review_reason if e is not None else None,
                        manual_review=None,
                    )
                    flags.append(flag)
                    if note:
                        reason_parts.append(f"{vtype}: {note}")
                total_violations = sum(1 for f in flags if f == 1)
                review_note = "; ".join(reason_parts)

                # track_id XUẤT RA ghép tiền tố video_id — tra cứu nội bộ (self._vehicle_class...)
                # vẫn dùng tid THÔ làm khoá, không đổi gì ở đây.
                writer.writerow([
                    f"{self.video_id}_{tid}", self._vehicle_class[tid], self._vehicle_entry_lane.get(tid, ""),
                    plate_history.get(tid, ""), *flags, total_violations, review_note,
                    self._vehicle_first_frame[tid], self._vehicle_last_frame[tid],
                    first_dt.isoformat(), last_dt.isoformat(),
                ])

    def _write_traffic_statistics(self, stats: StatsTracker, fps_source: float) -> None:
        with (self.dir / "traffic_statistics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["total_vehicles", stats.total_vehicles])
            for cls, count in sorted(stats.class_counts.items()):
                writer.writerow([f"vehicle_count_{cls}", count])
            if stats.has_detection_zone:
                writer.writerow(["detection_zone_total", stats.detection_zone_total])
            if stats.has_analysis_zone:
                writer.writerow(["analysis_zone_total", stats.analysis_zone_total])
            for lane_id, count in sorted(stats.lane_counts.items()):
                writer.writerow([f"lane_count_{lane_id}", count])
            writer.writerow(["total_violations", stats.total_violations])
            for vtype, count in sorted(stats.violation_counts.items()):
                writer.writerow([f"violation_count_{vtype}", count])
            # congestion_peak_vehicle_count/congestion_was_ever_congested ĐÃ BỎ khỏi bảng tổng kết
            # 1 lần này — số liệu ùn tắc chi tiết THEO THỜI GIAN giờ nằm ở traffic_flow.csv
            # (cột congestion_state từng bucket), đầy đủ và hữu ích hơn 2 con số tĩnh này.
            writer.writerow(["fps_source", round(fps_source, 2)])

    def _write_traffic_flow(self, base_datetime: datetime) -> None:
        """Nhịp đập lưu lượng theo khung thời gian cố định (mặc định 60s/dòng, xem
        `flow_bucket_seconds`) — khác `events.csv` (theo sự kiện) và `vehicle_summary.csv` (theo
        phương tiện), dùng để vẽ biểu đồ đường: lưu lượng/vi phạm/hiệu năng biến động ra sao theo
        thời gian trong suốt video."""
        with (self.dir / "traffic_flow.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "bucket_start", *[f"count_{c}" for c in FLOW_VEHICLE_CLASSES], "avg_fps", "congestion_state",
            ])
            for idx in sorted(self._flow_buckets):
                b = self._flow_buckets[idx]
                bucket_dt = base_datetime + timedelta(seconds=idx * self.flow_bucket_seconds)
                wall_elapsed = (b.wall_end - b.wall_start) if (b.wall_start is not None and b.wall_end is not None) else 0.0
                avg_fps = round(b.frame_count / wall_elapsed, 2) if wall_elapsed > 0 else ""
                writer.writerow([
                    bucket_dt.isoformat(),
                    *[b.class_counts.get(c, 0) for c in FLOW_VEHICLE_CLASSES],
                    avg_fps, b.congested,
                ])
