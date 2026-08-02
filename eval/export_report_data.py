"""Gộp dữ liệu log của MỌI video đã xử lý thành vài file tổng hợp — phục vụ viết báo cáo (rubric
yêu cầu "bảng thống kê vi phạm theo thời gian / loại xe / biển số"). Đây là script XUẤT 1 LẦN KHI
CẦN, KHÔNG phải tính năng chạy nền của hệ thống — đã cân nhắc & quyết định KHÔNG xây `global/` log
runtime liên tục theo đúng đặc tả `docs/thiet_ke_du_lieu_log.md` (tốn công sửa logger.py/evidence.py
+ mọi endpoint đọc log + frontend, trong khi nhu cầu thật chỉ là "có bảng Excel tổng hợp lúc viết
báo cáo" — rẻ hơn nhiều khi làm dạng script xuất theo yêu cầu).

Ghi ra `data/report_export/` — CỐ Ý KHÔNG ghi vào `data/logs/` để tránh bị `state.py::
list_recent_events()` (quét `data/logs/*/events.json`) đọc nhầm file tổng hợp này thành 1 "video"
riêng, gây đếm trùng lặp mọi vi phạm trên trang Sự kiện.

Dùng:
    python eval/export_report_data.py
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Console Windows mặc định dùng codepage cp1252 (không phải UTF-8) khi chạy qua 1 số trình bao
# (vd git-bash) — in tiếng Việt có dấu trực tiếp qua print() sẽ crash UnicodeEncodeError dù ghi
# file (đã dùng encoding="utf-8" tường minh) vẫn đúng. Ép stdout/stderr sang UTF-8 để chạy được
# nhất quán trên mọi terminal, không phụ thuộc codepage hệ thống.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

LOGS_DIR = Path("data/logs")
OUT_DIR = Path("data/report_export")
VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# track_id trong events.csv/vehicle_summary.csv nguồn giờ đã là chuỗi ghép "<video_id>_<số>" (duy
# nhất tuyệt đối giữa mọi video, xem logger.py/config_schema.py) — source_id (tên thư mục log) vẫn
# giữ lại riêng vì vài bảng khác (STATS_FIELDS/FLOW_FIELDS) không có track_id để tự suy ra video.
# "export_run_at": mỗi lần chạy script này GHI NỐI TIẾP (không ghi đè) vào file cũ — script chạy
# nhiều lần theo thời gian khi có video mới/xử lý lại (dữ liệu nguồn data/logs/*/ bị GHI ĐÈ mỗi
# lần 1 video được xử lý lại, xem logger.py — nên các lần export TRƯỚC là snapshot lịch sử duy
# nhất còn giữ được, không thể tái tạo lại nếu ghi đè mất) — cột này đánh dấu rõ dòng nào thuộc
# lần export nào để dễ lọc/so sánh khi viết báo cáo.
EVENTS_FIELDS = [
    "export_run_at", "violation_datetime", "track_id", "vehicle_class", "violation_type", "lane_id",
    "plate_number", "source_id", "video_time", "frame_idx", "evidence_path",
]
VEHICLE_FIELDS = [
    "export_run_at", "source_id", "track_id", "class", "entry_lane", "plate_number",
    "red_light", "wrong_lane", "wrong_way", "wrong_turn", "no_helmet", "total_violations",
    "review_note", "first_frame", "last_frame", "first_seen_datetime", "last_seen_datetime",
]
STATS_FIELDS = ["export_run_at", "source_id", "metric", "value"]
SUMMARY_FIELDS = ["export_run_at", "metric", "total_across_all_videos"]
FLOW_FIELDS = [
    "export_run_at", "source_id", "bucket_start", "count_car", "count_motorcycle", "count_bus", "count_truck",
    "count_bicycle", "avg_fps", "congestion_state",
]


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _append_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Ghi NỐI TIẾP vào file cũ (không ghi đè) — mỗi lần chạy script là 1 lần export mới, đánh dấu
    qua cột `export_run_at` (xem giải thích ở khai báo *_FIELDS). Đọc lại toàn bộ dòng cũ (nếu có)
    rồi ghi lại + nối thêm dòng mới — thay vì append thẳng ở mức hệ điều hành — để tự nâng cấp
    file tạo TRƯỚC khi có cột `export_run_at` này (schema cũ thiếu cột, append thẳng sẽ làm lệch
    cột giữa các dòng cũ/mới trong cùng 1 file)."""
    existing_rows: list[dict] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r.setdefault("export_run_at", "")
                existing_rows.append(r)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in existing_rows:
            writer.writerow(r)
        for r in rows:
            writer.writerow(r)


def main() -> None:
    if not LOGS_DIR.exists():
        print(f"Không tìm thấy {LOGS_DIR}/ — chưa có video nào được xử lý.")
        return
    video_dirs = sorted(p for p in LOGS_DIR.iterdir() if p.is_dir())
    if not video_dirs:
        print(f"{LOGS_DIR}/ rỗng — chưa có video nào được xử lý.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_at = datetime.now(VN_TZ).isoformat()

    all_events: list[dict] = []
    all_vehicles: list[dict] = []
    all_stats: list[dict] = []
    all_flow: list[dict] = []
    for vdir in video_dirs:
        for r in _read_csv_rows(vdir / "events.csv"):
            # events.csv giờ giữ lại cả dòng đã bị hệ thống RÚT LẠI (cột "retracted", xem
            # logger.py/docs/nhat-ky-ky-thuat.md — đổi để Web UI hiện được, trước đây bị loại
            # hẳn khỏi file) — báo cáo THỐNG KÊ CHÍNH THỨC chỉ tính vi phạm CÒN HIỆU LỰC, không
            # tính dòng đã rút lại (khớp đúng ý định ban đầu của báo cáo, không đổi số liệu).
            if r.get("retracted") == "True":
                continue
            all_events.append({"export_run_at": run_at, "source_id": vdir.name, **r})
        for r in _read_csv_rows(vdir / "vehicle_summary.csv"):
            all_vehicles.append({"export_run_at": run_at, "source_id": vdir.name, **r})
        for r in _read_csv_rows(vdir / "traffic_statistics.csv"):
            all_stats.append({"export_run_at": run_at, "source_id": vdir.name, **r})
        for r in _read_csv_rows(vdir / "traffic_flow.csv"):
            all_flow.append({"export_run_at": run_at, "source_id": vdir.name, **r})

    _append_csv(OUT_DIR / "global_events.csv", all_events, EVENTS_FIELDS)
    _append_csv(OUT_DIR / "global_vehicle_summary.csv", all_vehicles, VEHICLE_FIELDS)
    # Mỗi dòng là 1 chỉ số của 1 video, KHÔNG tự gộp theo lane — mã lane (L1, L2...) chỉ có ý
    # nghĩa TRONG PHẠM VI 1 video, "L1" của video A và video B là 2 làn hoàn toàn khác nhau, chỉ
    # trùng tên (xem mục 5.1 docs/thiet_ke_du_lieu_log.md) — cộng gộp sẽ cho số liệu vô nghĩa.
    _append_csv(OUT_DIR / "global_statistics_by_video.csv", all_stats, STATS_FIELDS)
    # Tương tự — bucket_start của video A và video B KHÔNG liên quan nhau (mỗi video tính từ mốc
    # riêng lúc bắt đầu xử lý, xem Pipeline._resolve_base_datetime), nên chỉ nối các dòng lại theo
    # đúng video, KHÔNG gộp/nội suy chung 1 trục thời gian giữa các video khác nhau.
    _append_csv(OUT_DIR / "global_traffic_flow.csv", all_flow, FLOW_FIELDS)

    # Bảng tổng hợp CHỈ cộng dồn những chỉ số dùng chung danh mục giữa mọi video (loại xe/loại vi
    # phạm/tổng số) — an toàn vì tên loại xe/loại vi phạm là danh mục cố định toàn hệ thống, khác
    # lane_count_*/detection_zone_total/... (quy mô vùng quan sát khác nhau giữa các video, cộng
    # dồn không có ý nghĩa thống nhất) nên KHÔNG đưa vào bảng này.
    totals: dict[str, float] = {}
    for r in all_stats:
        metric = r["metric"]
        if metric.startswith("vehicle_count_") or metric.startswith("violation_count_") \
                or metric in ("total_vehicles", "total_violations"):
            try:
                totals[metric] = totals.get(metric, 0) + float(r["value"])
            except ValueError:
                continue
    summary_rows = [{"export_run_at": run_at, "metric": "total_videos", "total_across_all_videos": len(video_dirs)}]
    for k, v in sorted(totals.items()):
        summary_rows.append({"export_run_at": run_at, "metric": k, "total_across_all_videos": int(v) if v == int(v) else v})
    _append_csv(OUT_DIR / "global_summary.csv", summary_rows, SUMMARY_FIELDS)

    print(f"Da them 1 lan export ({run_at}) cho {len(video_dirs)} video ({', '.join(p.name for p in video_dirs)}) vao {OUT_DIR}/ (noi tiep, khong ghi de lan truoc):")
    print(f"  - global_events.csv: {len(all_events)} dòng")
    print(f"  - global_vehicle_summary.csv: {len(all_vehicles)} dòng")
    print(f"  - global_statistics_by_video.csv: {len(all_stats)} dòng")
    print(f"  - global_traffic_flow.csv: {len(all_flow)} dòng")
    print(f"  - global_summary.csv: {len(summary_rows)} dòng (tổng quan toàn hệ thống)")


if __name__ == "__main__":
    main()
