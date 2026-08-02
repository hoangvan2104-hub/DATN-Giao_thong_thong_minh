"""Business logic cho Web UI — tách khỏi app.py (chỉ định tuyến HTTP), đúng quy ước Engine/Web đã
có từ đầu dự án (xem CLAUDE.md). Gồm: liệt kê camera/video, JobManager (chạy Pipeline nền theo
luồng riêng + snapshot/SSE cho xem trực tiếp), đọc/ghi config, đọc log lịch sử vi phạm, ảnh minh
chứng/thumbnail, cơ chế xem xét thủ công vi phạm, tổng hợp báo cáo.

Dựng lại từ đầu cùng đợt xoá `src/web/` cũ (2026-07-30) — không phải phục dựng nguyên văn code cũ,
nhưng tái dùng các API tầng engine đã kiểm chứng qua suốt dự án (Pipeline/ViolationLogger/
EvidenceWriter/config_schema...), không phát minh lại. `main.resolve_and_validate()` dùng chung
cho cả CLI (main.py) lẫn Web UI (đúng như docstring của hàm đó đã ghi từ trước).
"""
from __future__ import annotations

import csv
import json
import queue
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import imageio_ffmpeg
import numpy as np

from main import resolve_and_validate
from src.engine.config_schema import ConfigError, validate_config
from src.engine.evidence import ALL_VIOLATION_TYPES
from src.engine.logger import MANUAL_REVIEW_VERDICTS, resolve_violation_flag
from src.engine.pipeline import Pipeline
from src.utils.config_format import format_compact

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

INPUT_DIR = Path("data/input")
CONFIG_DIR = Path("config/videos")
OUTPUT_DIR = Path("data/output")
LOG_DIR = Path("data/logs")
EVIDENCE_DIR = Path("data/evidence")
THUMB_DIR = Path("data/thumbnails")
DEFAULT_MODEL = "models/pretrained/yolo11s.pt"
# Tên job cố định cho phiên xem trực tiếp webcam — không trùng bất kỳ tên video thật nào trong
# data/input/ (tiền tố "_" theo đúng quy ước "nội bộ/tạm" đã dùng xuyên suốt dự án, vd
# "_raw_<name>.mp4"), giúp `/api/status`/`/api/stream.mjpeg` phân biệt rõ đây không phải 1 video
# đã lưu — trang chi tiết camera (route theo tên) sẽ không bao giờ khớp tên này.
WEBCAM_JOB_NAME = "_webcam_live"

for _d in (OUTPUT_DIR, LOG_DIR, EVIDENCE_DIR, THUMB_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _webcam_config(video_id: str) -> dict:
    """Config tối giản cho phiên xem trực tiếp webcam — dựng thẳng trong bộ nhớ (không phải đọc
    từ `config/videos/`, vì webcam không có camera cố định để chuẩn bị trước zones/lines/đèn tín
    hiệu). Tắt hết 5 rule vi phạm (không có zones để đối chiếu, dù bật cũng không hoạt động có ý
    nghĩa gì) — chỉ còn phát hiện + theo dõi phương tiện thuần, đúng tinh thần "xem trực tiếp thử
    nghiệm", không phải giám sát 1 giao lộ đã cấu hình đầy đủ."""
    return {
        "meta": {"video_id": video_id, "video_name": f"webcam_{video_id}.mp4", "description": "Xem trực tiếp webcam"},
        "classes": ["car", "motorcycle", "bus", "truck", "bicycle"],
        "zones": [], "traffic_lights": [], "lines": [], "vectors": [],
        "system": {
            "features": {
                "detect_red_light": False, "detect_wrong_way": False, "detect_wrong_turn": False,
                "detect_wrong_lane": False, "detect_plate": False, "detect_helmet": False,
                "detect_congestion": False,
            },
        },
    }

# 2 verdict xem xét thủ công khiến vi phạm KHÔNG còn hiệu lực — dùng để quyết định "removed"/
# gạch-xám khi hiển thị (xem get_events/get_vehicle_summary). Khớp _MANUAL_VERDICTS_AS_RETRACTED
# trong logger.py (không import trực tiếp vì đó là biến private của module đó).
_MANUAL_RETRACT_VERDICTS = {"uu_tien_bo_qua", "loi_nhan_dien"}

# Mã lý do rút lại TỰ ĐỘNG (hệ thống) — CÁC CHUỖI NÀY là hằng số nội bộ dùng làm khoá match trong
# pipeline.py (không dấu, nối bằng "_" — không phải văn bản hiển thị cho người dùng cuối). Bảng
# dịch sang tiếng Việt có dấu, dễ đọc — user báo cột "Ghi chú xem xét" hiện nguyên mã (kiểu
# "he_thong_xac_nhan_..." dài, không khoảng trắng) vừa khó đọc vừa gây tràn ô do không có chỗ để
# ngắt dòng. CHỈ dịch ở tầng hiển thị (state.py) — không đổi giá trị GỐC lưu trong file log, tránh
# đụng vào logic parse/khớp chuỗi khác (`_parse_review_note`).
AUTO_REASON_LABELS = {
    "he_thong_xac_nhan_re_dung_huong_duoc_mien_tru_den_do":
        "Hệ thống xác nhận rẽ đúng hướng được miễn trừ vượt đèn đỏ",
    "he_thong_xac_nhan_huong_re_khong_bi_den_nay_quan_ly":
        "Hệ thống xác nhận hướng rẽ không bị đèn tín hiệu này quản lý",
    "he_thong_xac_nhan_loai_xe_nay_duoc_mien_tru_theo_huong_da_di":
        "Hệ thống xác nhận loại xe này được miễn trừ theo hướng đã đi",
    "he_thong_tu_dao_nguoc_phieu_bau_sau_khi_co_them_bang_chung":
        "Hệ thống tự đảo ngược kết luận sau khi có thêm bằng chứng",
}


def _translate_auto_reason(code: str | None) -> str | None:
    if not code:
        return None
    return AUTO_REASON_LABELS.get(code, code)


# ---------------------------------------------------------------------------
# Danh sách video/camera
# ---------------------------------------------------------------------------

def _video_path(name: str) -> Path | None:
    matches = sorted(INPUT_DIR.glob(f"{name}.*"))
    return matches[0] if matches else None


def list_videos() -> list[dict]:
    # CHỈ liệt kê camera có file video GỐC thật (data/input/) — dù đã có config/log/output đầy đủ
    # (vd "camera_test", thiếu file gốc vì lý do nào đó) vẫn KHÔNG hiện, theo đúng yêu cầu user:
    # không có video gốc thì không xem/xử lý lại được, hiện ra chỉ gây nhầm lẫn "camera chết".
    names: set[str] = set()
    for p in INPUT_DIR.glob("*"):
        # Bỏ qua file ẩn/placeholder (vd ".gitkeep" giữ thư mục rỗng trong git) — không phải
        # video thật, không nên hiện thành 1 "camera" rỗng vô nghĩa trên giao diện.
        if p.is_file() and not p.name.startswith("."):
            names.add(p.stem)

    result = []
    for name in sorted(names):
        config_path = CONFIG_DIR / f"{name}.json"
        video_id = None
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                video_id = cfg.get("meta", {}).get("video_id")
            except (json.JSONDecodeError, OSError):
                video_id = None
        input_path = _video_path(name)
        result.append({
            "name": name,
            "has_input": input_path is not None,
            # Tên file gốc THẬT (kèm đuôi thật — .mp4/.mov/...) — frontend cần giá trị này để dựng
            # đúng URL xem trước video gốc (`/data/input/<input_filename>`), không thể tự đoán
            # đuôi file chỉ từ `name`.
            "input_filename": input_path.name if input_path is not None else None,
            "has_config": config_path.exists(),
            "has_output": (OUTPUT_DIR / f"{name}.mp4").exists(),
            "has_log": (LOG_DIR / name / "events.json").exists(),
            "video_id": video_id,
            "status": job_manager.status_for(name),
        })
    return result


def delete_video(name: str) -> None:
    """Chỉ xoá video gốc (`data/input/`) — GIỮ config/log/evidence/output cũ (đúng nguyên tắc đã
    theo suốt dự án: không tự ý xoá dữ liệu người dùng ngoài phạm vi được yêu cầu)."""
    path = _video_path(name)
    if path is not None:
        path.unlink()


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")


def upload_video(name: str, suffix: str, content: bytes) -> str:
    """Chuẩn hoá tên file thành ký tự an toàn (chỉ chữ/số/'_'/'-') — tên gốc do người dùng chọn
    (tên file upload), dùng trực tiếp làm `name` embed vào HTML `onclick="...('${name}')"` ở
    frontend nên PHẢI an toàn tuyệt đối, không chỉ để tránh lỗi file path. Trả về tên đã chuẩn hoá
    để caller (app.py) trả lại đúng cho client (khác hẳn tên gốc nếu có ký tự bị thay thế)."""
    safe_name = _SAFE_NAME.sub("_", name).strip("_") or "video"
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    (INPUT_DIR / f"{safe_name}{suffix}").write_bytes(content)
    return safe_name


# Chặn tổng dung lượng tải qua URL — video "bình thường" của đồ án chỉ vài chục-vài trăm MB (xem
# CLAUDE.md: 30s-15 phút/video), 3GB đã rất dư dả; chặn sớm tránh 1 link trỏ tới file khổng lồ (cố
# ý hoặc nhầm) làm đầy ổ đĩa server.
MAX_URL_DOWNLOAD_BYTES = 3 * 1024 * 1024 * 1024
_VIDEO_EXT_BY_CONTENT_TYPE = {
    "video/mp4": ".mp4", "video/quicktime": ".mov", "video/x-msvideo": ".avi",
    "video/webm": ".webm", "video/x-matroska": ".mkv",
}


def upload_video_from_url(url: str, name: str | None) -> str:
    """Tải video từ 1 URL trực tiếp (link .mp4/.mov/... — KHÔNG phải link trang xem video kiểu
    YouTube/TikTok, cần trình tải riêng biệt ngoài phạm vi đồ án) về `data/input/`, dùng thư viện
    chuẩn `urllib` (không thêm dependency mới). Chỉ chấp nhận scheme http/https — chặn `file://`
    và tương tự để tránh server tự đọc file cục bộ qua đường vòng URL."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Chỉ hỗ trợ link http/https, không hỗ trợ '{parsed.scheme or url}'")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (DATN-traffic-system)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        url_suffix = Path(parsed.path).suffix.lower()
        suffix = url_suffix if url_suffix in _VIDEO_EXT_BY_CONTENT_TYPE.values() else (
            _VIDEO_EXT_BY_CONTENT_TYPE.get(content_type, url_suffix or ".mp4")
        )

        base_name = name.strip() if name and name.strip() else Path(parsed.path).stem or "video"
        safe_name = _SAFE_NAME.sub("_", base_name).strip("_") or "video"

        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        dest = INPUT_DIR / f"{safe_name}{suffix}"
        total = 0
        with dest.open("wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_URL_DOWNLOAD_BYTES:
                    f.close()
                    dest.unlink(missing_ok=True)
                    raise ValueError(
                        f"File vượt quá giới hạn {MAX_URL_DOWNLOAD_BYTES // (1024*1024*1024)}GB, đã huỷ tải."
                    )
                f.write(chunk)
        if total == 0:
            dest.unlink(missing_ok=True)
            raise ValueError("Không tải được nội dung nào từ link (file rỗng)")
    return safe_name


# ---------------------------------------------------------------------------
# Thumbnail
# ---------------------------------------------------------------------------

_thumbnail_failed: set[str] = set()


def get_thumbnail_path(name: str) -> Path:
    thumb_path = THUMB_DIR / f"{name}.jpg"
    if thumb_path.exists():
        return thumb_path
    if name in _thumbnail_failed:
        raise FileNotFoundError(f"Không tạo được thumbnail cho '{name}' (đã thử trước đó, video có thể hỏng)")
    video_path = _video_path(name)
    if video_path is None:
        raise FileNotFoundError(f"Không có video nguồn cho '{name}'")
    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(total * 0.1)))
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok:
            _thumbnail_failed.add(name)
            raise RuntimeError(f"Không đọc được khung hình nào từ '{name}'")
    finally:
        cap.release()
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(thumb_path), frame)
    return thumb_path


def invalidate_thumbnail(name: str) -> None:
    (THUMB_DIR / f"{name}.jpg").unlink(missing_ok=True)
    _thumbnail_failed.discard(name)


def get_video_meta(name: str) -> dict:
    """Độ phân giải/FPS/tổng số khung hình THẬT của 1 video — dùng cho Config Wizard (slider chọn
    frame nền để vẽ + quy đổi toạ độ canvas sang pixel gốc, xem `get_frame_jpeg`)."""
    video_path = _video_path(name)
    if video_path is None:
        raise FileNotFoundError(f"Không có video nguồn cho '{name}'")
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"width": width, "height": height, "fps": round(fps, 2), "total_frames": total_frames}


def get_frame_jpeg(name: str, frame_idx: int) -> bytes:
    """JPEG của 1 khung hình BẤT KỲ theo yêu cầu (không cache ra đĩa như thumbnail — Wizard cho
    phép user kéo slider tự do, cache theo frame_idx sẽ sinh vô số file rác). Độ phân giải GỐC
    (không hạ theo `max_dimension` — toạ độ config Wizard sinh ra luôn ở pixel gốc, khớp đúng quy
    ước config viết tay hiện có)."""
    video_path = _video_path(name)
    if video_path is None:
        raise FileNotFoundError(f"Không có video nguồn cho '{name}'")
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seek_frame = max(0, min(total - 1, frame_idx)) if total > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Không đọc được khung hình {frame_idx} từ '{name}'")
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError(f"Không encode được khung hình {frame_idx} thành JPEG")
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Config CRUD
# ---------------------------------------------------------------------------

def get_config(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Chưa có config cho '{name}'")
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_video_filename(name: str) -> str:
    path = _video_path(name)
    return path.name if path is not None else f"{name}.mp4"


def save_config(name: str, data: dict) -> list[str]:
    """Validate + ép `meta.video_name` khớp file thật + chặn `meta.video_id` trùng với config
    KHÁC (xem docs/nhat-ky-ky-thuat.md mục 22 — track ID toàn cục theo video, trùng video_id phá
    vỡ chính mục đích của cơ chế đó) + ghi ra đĩa theo định dạng gọn (`format_compact`). Trả về
    danh sách cảnh báo KHÔNG chặn (vd lane thiếu đèn tín hiệu) để caller hiển thị nếu muốn."""
    data.setdefault("meta", {})["video_name"] = _expected_video_filename(name)
    warnings = validate_config(data)

    video_id = data["meta"]["video_id"]
    for sibling in CONFIG_DIR.glob("*.json"):
        if sibling.stem == name:
            continue
        try:
            sibling_cfg = json.loads(sibling.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if sibling_cfg.get("meta", {}).get("video_id") == video_id:
            raise ConfigError(
                f"video_id '{video_id}' đã được dùng bởi config '{sibling.stem}.json' — "
                f"mỗi video cần 1 video_id DUY NHẤT, chọn mã khác."
            )

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / f"{name}.json").write_text(format_compact(data) + "\n", encoding="utf-8")
    return warnings


def delete_config(name: str) -> None:
    (CONFIG_DIR / f"{name}.json").unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Đọc log lịch sử (đã ghi xong bởi ViolationLogger) + cơ chế xem xét thủ công
# ---------------------------------------------------------------------------

def _review_log_path(name: str) -> Path:
    return LOG_DIR / name / "review_log.json"


def clear_review_log(name: str) -> None:
    """Gọi khi BẮT ĐẦU 1 job xử lý mới cho video này — xem xét thủ công của lần chạy TRƯỚC không
    còn ý nghĩa (track_id không đảm bảo khớp giữa 2 lần chạy khác nhau, model cache dùng chung
    giữa các video nên bộ đếm track_id không reset — nếu không dọn, "vi phạm bổ sung" cũ sẽ hiện
    lại sai lệch trên dữ liệu mới, bug thật đã gặp ở bản trước — xem CLAUDE.md)."""
    _review_log_path(name).unlink(missing_ok=True)


def _load_review_log(name: str) -> dict[tuple[str, str], dict]:
    path = _review_log_path(name)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {(r["track_id"], r["violation_type"]): r for r in raw}


def _save_review_log(name: str, overrides: dict[tuple[str, str], dict]) -> None:
    path = _review_log_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(list(overrides.values()), ensure_ascii=False, indent=2), encoding="utf-8",
    )


def _review_label(ov: dict) -> str:
    label = MANUAL_REVIEW_VERDICTS.get(ov["verdict"], ov["verdict"])
    note = (ov.get("note") or "").strip()
    return f"{label}: {note}" if note else label


def _parse_review_note(raw_note: str) -> dict[str, str]:
    """Tách cột `review_note` gộp (vd "red_light: ly_do_1; no_helmet: ly_do_2") thành dict theo
    từng loại vi phạm — an toàn vì lý do TỰ ĐỘNG (hệ thống rút lại) luôn là cụm từ cố định không
    chứa "; " (xem _RED_LIGHT_RETRACT_REASONS trong pipeline.py)."""
    result: dict[str, str] = {}
    for part in (raw_note or "").split("; "):
        if ": " in part:
            vtype, reason = part.split(": ", 1)
            if vtype in ALL_VIOLATION_TYPES:
                result[vtype] = reason
    return result


def get_vehicle_summary(name: str) -> list[dict]:
    """Đọc `vehicle_summary.csv` + áp đè kết quả xem xét thủ công (nếu có) lên đúng cặp
    (track_id, loại vi phạm) — KHÔNG ghi đè file gốc trên đĩa (đơn giản hoá có chủ đích so với
    thiết kế trước: overlay áp dụng tại thời điểm ĐỌC, review_log.json là nguồn duy nhất giữ lại
    quyết định của con người, tránh rủi ro ghi hỏng file log gốc). Dùng chung
    `resolve_violation_flag()` (đúng hàm ViolationLogger dùng lúc ghi lần đầu) để công thức tính
    cờ nhất quán tuyệt đối giữa 2 thời điểm."""
    path = LOG_DIR / name / "vehicle_summary.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    overrides = _load_review_log(name)
    for row in rows:
        auto_reasons = _parse_review_note(row.get("review_note", ""))
        reasons = []
        total = 0
        for vtype in ALL_VIOLATION_TYPES:
            raw_flag = row.get(vtype) or "0"
            auto_flag = int(raw_flag)
            ov = overrides.get((row["track_id"], vtype))
            manual_review = {"verdict": ov["verdict"], "note": ov.get("note")} if ov else None
            flag, note = resolve_violation_flag(
                auto_present=auto_flag != 0, auto_retracted=auto_flag == -1,
                auto_reason=_translate_auto_reason(auto_reasons.get(vtype)), manual_review=manual_review,
            )
            row[vtype] = str(flag)
            if flag == 1:
                total += 1
            if note:
                reasons.append(f"{vtype}: {note}")
        row["total_violations"] = str(total)
        row["review_note"] = "; ".join(reasons)
    return rows


def get_events(name: str) -> list[dict]:
    """Đọc `events.json` + đánh dấu `removed`/`review_note` theo xem xét thủ công.

    `events.json` giờ GIỮ LẠI cả dòng đã bị hệ thống tự rút lại (kèm sẵn `retracted`/`review_note`
    do `logger.py::_write_events()` ghi — xem docs/nhat-ky-ky-thuat.md) — mặc định (không có xem
    xét thủ công đè lên) dùng ĐÚNG trạng thái đó làm `removed`, để panel "Log vi phạm" lúc xem lại
    hiện được cả trường hợp hệ thống ghi nhận tạm rồi tự sửa (vd rẽ phải khi đèn đỏ được xác nhận
    muộn là hợp lệ), không chỉ lúc xem trực tiếp như thiết kế trước.

    KHÔNG tổng hợp dòng "ảo" cho vi phạm được BỔ SUNG thủ công (`verdict=bo_sung_thu_cong`) — theo
    yêu cầu user, loại này chỉ cần hiện trong "Log tất cả phương tiện" (qua cờ trong
    `get_vehicle_summary()`), không cần thêm vào "Log vi phạm"/"Hồ sơ vi phạm" (dùng chung hàm
    này) vì không có bằng chứng/ảnh minh chứng thật đi kèm như 1 sự kiện do hệ thống phát hiện."""
    path = LOG_DIR / name / "events.json"
    events = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    overrides = _load_review_log(name)
    for e in events:
        key = (e["track_id"], e["violation_type"])
        ov = overrides.get(key)
        if ov is not None:
            e["removed"] = ov["verdict"] in _MANUAL_RETRACT_VERDICTS
            e["review_note"] = _review_label(ov)
        else:
            e["removed"] = bool(e.get("retracted"))
            e["review_note"] = _translate_auto_reason(e.get("review_note"))
        e["manual_added"] = False
    # Moi nhat truoc (frame_idx giam dan) — nguoi xem theo doi log khong can cuon xuong duoi de
    # thay su kien vua xay ra, dung yeu cau user (khac ban dau sap theo thoi gian tang dan).
    events.sort(key=lambda e: e["frame_idx"], reverse=True)
    return events


def review_verdicts_static() -> list[dict]:
    """Toàn bộ 4 kết luận xem xét thủ công (KHÔNG phụ thuộc trạng thái hiện tại của 1 cặp cụ thể,
    khác `review_options()` bên dưới) — dùng cho nút "Chỉnh sửa" ở bảng "Log tất cả phương tiện":
    không gắn với 1 vi phạm cụ thể đã tồn tại nên không thể suy ra đúng 1-2 lựa chọn phù hợp như
    `review_options()`, để người xem tự chọn trong danh sách cố định."""
    return [{"value": k, "label": v} for k, v in MANUAL_REVIEW_VERDICTS.items()]


def review_options(name: str, track_id: str, violation_type: str) -> list[dict]:
    if violation_type not in ALL_VIOLATION_TYPES:
        raise ValueError(f"Loại vi phạm không hợp lệ: {violation_type}")
    row = next((r for r in get_vehicle_summary(name) if r["track_id"] == track_id), None)
    flag = int(row[violation_type]) if row and row.get(violation_type) not in (None, "") else 0
    if flag == 1:
        keys = ["uu_tien_bo_qua", "loi_nhan_dien"]
    elif flag == -1:
        keys = ["vi_pham_that", "uu_tien_bo_qua"]
    else:
        keys = ["bo_sung_thu_cong"]
    return [{"value": k, "label": MANUAL_REVIEW_VERDICTS[k]} for k in keys]


def submit_review(name: str, track_id: str, violation_type: str, verdict: str, note: str) -> None:
    if verdict not in MANUAL_REVIEW_VERDICTS:
        raise ValueError(f"Lựa chọn không hợp lệ: {verdict}")
    if violation_type not in ALL_VIOLATION_TYPES:
        raise ValueError(f"Loại vi phạm không hợp lệ: {violation_type}")
    overrides = _load_review_log(name)
    overrides[(track_id, violation_type)] = {
        "track_id": track_id, "violation_type": violation_type, "verdict": verdict,
        "note": note, "reviewed_at": datetime.now(VN_TZ).isoformat(),
    }
    _save_review_log(name, overrides)


def get_traffic_flow(name: str) -> list[dict]:
    path = LOG_DIR / name / "traffic_flow.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_traffic_light_states(name: str) -> list[dict]:
    path = LOG_DIR / name / "traffic_light_states.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def get_traffic_statistics(name: str) -> dict[str, str]:
    path = LOG_DIR / name / "traffic_statistics.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {row["metric"]: row["value"] for row in csv.DictReader(f)}


def list_evidence(name: str) -> dict[str, list[str]]:
    base = EVIDENCE_DIR / name
    result: dict[str, list[str]] = {}
    for vtype in ALL_VIOLATION_TYPES:
        d = base / vtype
        result[vtype] = sorted(p.name for p in d.glob("*.jpg")) if d.exists() else []
    return result


def list_recent_events(limit: int = 200) -> list[dict]:
    """Gộp vi phạm CÒN HIỆU LỰC (đã áp xem xét thủ công) từ MỌI video, mới nhất trước — dùng cho
    trang "Hồ sơ vi phạm" (toàn hệ thống) + luồng cảnh báo ở "Trang chủ". Quét lại toàn bộ
    `data/logs/` mỗi lần gọi — chấp nhận được với quy mô vài chục video của đồ án, không cần
    cache/index riêng."""
    all_events: list[dict] = []
    if not LOG_DIR.exists():
        return all_events
    for d in sorted(LOG_DIR.iterdir()):
        if not d.is_dir():
            continue
        for e in get_events(d.name):
            if e["removed"]:
                continue
            all_events.append({**e, "video_name": d.name})
    all_events.sort(key=lambda e: e.get("violation_datetime") or "", reverse=True)
    return all_events[:limit]


def get_report_overview() -> dict:
    """Tổng hợp số liệu MỌI video đã có log — dùng cho trang 'Báo cáo & Thống kê'. Đọc trực tiếp
    `traffic_statistics.csv` từng video (đã có sẵn số liệu tổng hợp cuối cùng của lần chạy đó) —
    KHÔNG áp xem xét thủ công vào đây (chỉ ảnh hưởng bảng chi tiết theo xe/sự kiện), vì mục đích
    trang này là bức tranh tổng quan nhanh, không phải hồ sơ pháp lý từng vi phạm."""
    total_vehicles = 0
    total_violations = 0
    violation_by_type: dict[str, int] = {}
    vehicle_by_class: dict[str, int] = {}
    videos: list[dict] = []
    if LOG_DIR.exists():
        for d in sorted(LOG_DIR.iterdir()):
            if not d.is_dir():
                continue
            stats = get_traffic_statistics(d.name)
            if not stats:
                continue
            tv = int(float(stats.get("total_vehicles") or 0))
            vv = int(float(stats.get("total_violations") or 0))
            total_vehicles += tv
            total_violations += vv
            for k, v in stats.items():
                if k.startswith("violation_count_"):
                    vtype = k[len("violation_count_"):]
                    violation_by_type[vtype] = violation_by_type.get(vtype, 0) + int(float(v or 0))
                elif k.startswith("vehicle_count_"):
                    cls = k[len("vehicle_count_"):]
                    vehicle_by_class[cls] = vehicle_by_class.get(cls, 0) + int(float(v or 0))
            videos.append({
                "name": d.name, "total_vehicles": tv, "total_violations": vv,
                "fps_source": float(stats.get("fps_source") or 0),
            })
    return {
        "total_vehicles": total_vehicles, "total_violations": total_violations,
        "violation_by_type": violation_by_type, "vehicle_by_class": vehicle_by_class,
        "videos": videos,
    }


# ---------------------------------------------------------------------------
# Lịch sử toàn cục APPEND-ONLY — trang 'Nhật ký hệ thống AI'. Khác hẳn
# `data/logs/<video>/*` (BỊ GHI ĐÈ mỗi lần video đó chạy lại, chỉ phản ánh lần chạy GẦN NHẤT) —
# 2 file dưới đây chỉ CỘNG THÊM dòng mới mỗi khi 1 lần xử lý hoàn tất, không bao giờ ghi đè, nên
# chạy lại 1 video nhiều lần KHÔNG làm mất lịch sử các lần chạy trước (theo đúng yêu cầu user).
# Đặt trực tiếp trong `data/logs/` (không phải thư mục con) — an toàn vì mọi nơi khác chỉ quét
# CÁC THƯ MỤC con của LOG_DIR (`if not d.is_dir(): continue`), tự động bỏ qua 2 file này.
# ---------------------------------------------------------------------------

GLOBAL_RUN_HISTORY_PATH = LOG_DIR / "_global_run_history.csv"
GLOBAL_VEHICLE_HISTORY_PATH = LOG_DIR / "_global_vehicle_history.csv"


def _append_run_history(name: str, video_id: str, result: dict) -> None:
    """Ghi 1 dòng cho MỖI LẦN xử lý xong (append) — tổng hợp nhanh 1 lần chạy, dùng cho bảng
    'Lịch sử xử lý' ở trang Nhật ký hệ thống AI."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not GLOBAL_RUN_HISTORY_PATH.exists()
    with GLOBAL_RUN_HISTORY_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow([
                "video_name", "video_id", "ran_at", "fps_source", "fps_processed",
                "total_vehicles", "total_violations",
            ])
        writer.writerow([
            name, video_id, datetime.now(VN_TZ).isoformat(),
            result.get("fps_source", ""), result.get("fps_processed", ""),
            result.get("vehicle_counts", {}).get("total", 0),
            sum(result.get("violations", {}).values()),
        ])


def _append_vehicle_history(name: str, video_id: str) -> None:
    """Đọc `vehicle_summary.csv` VỪA ghi xong (ngay lúc xử lý hoàn tất, trước khi ai kịp xem xét
    thủ công — đây là snapshot TẠI THỜI ĐIỂM CHẠY, không phản ánh chỉnh sửa sau này, đúng ý nghĩa
    'lịch sử') và ghi lại (append) từng dòng vào file lịch sử toàn cục — bảo tồn dữ liệu TỪNG XE
    của MỌI lần chạy, kể cả khi `vehicle_summary.csv` gốc bị ghi đè bởi lần chạy sau."""
    src = LOG_DIR / name / "vehicle_summary.csv"
    if not src.exists():
        return
    with src.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    ran_at = datetime.now(VN_TZ).isoformat()
    fieldnames = ["video_name", "video_id", "ran_at", *rows[0].keys()]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not GLOBAL_VEHICLE_HISTORY_PATH.exists()
    with GLOBAL_VEHICLE_HISTORY_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow({"video_name": name, "video_id": video_id, "ran_at": ran_at, **row})


def get_run_history() -> list[dict]:
    """Lịch sử MỌI lần xử lý đã hoàn tất (không chỉ lần gần nhất của mỗi video) — đọc từ file
    append-only, mới nhất trước."""
    if not GLOBAL_RUN_HISTORY_PATH.exists():
        return []
    with GLOBAL_RUN_HISTORY_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    runs = [{
        "name": r["video_name"], "ran_at": r["ran_at"],
        "fps_processed": float(r["fps_processed"]) if r.get("fps_processed") else None,
        "total_vehicles": int(r["total_vehicles"]) if r.get("total_vehicles") else 0,
        "total_violations": int(r["total_violations"]) if r.get("total_violations") else 0,
    } for r in rows]
    runs.sort(key=lambda r: r["ran_at"], reverse=True)
    return runs


def get_vehicle_history(limit: int = 2000) -> list[dict]:
    """Lịch sử TỪNG PHƯƠNG TIỆN qua MỌI lần chạy (không mất khi 1 video được xử lý lại) — dùng
    cho trang 'Nhật ký hệ thống AI'. `limit`: giới hạn số dòng MỚI NHẤT trả về (file lớn dần theo
    thời gian sử dụng thật)."""
    if not GLOBAL_VEHICLE_HISTORY_PATH.exists():
        return []
    with GLOBAL_VEHICLE_HISTORY_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: r.get("ran_at", ""), reverse=True)
    return rows[:limit]


def get_violations_over_time() -> dict:
    """Gộp vi phạm CÒN HIỆU LỰC từ MỌI camera theo `violation_datetime` (giờ Việt Nam thật) —
    dùng cho 2 biểu đồ 'theo ngày'/'theo giờ trong ngày' ở trang Báo cáo & Thống kê. Không giới
    hạn `limit` như `list_recent_events()` (trang đó chỉ cần vài trăm dòng mới nhất để hiển thị
    danh sách, còn thống kê cần TOÀN BỘ để không lệch số)."""
    by_day: dict[str, int] = {}
    by_hour: dict[str, int] = {}
    for e in list_recent_events(limit=1_000_000):
        dt = e.get("violation_datetime") or ""
        if len(dt) < 13:
            continue
        day, hour = dt[:10], dt[11:13]
        by_day[day] = by_day.get(day, 0) + 1
        by_hour[hour] = by_hour.get(hour, 0) + 1
    return {"by_day": by_day, "by_hour": by_hour}


def get_congestion_overview() -> dict:
    """Tổng hợp ùn tắc theo từng camera — đọc `traffic_flow.csv` (LUÔN có cột `congestion_state`
    dù tính năng ùn tắc video đó có bật hay không, xem `logger.py`) — CHỈ tính camera nào thực sự
    khai báo bật `system.features.detect_congestion` trong config, tránh hiểu nhầm cột luôn tồn
    tại thành 'đã bật cho mọi video'."""
    cameras: list[dict] = []
    if not LOG_DIR.exists():
        return {"cameras": cameras}
    for d in sorted(LOG_DIR.iterdir()):
        if not d.is_dir():
            continue
        try:
            cfg = get_config(d.name)
        except FileNotFoundError:
            continue
        if not cfg.get("system", {}).get("features", {}).get("detect_congestion", False):
            continue
        flow = get_traffic_flow(d.name)
        if not flow:
            continue
        total = len(flow)
        congested = sum(1 for row in flow if row.get("congestion_state") == "True")
        cameras.append({
            "name": d.name, "total_buckets": total, "congested_buckets": congested,
            "congested_ratio": round(congested / total, 3) if total else 0.0,
        })
    return {"cameras": cameras}


# ---------------------------------------------------------------------------
# Xuất bảng ra CSV/Excel
# ---------------------------------------------------------------------------

def export_table(rows: list[dict], fmt: str) -> tuple[bytes, str, str]:
    """Trả về (nội dung, content-type, đuôi file) — dùng chung cho xuất events/vehicle_summary."""
    import io

    if not rows:
        rows = []
    if fmt == "xlsx":
        import pandas as pd

        buf = io.BytesIO()
        pd.DataFrame(rows).to_excel(buf, index=False, engine="openpyxl")
        return buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"

    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return buf.getvalue().encode("utf-8-sig"), "text/csv", "csv"


# ---------------------------------------------------------------------------
# JobManager — chạy Pipeline nền + snapshot/SSE cho xem trực tiếp
# ---------------------------------------------------------------------------

def _reencode_for_browser(src_path: Path, dst_path: Path) -> None:
    """`cv2.VideoWriter` ghi codec `mp4v` (MPEG-4 Part 2) — hầu hết trình duyệt hiện đại KHÔNG
    giải mã được trong thẻ <video> HTML5 (chỉ hỗ trợ H.264/VP9/AV1, khác VLC/desktop player vẫn
    mở được). Re-encode bằng ffmpeg tĩnh (imageio-ffmpeg, không cần cài hệ thống) sang H.264 +
    `+faststart` (đọc được metadata ngay từ đầu file, tua nhanh không cần tải hết) — bug/fix này
    đã gặp và xác nhận ở bản Web UI trước, xem CLAUDE.md."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg_exe, "-y", "-i", str(src_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(dst_path),
        ],
        check=True, capture_output=True,
    )


@dataclass
class JobState:
    name: str
    stage: str = "processing"  # processing | encoding | done | error
    error: str | None = None
    result: dict | None = None
    started_at: float = field(default_factory=time.time)
    stats: dict = field(default_factory=dict)
    vehicles: dict[str, dict] = field(default_factory=dict)
    active_violations: set[tuple[str, str]] = field(default_factory=set)
    # Loại vi phạm TỪNG bị gắn cờ (dù hiện tại còn hiệu lực hay đã rút lại) — dùng để tính cờ
    # -1/0/1 cho bảng "Log tất cả phương tiện" hiện NGAY LÚC ĐANG XỬ LÝ (trước đây chỉ có ở bảng
    # xem lại sau khi xong, xem CLAUDE.md).
    ever_violations: dict[str, set[str]] = field(default_factory=dict)
    frame_idx: int = 0
    latest_jpeg: bytes | None = None
    # Tăng dần mỗi lần latest_jpeg đổi — cho phép generator MJPEG (xem JobManager.mjpeg_frames)
    # phát hiện "có khung hình mới" bằng so sánh số nguyên rẻ, không phải so sánh cả chuỗi byte
    # JPEG (có thể vài chục KB) mỗi lần polling nội bộ.
    jpeg_seq: int = 0


class JobManager:
    """1 job xử lý video tại 1 thời điểm (đúng ràng buộc "1 nguồn xử lý tại 1 thời điểm" của đồ
    án — xem CLAUDE.md). Model cache (singleton theo (model_path, device)) đã nằm sẵn ở tầng
    engine (tracker.py/plate_ocr.py/no_helmet.py) — JobManager chỉ cần tạo `Pipeline` mới mỗi
    job, không cần tự quản lý cache model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.job: JobState | None = None
        self._stop_flag: threading.Event | None = None
        self._subscribers: list[queue.Queue] = []
        self._last_snapshot_wall = 0.0

    def status_for(self, name: str) -> str:
        with self._lock:
            if self.job is not None and self.job.name == name:
                return self.job.stage
        return "idle"

    def status(self) -> dict | None:
        with self._lock:
            j = self.job
            if j is None:
                return None
            return {
                "name": j.name, "stage": j.stage, "error": j.error, "result": j.result,
                "stats": j.stats, "frame_idx": j.frame_idx,
                # Kèm sẵn danh sách phương tiện trong response — trang chi tiết camera lúc đang xử
                # lý trực tiếp cần hiện "Log tất cả phương tiện" NGAY (không đợi xử lý xong như
                # trước), dùng lại /api/status (đã poll sẵn mỗi 900ms) thay vì gọi thêm 1 API riêng.
                "vehicles": list(j.vehicles.values()),
            }

    def vehicles(self) -> list[dict]:
        with self._lock:
            if self.job is None:
                return []
            return list(self.job.vehicles.values())

    def start(self, name: str) -> None:
        with self._lock:
            if self.job is not None and self.job.stage in ("processing", "encoding"):
                raise RuntimeError(
                    f"Đang xử lý '{self.job.name}' — đợi xong hoặc dừng lại trước khi chạy video khác."
                )
            video_path, config = resolve_and_validate(name)
            self.job = JobState(name=name)
            self._stop_flag = threading.Event()
        clear_review_log(name)
        threading.Thread(
            target=self._run, args=(name, str(video_path), config, self._stop_flag), daemon=True,
        ).start()

    def start_webcam(self, camera_index: int = 0) -> str:
        """Xem TRỰC TIẾP webcam qua chính Pipeline (bbox/tracking thật, không phải mô phỏng) —
        khác hẳn 1 bản ghi thô đã bỏ trước đây (xem CLAUDE.md). Webcam chưa có config zone/lane
        riêng nên chỉ nhận diện + theo dõi thuần (mọi rule vi phạm tắt qua `_webcam_config` —
        không có zones/lines để rule nào hoạt động có ý nghĩa). Dùng chung khoá "1 job tại 1 thời
        điểm" với video file — tái dùng nguyên vẹn `/api/status`/`/api/stream.mjpeg`/
        `/api/vehicles/live` đã có, không cần endpoint đọc riêng cho webcam."""
        with self._lock:
            if self.job is not None and self.job.stage in ("processing", "encoding"):
                raise RuntimeError(
                    f"Đang xử lý '{self.job.name}' — đợi xong hoặc dừng lại trước khi xem webcam."
                )
            name = WEBCAM_JOB_NAME
            config = _webcam_config(video_id=f"CAM{camera_index}")
            self.job = JobState(name=name)
            self._stop_flag = threading.Event()
        threading.Thread(
            target=self._run_webcam, args=(name, camera_index, config, self._stop_flag), daemon=True,
        ).start()
        return name

    def stop(self) -> None:
        with self._lock:
            if self._stop_flag is not None:
                self._stop_flag.set()

    def _run_webcam(self, name: str, camera_index: int, config: dict, stop_flag: threading.Event) -> None:
        """Không ghi file output/log lịch sử (đúng tinh thần "xem trực tiếp", không phải ghi hình
        — webcam không có điểm kết thúc tự nhiên như file, chỉ dừng khi người dùng bấm Dừng)."""
        try:
            pipeline = Pipeline(config=config, model_path=DEFAULT_MODEL)
            result = pipeline.run(
                video_path=camera_index, output_path=None, show=False,
                log_name=None, frame_callback=self._make_callback(pipeline),
                stop_flag=stop_flag,
            )
            with self._lock:
                if self.job is not None and self.job.name == name:
                    self.job.stage = "done"
                    self.job.result = result
            self._broadcast({"type": "done", "name": name})
        except Exception as exc:  # noqa: BLE001 - báo lỗi cho UI, không để job treo im lặng
            with self._lock:
                if self.job is not None and self.job.name == name:
                    self.job.stage = "error"
                    self.job.error = str(exc)
            self._broadcast({"type": "error", "name": name, "error": str(exc)})

    def _run(self, name: str, video_path: str, config: dict, stop_flag: threading.Event) -> None:
        try:
            pipeline = Pipeline(config=config, model_path=DEFAULT_MODEL)
            raw_output = OUTPUT_DIR / f"_raw_{name}.mp4"
            final_output = OUTPUT_DIR / f"{name}.mp4"
            result = pipeline.run(
                video_path=video_path, output_path=str(raw_output), show=False,
                log_name=name, frame_callback=self._make_callback(pipeline),
                stop_flag=stop_flag,
            )
            with self._lock:
                if self.job is not None:
                    self.job.stage = "encoding"
            _reencode_for_browser(raw_output, final_output)
            raw_output.unlink(missing_ok=True)
            invalidate_thumbnail(name)
            video_id = config["meta"]["video_id"]
            _append_run_history(name, video_id, result)
            _append_vehicle_history(name, video_id)
            with self._lock:
                if self.job is not None:
                    self.job.stage = "done"
                    self.job.result = result
            self._broadcast({"type": "done", "name": name})
        except Exception as exc:  # noqa: BLE001 - báo lỗi cho UI, không để job treo im lặng
            with self._lock:
                if self.job is not None:
                    self.job.stage = "error"
                    self.job.error = str(exc)
            self._broadcast({"type": "error", "name": name, "message": str(exc)})

    def _make_callback(self, pipeline: Pipeline):
        video_id = pipeline.config["meta"]["video_id"]

        def callback(frame: np.ndarray, frame_idx: int, tracks: list) -> None:
            now = time.time()
            with self._lock:
                job = self.job
                if job is None:
                    return
                job.frame_idx = frame_idx
                job.stats = {
                    "total_vehicles": pipeline.stats.total_vehicles,
                    "class_counts": dict(pipeline.stats.class_counts),
                    "violation_counts": dict(pipeline.stats.violation_counts),
                    "total_violations": pipeline.stats.total_violations,
                    "traffic_light_states": dict(pipeline.stats.traffic_light_states),
                    "congestion_enabled": pipeline.stats.congestion_enabled,
                    "congestion_vehicle_count": pipeline.stats.congestion_vehicle_count,
                    "is_congested": pipeline.stats.is_congested,
                    "lane_counts": dict(pipeline.stats.lane_counts),
                }
                for t in tracks:
                    key = f"{video_id}_{t.track_id}"
                    is_new = key not in job.vehicles
                    active_vtypes = set(pipeline.violation_events.get(t.track_id, {}).keys())
                    ever = job.ever_violations.setdefault(key, set())
                    ever |= active_vtypes
                    flags = {
                        vtype: (1 if vtype in active_vtypes else (-1 if vtype in ever else 0))
                        for vtype in ALL_VIOLATION_TYPES
                    }
                    lane_id = job.vehicles.get(key, {}).get("entry_lane")
                    if lane_id is None and pipeline.zone_map is not None:
                        found = pipeline.zone_map.find_lane(t.bbox)
                        lane_id = found.id if found else None
                    job.vehicles[key] = {
                        "track_id": key, "class": t.cls_name,
                        "plate_number": pipeline.plate_numbers.get(t.track_id),
                        "first_frame": job.vehicles.get(key, {}).get("first_frame", frame_idx),
                        "entry_lane": lane_id, "flags": flags,
                    }
                    if is_new:
                        self._broadcast_locked({"type": "vehicle", "name": job.name, "vehicle": job.vehicles[key]})

                current_keys: set[tuple[str, str]] = set()
                # Track VẪN CÒN SỐNG trong pipeline.violation_events (dù giá trị dict con của nó
                # rỗng hay không) — phân biệt với track đã bị dọn TOÀN BỘ entry lúc rời khung hình
                # (tracker.last_removed_ids, xem pipeline.py) — 2 tình huống hoàn toàn khác nhau,
                # chỉ tình huống ĐẦU mới là "rút lại" thật (xem bug fix bên dưới).
                alive_prefixed = {f"{video_id}_{tid}" for tid in pipeline.violation_events}
                for tid, types in pipeline.violation_events.items():
                    prefix = f"{video_id}_{tid}"
                    for vtype in types:
                        vkey = (prefix, vtype)
                        current_keys.add(vkey)
                        if vkey not in job.active_violations:
                            self._broadcast_locked({
                                "type": "violation", "name": job.name,
                                "track_id": prefix, "violation_type": vtype,
                            })
                for track_id, vtype in job.active_violations - current_keys:
                    # BUG ĐÃ SỬA: trước đây COI MỌI khoá biến mất khỏi active_violations là "rút
                    # lại" — nhưng khoá biến mất vì xe RỜI KHUNG HÌNH (track kết thúc bình thường,
                    # tracker.last_removed_ids dọn TOÀN BỘ entry của track đó) trông giống hệt về
                    # mặt dữ liệu với khoá biến mất vì bị RÚT LẠI THẬT (RedLightRule/NoHelmetRule
                    # tự sửa, track vẫn còn sống) — gây gạch nhầm TOÀN BỘ vi phạm của mọi xe vừa
                    # rời khung hình trên giao diện (đã xác nhận qua ảnh chụp màn hình user gửi).
                    # Chỉ phát sự kiện "retraction" khi track ĐÓ vẫn còn sống.
                    if track_id in alive_prefixed:
                        self._broadcast_locked({
                            "type": "retraction", "name": job.name,
                            "track_id": track_id, "violation_type": vtype,
                        })
                job.active_violations = current_keys

                # Throttle encode snapshot xuống tối đa ~20 lần/giây (0.05s) — đủ mượt cho mắt
                # người xem trực tiếp (khác hẳn 0.3s trước đó, vốn chỉ đủ cho polling ảnh tĩnh
                # ~900ms/lần — nay dùng để nuôi luồng MJPEG thật qua mjpeg_frames() nên cần tần
                # suất cao hơn nhiều). Vẫn CÓ throttle (không encode MỌI khung hình xử lý được,
                # có thể tới 40+ fps) để không lãng phí CPU vượt quá mức mắt người phân biệt được.
                if now - self._last_snapshot_wall >= 0.05:
                    self._last_snapshot_wall = now
                    ok, buf = cv2.imencode(".jpg", frame)
                    if ok:
                        job.latest_jpeg = buf.tobytes()
                        job.jpeg_seq += 1

        return callback

    def snapshot(self) -> bytes | None:
        with self._lock:
            return self.job.latest_jpeg if self.job is not None else None

    def mjpeg_frames(self, name: str):
        """Generator cho luồng MJPEG thật (`multipart/x-mixed-replace`) — thay cho cách cũ
        frontend tự polling `/api/snapshot.jpg` mỗi 900ms (chỉ ~1 ảnh/giây, trông giật dù xử lý
        không hề chậm). Trình duyệt hiển thị `<img src="/api/stream.mjpeg?...">` sẽ tự động vẽ
        lại MỖI khi nhận được 1 phần multipart mới — không cần vòng lặp JS polling cho riêng ảnh
        nữa. Chỉ phát khung hình khi job HIỆN TẠI đúng tên `name` truyền vào (đề phòng camera
        khác bắt đầu chạy giữa chừng — dù hệ thống chỉ cho 1 job/lúc, tránh lẫn dữ liệu); dừng
        hẳn khi job không còn khớp tên hoặc đã kết thúc (client `<img>` sẽ tự ngắt kết nối, trang
        đã điều hướng đi nơi khác từ trước đó qua cơ chế poll `/api/status` sẵn có)."""
        last_seq = -1
        while True:
            with self._lock:
                job = self.job
                if job is None or job.name != name:
                    return
                if job.stage not in ("processing", "encoding"):
                    if job.jpeg_seq == last_seq:
                        return
                jpeg = job.latest_jpeg
                seq = job.jpeg_seq
            if jpeg is not None and seq != last_seq:
                last_seq = seq
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
            time.sleep(0.02)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _broadcast_locked(self, event: dict) -> None:
        """Gọi khi ĐÃ giữ self._lock (từ callback) — không lock lại (tránh deadlock)."""
        for q in self._subscribers:
            q.put(event)

    def _broadcast(self, event: dict) -> None:
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            q.put(event)


job_manager = JobManager()
