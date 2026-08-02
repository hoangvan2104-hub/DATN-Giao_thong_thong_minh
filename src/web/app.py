"""FastAPI app cho Web UI. `index.html` (Header/Footer, xem CLAUDE.md) tự chứa — các route dưới
đây phục vụ phần THÂN TRANG mới: Trang chủ / Màn hình giám sát chung / Hồ sơ vi phạm / Báo cáo &
Thống kê / Nhật ký hệ thống AI. Route chỉ định tuyến + validate request — mọi logic nghiệp vụ nằm
ở `state.py` (đúng quy ước tách lớp của dự án).
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.engine.config_schema import ConfigError
from src.web import state

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = Path("data")

app = FastAPI(title="Hệ thống giám sát vi phạm giao thông")

DATA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.exception_handler(FileNotFoundError)
def _not_found(request: Request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ConfigError)
def _config_error(request: Request, exc: ConfigError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(ValueError)
def _value_error(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
def _runtime_error(request: Request, exc: RuntimeError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Video/camera
# ---------------------------------------------------------------------------

@app.get("/api/videos")
def api_list_videos() -> list[dict]:
    return state.list_videos()


@app.post("/api/upload")
async def api_upload(file: UploadFile) -> dict:
    name = Path(file.filename or "video").stem
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    content = await file.read()
    saved_name = state.upload_video(name, suffix, content)
    return {"name": saved_name}


@app.post("/api/upload-url")
async def api_upload_url(request: Request) -> dict:
    """Thêm video bằng link tải trực tiếp (vd .mp4 công khai) — KHÔNG hỗ trợ link trang xem video
    (YouTube/TikTok/Facebook...), cần trình tải chuyên dụng riêng ngoài phạm vi đồ án."""
    body = await request.json()
    url = (body.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=422, detail="Thiếu url")
    saved_name = state.upload_video_from_url(url, body.get("name"))
    return {"name": saved_name}


@app.delete("/api/videos/{name}")
def api_delete_video(name: str) -> dict:
    state.delete_video(name)
    return {"ok": True}


@app.post("/api/videos/{name}/process")
def api_start_processing(name: str) -> dict:
    state.job_manager.start(name)
    return {"ok": True}


@app.post("/api/videos/{name}/stop")
def api_stop_processing(name: str) -> dict:
    state.job_manager.stop()
    return {"ok": True}


@app.post("/api/webcam/start")
def api_webcam_start(camera_index: int = 0) -> dict:
    name = state.job_manager.start_webcam(camera_index)
    return {"name": name}


@app.post("/api/webcam/stop")
def api_webcam_stop() -> dict:
    state.job_manager.stop()
    return {"ok": True}


@app.get("/api/status")
def api_status() -> dict | None:
    return state.job_manager.status()


@app.get("/api/vehicles/live")
def api_vehicles_live() -> list[dict]:
    return state.job_manager.vehicles()


@app.get("/api/snapshot.jpg")
def api_snapshot() -> Response:
    jpeg = state.job_manager.snapshot()
    if jpeg is None:
        raise HTTPException(status_code=404, detail="Chưa có khung hình nào (chưa xử lý video nào)")
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/api/stream.mjpeg")
def api_stream_mjpeg(name: str) -> StreamingResponse:
    """Luồng MJPEG thật cho trang chi tiết camera lúc đang xử lý — xem docstring
    `JobManager.mjpeg_frames()`. Thay cho polling `/api/snapshot.jpg` mỗi 900ms trước đây (mượt
    hơn hẳn, ~20 khung/giây thay vì ~1 khung/giây)."""
    return StreamingResponse(
        state.job_manager.mjpeg_frames(name),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/events/stream")
def api_events_stream() -> StreamingResponse:
    q = state.job_manager.subscribe()

    def gen():
        try:
            while True:
                event = q.get()
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            state.job_manager.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/thumbnail/{name}.jpg")
def api_thumbnail(name: str) -> FileResponse:
    path = state.get_thumbnail_path(name)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/video-meta/{name}")
def api_video_meta(name: str) -> dict:
    """Độ phân giải/FPS/tổng số khung hình THẬT — dùng cho Config Wizard (slider chọn frame nền +
    quy đổi toạ độ canvas sang pixel gốc)."""
    return state.get_video_meta(name)


@app.get("/api/frame/{name}.jpg")
def api_frame(name: str, frame_idx: int = 0) -> Response:
    """Khung hình BẤT KỲ theo yêu cầu (không cache — khác `/api/thumbnail`) — dùng cho Config
    Wizard lúc user kéo slider chọn frame nền để vẽ."""
    try:
        jpeg_bytes = state.get_frame_jpeg(name, frame_idx)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(content=jpeg_bytes, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------------------
# Config CRUD
# ---------------------------------------------------------------------------

@app.get("/api/config/{name}")
def api_get_config(name: str) -> dict:
    return state.get_config(name)


@app.put("/api/config/{name}")
async def api_save_config(name: str, request: Request) -> dict:
    data = await request.json()
    warnings = state.save_config(name, data)
    return {"ok": True, "warnings": warnings}


@app.delete("/api/config/{name}")
def api_delete_config(name: str) -> dict:
    state.delete_config(name)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Log lịch sử (đã xử lý xong)
# ---------------------------------------------------------------------------

@app.get("/api/logs/{name}/events")
def api_log_events(name: str) -> list[dict]:
    return state.get_events(name)


@app.get("/api/logs/{name}/vehicle-summary")
def api_log_vehicle_summary(name: str) -> list[dict]:
    return state.get_vehicle_summary(name)


@app.get("/api/logs/{name}/traffic-flow")
def api_log_traffic_flow(name: str) -> list[dict]:
    return state.get_traffic_flow(name)


@app.get("/api/logs/{name}/traffic-lights")
def api_log_traffic_lights(name: str) -> list[dict]:
    return state.get_traffic_light_states(name)


@app.get("/api/logs/{name}/statistics")
def api_log_statistics(name: str) -> dict:
    return state.get_traffic_statistics(name)


@app.get("/api/logs/{name}/evidence")
def api_log_evidence(name: str) -> dict:
    return state.list_evidence(name)


@app.get("/api/logs/{name}/events-export")
def api_events_export(name: str, fmt: str = "csv") -> Response:
    rows = [{k: v for k, v in e.items() if k not in ("removed", "manual_added")} for e in state.get_events(name)]
    content, media_type, ext = state.export_table(rows, fmt)
    return Response(
        content=content, media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{name}_events.{ext}"'},
    )


@app.get("/api/logs/{name}/vehicles-export")
def api_vehicles_export(name: str, fmt: str = "csv") -> Response:
    content, media_type, ext = state.export_table(state.get_vehicle_summary(name), fmt)
    return Response(
        content=content, media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{name}_vehicle_summary.{ext}"'},
    )


# ---------------------------------------------------------------------------
# Xem xét thủ công vi phạm
# ---------------------------------------------------------------------------

@app.get("/api/review-verdicts")
def api_review_verdicts_static() -> list[dict]:
    return state.review_verdicts_static()


@app.get("/api/review-options")
def api_review_options(video: str, track_id: str, violation_type: str) -> list[dict]:
    return state.review_options(video, track_id, violation_type)


@app.post("/api/review")
async def api_submit_review(request: Request) -> dict:
    body = await request.json()
    state.submit_review(
        body["video"], body["track_id"], body["violation_type"], body["verdict"], body.get("note", ""),
    )
    return {"ok": True}


# ---------------------------------------------------------------------------
# Tổng hợp toàn hệ thống — Hồ sơ vi phạm / Trang chủ / Báo cáo / Nhật ký AI
# ---------------------------------------------------------------------------

@app.get("/api/events/recent")
def api_events_recent(limit: int = 200) -> list[dict]:
    return state.list_recent_events(limit)


@app.get("/api/reports/overview")
def api_reports_overview() -> dict:
    return state.get_report_overview()


@app.get("/api/reports/violations-over-time")
def api_violations_over_time() -> dict:
    return state.get_violations_over_time()


@app.get("/api/reports/congestion")
def api_congestion_overview() -> dict:
    return state.get_congestion_overview()


@app.get("/api/runs")
def api_runs() -> list[dict]:
    return state.get_run_history()


@app.get("/api/vehicle-history")
def api_vehicle_history(limit: int = 2000) -> list[dict]:
    return state.get_vehicle_history(limit)


@app.get("/api/vehicle-history/export")
def api_vehicle_history_export(fmt: str = "csv") -> Response:
    content, media_type, ext = state.export_table(state.get_vehicle_history(limit=1_000_000), fmt)
    return Response(
        content=content, media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="lich_su_phuong_tien.{ext}"'},
    )
