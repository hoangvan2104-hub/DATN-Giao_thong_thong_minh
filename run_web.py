"""Chạy Web UI: python run_web.py rồi mở http://localhost:8000"""
import asyncio
import sys

import uvicorn

if __name__ == "__main__":
    if sys.platform == "win32":
        # ProactorEventLoop (mặc định trên Windows) in traceback vô hại
        # (ConnectionResetError WinError 10054) mỗi khi trình duyệt đóng kết nối HTTP nửa chừng
        # (vd tua video <video> huỷ request đang tải dở) - SelectorEventLoop xử lý êm hơn. An
        # toàn đổi vì subprocess ffmpeg (state.py::_reencode_for_browser) dùng subprocess.run
        # đồng bộ, không phụ thuộc event loop policy.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run("src.web.app:app", host="0.0.0.0", port=8000, reload=False)
