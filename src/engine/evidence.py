"""Ghi ảnh minh chứng vi phạm — bất đồng bộ (queue + thread riêng), không chặn vòng lặp xử lý
chính (xem CLAUDE.md mục "Quy ước code"). Lưu theo
data/evidence/<video_name>/<loai_vi_pham>/<ID_xe>_<loai_vi_pham>_<ngay_gio>.jpg — khung hình
truyền vào đã được chuẩn bị sẵn ở pipeline.py (ảnh SẠCH, chỉ 1 bbox cho đúng xe vi phạm, xem
`src/render/overlay.py::draw_evidence_box`), module này chỉ lo việc ghi đĩa bất đồng bộ.
"""
from __future__ import annotations

import queue
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Định dạng ngày giờ trong tên file — an toàn cho tên file (không dấu ":"/"/"), vẫn sắp xếp được
# theo thứ tự thời gian khi liệt kê thư mục.
_FILENAME_DT_FORMAT = "%Y%m%d_%H%M%S"

# Toàn bộ loại vi phạm hệ thống hỗ trợ — LUÔN tạo đủ thư mục cho CẢ 5 loại này với MỌI video, bất
# kể video/config đó có thật sự bật rule đó hay có phát sinh vi phạm loại đó hay không (vd video
# không cấu hình đèn tín hiệu vẫn có thư mục `red_light` rỗng) — giữ cấu trúc thư mục nhất quán
# giữa mọi video, tiện duyệt/so sánh hàng loạt khi làm báo cáo thay vì phải đoán "thư mục không
# tồn tại nghĩa là video không có loại vi phạm này hay chỉ đơn giản là chưa từng xảy ra".
ALL_VIOLATION_TYPES = ["red_light", "wrong_lane", "wrong_way", "wrong_turn", "no_helmet"]


class EvidenceWriter:
    def __init__(self, video_name: str, video_id: str, base_dir: str | Path = "data/evidence"):
        self.dir = Path(base_dir) / video_name
        # video_id: mã ngắn từ meta.video_id (bắt buộc, xem config_schema.py) — ghép vào tên file
        # thành ID DUY NHẤT TUYỆT ĐỐI giữa mọi video (vd "VD1_57"), không chỉ track_id thô (dễ
        # trùng giữa các video khác nhau — xem docs/nhat-ky-ky-thuat.md).
        self.video_id = video_id
        for vtype in ALL_VIOLATION_TYPES:
            (self.dir / vtype).mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def capture(
        self, track_id: int, violation_dt: datetime, violation_type: str, frame: np.ndarray,
    ) -> str:
        """Đưa khung hình (đã vẽ sẵn 1 bbox evidence, xem overlay.draw_evidence_box) vào hàng đợi
        ghi đĩa, trả về đường dẫn NGAY (không đợi ghi xong — việc ghi thật sự xảy ra bất đồng bộ
        ở thread riêng) để logger dùng làm tham chiếu. `violation_dt`: thời điểm vi phạm THEO
        NGÀY GIỜ THẬT (không phải giây/frame tính từ đầu video) — Pipeline tính từ
        `meta.recording_started_at` trong config (nếu có khai báo) hoặc thời gian sửa đổi file
        video làm mốc, cộng thêm số giây đã trôi qua trong video tới lúc vi phạm."""
        stamp = violation_dt.strftime(_FILENAME_DT_FORMAT)
        rel_path = f"{violation_type}/{self.video_id}_{track_id}_{violation_type}_{stamp}.jpg"
        self._queue.put((rel_path, frame.copy()))
        # Luôn dùng "/" (POSIX) trong đường dẫn ghi vào log — path này lưu vào CSV/JSON để Web
        # UI/báo cáo đọc lại sau, không nên phụ thuộc dấu phân tách của hệ điều hành lúc ghi.
        return (self.dir / rel_path).as_posix()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            rel_path, frame = item
            path = self.dir / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), frame)
            self._queue.task_done()

    def close(self) -> None:
        """Chờ ghi hết ảnh còn trong hàng đợi rồi dừng thread — gọi cuối `Pipeline.run()`."""
        self._queue.put(None)
        self._thread.join()
