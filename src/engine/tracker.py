"""Theo dõi phương tiện xuyên khung hình — dùng ByteTrack có sẵn của Ultralytics.

Quyết định kiến trúc (xem CLAUDE.md): dùng model.track() built-in thay vì tự viết
tracker IOU thuần Python (chậm, O(n^2), hệ thống cũ đã gặp vấn đề này).

Bản đơn giản hoá (2026-07-08, theo yêu cầu kiểm tra lại): dùng thẳng track_id gốc của
ByteTrack, không ánh xạ sang ID tuần tự/không lọc min_hits nữa.

Mọi tham số tinh chỉnh (tracker_params, nms_iou...) đọc từ block `system` trong config
JSON của từng video — KHÔNG có file YAML riêng để giữ đúng nguyên tắc "1 file config duy
nhất". File YAML mà Ultralytics yêu cầu được tự sinh tạm thời từ tracker_params khi khởi
tạo tracker (xem `_write_tracker_yaml`), không phải file lưu trong repo.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass

import cv2
import torch
import yaml
from ultralytics import YOLO

# Giá trị mặc định của Ultralytics bytetrack.yaml — override từng phần qua tracker_params.
DEFAULT_TRACKER_PARAMS = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
}


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]  # x1,y1,x2,y2 (pixel)
    cls_name: str
    conf: float


# Cache model YOLO đã load theo (model_path, device) — tránh nạp lại từ đĩa + đẩy lên GPU mỗi lần
# khởi tạo VehicleTracker mới. Quan trọng cho Web UI (src/web/): mỗi video xử lý cần 1
# VehicleTracker riêng (config/zones khác nhau), nhưng model_path THƯỜNG giống nhau giữa các
# video — nếu không cache, model bị nạp lại mỗi lần xử lý video mới, đúng nhược điểm hệ thống cũ
# (xem CLAUDE.md: "fix: load model 1 lần (singleton), không load lại mỗi request").
_MODEL_CACHE: dict[tuple[str, str], YOLO] = {}


def _load_yolo_cached(model_path: str, device: str) -> YOLO:
    key = (model_path, device)
    if key not in _MODEL_CACHE:
        model = YOLO(model_path)
        model.to(device)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


class VehicleTracker:
    def __init__(
        self,
        model_path: str = "yolo11s.pt",
        classes: list[str] | None = None,
        device: str | None = None,
        conf: float = 0.1,
        imgsz: int = 640,
        nms_iou: float = 0.7,
        tracker_params: dict | None = None,
        min_visible_conf: float = 0.3,
        bbox_smoothing_alpha: float = 0.5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _load_yolo_cached(model_path, self.device)
        # `model.track(..., persist=True)` giữ trạng thái tracker (ByteTrack) gắn liền với
        # `model.predictor` giữa các lần gọi — khi model được TÁI SỬ DỤNG từ cache cho 1 VIDEO
        # KHÁC (Web UI xử lý nhiều video liên tiếp), predictor cũ (kèm track ID/Kalman filter của
        # video TRƯỚC) vẫn còn nguyên. ĐÃ THỬ đặt `model.predictor = None` để buộc Ultralytics
        # khởi tạo lại "sạch" — nhưng đo trực tiếp phát hiện đây là quyết định SAI: predictor
        # được recreate qua đường này bị THIẾU detect nhiều xe thật xuất hiện sau này trong video
        # (kiểm chứng: 7 xe thật giảm còn 3 khi video là video thứ 2 trong phiên, dù raw
        # `model.predict()` trên cùng khung hình vẫn thấy đủ box — có gì đó trong việc Ultralytics
        # tự tạo lại predictor "từ đầu" qua `predictor=None` KHÔNG tương đương hoàn toàn với
        # predictor được tạo tự nhiên lúc `.track()` gọi lần đầu, dù không rõ chi tiết nội bộ).
        # KHÔNG reset gì cả mới là lựa chọn ĐÚNG: track ID chỉ tiếp tục tăng thay vì reset về 1
        # cho mỗi video mới (mỹ quan, không ảnh hưởng logic — track_id chỉ dùng làm khoá dict/tên
        # file, hoạt động đúng với BẤT KỲ số nguyên nào) — đã đo lại: giữ nguyên đủ 7/7 xe thật,
        # số liệu khớp hệt trường hợp video chạy độc lập (cold), chỉ khác ID bắt đầu cao hơn 1.
        # Nguy cơ 1 track cũ "hồi sinh" nhầm sang video mới (IoU khớp tình cờ với vị trí dự đoán
        # Kalman filter cũ) gần như không thể xảy ra thực tế vì khung hình đầu video MỚI thường
        # khác hẳn góc camera/nội dung video CŨ — chấp nhận rủi ro lý thuyết này để đổi lấy đúng
        # hành vi detect/track thật (nghiêm trọng hơn nhiều so với rủi ro cosmetic của ID).
        # conf thấp (khớp track_low_thresh mặc định của ByteTrack, ~0.1) để tracker được
        # thấy cả detection tin cậy thấp dùng cho việc nối track — nếu lọc sớm ở đây
        # (vd 0.35) sẽ làm mất khả năng "cứu" track qua các frame detection yếu, gây đổi ID
        # dù không hề bị che khuất. Việc lọc hiển thị theo conf xử lý riêng ở min_visible_conf.
        self.conf = conf
        self.imgsz = imgsz
        self.nms_iou = nms_iou
        self.class_names: dict[int, str] = self.model.names
        self.allowed_class_ids = self._resolve_class_ids(classes)

        merged_params = {**DEFAULT_TRACKER_PARAMS, **(tracker_params or {})}
        self.tracker_cfg = self._write_tracker_yaml(merged_params)
        # lost_buffer luôn tự khớp track_buffer thực tế đang dùng (không còn phải tự đồng bộ
        # tay 2 nơi như trước) — số frame vắng mặt liên tục trước khi coi 1 track mất hẳn và
        # dọn state.
        self.lost_buffer = int(merged_params["track_buffer"])

        # min_visible_conf: ngưỡng conf tối thiểu để 1 box được VẼ ra — track vẫn "sống"
        # trong bộ nhớ tracker dù conf tụt thấp 1 vài frame (ID không đổi khi hồi phục),
        # chỉ tạm ẩn box để tránh vẽ box nhiễu/trôi lệch (bóng xe, 1 phần xe...).
        self.min_visible_conf = min_visible_conf
        self._missing_count: dict[int, int] = {}
        self._active_ids: set[int] = set()
        self.last_removed_ids: set[int] = set()

        # Làm mượt bbox theo thời gian (EMA) — box detect thô dao động nhẹ về vị trí/kích
        # thước giữa các frame liên tiếp dù xe di chuyển đều, gây cảm giác "giật" khi xem
        # (khác với video bị giật thật — đây là bbox không mượt, không phải encode/FPS).
        # alpha thấp = mượt hơn nhưng trễ theo chuyển động thật hơn; 1.0 = tắt hẳn làm mượt.
        self.bbox_smoothing_alpha = bbox_smoothing_alpha
        self._smoothed_bbox: dict[int, tuple[float, float, float, float]] = {}

    @staticmethod
    def _write_tracker_yaml(params: dict) -> str:
        """Ultralytics chỉ nhận cấu hình tracker qua đường dẫn file YAML — sinh file tạm
        (không lưu trong repo) từ tracker_params để tránh phải duy trì 1 file YAML riêng."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
        yaml.safe_dump(params, tmp)
        tmp.close()
        return tmp.name

    def _resolve_class_ids(self, classes: list[str] | None) -> list[int] | None:
        if not classes:
            return None
        name_to_id = {v: k for k, v in self.class_names.items()}
        missing = [c for c in classes if c not in name_to_id]
        if missing:
            raise ValueError(f"Model không có class: {missing}. Class hỗ trợ: {sorted(name_to_id)}")
        return [name_to_id[c] for c in classes]

    def update(self, frame, roi_mask=None) -> list[Track]:
        """Trả về danh sách track hiện tại (track_id = ID gốc của ByteTrack).
        `self.last_removed_ids` chứa các track_id vừa bị tracker xoá ở lần gọi này —
        dùng để dọn state ở rules/logger, tránh rò rỉ bộ nhớ như hệ thống cũ.

        `roi_mask`: mask nhị phân (0/255, cùng kích thước frame) — nếu có, chỉ nhận diện
        trong vùng mask=255 (zone type=road role=detection). None = nhận diện cả khung hình.
        """
        if roi_mask is not None:
            frame = cv2.bitwise_and(frame, frame, mask=roi_mask)

        result = self.model.track(
            frame,
            persist=True,
            device=self.device,
            conf=self.conf,
            iou=self.nms_iou,
            imgsz=self.imgsz,
            classes=self.allowed_class_ids,
            tracker=self.tracker_cfg,
            # agnostic_nms=True: gộp box chồng lấn bất kể nhãn lớp. Bắt buộc phải bật khi
            # conf thấp (0.1) — nếu không, 1 xe có thể sinh 2 box gần trùng vị trí nhưng khác
            # lớp (vd "car" và "truck") không bị NMS gộp (NMS mặc định chỉ so trong cùng lớp).
            agnostic_nms=True,
            verbose=False,
            # LƯU Ý: đã thử FP16 (quantize=16) cho model chính này — nhanh hơn thật (vd vid_test
            # 29 FPS thay vì ~25, vượt real-time), số vi phạm/số lần track chồng lấp không đổi,
            # NHƯNG đo sâu hơn trên bienso0.mp4 phát hiện quantize=16 làm lệch NHẸ track ID (sai
            # số làm tròn fp16 đủ để đổi quyết định NMS/spawn track ở vài khung hình biên) —
            # lệch này KHÔNG đổi số vi phạm/tổng xe đếm được, nhưng GIÁN TIẾP làm biển số 1 xe bị
            # bỏ lỡ đọc (track "khác" nhận throttle timing khác, không đọc trúng khung hình rõ
            # nhất) — xem docs/nhat-ky-ky-thuat.md mục tối ưu FPS. Ưu tiên khớp TUYỆT ĐỐI baseline
            # đã dùng xuyên suốt dự án hơn tốc độ — giữ FP32 (quantize=None).
            quantize=None,
        )[0]

        tracks: list[Track] = []
        seen_raw_ids: set[int] = set()
        if result.boxes is not None and result.boxes.id is not None:
            for box in result.boxes:
                raw_id = int(box.id[0])
                conf = float(box.conf[0])
                seen_raw_ids.add(raw_id)
                self._missing_count[raw_id] = 0
                self._active_ids.add(raw_id)

                if conf < self.min_visible_conf:
                    continue  # box yếu/trôi lệch — track vẫn sống, tạm không vẽ frame này

                raw_bbox = tuple(box.xyxy[0].tolist())
                prev_bbox = self._smoothed_bbox.get(raw_id)
                if prev_bbox is None:
                    smoothed_bbox = raw_bbox
                else:
                    a = self.bbox_smoothing_alpha
                    smoothed_bbox = tuple(a * r + (1 - a) * p for r, p in zip(raw_bbox, prev_bbox))
                self._smoothed_bbox[raw_id] = smoothed_bbox

                cls_id = int(box.cls[0])
                tracks.append(Track(
                    track_id=raw_id,
                    bbox=smoothed_bbox,
                    cls_name=self.class_names[cls_id],
                    conf=conf,
                ))

        removed_ids: set[int] = set()
        for raw_id in list(self._active_ids):
            if raw_id in seen_raw_ids:
                continue
            self._missing_count[raw_id] = self._missing_count.get(raw_id, 0) + 1
            if self._missing_count[raw_id] >= self.lost_buffer:
                removed_ids.add(raw_id)
                self._active_ids.discard(raw_id)
                del self._missing_count[raw_id]
                self._smoothed_bbox.pop(raw_id, None)

        self.last_removed_ids = removed_ids
        return tracks
