"""Phát hiện + đọc biển số xe — kế thừa 2 model YOLOv5 đã proven từ hệ thống cũ
(LP_detector.pt, LP_ocr.pt — gốc từ repo trungdinh22/License-Plate-Recognition).

Model YOLOv5 gốc (không phải Ultralytics YOLO11) — checkpoint không tương thích thẳng với
`ultralytics.YOLO()` (đã kiểm chứng: báo lỗi "NOT forwards compatible"). Phải tải qua
`torch.hub` trỏ vào `third_party/yolov5` (vendor sẵn trong repo — không phụ thuộc đường dẫn
máy khác) + tạm nới `weights_only` (PyTorch >=2.6 mặc định chặn unpickle class tuỳ biến của
YOLOv5 cũ; model này là của chính user, đáng tin cậy).

Khác biệt so với hệ thống cũ (nguyên nhân chính gây chậm trước đây): throttle nghiêm ngặt
theo track — không chạy detect+OCR mọi frame cho mọi xe, và dừng hẳn sau `max_attempts` lần
thử thất bại liên tiếp (hệ thống cũ retry vô hạn khi biển mờ).
"""
from __future__ import annotations

import logging
import math
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch

# Code yolov5 vendor dùng API torch.cuda.amp.autocast cũ (deprecated ở PyTorch mới nhưng vẫn
# hoạt động đúng) — im lặng cảnh báo này thay vì sửa code vendor (giữ nguyên để dễ đối chiếu
# với repo gốc).
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)

_YOLOV5_DIR = Path(__file__).resolve().parent.parent.parent / "third_party" / "yolov5"

# Kích thước tối thiểu (px) mỗi chiều để 1 box phát hiện được coi là biển số thật, không phải
# nhiễu — biển số cần ít nhất vài chục pixel mỗi chiều mới đủ chi tiết cho OCR ký tự.
MIN_PLATE_DIM = 15
# Kích thước tối thiểu (px) mỗi chiều của CHÍNH chiếc xe (không phải biển số) để còn đáng chạy
# detector biển số — xe nhỏ hơn mức này thì biển số bên trong chắc chắn dưới MIN_PLATE_DIM,
# bỏ qua ngay để tiết kiệm 1 lần gọi model (chi phí GPU thật) — cải thiện FPS cho video nhiều
# xe ở xa/nhỏ trong khung hình.
MIN_VEHICLE_DIM_FOR_PLATE = 40
# Tỉ lệ (0-1) phần TRÊN CÙNG của bbox xe bị bỏ qua khi tìm biển số — xem _crop_plate().
SEARCH_TOP_SKIP = 0.35


@contextmanager
def _trusted_torch_load():
    """PyTorch >=2.6 mặc định `weights_only=True`, chặn checkpoint YOLOv5 cũ (dùng class tuỳ
    biến `models.yolo.Model`). Chỉ nới tạm thời trong lúc load model tin cậy này, khôi phục
    ngay sau đó — không ảnh hưởng các `torch.load` khác trong hệ thống."""
    original = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = original


# Cache model YOLOv5 đã load theo (weights_path, device) — việc load qua torch.hub khá tốn kém
# (evict/khôi phục sys.modules, dọn logging handler...) nên càng đáng cache hơn các model
# Ultralytics thường. Quan trọng cho Web UI (src/web/): tránh nạp lại 2 model biển số mỗi lần xử
# lý 1 video mới trong cùng phiên server (xem CLAUDE.md: "load model 1 lần, không load lại mỗi
# request"). Model YOLOv5 ở đây chỉ dùng cho forward pass thuần (detect/OCR ký tự), KHÔNG giữ
# trạng thái track giữa các lần gọi như VehicleTracker — an toàn để dùng lại thẳng, không cần
# reset gì thêm.
_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _load_yolov5_model(weights_path: str, device: str):
    """Load model YOLOv5 qua torch.hub — né xung đột package `utils` giữa yolov5 vendor và
    `src/utils` của chính hệ thống bằng cách tạm evict khỏi sys.modules trong lúc load (giữ
    nguyên kỹ thuật đã proven từ hệ thống cũ)."""
    key = (weights_path, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    yolov5_dir = str(_YOLOV5_DIR)
    if not Path(yolov5_dir).exists():
        raise RuntimeError(f"Không tìm thấy {yolov5_dir} — cần vendor yolov5 vào third_party/")

    orig_sys_path = sys.path.copy()
    evict_prefixes = ("utils", "utils.", "tqdm", "tqdm.")
    evicted = {
        k: v for k, v in sys.modules.items()
        if k == "utils" or any(k.startswith(p) for p in evict_prefixes)
    }
    try:
        for k in list(evicted):
            sys.modules.pop(k, None)
        while yolov5_dir in sys.path:
            sys.path.remove(yolov5_dir)
        sys.path.insert(0, yolov5_dir)
        # Re-import utils.general (evict ở trên buộc import lại) sẽ chạy lại set_logging()
        # (module-level, utils/general.py:95), hàm này LUÔN gắn thêm 1 StreamHandler mới vào
        # ROOT logger (logging.getLogger(name=None)) mỗi lần — logging cache theo tên logger,
        # không theo module instance, nên các lần load model sau sẽ có N handler -> in log
        # trùng N lần. Dọn sạch TRƯỚC khi load để mỗi lần chỉ còn đúng 1 handler mới.
        logging.root.handlers.clear()
        with _trusted_torch_load():
            model = torch.hub.load(
                yolov5_dir, "custom", path=weights_path, source="local",
                force_reload=False, verbose=False,
            )
        model.to(device)
        model.eval()
        _MODEL_CACHE[key] = model
        return model
    finally:
        sys.path.clear()
        sys.path.extend(orig_sys_path)
        for k in [
            k for k in sys.modules
            if k == "utils" or any(k.startswith(p) for p in evict_prefixes)
        ]:
            sys.modules.pop(k, None)
        sys.modules.update(evicted)


# ─── Deskew + đọc ký tự (thuật toán từ repo trungdinh22/License-Plate-Recognition) ─────

def _change_contrast(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)


def _compute_skew(src_img: np.ndarray, center_thres: int) -> float:
    h, w = src_img.shape[:2]
    img = cv2.medianBlur(src_img, 3)
    edges = cv2.Canny(img, 30, 100, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, 30, minLineLength=w / 1.5, maxLineGap=h / 3.0)
    if lines is None:
        return 1.0
    # cv2.HoughLinesP trả shape (N,1,4) hoặc (N,4) tuỳ bản OpenCV — chuẩn hoá về (N,4) để
    # unpack trực tiếp (bản build đang dùng trả (N,4), code gốc giả định (N,1,4) nên unpack
    # sai gây "cannot unpack non-iterable" — đã kiểm chứng bằng repro trực tiếp).
    lines = lines.reshape(-1, 4)

    min_line, min_line_pos = 100, 0
    for i, (x1, y1, x2, y2) in enumerate(lines):
        cy = (y1 + y2) / 2
        if center_thres == 1 and cy < 7:
            continue
        if cy < min_line:
            min_line, min_line_pos = cy, i

    x1, y1, x2, y2 = lines[min_line_pos]
    ang = np.arctan2(y2 - y1, x2 - x1)
    if math.fabs(ang) > 30:
        return 0.0
    return ang * 180 / math.pi


def _deskew(src_img: np.ndarray, change_contrast: bool, center_thres: int) -> np.ndarray:
    img = _change_contrast(src_img) if change_contrast else src_img
    return _rotate_image(img, _compute_skew(img, center_thres))


def _linear_equation(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    if x2 == x1:
        return 0, y1
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1 if x1 != 0 else (y2 - y1) / (x2 - x1)
    return a, b


def _check_point_linear(x: float, y: float, x1: float, y1: float, x2: float, y2: float) -> bool:
    a, b = _linear_equation(x1, y1, x2, y2)
    return math.isclose(a * x + b, y, abs_tol=3)


def _parse_plate_from_detection_rows(rows: list[list], names: dict[int, str]) -> str:
    """Đọc chuỗi biển số từ danh sách detect ký tự thô `[x1,y1,x2,y2,conf,cls]` (đã tách khỏi
    `_read_plate_variants` để tái sử dụng chung cho cả đường batch lẫn đơn) — tự phân biệt biển 1
    dòng (ô tô) hay 2 dòng (xe máy VN) theo việc các ký tự có thẳng hàng với 2 điểm trái/phải
    nhất hay không."""
    if not rows or not (7 <= len(rows) <= 10):
        return ""  # biển số VN hợp lệ có 7-10 ký tự

    center_list = []
    y_sum = 0.0
    for x1, y1, x2, y2, _conf, cls in rows:
        x_c, y_c = (x1 + x2) / 2, (y1 + y2) / 2
        y_sum += y_c
        center_list.append([x_c, y_c, names[int(cls)]])

    l_point = min(center_list, key=lambda c: c[0])
    r_point = max(center_list, key=lambda c: c[0])

    is_two_lines = False
    if l_point[0] != r_point[0]:
        for c in center_list:
            if not _check_point_linear(c[0], c[1], l_point[0], l_point[1], r_point[0], r_point[1]):
                is_two_lines = True
                break

    if is_two_lines:
        y_mean = int(y_sum / len(rows))
        line1 = sorted((c for c in center_list if int(c[1]) <= y_mean), key=lambda c: c[0])
        line2 = sorted((c for c in center_list if int(c[1]) > y_mean), key=lambda c: c[0])
        return "".join(str(c[2]) for c in line1) + "-" + "".join(str(c[2]) for c in line2)
    return "".join(str(c[2]) for c in sorted(center_list, key=lambda c: c[0]))


def _read_plate_from_variants(model, imgs: list[np.ndarray]) -> str:
    """Đọc biển số từ NHIỀU biến thể deskew cùng lúc trong 1 lần forward pass (batch), thay vì
    gọi model tuần tự cho từng biến thể — YOLOv5 AutoShape hỗ trợ sẵn input dạng list (xem
    third_party/yolov5/models/common.py::AutoShape.forward). Mỗi lần gọi model có chi phí cố định
    (kernel launch + đồng bộ CUDA) không giảm theo kích thước ảnh nhỏ, nên gộp N lần gọi tuần tự
    thành 1 lần gọi batch giảm đáng kể chi phí này (đo thực tế: xem docs/nhat-ky-ky-thuat.md mục
    tối ưu FPS) — kết quả từng ảnh trong batch độc lập với ảnh khác (eval mode, không có
    batchnorm cập nhật running stats), không đổi độ chính xác so với gọi tuần tự từng ảnh.
    Trả về kết quả HỢP LỆ ĐẦU TIÊN theo đúng thứ tự ưu tiên `imgs` (giữ nguyên ngữ nghĩa "thử
    lần lượt, dừng ở lần đầu thành công" của vòng lặp cũ)."""
    results = model(imgs)
    names = model.names
    for i in range(len(imgs)):
        rows = results.xyxy[i].tolist()
        text = _parse_plate_from_detection_rows(rows, names)
        if text:
            return text
    return ""


class PlateReader:
    """Phát hiện vùng biển số trong bbox xe + đọc ký tự, throttle nghiêm ngặt theo track."""

    def __init__(
        self,
        detector_path: str = "models/pretrained/LP_detector.pt",
        ocr_path: str = "models/pretrained/LP_ocr.pt",
        device: str | None = None,
        throttle_frames: int = 10,
        max_attempts: int = 5,
        detect_conf: float = 0.5,
        ocr_conf: float = 0.6,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = _load_yolov5_model(detector_path, self.device)
        self.detector.conf = detect_conf
        self.ocr = _load_yolov5_model(ocr_path, self.device)
        self.ocr.conf = ocr_conf
        # LƯU Ý: đã thử bật AMP (fp16, `.amp = True`) cho cả detector lẫn ocr — đo trực tiếp phát
        # hiện gây SAI ký tự thật (regression: "29G1-18651" đọc nhầm thành "296G1-18651", "29L5-
        # 56817" biến mất) trên bienso0.mp4 dù model biển số chính (vehicle tracker) và model mũ
        # bảo hiểm dùng FP16 không hề ảnh hưởng độ chính xác. OCR ký tự nhỏ nhạy cảm hơn nhiều với
        # sai số làm tròn fp16 so với detect object thông thường — ĐÃ BỎ AMP ở đây, giữ fp32 cho
        # cả 2 model biển số. Bài học: FP16 KHÔNG an toàn đồng nhất cho mọi model, phải đo riêng
        # từng model chứ không giả định "đã đúng ở model A thì đúng ở model B".

        # throttle_frames: số frame tối thiểu giữa 2 lần thử cho CÙNG 1 track (không chạy
        # mọi frame — nguyên nhân chính gây chậm ở hệ thống cũ). max_attempts: số lần thử
        # thất bại tối đa trước khi bỏ cuộc hẳn với track đó (hệ thống cũ retry vô hạn).
        self.throttle_frames = throttle_frames
        self.max_attempts = max_attempts

        self._plates: dict[int, str] = {}
        self._attempts: dict[int, int] = {}
        self._last_tried_frame: dict[int, int] = {}
        # Lịch sử vĩnh viễn, KHÔNG bị dọn khi track bị xoá (khác _plates/_attempts, vốn chỉ
        # phục vụ throttle theo vòng đời track còn sống) — dùng để thống kê/ghi log toàn video
        # và cho evidence writer sau này (Phase 3), vì xe rời khung hình xong mới cần tra lại
        # biển số của nó lúc ghi log vi phạm.
        self.history: dict[int, str] = {}
        self.ever_attempted: set[int] = set()
        # Ảnh crop vùng biển số ĐÃ đọc thành công — vĩnh viễn (giống history), dùng để ghép vào
        # ảnh minh chứng vi phạm (evidence) làm ảnh cận cảnh biển số kèm theo, xem
        # overlay.py::draw_evidence_box + pipeline.py.
        self.plate_crops: dict[int, np.ndarray] = {}
        # Vị trí biển số dạng TỈ LỆ (0-1) so với chính bbox xe tại lúc đọc thành công — KHÔNG lưu
        # toạ độ tuyệt đối vì xe di chuyển liên tục mỗi khung hình trong khi OCR chỉ chạy 1 lần
        # (throttle). overlay.py tái chiếu tỉ lệ này lên bbox xe HIỆN TẠI mỗi khung hình để khung
        # biển số luôn bám đúng vị trí xe đang di chuyển, không cần chạy lại detector biển số.
        self.plate_relative_box: dict[int, tuple[float, float, float, float]] = {}
        # Kích thước (rộng, cao) bbox xe tại ĐÚNG lúc chụp `plate_relative_box` — vì OCR chỉ chạy
        # 1 lần rồi cache vĩnh viễn, xe có thể tiến/lùi rất nhiều so với camera trong phần còn lại
        # video, làm bbox đổi kích thước/tỉ lệ khung hình nhiều (phối cảnh) khiến tỉ lệ vị trí biển
        # số ĐÃ GHI không còn đúng nữa (đã gặp thật: khung trôi xuống dưới yên xe khi xe tới gần
        # camera hơn hẳn lúc đọc được biển số) — overlay.py so kích thước HIỆN TẠI với giá trị này,
        # ẩn khung nếu lệch quá xa thay vì vẽ sai vị trí.
        self.plate_capture_size: dict[int, tuple[float, float]] = {}

    def read(
        self, frame: np.ndarray, track_id: int,
        bbox: tuple[float, float, float, float], frame_idx: int,
    ) -> str | None:
        """Trả về biển số đã đọc được (cache), hoặc None nếu chưa đọc được/chưa tới lượt thử."""
        if track_id in self._plates:
            return self._plates[track_id]

        if self._attempts.get(track_id, 0) >= self.max_attempts:
            return None

        last_tried = self._last_tried_frame.get(track_id, -10**9)
        if frame_idx - last_tried < self.throttle_frames:
            return None

        # Xe quá nhỏ/xa thì biển số bên trong chắc chắn dưới MIN_PLATE_DIM — bỏ qua NGAY, không
        # tốn 1 lần gọi model detector biển số (chi phí GPU thật, khác nhánh "chưa thấy vùng
        # biển" bên dưới vốn đã tốn 1 lần chạy detector rồi mới biết là không có). Giảm tải rõ
        # rệt cho video nhiều xe ở xa/nhỏ trong khung hình.
        x1, y1, x2, y2 = bbox
        if (x2 - x1) < MIN_VEHICLE_DIM_FOR_PLATE or (y2 - y1) < MIN_VEHICLE_DIM_FOR_PLATE:
            return None

        self._last_tried_frame[track_id] = frame_idx
        self.ever_attempted.add(track_id)

        found = self._crop_plate(frame, bbox)
        if found is None:
            # Chưa thấy VÙNG biển số nào cả (xe đang quay ngang, quá xa...) — KHÔNG tính vào
            # max_attempts, vì chưa hề tốn 1 lần chạy OCR thật nào. Nếu tính, xe vào khung hình
            # từ góc chưa lộ biển số (vd nhìn nghiêng) sẽ tốn hết ngân sách trước khi kịp quay
            # sang góc lộ biển số rõ ràng — đã gặp thật (bug), sửa để tiếp tục chờ, chỉ giới
            # hạn max_attempts cho các lần ĐÃ tìm thấy vùng biển số nhưng đọc ký tự thất bại.
            return None
        crop, relative_box = found

        self._attempts[track_id] = self._attempts.get(track_id, 0) + 1

        # 4 bien the deskew (2 change_contrast x 2 center_thres) duoc xu ly trong 1 LAN GOI MODEL
        # DUY NHAT (batch) thay vi 4 lan goi tuan tu — xem docstring _read_plate_from_variants.
        variants: list[np.ndarray] = []
        for change_contrast in (False, True):
            for center_thres in (0, 1):
                try:
                    variants.append(_deskew(crop, change_contrast, center_thres))
                except Exception:
                    continue
        if not variants:
            return None
        try:
            text = _read_plate_from_variants(self.ocr, variants)
        except Exception:
            return None
        if text:
            self._plates[track_id] = text
            self.history[track_id] = text
            self.plate_crops[track_id] = crop
            self.plate_relative_box[track_id] = relative_box
            self.plate_capture_size[track_id] = (x2 - x1, y2 - y1)
            return text
        return None

    def _crop_plate(
        self, frame: np.ndarray, bbox: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
        """Trả về (crop biển số, box biển số dạng TỈ LỆ 0-1 so với chính `bbox` xe truyền vào —
        vd (0.1, 0.6, 0.3, 0.8) nghĩa là biển số nằm ở 10-30% chiều rộng, 60-80% chiều cao bbox
        xe). Tỉ lệ này không đổi khi tái chiếu lên bbox xe ở bất kỳ khung hình nào sau đó (xem
        `plate_relative_box`), bất kể xe đã di chuyển/co giãn bao nhiêu trong khung hình."""
        x1, y1, x2, y2 = bbox
        fx1, fy1 = max(0, int(x1)), max(0, int(y1))
        fx2, fy2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
        roi = frame[fy1:fy2, fx1:fx2]
        if roi.size == 0:
            return None

        results = self.detector(roi)
        # Dùng thẳng tensor thô `results.xyxy[0]` ([x1,y1,x2,y2,conf,cls]) thay vì
        # `results.pandas()` — pandas dựng DataFrame từ đúng CÙNG dữ liệu này (xem
        # third_party/yolov5/models/common.py::Detections.pandas()), chỉ tốn thêm chi phí dựng
        # object cho vài dòng dữ liệu, không đổi kết quả lọc/chọn bên dưới.
        det = results.xyxy[0].cpu().numpy()
        if det.shape[0] == 0:
            return None
        # Loại box quá nhỏ (nhiễu, không đủ pixel để đọc ký tự) trước khi chọn — đã gặp thật:
        # detector đôi khi bắt trúng vùng nhiễu vài pixel với confidence cao nhất trong khung
        # hình đó, tốn hết max_attempts trên rác thay vì chờ khung hình rõ hơn.
        w, h = det[:, 2] - det[:, 0], det[:, 3] - det[:, 1]
        det = det[(w >= MIN_PLATE_DIM) & (h >= MIN_PLATE_DIM)]
        if det.shape[0] == 0:
            return None
        # Loại box có TÂM nằm trong phần TRÊN CÙNG (SEARCH_TOP_SKIP) của bbox xe — biển số
        # thật luôn nằm ở nửa dưới xe (gần bánh), phần trên là người lái/nóc xe. Đã gặp bug thật
        # (bienso0): 2 xe đứng gần nhau theo chiều sâu (không chồng lấn bbox nhiều) nhưng xe PHÍA
        # SAU vẫn lọt vào phần trên bbox của xe phía trước do góc camera — detector bắt nhầm biển
        # số xe khác. LƯU Ý: lọc SAU khi detect trên ROI ĐẦY ĐỦ (không cắt pixel trước khi đưa
        # vào detector như bản cũ) — cắt trước từng gây bug khác: xe đứng sát camera có biển số
        # nằm ngay trên ranh giới skip, cắt pixel trước khiến detector chỉ thấy nửa dưới của
        # chính biển số đó (mất dòng trên của biển 2 dòng) thay vì bị cắt vì đó là biển xe khác.
        skip_y = roi.shape[0] * SEARCH_TOP_SKIP
        center_y = (det[:, 1] + det[:, 3]) / 2
        det = det[center_y >= skip_y]
        if det.shape[0] == 0:
            return None
        best = det[det[:, 4].argmax()]

        px1, py1 = max(0, int(best[0])), max(0, int(best[1]))
        px2, py2 = min(roi.shape[1], int(best[2])), min(roi.shape[0], int(best[3]))
        crop = roi[py1:py2, px1:px2]
        if crop.size == 0:
            return None
        roi_w, roi_h = roi.shape[1], roi.shape[0]
        relative_box = (px1 / roi_w, py1 / roi_h, px2 / roi_w, py2 / roi_h)
        return crop, relative_box

    def remove(self, track_ids: set[int]) -> None:
        for tid in track_ids:
            self._plates.pop(tid, None)
            self._attempts.pop(tid, None)
            self._last_tried_frame.pop(tid, None)
