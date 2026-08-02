"""Rule: không đội mũ bảo hiểm (xe máy/xe đạp).

Khác 4 rule còn lại (không dựa vào hình học zone/line/vector) — dùng model phân loại riêng
(mặc định `bestyolo.pt`, YOLO11, 2 nhãn `helmet`/`no_helmet` — xem CLAUDE.md/nhat-ky-ky-thuat.md)
để xác định có đội mũ hay không. `no_helmet.pt` (model tự fine-tune trên Colab, xem
E:/Nohelmet/helmet_proposed_F-2/) VẪN LÀ model dùng cho bảng so sánh Baseline vs Proposed bắt
buộc theo rubric — chỉ không còn là model CHẠY THẬT trong hệ thống (đã đổi sang `bestyolo.pt` vì
độ chính xác tốt hơn trên mũ màu tối, xem "Model mũ bảo hiểm mới bestyolo.pt" trong CLAUDE.md).
Cần đọc pixel frame (không chỉ vị trí bbox) nên KHÔNG nằm trong
`Pipeline.rules` (interface `update(tracks, frame_idx)` thuần hình học) — xử lý tương tự
`PlateReader` (đối tượng riêng trong pipeline.py, được gọi kèm `frame`).

Throttle theo track (không chạy detect mọi khung hình cho mọi xe — cùng nguyên tắc với
PlateReader, tránh lặp lại nguyên nhân chậm của hệ thống cũ).

Vote theo ĐA SỐ trong cửa sổ trượt (KHÔNG bắt buộc liên tiếp tuyệt đối) + CÓ THỂ RÚT LẠI vi
phạm nếu bằng chứng sau đó đảo ngược đủ mạnh — phiên bản đầu dùng "N lần liên tiếp giống nhau,
khoá vĩnh viễn khi đã xác nhận" nhưng gặp báo sai thật: xe bị nhận nhầm "không đội mũ" khi đi xa
(ảnh nhỏ/mờ) hoặc qua vùng bóng râm (ánh sáng xấu tạm thời) trong đúng 3 khung hình liên tiếp,
sau đó dù xe quay lại điều kiện tốt vẫn bị ghi sai vĩnh viễn vì đã dừng kiểm tra. Thiết kế mới
tiếp tục theo dõi CẢ SAU KHI đã xác nhận vi phạm, cho phép rút lại nếu đa số phiếu sau đó đảo
chiều rõ ràng (có độ trễ/hysteresis giữa ngưỡng xác nhận và ngưỡng rút lại để tránh dao động qua
lại liên tục ở ranh giới).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

from src.engine.tracker import Track

# Chỉ xe máy/xe gắn máy bắt buộc đội mũ bảo hiểm theo luật giao thông Việt Nam (Nghị định
# 100/2019 và sửa đổi) — xe đạp thường (class "bicycle", khác xe đạp điện) KHÔNG bắt buộc, nên
# không đưa vào kiểm tra để tránh báo vi phạm sai đối tượng.
CHECK_CLASSES = {"motorcycle"}
# Kích thước tối thiểu (px) mỗi chiều của xe để còn đáng chạy model mũ bảo hiểm — xe nhỏ hơn
# mức này thì vùng đầu (dù đã mở rộng head_extend_ratio) vẫn quá ít chi tiết để phân loại đáng
# tin cậy, bỏ qua ngay để tiết kiệm 1 lần gọi model (chi phí GPU thật).
MIN_VEHICLE_DIM = 30
# Crop nhỏ hơn tỉ lệ này so với lần LỚN NHẤT từng thấy của CHÍNH track đó bị coi là không đủ
# tin cậy để tính vào phiếu — xe đi CÀNG XA thì ảnh CÀNG kém tin cậy hơn (không phải ngược lại),
# nếu không chặn, các lần đọc muộn/xa/nhỏ có thể ghi đè phán quyết đúng lúc xe còn gần/rõ (đã
# gặp thật: vi phạm xác nhận đúng lúc gần bị RÚT LẠI SAI do các lần đọc sau đó xa hơn).
MIN_AREA_RATIO_FOR_VOTE = 0.5


def _is_no_helmet_label(label: str) -> bool:
    """So khớp tên nhãn "không đội mũ" KHÔNG phân biệt hoa/thường và định dạng phân tách
    (underscore/gạch ngang/khoảng trắng) — model gốc dùng "NoHelmet" nhưng người dùng có thể
    đổi sang model khác dùng quy ước khác (vd "no_helmet", "NO-HELMET"...). So khớp cứng đúng
    1 chuỗi sẽ âm thầm không bao giờ khớp nếu đổi model, khiến rule luôn báo 0 vi phạm mà không
    có lỗi/cảnh báo gì — nguy hiểm hơn nhiều so với việc chậm 1 chút để chuẩn hoá chuỗi."""
    normalized = label.lower().replace("_", "").replace("-", "").replace(" ", "")
    return normalized == "nohelmet"


# Cache model đã load theo (model_path, device) — model này chỉ dùng forward pass thuần (phân
# loại đầu có/không mũ), không giữ trạng thái track giữa các lần gọi như VehicleTracker, an toàn
# để dùng lại thẳng không cần reset gì thêm. Quan trọng cho Web UI (src/web/): tránh nạp lại mỗi
# lần xử lý video mới trong cùng phiên server.
_MODEL_CACHE: dict[tuple[str, str], YOLO] = {}


def _load_yolo_cached(model_path: str, device: str) -> YOLO:
    key = (model_path, device)
    if key not in _MODEL_CACHE:
        model = YOLO(model_path)
        model.to(device)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


@dataclass
class NoHelmetViolation:
    track_id: int
    frame_idx: int


class NoHelmetRule:
    def __init__(
        self,
        model_path: str = "models/pretrained/bestyolo.pt",
        device: str | None = None,
        throttle_frames: int = 10,
        confirm_votes: int = 3,
        vote_window: int = 6,
        confirm_ratio: float = 0.8,
        conf: float = 0.4,
        head_extend_ratio: float = 1.5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _load_yolo_cached(model_path, self.device)
        # throttle_frames: số khung hình tối thiểu giữa 2 lần thử cho CÙNG 1 track.
        # vote_window: số lần kiểm tra GẦN NHẤT được xét (cửa sổ trượt, không nhất thiết liên
        # tiếp giống nhau tuyệt đối). confirm_votes: SÀN số phiếu tối thiểu đã thu thập trước khi
        # xét xác nhận (giữ khả năng bắt vi phạm thật ở track sống ngắn, chỉ có vài phiếu trước
        # khi rời khung hình). confirm_ratio: tỉ lệ phiếu "không mũ" tối thiểu trong SỐ PHIẾU HIỆN
        # CÓ (không phải cửa sổ đầy) — xem lý do đổi từ ngưỡng tuyệt đối cố định sang tỉ lệ ở
        # update(): ngưỡng tuyệt đối cố định (vd 3) so với cửa sổ NGÀY CÀNG ĐẦY (tối đa
        # vote_window) tương đương tỉ lệ ngày càng LỎNG dần (3/3=100% lúc mới, 3/5=60% lúc đầy) —
        # 60% quá dễ trúng với nhiễu rải rác (đã đo thật: track có mũ bảo hiểm rõ ràng vẫn bị gắn
        # cờ sai vì tình cờ 3/5 phiếu rơi vào nhiễu). Tỉ lệ cố định giữ ngưỡng nghiêm ngặt nhất
        # quán bất kể track đã sống bao lâu. RÚT LẠI vi phạm đã xác nhận dùng ngưỡng KHÁC hẳn, khó
        # hơn nhiều (toàn bộ cửa sổ đều "có mũ") — xem lý do chi tiết ở update().
        self.throttle_frames = throttle_frames
        self.confirm_votes = confirm_votes
        self.vote_window = vote_window
        self.confirm_ratio = confirm_ratio
        self.conf = conf
        # bbox track "motorcycle"/"bicycle" từ detector chính chỉ bao đúng THÂN XE, không bao
        # gồm đầu người lái (đã xác nhận bằng ảnh thật — model mũ bảo hiểm luôn ra 0 box vì
        # crop không hề chứa đầu) — mở rộng crop lên trên thêm head_extend_ratio lần chiều cao
        # bbox gốc để chắc chắn chứa vùng đầu/mũ bảo hiểm.
        self.head_extend_ratio = head_extend_ratio

        self._last_tried_frame: dict[int, int] = {}
        self._votes: dict[int, list[bool]] = {}
        # Diện tích bbox LỚN NHẤT từng thấy của mỗi track — dùng để lọc phiếu không đủ tin cậy
        # (crop quá nhỏ so với lần tốt nhất), xem MIN_AREA_RATIO_FOR_VOTE.
        self._max_bbox_area: dict[int, float] = {}
        # Track ĐANG bị coi là vi phạm — không còn "vĩnh viễn", có thể bị .discard() lại nếu
        # đa số phiếu sau đó đảo chiều (xem update()).
        self._reported: set[int] = set()

        # Vị trí đầu/mũ bảo hiểm dạng TỈ LỆ (0-1) so với vùng crop đã MỞ RỘNG lên trên
        # (head_extend_ratio) tại lần kiểm tra GẦN NHẤT — không phải toạ độ tuyệt đối, vì xe di
        # chuyển mỗi khung hình trong khi model chỉ chạy throttle. `overlay.py` tái chiếu tỉ lệ
        # này lên vùng mở rộng tính từ bbox xe HIỆN TẠI (dùng `head_extend_ratio` bên dưới) mỗi
        # khung hình để khung luôn bám đúng vị trí đầu đang di chuyển. Dọn theo vòng đời track
        # (khác `plate_crops`/`history` của PlateReader — không cần giữ lại sau khi xe rời khung
        # hình, không phục vụ evidence).
        self.head_relative_box: dict[int, tuple[float, float, float, float]] = {}
        self.head_label: dict[int, str] = {}
        # Kích thước (rộng, cao) vùng crop MỞ RỘNG tại đúng lúc chụp `head_relative_box` — dùng
        # để overlay.py so với kích thước hiện tại, ẩn khung nếu lệch quá xa (cùng lý do
        # `plate_capture_size` trong plate_ocr.py — throttle khiến giá trị có thể cũ hơn vài
        # khung hình, xe di chuyển/đổi khoảng cách trong lúc đó).
        self.head_capture_size: dict[int, tuple[float, float]] = {}

    def update(
        self, tracks: list[Track], frame_idx: int, frame: np.ndarray,
        native_frame: np.ndarray | None = None, scale: float = 1.0,
    ) -> tuple[list[NoHelmetViolation], list[int]]:
        """`native_frame`/`scale`: nếu pipeline đang xử lý ở độ phân giải đã hạ (xem
        `Pipeline._apply_max_dimension`), crop để nhận diện mũ bảo hiểm nên lấy từ khung hình
        GỐC (chưa hạ) để giữ đủ chi tiết vùng đầu — vùng này vốn đã nhỏ (crop trong 1 bbox xe),
        hạ độ phân giải cả pipeline trước rồi mới crop sẽ mất chi tiết ngay chỗ cần nhất.

        Trả về (vi phạm MỚI xác nhận, track_id vừa được RÚT LẠI vi phạm do bằng chứng sau đó
        đảo ngược đủ mạnh)."""
        violations: list[NoHelmetViolation] = []
        retracted: list[int] = []
        crop_frame = native_frame if native_frame is not None else frame

        # Giai đoạn 1: lọc track đủ điều kiện kiểm tra khung hình này (throttle/kích thước/độ
        # tin cậy crop) + cắt ảnh đầu — CHƯA gọi model.
        eligible: list[tuple[Track, np.ndarray]] = []
        for t in tracks:
            # KHÔNG còn skip khi t.track_id in self._reported — phải tiếp tục theo dõi track đã
            # bị gắn cờ để có cơ hội rút lại nếu sau đó chứng minh sai (xem docstring class).
            if t.cls_name not in CHECK_CLASSES:
                continue

            bx1, by1, bx2, by2 = t.bbox
            if (bx2 - bx1) < MIN_VEHICLE_DIM * scale or (by2 - by1) < MIN_VEHICLE_DIM * scale:
                continue

            area = (bx2 - bx1) * (by2 - by1)
            max_area = self._max_bbox_area.get(t.track_id, 0.0)
            if area > max_area:
                self._max_bbox_area[t.track_id] = area
            elif area < max_area * MIN_AREA_RATIO_FOR_VOTE:
                # Xe hiện nhỏ hơn hẳn lần lớn nhất từng thấy (đang đi xa dần) — crop lúc này
                # kém tin cậy hơn, không tính vào phiếu để tránh ghi đè phán quyết lúc còn gần.
                continue

            last_tried = self._last_tried_frame.get(t.track_id, -10**9)
            if frame_idx - last_tried < self.throttle_frames:
                continue
            self._last_tried_frame[t.track_id] = frame_idx

            bbox = tuple(v / scale for v in t.bbox) if native_frame is not None else t.bbox
            crop = self._crop(crop_frame, bbox)
            if crop is None:
                continue
            eligible.append((t, crop))

        if not eligible:
            return violations, retracted

        # Giai đoạn 2: gộp TẤT CẢ crop đủ điều kiện trong khung hình này vào 1 lần gọi model
        # (batch) thay vì gọi tuần tự từng xe — mỗi lần gọi model có chi phí cố định (kernel
        # launch + đồng bộ CUDA) không giảm theo số lượng ảnh nhỏ, gộp N xe/khung hình thành 1
        # lần gọi giảm đáng kể chi phí này khi nhiều xe máy cùng đủ điều kiện kiểm tra 1 lúc
        # (giao lộ đông xe) — kết quả từng ảnh độc lập với ảnh khác (eval mode), không đổi độ
        # chính xác so với gọi tuần tự từng ảnh.
        quantize = 16 if self.device != "cpu" else None
        results = self.model([c for _, c in eligible], conf=self.conf, verbose=False, quantize=quantize)
        names = self.model.names

        # Giai đoạn 3: áp dụng logic vote như cũ, theo đúng thứ tự eligible/results khớp nhau.
        for (t, crop), result in zip(eligible, results):
            if result.boxes is None or len(result.boxes) == 0:
                continue  # không thấy đầu rõ (góc khuất/xa) — bỏ qua lần này, thử lại sau

            # Lấy nhãn của box tin cậy NHẤT thay vì "có ít nhất 1 box NoHelmet" — model đôi khi
            # ra 2 box chồng lấp cho CÙNG 1 đầu (1 Helmet + 1 NoHelmet, do model chưa hoàn hảo
            # với mũ màu tối/góc khuất — xem CLAUDE.md), "any" sẽ tính vi phạm dù box Helmet tin
            # cậy hơn hẳn. Không loại bỏ hết sai số của model (giới hạn thật của model, không
            # phải bug tích hợp) nhưng tránh cộng thêm sai số logic phía trên.
            best_box = max(result.boxes, key=lambda b: float(b.conf[0]))
            has_no_helmet = _is_no_helmet_label(names[int(best_box.cls[0])])

            # Lưu vị trí + nhãn MỚI NHẤT để vẽ lên video (xem docstring `head_relative_box`) —
            # cập nhật ở MỌI lần model chạy ra kết quả, không đợi đủ vote/xác nhận vi phạm, vì
            # đây là hiển thị trực quan "model đang thấy gì", khác hẳn is_no_helmet_votes vốn cần
            # đa số phiếu mới kết luận vi phạm chính thức.
            ch, cw = crop.shape[:2]
            bx1, by1, bx2, by2 = best_box.xyxy[0].tolist()
            self.head_relative_box[t.track_id] = (bx1 / cw, by1 / ch, bx2 / cw, by2 / ch)
            self.head_label[t.track_id] = "no_helmet" if has_no_helmet else "helmet"
            self.head_capture_size[t.track_id] = (cw, ch)

            votes = self._votes.setdefault(t.track_id, [])
            votes.append(has_no_helmet)
            if len(votes) > self.vote_window:
                votes.pop(0)
            if len(votes) < self.confirm_votes:
                continue

            no_helmet_votes = sum(votes)
            was_reported = t.track_id in self._reported
            required = math.ceil(len(votes) * self.confirm_ratio)

            if not was_reported and no_helmet_votes >= required:
                self._reported.add(t.track_id)
                violations.append(NoHelmetViolation(track_id=t.track_id, frame_idx=frame_idx))
            elif was_reported and len(votes) == self.vote_window and no_helmet_votes == 0:
                # RÚT LẠI cố ý khó hơn NHIỀU so với xác nhận (bất đối xứng có chủ đích) — xác
                # nhận chỉ cần đa số (confirm_votes/vote_window), nhưng rút lại cần TOÀN BỘ cửa
                # sổ đều "có mũ" (bằng chứng áp đảo tuyệt đối). Lý do: đã gặp thật 2 chiều lỗi
                # đối lập — (1) 1-2 khung hình nhiễu thoáng qua (bóng râm/góc khuất) làm KHOÁ
                # NHẦM 1 xe đội mũ đúng luật thành vi phạm (rút lại đúng lúc cần); nhưng (2) xe
                # đi CÀNG XA thì ảnh CÀNG kém tin cậy hơn, không phải hơn — nếu rút lại cũng dễ
                # như xác nhận, 1-2 lần đọc nhiễu ở xa có thể XOÁ NHẦM 1 vi phạm THẬT đã xác
                # nhận đúng lúc còn gần/rõ (đã đo trực tiếp: vi phạm thật xác nhận đúng ở khung
                # hình 226 nhưng bị rút lại sai ở khung hình 256 do các lần đọc sau đó xa hơn,
                # kém tin cậy hơn, tình cờ ra "có mũ"). Yêu cầu bằng chứng áp đảo cho rút lại
                # giảm rủi ro này trong khi vẫn giữ được khả năng tự sửa khi cần.
                self._reported.discard(t.track_id)
                retracted.append(t.track_id)

        return violations, retracted

    def _crop(self, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox
        extended_y1 = y1 - (y2 - y1) * self.head_extend_ratio
        fx1, fy1 = max(0, int(x1)), max(0, int(extended_y1))
        fx2, fy2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
        roi = frame[fy1:fy2, fx1:fx2]
        return roi if roi.size > 0 else None

    def remove(self, track_ids: set[int]) -> None:
        for tid in track_ids:
            self._last_tried_frame.pop(tid, None)
            self._votes.pop(tid, None)
            self._max_bbox_area.pop(tid, None)
            self._reported.discard(tid)
            self.head_relative_box.pop(tid, None)
            self.head_label.pop(tid, None)
            self.head_capture_size.pop(tid, None)
