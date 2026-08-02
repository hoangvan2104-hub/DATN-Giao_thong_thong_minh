"""Phát hiện & phân loại phương tiện bằng YOLO (không tracking — dùng cho eval/benchmark mAP)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from ultralytics import YOLO


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1,y1,x2,y2 (pixel)
    cls_name: str
    conf: float


class VehicleDetector:
    def __init__(
        self,
        model_path: str = "yolo11s.pt",
        classes: list[str] | None = None,
        device: str | None = None,
        conf: float = 0.35,
        imgsz: int = 640,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.conf = conf
        self.imgsz = imgsz
        self.class_names: dict[int, str] = self.model.names
        self.allowed_class_ids = self._resolve_class_ids(classes)

    def _resolve_class_ids(self, classes: list[str] | None) -> list[int] | None:
        if not classes:
            return None
        name_to_id = {v: k for k, v in self.class_names.items()}
        missing = [c for c in classes if c not in name_to_id]
        if missing:
            raise ValueError(
                f"Model '{self.model.model_name if hasattr(self.model, 'model_name') else ''}' "
                f"không có class: {missing}. Class hỗ trợ: {sorted(name_to_id)}"
            )
        return [name_to_id[c] for c in classes]

    def detect(self, frame) -> list[Detection]:
        result = self.model.predict(
            frame,
            device=self.device,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=self.allowed_class_ids,
            agnostic_nms=True,
            verbose=False,
        )[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append(Detection(bbox=(x1, y1, x2, y2), cls_name=self.class_names[cls_id], conf=conf))
        return detections
