"""YOLO object detection with automatic CPU/GPU selection."""
from pathlib import Path
import logging
import torch
from ultralytics import YOLO
from config.config import get_settings

log = logging.getLogger(__name__)


class ObjectDetector:
    def __init__(self, model_version: str | None = None, device: str = "auto"):
        settings = get_settings()
        requested = device if device != "auto" else settings.device
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA requested but unavailable; using CPU for object detection")
            self.device = "cpu"
        self.model_path = model_version or settings.yolo_model
        self.model = YOLO(self.model_path)

    def detect_objects(self, image_path: str | Path, confidence_threshold: float = 0.3) -> list[dict]:
        results = self.model.predict(source=str(image_path), conf=confidence_threshold,
                                     device=self.device, verbose=False)
        if not results:
            return []
        result = results[0]
        detections = []
        for box in result.boxes or []:
            class_id = int(box.cls[0].item())
            detections.append({
                "object": result.names[class_id],
                "confidence": round(float(box.conf[0].item()), 3),
                "box": [round(float(value), 1) for value in box.xyxy[0].tolist()],
            })
        return detections
