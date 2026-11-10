import os
import torch
import urllib.request
from pathlib import Path
from typing import List, Dict, Any
from ultralytics import YOLOv10

class ObjectDetector:
    def __init__(self, model_version: str = "yolov10n", device: str = "cuda"):
        """
        Initializes YOLOv10. Auto-downloads weights if missing in working directory.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Verify weight file exists, download if missing
        model_filename = f"{model_version}.pt"
        self.model_path = Path(model_filename)
        
        if not self.model_path.exists():
            print(f"[{model_version}] weights not found locally. Initiating download...")
            download_url = f"https://github.com/THU-MIG/yolov10/releases/download/v1.1/{model_filename}"
            try:
                urllib.request.urlretrieve(download_url, self.model_path)
                print(f"[{model_version}] Weight file successfully saved to {self.model_path.absolute()}")
            except Exception as e:
                raise RuntimeError(f"Failed to download YOLOv10 weights from {download_url}. Error: {e}")

        print(f"Initializing YOLOv10 ({model_version}) on device: {self.device}")
        self.model = YOLOv10(str(self.model_path)).to(self.device)

    def detect_objects(self, image_path: str, confidence_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Runs object detection on a target frame.
        """
        results = self.model(image_path, conf=confidence_threshold, verbose=False)
        detections = []
        
        if not results:
            return detections
            
        result = results[0]
        names = result.names
        
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = names[class_id]
            confidence = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
            
            detections.append({
                "object": class_name,
                "confidence": round(confidence, 3),
                "box": [round(coord, 1) for coord in xyxy]
            })
            
        return detections