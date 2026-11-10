"""Video action recognition using a bounded 16-frame sample."""
from pathlib import Path
import av
import logging
import numpy as np
import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from config.config import get_settings

log = logging.getLogger(__name__)


class SceneRecognizer:
    def __init__(self, model_name: str = "MCG-NJU/videomae-base-finetuned-kinetics"):
        requested = get_settings().device
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_name).to(self.device).eval()

    def _load_video(self, path: str | Path) -> list:
        frames = []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            total = stream.frames or 0
            if total:
                targets = set(np.linspace(0, max(total - 1, 0), 16, dtype=int).tolist())
                for index, frame in enumerate(container.decode(video=0)):
                    if index in targets:
                        frames.append(frame.to_image().convert("RGB"))
            else:
                decoded = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
                if decoded:
                    indices = np.linspace(0, len(decoded) - 1, min(16, len(decoded)), dtype=int)
                    frames = [decoded[index] for index in indices]
        if not frames:
            raise ValueError(f"No video frames could be decoded from {path}")
        if len(frames) < 16:
            frames.extend([frames[-1]] * (16 - len(frames)))
        return frames[:16]

    def recognize_actions(self, video_path: str | Path, top_k: int = 5) -> list[str]:
        frames = self._load_video(video_path)
        inputs = self.processor(frames, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            indices = logits.softmax(-1).topk(min(top_k, logits.shape[-1])).indices[0]
        return [self.model.config.id2label[index.item()] for index in indices]
