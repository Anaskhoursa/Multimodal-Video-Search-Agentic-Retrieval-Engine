"""CLIP image encoder for normalized 512-dimensional visual vectors."""
import logging
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from config.config import get_settings

log = logging.getLogger(__name__)


class VisionEmbedder:
    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        requested = settings.device
        if requested == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA requested but unavailable; using CPU for CLIP")
            self.device = "cpu"
        else:
            self.device = requested
        name = model_name or settings.clip_model
        self.model = CLIPModel.from_pretrained(name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(name)

    def generate_frame_embedding(self, image_path: str) -> list[float]:
        try:
            with Image.open(image_path) as image:
                inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
            with torch.inference_mode():
                features = torch.nn.functional.normalize(self.model.get_image_features(**inputs), dim=-1)
            return features[0].detach().cpu().tolist()
        except Exception:
            log.exception("Could not create CLIP embedding for %s", image_path)
            return []
