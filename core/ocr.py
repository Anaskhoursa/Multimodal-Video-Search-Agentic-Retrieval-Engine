"""Lazy, device-aware OCR wrapper."""
import logging
import torch
import easyocr

log = logging.getLogger(__name__)


class OCREngine:
    def __init__(self, languages: list[str] | None = None, device: str = "auto"):
        requested = device
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA requested but unavailable; using CPU for OCR")
            requested = "cpu"
        self.reader = easyocr.Reader(languages or ["en"], gpu=requested.startswith("cuda"))

    def extract_text(self, image_path: str) -> str:
        try:
            lines = self.reader.readtext(image_path, detail=0, paragraph=True)
            return " ".join(str(line).strip() for line in lines if str(line).strip())
        except Exception:
            log.exception("OCR failed for %s", image_path)
            return ""
