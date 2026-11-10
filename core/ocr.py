import torch
import easyocr
from typing import List

class OCREngine:
    def __init__(self, languages: List[str] = ["en"], device: str = "cuda"):
        """
        Initializes EasyOCR reader on GPU or CPU.
        """
        self.use_gpu = torch.cuda.is_available() and device == "cuda"
        print(f"Initializing EasyOCR Reader on device: {'cuda' if self.use_gpu else 'cpu'}")
        
        # This will download the OCR weights automatically on the first boot
        self.reader = easyocr.Reader(languages, gpu=self.use_gpu)

    def extract_text(self, image_path: str) -> str:
        """
        Reads an image file and flattens all visible text blocks into a single string.
        """
        try:
            results = self.reader.readtext(image_path, detail=0) # detail=0 returns text strings only
            if results:
                return " ".join(results).strip()
        except Exception as e:
            print(f"OCR Exception processing {image_path}: {e}")
            
        return ""