"""Speech transcription and sentence-transformer embeddings."""
from pathlib import Path
import subprocess
import logging
import torch
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from config.config import get_settings

log = logging.getLogger(__name__)


class AudioProcessor:
    def __init__(self, whisper_model_size: str = "base", device: str = "auto",
                 text_model: str | None = None):
        settings = get_settings()
        selected = device if device != "auto" else settings.device
        if selected == "auto":
            selected = "cuda" if torch.cuda.is_available() else "cpu"
        if selected.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA requested but unavailable; using CPU for audio models")
            selected = "cpu"
        self.device = selected
        compute_type = "float16" if self.device.startswith("cuda") else "int8"
        self.whisper = WhisperModel(whisper_model_size, device=self.device, compute_type=compute_type)
        self.embedder = SentenceTransformer(text_model or settings.text_embedding_model, device=self.device)

    def extract_audio_from_chunk(self, video_chunk_path: Path, output_audio_path: Path) -> bool:
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_chunk_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(output_audio_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            output_audio_path.unlink(missing_ok=True)
            log.debug("No extractable audio in %s: %s", video_chunk_path, result.stderr[-500:])
            return False
        return True

    def transcribe_and_embed(self, audio_path: Path) -> dict:
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            return {"text": "", "embedding": []}
        segments, _ = self.whisper.transcribe(str(audio_path), beam_size=5, vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        if not text:
            return {"text": "", "embedding": []}
        embedding = self.embedder.encode(text, normalize_embeddings=True).tolist()
        return {"text": text, "embedding": embedding}
