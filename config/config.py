"""Environment-backed application settings.

Load a local .env file for developer runs. Expensive models and network clients
are still created by the FastAPI lifespan, never during module import.
"""
from functools import lru_cache
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    storage_dir: str = os.getenv("STORAGE_DIR", "./storage")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3")
    whisper_model: str = os.getenv("WHISPER_MODEL", "base")
    text_embedding_model: str = os.getenv(
        "TEXT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    clip_model: str = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
    yolo_model: str = os.getenv("YOLO_MODEL", "yolov10n.pt")
    device: str = os.getenv("MODEL_DEVICE", "auto").lower()
    enable_scene_recognition: bool = _bool("ENABLE_SCENE_RECOGNITION", True)
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024**3)))
    max_video_chunk_seconds: float = float(os.getenv("MAX_VIDEO_CHUNK_SECONDS", "30"))
    retrieval_limit: int = int(os.getenv("RETRIEVAL_LIMIT", "5"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
