import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer

class AudioProcessor:
    def __init__(self, whisper_model_size: str = "base", device: str = "cuda"):
        """
        Initializes Faster-Whisper and a text embedder on the specified hardware (cuda/cpu).
        """
        print(f"Initializing Faster-Whisper ({whisper_model_size}) on device: {device}")
        # compute_type="float16" forces efficient GPU processing. Swap to "int8" if low on VRAM.
        self.whisper = WhisperModel(whisper_model_size, device=device, compute_type="int8")
        
        print(f"Initializing Text Embedder (all-MiniLM-L6-v2) on device: {device}")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    def extract_audio_from_chunk(self, video_chunk_path: Path, output_audio_path: Path) -> bool:
        """Rips raw 16kHz mono audio track from an MP4 chunk for Whisper."""
        command = [
            "ffmpeg", "-y", "-i", str(video_chunk_path),
            "-vn",                  # Disable video recording stream
            "-acodec", "pcm_s16le", # Convert audio to linear PCM 16-bit
            "-ar", "16000",         # Set audio sample rate to 16kHz (Whisper native)
            "-ac", "1",             # Reduce to 1 audio channel (mono)
            str(output_audio_path)
        ]
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0

    def transcribe_and_embed(self, audio_path: Path) -> Dict[str, Any]:
        """Transcribes the audio chunk and generates its semantic text embedding."""
        if not audio_path.exists():
            return {"text": "", "embedding": []}

        # Transcribe audio file
        segments, info = self.whisper.transcribe(str(audio_path), beam_size=5)
        
        # Combine all sentence segments found in this chunk into one unified block
        text_content = " ".join([seg.text for seg in segments]).strip()
        
        if not text_content:
            return {"text": "", "embedding": []}

        # Generate a mathematical vector for this text string (returns a 384-dimensional array)
        embedding = self.embedder.encode(text_content).tolist()
        
        return {
            "text": text_content,
            "embedding": embedding
        }