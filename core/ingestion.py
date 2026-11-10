"""FFmpeg-backed video preparation with bounded, timestamped scene chunks."""
from pathlib import Path
import logging
import re
import subprocess

log = logging.getLogger(__name__)


class VideoIngestionPipeline:
    def __init__(self, base_storage_dir: str, max_chunk_seconds: float = 30.0):
        self.base_dir = Path(base_storage_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.max_chunk_seconds = max(5.0, float(max_chunk_seconds))
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def create_video_workspace(self, video_id: str) -> dict[str, Path]:
        root = self.processed_dir / video_id
        paths = {"workspace": root, "chunks": root / "chunks",
                 "frames": root / "frames", "audio": root / "audio"}
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return paths

    @staticmethod
    def _run(command: list[str]) -> subprocess.CompletedProcess:
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        except FileNotFoundError as exc:
            raise RuntimeError("FFmpeg and FFprobe must be installed and available on PATH.") from exc
        if result.returncode != 0:
            raise RuntimeError(f"Command failed ({command[0]}): {result.stderr[-2000:]}")
        return result

    def duration(self, video_path: Path) -> float:
        result = self._run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ])
        try:
            duration = float(result.stdout.strip())
        except ValueError as exc:
            raise RuntimeError("Could not read video duration.") from exc
        if duration <= 0:
            raise RuntimeError("The uploaded video has no playable duration.")
        return duration

    def _scene_boundaries(self, video_path: Path, duration: float,
                          threshold: float = 0.4) -> list[float]:
        # select(scene) emits showinfo only for selected scene-change frames.
        # Parsing all metadata frames (the previous implementation) created a
        # boundary at almost every frame and exploded the number of clips.
        try:
            result = self._run([
                "ffmpeg", "-hide_banner", "-i", str(video_path),
                "-vf", f"select='gt(scene,{threshold})',showinfo",
                "-an", "-f", "null", "-",
            ])
            points = [0.0]
            for match in re.finditer(r"pts_time:([0-9.]+)", result.stderr):
                point = float(match.group(1))
                if 0 < point < duration:
                    points.append(point)
            return sorted(set(points + [duration]))
        except (RuntimeError, ValueError) as exc:
            log.warning("Scene detection failed; using fixed chunks: %s", exc)
            return [0.0, duration]

    def chunk_video(self, video_path: Path, output_chunk_dir: Path,
                    output_frame_dir: Path, threshold: float = 0.4) -> list[tuple[float, float]]:
        output_chunk_dir.mkdir(parents=True, exist_ok=True)
        output_frame_dir.mkdir(parents=True, exist_ok=True)
        duration = self.duration(video_path)
        boundaries = self._scene_boundaries(video_path, duration, threshold)
        segments: list[tuple[float, float]] = []
        for left, right in zip(boundaries, boundaries[1:]):
            cursor = left
            while cursor < right:
                end = min(right, cursor + self.max_chunk_seconds)
                if end - cursor < 0.25:
                    break
                segments.append((cursor, end))
                cursor = end
        if not segments:
            segments = [(0.0, duration)]

        for start, end in segments:
            stem = f"scene_{start:.2f}_to_{end:.2f}"
            chunk_path = output_chunk_dir / f"{stem}.mp4"
            self._run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{start:.3f}", "-i", str(video_path),
                "-t", f"{end - start:.3f}", "-map", "0:v:0", "-map", "0:a:0?",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-movflags", "+faststart", str(chunk_path),
            ])
            pattern = output_frame_dir / f"{stem}_frame%03d.jpg"
            self._run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(chunk_path), "-vf", "fps=0.5", "-q:v", "3", str(pattern),
            ])
        log.info("Prepared %d scene chunks and timestamped frames for %s", len(segments), video_path.name)
        return segments
