import os
import subprocess
import shutil
from pathlib import Path
import subprocess
import re
from pathlib import Path
from typing import List, Tuple
import numpy as np

class VideoIngestionPipeline:
    def __init__(self, base_storage_dir: str):
        self.base_dir = Path(base_storage_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        
        # Ensure workspace directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def create_video_workspace(self, video_id: str) -> dict:
        """Creates specialized subdirectories for a specific video processing job."""
        video_workspace = self.processed_dir / video_id
        paths = {
            "workspace": video_workspace,
            "chunks": video_workspace / "chunks",
            "frames": video_workspace / "frames",
            "audio": video_workspace / "audio"
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return paths

    def extract_audio(self, video_path: Path, output_audio_path: Path) -> bool:
        """Strips audio channel from video and converts it to a clean WAV format for Whisper."""
        command = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", # Disable video recording
            "-acodec", "pcm_s16le", # Convert to 16-bit PCM uncompressed WAV
            "-ar", "16000", # Resample to 16kHz (Optimal for speech models like Whisper)
            "-ac", "1", # Downmix to mono channel
            str(output_audio_path)
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0

    def extract_frames(self, video_path: Path, output_frame_dir: Path, fps: float = 1.0) -> bool:
        """Extracts keyframes at a specific Frame Per Second (FPS) rate for CLIP analysis."""
        command = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"fps={fps}", # Video filter option: set frame rate extraction
            f"{output_frame_dir}/frame_%04d.jpg" # Output naming pattern (e.g. frame_0001.jpg)
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0



    def chunk_video(self, video_path: Path, output_chunk_dir: Path, output_frame_dir: Path, threshold: float = 0.4) -> List[Tuple[float, float]]:
        timestamp_file = output_chunk_dir / "timestamps.txt"
        
        # STEP 1: Detect scene changes
        detect_command = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"scdet=s=1,metadata=print:file={timestamp_file.as_posix()}",
            "-an",
            "-f", "null", "-"
        ]
        result = subprocess.run(
            detect_command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )

        # STEP 2: Parse timestamps
        timestamps = [0.0]
        if timestamp_file.exists():
            with open(timestamp_file, "r", encoding="utf-8") as f:
                for line in f:
                    if "pts_time:" in line:
                        try:
                            time_str = line.split("pts_time:")[1].strip().split()[0]
                            val = round(float(time_str), 2)
                            if val > 0.0 and val not in timestamps:
                                timestamps.append(val)
                        except (IndexError, ValueError):
                            continue
            timestamp_file.unlink()
        else:
            print("WARNING: timestamp file not found. FFmpeg stderr:")
            print(result.stderr.decode())

        # STEP 3: Get total duration
        duration_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        total_duration = round(float(subprocess.check_output(duration_cmd).decode().strip()), 2)
        timestamps.append(total_duration)
        timestamps = sorted(list(set(timestamps)))

        # STEP 4: Merge very short chunks (< 2 seconds)
        merged = [timestamps[0]]
        for t in timestamps[1:]:
            if t - merged[-1] >= 2.0:
                merged.append(t)
        merged[-1] = total_duration
        timestamps = merged
        
        from core.detector import ObjectDetector

        # Initialize once (outside the scene loop)
        detector = ObjectDetector()

        frames_dir = output_frame_dir
        frames_dir.mkdir(exist_ok=True)

        scene_segments = []

        for i in range(len(timestamps) - 1):
            start = timestamps[i]
            end = timestamps[i + 1]
            duration = end - start
            scene_segments.append((start, end))

            # ---------------- Slice scene ---------------- #
            output_path = output_chunk_dir / f"scene_{start:.2f}_to_{end:.2f}.mp4"

            slice_command = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-to", str(end),
                "-i", str(video_path),
                "-c", "copy",
                str(output_path)
            ]
            subprocess.run(slice_command,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)

            # ---------------- Sample timestamps ---------------- #

            sample_interval = min(2.0, max(0.5, duration / 20))

            sample_points = np.arange(start, end, sample_interval).tolist()

            if len(sample_points) == 0:
                sample_points = [start]

            previous_objects = None
            saved_idx = 0

            for ts in sample_points:

                temp_frame = frames_dir / "__temp.jpg"

                extract_command = [
                    "ffmpeg", "-y",
                    "-ss", str(round(ts, 3)),
                    "-i", str(video_path),
                    "-frames:v", "1",
                    "-q:v", "2",
                    str(temp_frame)
                ]

                subprocess.run(
                    extract_command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

                detections = detector.detect_objects(str(temp_frame))

                current_objects = {
                    d["object"]
                    for d in detections
                }

                # Save first frame of every scene
                should_save = previous_objects is None

                # Save whenever object set changes
                if previous_objects is not None:
                    if current_objects != previous_objects:
                        should_save = True

                if should_save:

                    frame_filename = (
                        f"chunk_{start:.2f}_{end:.2f}_frame{saved_idx}.jpg"
                    )

                    final_path = frames_dir / frame_filename

                    os.replace(temp_frame, final_path)

                    saved_idx += 1

                    previous_objects = current_objects

                else:
                    temp_frame.unlink(missing_ok=True)

        print(f"Created {len(scene_segments)} scene chunks.")
        print(f"Keyframes saved to: {frames_dir}")
        return scene_segments