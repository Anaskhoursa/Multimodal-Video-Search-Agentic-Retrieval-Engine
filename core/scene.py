from pathlib import Path
from typing import List
import torch

from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification
)
import av


class SceneRecognizer:

    def __init__(
        self,
        model_name="MCG-NJU/videomae-base-finetuned-kinetics"
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)

        self.model = (
            VideoMAEForVideoClassification
            .from_pretrained(model_name)
            .to(self.device)
        )

        self.model.eval()

    def recognize_actions(
        self,
        video_path: str,
        top_k: int = 5
    ) -> List[str]:

        frames = self._load_video(video_path)

        inputs = self.processor(
            frames,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = outputs.logits.softmax(-1)

        values, indices = probs.topk(top_k)

        labels = []

        for idx in indices[0]:
            labels.append(
                self.model.config.id2label[idx.item()]
            )

        return labels

    def _load_video(self, path: str) -> List:
            import numpy as np
            
            container = av.open(path)
            stream = container.streams.video[0]
            
            # 1. Read all raw frames from the video container chunk
            all_frames = []
            for frame in container.decode(video=0):
                all_frames.append(frame.to_image())
                
            container.close()
            
            total_frames = len(all_frames)
            
            # Fallback if the video segment contains no readable frames
            if total_frames == 0:
                raise ValueError(f"No video frames could be decoded from path: {path}")
                
            # 2. Uniformly sample exactly 16 frame indices across the sequence duration
            target_frame_count = 16
            indices = np.linspace(0, total_frames - 1, num=target_frame_count, dtype=int)
            
            # 3. Pull out just those 16 target frames
            sampled_frames = [all_frames[idx] for idx in indices]
            
            return sampled_frames