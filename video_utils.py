"""Video utilities for frame extraction."""

import cv2
from pathlib import Path
from typing import List, Optional
from PIL import Image
import numpy as np


def extract_frames_from_video(
    video_path: str,
    skip: int = 1,
    max_frames: Optional[int] = None,
) -> List[Image.Image]:
    """
    Extract frames from video file.

    Args:
        video_path: Path to video file
        skip: Frame skip interval (1 = all frames)
        max_frames: Max frames to extract (None = all)

    Returns:
        List of PIL Images
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip == 0:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_frame = Image.fromarray(frame)
            frames.append(pil_frame)

            if max_frames and len(frames) >= max_frames:
                break

        frame_count += 1

    cap.release()
    return frames


def frames_to_tensors(frames: List[Image.Image]) -> List[np.ndarray]:
    """Convert PIL Images to numpy arrays."""
    return [np.array(f) for f in frames]


def frames_to_file_paths(frames: List[Image.Image], temp_dir: str = "/tmp/physion_frames") -> List[str]:
    """
    Save frames to temp directory and return file paths.
    Useful for metrics that expect file paths.
    """
    import tempfile
    import os

    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    for i, frame in enumerate(frames):
        frame_path = temp_dir / f"frame_{i:06d}.png"
        frame.save(frame_path)
        frame_paths.append(str(frame_path))

    return frame_paths
