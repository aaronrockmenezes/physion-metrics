"""
Wrapper classes for WorldScore metrics adapted for video-only evaluation.
Paper: WorldScore - A Unified Evaluation Benchmark for World Generation (2504.00983)
"""

import sys
from pathlib import Path
from typing import List, Tuple
import torch
import numpy as np
from PIL import Image

# Add local WorldScore to path (used if not pip-installed)
WORLDSCORE_PATH = Path(__file__).parent.parent / "WorldScore"
sys.path.insert(0, str(WORLDSCORE_PATH))


def _inject_third_party_paths():
    """Inject third_party subdirs into sys.path before worldscore imports.
    Handles both pip-installed and local repo."""
    import importlib.util
    subdirs = ["droid_slam", "groundingdino", "sam2", "VFIMamba", "SEA-RAFT"]

    spec = importlib.util.find_spec("worldscore")
    if spec and spec.origin:
        ws_pkg = Path(spec.origin).parent
        third_party = ws_pkg / "benchmark" / "metrics" / "third_party"
        if third_party.exists():
            for sub in subdirs:
                p = str(third_party / sub)
                if p not in sys.path:
                    sys.path.insert(0, p)
            return

    # Fallback: local repo
    third_party = WORLDSCORE_PATH / "worldscore" / "benchmark" / "metrics" / "third_party"
    for sub in subdirs:
        p = str(third_party / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


_inject_third_party_paths()

from video_utils import frames_to_file_paths
import os
import contextlib


def _get_worldscore_root() -> Path:
    """Return the parent dir of the worldscore package (CWD expected by WorldScore metrics)."""
    import importlib.util
    spec = importlib.util.find_spec("worldscore")
    if spec and spec.origin:
        return Path(spec.origin).parent.parent  # site-packages/
    return WORLDSCORE_PATH  # local repo: WorldScore/ contains worldscore/


@contextlib.contextmanager
def _worldscore_cwd():
    """Temporarily chdir to worldscore root so relative paths resolve correctly."""
    original = os.getcwd()
    try:
        os.chdir(_get_worldscore_root())
        yield
    finally:
        os.chdir(original)


def _pil_to_tensor(frame: Image.Image, device: torch.device, size: int = 512) -> torch.Tensor:
    """PIL Image → (1, 3, H, W) float tensor on device, resized to size x size."""
    arr = np.array(frame.convert("RGB")).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t


class CLIPIQAPlusMetric:
    """Subjective Quality (Image) - CLIP-IQA+."""

    def __init__(self):
        import pyiqa
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else "cpu")
        self.metric = pyiqa.create_metric("clipiqa+").to(self.device)

    def compute(self, frames: List[Image.Image]) -> float:
        scores = []
        with torch.no_grad():
            for frame in frames:
                t = _pil_to_tensor(frame, self.device)
                scores.append(self.metric(t).item())
        return float(np.mean(scores)) if scores else None


class CLIPAestheticMetric:
    """Subjective Quality (Aesthetic) - CLIP Aesthetic."""

    def __init__(self):
        import pyiqa
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else "cpu")
        self.metric = pyiqa.create_metric("laion_aes").to(self.device)

    def compute(self, frames: List[Image.Image]) -> float:
        scores = []
        with torch.no_grad():
            for frame in frames:
                t = _pil_to_tensor(frame, self.device)
                scores.append(self.metric(t).item())
        return float(np.mean(scores)) if scores else None


class OpticalFlowMetric:
    """Motion Magnitude - Optical Flow (SEA-RAFT)."""

    def __init__(self):
        with _worldscore_cwd():
            from worldscore.benchmark.metrics.third_party.flow_metrics import OpticalFlowMetric as _WS
            self._metric = _WS()

    def compute(self, frames: List[Image.Image]) -> float:
        frame_paths = frames_to_file_paths(frames, temp_dir="/tmp/physion_optical_flow")
        with _worldscore_cwd():
            return float(self._metric._compute_scores(frame_paths))


class OpticalFlowAEPEMetric:
    """Photometric Consistency - Optical Flow AEPE (SEA-RAFT)."""

    def __init__(self):
        with _worldscore_cwd():
            from worldscore.benchmark.metrics.third_party.flow_aepe_metrics import (
                OpticalFlowAverageEndPointErrorMetric as _WS,
            )
            self._metric = _WS()

    def compute(self, frames: List[Image.Image]) -> float:
        frame_paths = frames_to_file_paths(frames, temp_dir="/tmp/physion_optical_flow_aepe")
        with _worldscore_cwd():
            return float(self._metric._compute_scores(frame_paths))


class StyleConsistencyMetric:
    """Style Consistency - Gram Matrix (VGG19)."""

    def __init__(self):
        from worldscore.benchmark.metrics.third_party.gram_matrix_metrics import (
            GramMatrixMetric as _WS,
        )
        self._metric = _WS()

    def compute(self, frames: List[Image.Image]) -> float:
        if len(frames) < 2:
            return 0.0
        frame_paths = frames_to_file_paths(frames, temp_dir="/tmp/physion_style_consistency")
        return float(self._metric._compute_scores(frame_paths[0], frame_paths[1:]))


class MotionSmoothnessMetric:
    """Motion Smoothness - VFIMamba (MSE, SSIM, LPIPS)."""

    def __init__(self):
        with _worldscore_cwd():
            from worldscore.benchmark.metrics.third_party.motion_smoothness_metrics import (
                MotionSmoothnessMetric as _WS,
            )
            self._metric = _WS()

    def compute(self, frames: List[Image.Image]) -> Tuple[float, float, float]:
        frame_paths = frames_to_file_paths(frames, temp_dir="/tmp/physion_motion_smoothness")
        with _worldscore_cwd():
            mse, ssim, lpips = self._metric._compute_scores(frame_paths)
        return float(mse), float(ssim), float(lpips)
