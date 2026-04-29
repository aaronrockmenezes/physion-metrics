"""
Wrapper classes for WorldScore metrics adapted for video-only evaluation.
"""

import sys
from pathlib import Path
from typing import List, Union, Tuple
import torch
import numpy as np
from PIL import Image
import cv2
import torch.nn.functional as F

# Add local WorldScore to path (used if not pip-installed)
WORLDSCORE_PATH = Path(__file__).parent.parent / "WorldScore"
sys.path.insert(0, str(WORLDSCORE_PATH))

# Inject third_party paths before worldscore imports.
# Handles both: (a) local WorldScore repo, (b) pip-installed worldscore package.
def _inject_third_party_paths():
    import importlib.util
    subdirs = ["droid_slam", "groundingdino", "sam2", "VFIMamba", "SEA-RAFT"]

    # Try pip-installed worldscore location first
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

# Import only what we need, avoid lietorch dependencies
from worldscore.benchmark.metrics.base_metrics import IQAPytorchMetric, BaseMetric
from worldscore.common.gpu_utils import get_torch_device
from video_utils import frames_to_file_paths

# Lazy imports to avoid loading metrics/__init__.py which imports lietorch
def _import_optical_flow_metric():
    from worldscore.benchmark.metrics.third_party.flow_metrics import (
        OpticalFlowMetric as WSOpticalFlowMetric,
    )
    return WSOpticalFlowMetric

def _import_optical_flow_aepe_metric():
    from worldscore.benchmark.metrics.third_party.flow_aepe_metrics import (
        OpticalFlowAverageEndPointErrorMetric as WSOpticalFlowAEPEMetric,
    )
    return WSOpticalFlowAEPEMetric

def _import_motion_smoothness_metric():
    from worldscore.benchmark.metrics.third_party.motion_smoothness_metrics import (
        MotionSmoothnessMetric as WSMotionSmoothnessMetric,
    )
    return WSMotionSmoothnessMetric


class CLIPIQAPlusMetric(IQAPytorchMetric):
    """CLIP-IQA+ wrapper for video frames."""

    def __init__(self):
        super().__init__(metric_name="clipiqa+")

    def compute(self, frames: List[Image.Image]) -> float:
        """Compute CLIP-IQA+ score for frames."""
        imgs = self._process_image(frames)
        scores = []
        for img in imgs:
            score = self._metric(img.unsqueeze(0)).item()
            scores.append(score)
        return np.mean(scores)


class CLIPAestheticMetric(IQAPytorchMetric):
    """CLIP Aesthetic wrapper for video frames."""

    def __init__(self):
        super().__init__(metric_name="laion_aes")

    def compute(self, frames: List[Image.Image]) -> float:
        """Compute CLIP Aesthetic score for frames."""
        imgs = self._process_image(frames)
        scores = []
        for img in imgs:
            score = self._metric(img.unsqueeze(0)).item()
            scores.append(score)
        return np.mean(scores)


class OpticalFlowMetric(BaseMetric):
    """Optical flow magnitude (motion magnitude) wrapper."""

    def __init__(self):
        super().__init__()
        WSOpticalFlowMetric = _import_optical_flow_metric()
        self.ws_metric = WSOpticalFlowMetric()

    def compute(self, frames: List[Image.Image]) -> float:
        """Compute optical flow magnitude."""
        frame_paths = frames_to_file_paths(frames, temp_dir="/tmp/physion_optical_flow")
        score = self.ws_metric._compute_scores(frame_paths)
        return score


class OpticalFlowAEPEMetric(BaseMetric):
    """Optical flow AEPE (Average End-Point Error) wrapper."""

    def __init__(self):
        super().__init__()
        # Import here to avoid early loading
        from worldscore.benchmark.metrics.third_party.flow_aepe_metrics import (
            OpticalFlowAverageEndPointErrorMetric as WSOpticalFlowAEPEMetric,
        )

        self.ws_metric = WSOpticalFlowAEPEMetric()

    def compute(self, frames: List[Image.Image]) -> float:
        """Compute optical flow AEPE."""
        frame_paths = frames_to_file_paths(frames, temp_dir="/tmp/physion_optical_flow_aepe")
        score = self.ws_metric._compute_scores(frame_paths)
        return score


class MotionSmoothnessMetric(BaseMetric):
    """Motion smoothness (frame interpolation quality) wrapper."""

    def __init__(self):
        super().__init__()
        # Import here to avoid early loading
        from worldscore.benchmark.metrics.third_party.motion_smoothness_metrics import (
            MotionSmoothnessMetric as WSMotionSmoothnessMetric,
        )

        self.ws_metric = WSMotionSmoothnessMetric()

    def compute(self, frames: List[Image.Image]) -> Tuple[float, float, float]:
        """
        Compute motion smoothness metrics.

        Returns:
            Tuple of (MSE, SSIM, LPIPS)
        """
        frame_paths = frames_to_file_paths(frames, temp_dir="/tmp/physion_motion_smoothness")
        # Motion smoothness expects even/odd frame pairing
        # It will subsample automatically
        scores = self.ws_metric._compute_scores(frame_paths)
        # Returns (mse, ssim, lpips)
        return scores


class StyleConsistencyMetric(BaseMetric):
    """Style consistency (Gram matrix) wrapper for video frames."""

    def __init__(self):
        super().__init__()
        from worldscore.benchmark.metrics.third_party.gram_matrix_metrics import (
            GramMatrixMetric as WSGramMatrixMetric,
        )
        self.ws_metric = WSGramMatrixMetric()

    def compute(self, frames: List[Image.Image]) -> float:
        """Compute style consistency between first and last frame."""
        if len(frames) < 2:
            return 0.0

        frame_paths = frames_to_file_paths(frames, temp_dir="/tmp/physion_style_consistency")
        # Compare first frame (style reference) against all frames
        score = self.ws_metric._compute_scores(frame_paths[0], frame_paths[1:])
        return float(score)
