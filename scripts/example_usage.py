#!/usr/bin/env python3
"""Example usage of individual metric classes."""

from pathlib import Path

# Repo root → finds physion_metrics package
sys.path.insert(0, str(Path(__file__).parent.parent))
from physion_metrics.video_utils import extract_frames_from_video
from physion_metrics.metrics_wrapper import (
    CLIPIQAPlusMetric,
    CLIPAestheticMetric,
    OpticalFlowMetric,
    OpticalFlowAEPEMetric,
    MotionSmoothnessMetric,
)


def main():
    # Example: Process a single video
    video_path = "example_video.mp4"

    if not Path(video_path).exists():
        print(f"Video not found: {video_path}")
        print("Create a test video or provide a path to an existing one.")
        return

    print(f"Loading video: {video_path}")
    frames = extract_frames_from_video(video_path, skip=1)
    print(f"Extracted {len(frames)} frames")

    # Compute each metric
    print("\n" + "=" * 80)
    print("Computing metrics...")
    print("=" * 80)

    # 1. CLIP-IQA+
    print("\n1. CLIP-IQA+ (quality assessment)")
    metric = CLIPIQAPlusMetric()
    score = metric.compute(frames)
    print(f"   Score: {score:.4f} (range 0-1, higher is better)")

    # 2. CLIP Aesthetic
    print("\n2. CLIP Aesthetic (aesthetic quality)")
    metric = CLIPAestheticMetric()
    score = metric.compute(frames)
    print(f"   Score: {score:.4f} (range 0-10, higher is better)")

    # 3. Optical Flow
    print("\n3. Optical Flow (motion magnitude)")
    metric = OpticalFlowMetric()
    score = metric.compute(frames)
    print(f"   Score: {score:.4f} (higher = more motion)")

    # 4. Optical Flow AEPE
    print("\n4. Optical Flow AEPE (photometric consistency)")
    metric = OpticalFlowAEPEMetric()
    score = metric.compute(frames)
    print(f"   Score: {score:.4f} (lower is better)")

    # 5. Motion Smoothness
    print("\n5. Motion Smoothness (temporal smoothness)")
    metric = MotionSmoothnessMetric()
    mse, ssim, lpips = metric.compute(frames)
    print(f"   MSE:   {mse:.4f} (lower is better)")
    print(f"   SSIM:  {ssim:.4f} (higher is better, range 0-1)")
    print(f"   LPIPS: {lpips:.4f} (lower is better)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
