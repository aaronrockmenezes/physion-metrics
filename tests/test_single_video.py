#!/usr/bin/env python3
"""
Quick sanity test: run all metrics on a single video and print results.
Usage:
    python tests/test_single_video.py --video videos/Potato_Catches_Fire_On_Beach.mp4
    python tests/test_single_video.py --video /path/to/any.mp4
"""

import sys
from pathlib import Path
import argparse
import json
import time

# Repo root → finds physion_metrics package
sys.path.insert(0, str(Path(__file__).parent.parent))

from physion_metrics.video_utils import extract_frames_from_video
from physion_metrics.score_utils import compute_worldscore


def test_video(video_path: str, frame_skip: int = 1, max_frames: int = None):
    import torch

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Video:  {video_path}")
    print("=" * 60)

    # --- Frames ---
    t0 = time.time()
    frames = extract_frames_from_video(video_path, skip=frame_skip, max_frames=max_frames)
    print(f"Frames: {len(frames)} ({time.time()-t0:.2f}s)")
    print()

    results = {}

    # --- CLIP-IQA+ ---
    try:
        t0 = time.time()
        from physion_metrics.metrics_wrapper import CLIPIQAPlusMetric
        score = CLIPIQAPlusMetric().compute(frames)
        results["subjective_quality_image"] = score
        print(f"[✓] CLIP-IQA+              {score:.4f}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"[✗] CLIP-IQA+              {e}")

    # --- CLIP Aesthetic ---
    try:
        t0 = time.time()
        from physion_metrics.metrics_wrapper import CLIPAestheticMetric
        score = CLIPAestheticMetric().compute(frames)
        results["subjective_quality_aesthetic"] = score
        print(f"[✓] CLIP Aesthetic         {score:.4f}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"[✗] CLIP Aesthetic         {e}")

    # --- Optical Flow (Motion Magnitude) ---
    try:
        t0 = time.time()
        from physion_metrics.metrics_wrapper import OpticalFlowMetric
        score = OpticalFlowMetric().compute(frames)
        results["motion_magnitude"] = score
        print(f"[✓] Motion Magnitude       {score:.4f}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"[✗] Motion Magnitude       {e}")

    # --- Optical Flow AEPE (Photometric Consistency) ---
    try:
        t0 = time.time()
        from physion_metrics.metrics_wrapper import OpticalFlowAEPEMetric
        score = OpticalFlowAEPEMetric().compute(frames)
        results["photometric_consistency"] = score
        print(f"[✓] Photometric Consistency {score:.4f}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"[✗] Photometric Consistency {e}")

    # --- Style Consistency ---
    try:
        t0 = time.time()
        from physion_metrics.metrics_wrapper import StyleConsistencyMetric
        score = StyleConsistencyMetric().compute(frames)
        results["style_consistency"] = score
        print(f"[✓] Style Consistency      {score:.4f}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"[✗] Style Consistency      {e}")

    # --- Motion Smoothness ---
    try:
        t0 = time.time()
        from physion_metrics.metrics_wrapper import MotionSmoothnessMetric
        mse, ssim, lpips = MotionSmoothnessMetric().compute(frames)
        results["motion_smoothness_mse"]   = mse
        results["motion_smoothness_ssim"]  = ssim
        results["motion_smoothness_lpips"] = lpips
        print(f"[✓] Motion Smoothness      MSE={mse:.4f} SSIM={ssim:.4f} LPIPS={lpips:.4f}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"[✗] Motion Smoothness      {e}")

    # --- WorldScore ---
    print()
    print("=" * 60)
    ws = compute_worldscore(results)
    for k, v in ws.items():
        print(f"  {k:<30} {v:.4f}")

    # Save
    out = Path("tests/test_output.json")
    with open(out, "w") as f:
        json.dump({**results, **ws}, f, indent=2)
    print(f"\nSaved → {out}")
    return results, ws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      type=str, default="videos/Potato_Catches_Fire_On_Beach.mp4")
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    test_video(args.video, args.frame_skip, args.max_frames)


if __name__ == "__main__":
    main()
