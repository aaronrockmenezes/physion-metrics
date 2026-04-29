#!/usr/bin/env python3
"""
HPC-optimized metric computation for Physion Eval dataset.
5 video-only metrics optimized for GPU compute (H100/B200/A5000/etc).
Paper: WorldScore - A Unified Evaluation Benchmark for World Generation (2504.00983)
"""

import sys
from pathlib import Path
# Repo root → finds physion_metrics package
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
import numpy as np
from tqdm import tqdm
import time

# Add WorldScore to path

from physion_metrics.video_utils import extract_frames_from_video
from physion_metrics.metrics_wrapper import (
    CLIPIQAPlusMetric,
    CLIPAestheticMetric,
    OpticalFlowMetric,
    OpticalFlowAEPEMetric,
    MotionSmoothnessMetric,
    StyleConsistencyMetric,
)
from physion_metrics.score_utils import compute_worldscore


def get_device():
    """Get CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Use compute_metrics.py for CPU")
    return torch.device("cuda")


def clear_cache():
    """Clear GPU cache between videos."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_all_metrics(frames):
    """Compute all 5 metrics for frames."""
    results = {}

    # 1. Subjective Quality (Image) - CLIP-IQA+
    try:
        metric = CLIPIQAPlusMetric()
        score = metric.compute(frames)
        results["subjective_quality_image"] = float(score) if score else None
    except Exception as e:
        print(f"    CLIP-IQA+ error: {e}")
        results["subjective_quality_image"] = None

    # 2. Subjective Quality (Aesthetic) - CLIP Aesthetic
    try:
        metric = CLIPAestheticMetric()
        score = metric.compute(frames)
        results["subjective_quality_aesthetic"] = float(score) if score else None
    except Exception as e:
        print(f"    CLIP Aesthetic error: {e}")
        results["subjective_quality_aesthetic"] = None

    # 3. Motion Magnitude - Optical Flow
    try:
        metric = OpticalFlowMetric()
        score = metric.compute(frames)
        results["motion_magnitude"] = float(score) if score else None
    except Exception as e:
        print(f"    Motion Magnitude error: {e}")
        results["motion_magnitude"] = None

    # 4. Photometric Consistency - Optical Flow AEPE
    try:
        metric = OpticalFlowAEPEMetric()
        score = metric.compute(frames)
        results["photometric_consistency"] = float(score) if score else None
    except Exception as e:
        print(f"    Photometric Consistency error: {e}")
        results["photometric_consistency"] = None

    # 5. Style Consistency - Gram Matrix
    try:
        metric = StyleConsistencyMetric()
        score = metric.compute(frames)
        results["style_consistency"] = float(score) if score else None
    except Exception as e:
        print(f"    Style Consistency error: {e}")
        results["style_consistency"] = None

    # 6. Motion Smoothness - VFIMamba (MSE/SSIM/LPIPS)
    try:
        metric = MotionSmoothnessMetric()
        mse, ssim, lpips = metric.compute(frames)
        results["motion_smoothness_mse"] = float(mse) if mse else None
        results["motion_smoothness_ssim"] = float(ssim) if ssim else None
        results["motion_smoothness_lpips"] = float(lpips) if lpips else None
    except Exception as e:
        print(f"    Motion Smoothness error: {e}")
        results["motion_smoothness_mse"] = None
        results["motion_smoothness_ssim"] = None
        results["motion_smoothness_lpips"] = None

    clear_cache()
    return results


def process_video(video_path, frame_skip=1, max_frames=None):
    """Process single video."""
    start_time = time.time()
    print(f"\nProcessing: {Path(video_path).name}")

    try:
        frames = extract_frames_from_video(video_path, skip=frame_skip, max_frames=max_frames)
        print(f"  Extracted {len(frames)} frames")
    except Exception as e:
        print(f"  Error loading video: {e}")
        return None

    results = {
        "video": Path(video_path).name,
        "frames": len(frames),
    }

    metrics = compute_all_metrics(frames)
    results.update(metrics)

    # Compute normalized WorldScore
    ws = compute_worldscore(results)
    results.update(ws)

    elapsed = time.time() - start_time
    results["processing_time_seconds"] = elapsed
    print(f"  Completed in {elapsed:.2f}s")
    if "worldscore" in ws:
        print(f"  WorldScore: {ws['worldscore']:.4f}  (static={ws.get('worldscore_static', 'N/A'):.4f}, dynamic={ws.get('worldscore_dynamic', 'N/A'):.4f})")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="HPC: Compute 5 video-only metrics (requires CUDA)")
    parser.add_argument("--video", type=str, help="Single video file")
    parser.add_argument("--video-dir", type=str, help="Directory with videos")
    parser.add_argument("--output", type=str, default="physion_metrics_hpc.json")
    parser.add_argument("--frame-skip", type=int, default=1, help="Frame skip factor")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames per video")
    parser.add_argument("--max-videos", type=int, default=None, help="Max videos to process")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers")

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print("=" * 80)
    print("WorldScore: 5 Video-Only Metrics (HPC Optimized)")
    print("=" * 80)
    print("1. Subjective Quality (Image)      [CLIP-IQA+]")
    print("2. Subjective Quality (Aesthetic)  [CLIP Aesthetic]")
    print("3. Motion Magnitude                [Optical Flow]")
    print("4. Photometric Consistency         [Optical Flow AEPE]")
    print("5. Style Consistency               [Gram Matrix + VGG19]")
    print("6. Motion Smoothness (MSE/SSIM/LPIPS) [VFIMamba]")
    print("=" * 80)

    results = []
    start_total = time.time()

    if args.video:
        result = process_video(args.video, frame_skip=args.frame_skip, max_frames=args.max_frames)
        if result:
            results.append(result)
    elif args.video_dir:
        video_dir = Path(args.video_dir)
        videos = sorted(
            list(video_dir.glob("*.mp4")) +
            list(video_dir.glob("*.avi")) +
            list(video_dir.glob("*.mov"))
        )
        print(f"Found {len(videos)} videos\n")

        if args.max_videos:
            videos = videos[:args.max_videos]

        for video_path in tqdm(videos, desc="Videos"):
            result = process_video(str(video_path), frame_skip=args.frame_skip, max_frames=args.max_frames)
            if result:
                results.append(result)
    else:
        print("Provide --video or --video-dir")
        return

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - start_total

    print(f"\n\nResults saved to {output_path}")
    print(f"Total processing time: {total_time:.2f}s")
    if results:
        print(f"Average per video: {total_time / len(results):.2f}s")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results:
        subj_img = [r["subjective_quality_image"] for r in results if r.get("subjective_quality_image")]
        subj_aes = [r["subjective_quality_aesthetic"] for r in results if r.get("subjective_quality_aesthetic")]
        motion_mag = [r["motion_magnitude"] for r in results if r.get("motion_magnitude")]
        photo_cons = [r["photometric_consistency"] for r in results if r.get("photometric_consistency")]
        style_cons = [r["style_consistency"] for r in results if r.get("style_consistency")]
        mse = [r["motion_smoothness_mse"] for r in results if r.get("motion_smoothness_mse")]
        ssim = [r["motion_smoothness_ssim"] for r in results if r.get("motion_smoothness_ssim")]
        lpips = [r["motion_smoothness_lpips"] for r in results if r.get("motion_smoothness_lpips")]

        if subj_img:
            print(f"Subjective Quality (Image):     mean={np.mean(subj_img):.4f}, std={np.std(subj_img):.4f}")
        if subj_aes:
            print(f"Subjective Quality (Aesthetic): mean={np.mean(subj_aes):.4f}, std={np.std(subj_aes):.4f}")
        if motion_mag:
            print(f"Motion Magnitude:               mean={np.mean(motion_mag):.4f}, std={np.std(motion_mag):.4f}")
        if photo_cons:
            print(f"Photometric Consistency:        mean={np.mean(photo_cons):.4f}, std={np.std(photo_cons):.4f}")
        if style_cons:
            print(f"Style Consistency:              mean={np.mean(style_cons):.4f}, std={np.std(style_cons):.4f}")
        if mse:
            print(f"Motion Smoothness (MSE):        mean={np.mean(mse):.4f}, std={np.std(mse):.4f}")
        if ssim:
            print(f"Motion Smoothness (SSIM):       mean={np.mean(ssim):.4f}, std={np.std(ssim):.4f}")
        if lpips:
            print(f"Motion Smoothness (LPIPS):      mean={np.mean(lpips):.4f}, std={np.std(lpips):.4f}")

        # WorldScore
        ws_static  = [r["worldscore_static"]  for r in results if r.get("worldscore_static")]
        ws_dynamic = [r["worldscore_dynamic"] for r in results if r.get("worldscore_dynamic")]
        ws_all     = [r["worldscore"]         for r in results if r.get("worldscore")]

        if ws_static or ws_dynamic or ws_all:
            print()
            print("=" * 80)
            print("WORLDSCORE (normalized 0-1)")
            print("=" * 80)
        if ws_static:
            print(f"WorldScore-Static:              mean={np.mean(ws_static):.4f}, std={np.std(ws_static):.4f}")
        if ws_dynamic:
            print(f"WorldScore-Dynamic:             mean={np.mean(ws_dynamic):.4f}, std={np.std(ws_dynamic):.4f}")
        if ws_all:
            print(f"WorldScore:                     mean={np.mean(ws_all):.4f}, std={np.std(ws_all):.4f}")


if __name__ == "__main__":
    main()
