#!/usr/bin/env python3
"""
Mac M4 test: CLIP metrics only (minimal dependencies).
Paper: WorldScore - A Unified Evaluation Benchmark for World Generation (2504.00983)
"""

import sys
from pathlib import Path
import json
import torch
import pyiqa
from PIL import Image
import numpy as np
from tqdm import tqdm
from video_utils import extract_frames_from_video


def get_device():
    """Get best device (MPS for Mac, CUDA if available, CPU fallback)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_clip_iqa_plus(frames, device):
    """CLIP-IQA+ score (Subjective Quality Image)."""
    metric = pyiqa.create_metric("clipiqa+").to(device)
    scores = []

    for frame in frames:
        img_tensor = torch.from_numpy(np.array(frame)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        img_tensor = torch.nn.functional.interpolate(
            img_tensor,
            size=(512, 512),
            mode="bilinear",
            align_corners=False
        )

        with torch.no_grad():
            score = metric(img_tensor).item()
        scores.append(score)

    return sum(scores) / len(scores) if scores else None


def compute_clip_aesthetic(frames, device):
    """CLIP Aesthetic score (Subjective Quality Aesthetic)."""
    metric = pyiqa.create_metric("laion_aes").to(device)
    scores = []

    for frame in frames:
        img_tensor = torch.from_numpy(np.array(frame)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        img_tensor = torch.nn.functional.interpolate(
            img_tensor,
            size=(512, 512),
            mode="bilinear",
            align_corners=False
        )

        with torch.no_grad():
            score = metric(img_tensor).item()
        scores.append(score)

    return sum(scores) / len(scores) if scores else None


def process_video(video_path, device):
    """Process single video."""
    print(f"\nProcessing: {Path(video_path).name}")

    try:
        frames = extract_frames_from_video(video_path, skip=2, max_frames=50)
        print(f"  Extracted {len(frames)} frames")
    except Exception as e:
        print(f"  Error loading video: {e}")
        return None

    results = {
        "video": Path(video_path).name,
        "frames": len(frames),
    }

    # CLIP-IQA+
    try:
        print("  Computing CLIP-IQA+ (Subjective Quality - Image)...")
        score = compute_clip_iqa_plus(frames, device)
        results["subjective_quality_image"] = float(score) if score else None
        print(f"    Score: {score:.4f}" if score else "    Failed")
    except Exception as e:
        print(f"    Error: {e}")
        results["subjective_quality_image"] = None

    # CLIP Aesthetic
    try:
        print("  Computing CLIP Aesthetic (Subjective Quality - Aesthetic)...")
        score = compute_clip_aesthetic(frames, device)
        results["subjective_quality_aesthetic"] = float(score) if score else None
        print(f"    Score: {score:.4f}" if score else "    Failed")
    except Exception as e:
        print(f"    Error: {e}")
        results["subjective_quality_aesthetic"] = None

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Mac M4 test: CLIP metrics only")
    parser.add_argument("--video", type=str, help="Single video file")
    parser.add_argument("--video-dir", type=str, help="Directory with videos")
    parser.add_argument("--output", type=str, default="test_results_mac.json")

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    print("=" * 80)
    print("WorldScore: Mac M4 Test (CLIP Metrics Only)")
    print("=" * 80)
    print("1. Subjective Quality (Image)      [CLIP-IQA+]")
    print("2. Subjective Quality (Aesthetic)  [CLIP Aesthetic]")
    print("=" * 80)

    results = []

    if args.video:
        result = process_video(args.video, device)
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

        for video_path in tqdm(videos, desc="Videos"):
            result = process_video(str(video_path), device)
            if result:
                results.append(result)
    else:
        print("Provide --video or --video-dir")
        return

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to {args.output}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results:
        subj_img = [r["subjective_quality_image"] for r in results if r.get("subjective_quality_image")]
        subj_aes = [r["subjective_quality_aesthetic"] for r in results if r.get("subjective_quality_aesthetic")]

        if subj_img:
            print(f"Subjective Quality (Image):     mean={np.mean(subj_img):.4f}, std={np.std(subj_img):.4f}")

        if subj_aes:
            print(f"Subjective Quality (Aesthetic): mean={np.mean(subj_aes):.4f}, std={np.std(subj_aes):.4f}")


if __name__ == "__main__":
    main()
