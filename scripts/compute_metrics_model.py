#!/usr/bin/env python3
"""
Compute WorldScore metrics for a specific model from Physion-Eval dataset.
Reads metadata from Physion JSON, filters by model, runs metrics on matching videos.

Usage:
    WORLDSCORE_ROOT=/path/to/WorldScore python compute_metrics_model.py \
        --json Physion_Eval_20260322.json \
        --video-dir /path/to/videos \
        --model "Sora 2" \
        --output sora2_metrics.json
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

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".MP4", ".AVI", ".MOV"]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_video(video_dir: Path, filename: str) -> Path | None:
    """Find video file matching filename (with any extension)."""
    for ext in VIDEO_EXTENSIONS:
        p = video_dir / (filename + ext)
        if p.exists():
            return p
    # Glob fallback (case issues, extra suffix)
    matches = list(video_dir.glob(f"{filename}*"))
    matches = [m for m in matches if m.suffix.lower() in [e.lower() for e in VIDEO_EXTENSIONS]]
    return matches[0] if matches else None


def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_metrics():
    """Load all metric models once. Pass returned dict to compute_all_metrics()."""
    metrics = {}
    loaders = {
        "clip_iqa":    CLIPIQAPlusMetric,
        "clip_aes":    CLIPAestheticMetric,
        "opt_flow":    OpticalFlowMetric,
        "opt_flow_aepe": OpticalFlowAEPEMetric,
        "style":       StyleConsistencyMetric,
        "smoothness":  MotionSmoothnessMetric,
    }
    for key, cls in loaders.items():
        try:
            metrics[key] = cls()
            print(f"  [loaded] {cls.__name__}")
        except Exception as e:
            print(f"  [failed] {cls.__name__}: {e}")
            metrics[key] = None
    return metrics


def compute_all_metrics(frames, metrics: dict):
    results = {}

    try:
        if metrics["clip_iqa"]:
            results["subjective_quality_image"] = float(metrics["clip_iqa"].compute(frames))
    except Exception as e:
        print(f"    CLIP-IQA+ error: {e}")
        results["subjective_quality_image"] = None

    try:
        if metrics["clip_aes"]:
            results["subjective_quality_aesthetic"] = float(metrics["clip_aes"].compute(frames))
    except Exception as e:
        print(f"    CLIP Aesthetic error: {e}")
        results["subjective_quality_aesthetic"] = None

    try:
        if metrics["opt_flow"]:
            results["motion_magnitude"] = float(metrics["opt_flow"].compute(frames))
    except Exception as e:
        print(f"    Motion Magnitude error: {e}")
        results["motion_magnitude"] = None

    try:
        if metrics["opt_flow_aepe"]:
            results["photometric_consistency"] = float(metrics["opt_flow_aepe"].compute(frames))
    except Exception as e:
        print(f"    Photometric Consistency error: {e}")
        results["photometric_consistency"] = None

    try:
        if metrics["style"]:
            results["style_consistency"] = float(metrics["style"].compute(frames))
    except Exception as e:
        print(f"    Style Consistency error: {e}")
        results["style_consistency"] = None

    try:
        if metrics["smoothness"]:
            mse, ssim, lpips = metrics["smoothness"].compute(frames)
            results["motion_smoothness_mse"] = float(mse)
            results["motion_smoothness_ssim"] = float(ssim)
            results["motion_smoothness_lpips"] = float(lpips)
    except Exception as e:
        print(f"    Motion Smoothness error: {e}")
        results["motion_smoothness_mse"] = None
        results["motion_smoothness_ssim"] = None
        results["motion_smoothness_lpips"] = None

    clear_cache()
    return results


def process_entry(entry: dict, video_dir: Path, frame_skip: int, max_frames: int | None, metrics: dict):
    filename = entry["filename"]
    video_path = find_video(video_dir, filename)

    if video_path is None:
        print(f"  [SKIP] Video not found: {filename}")
        return None

    start = time.time()
    print(f"\n[{entry['id']}] {filename}")

    try:
        frames = extract_frames_from_video(str(video_path), skip=frame_skip, max_frames=max_frames)
        print(f"  Extracted {len(frames)} frames")
    except Exception as e:
        print(f"  Error loading video: {e}")
        return None

    raw_metrics = compute_all_metrics(frames, metrics)
    ws = compute_worldscore(raw_metrics)

    result = {
        # Physion metadata
        "id": entry["id"],
        "filename": filename,
        "model": entry["model"],
        "has_glitches": entry["has_glitches"],
        "glitch_severity": entry["glitch_severity"],
        "glitch_category": entry.get("Glitch_Category", ""),
        # Raw metrics
        "frames": len(frames),
        **raw_metrics,
        # Normalized WorldScore
        **ws,
        "processing_time_seconds": time.time() - start,
    }

    ws_val = ws.get("worldscore")
    if ws_val is not None:
        print(f"  WorldScore: {ws_val:.4f}  (static={ws.get('worldscore_static', 0):.4f}, dynamic={ws.get('worldscore_dynamic', 0):.4f})")

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute WorldScore for a specific model in Physion-Eval")
    parser.add_argument("--json",       type=str, required=True,  help="Path to Physion_Eval JSON file")
    parser.add_argument("--video-dir",  type=str, required=True,  help="Directory containing videos")
    parser.add_argument("--model",      type=str, required=True,  help='Model name e.g. "Sora 2", "Kling 2.5"')
    parser.add_argument("--output",     type=str, default=None,   help="Output JSON (default: <model>_metrics.json)")
    parser.add_argument("--frame-skip", type=int, default=1,      help="Frame skip factor (default: 1 = all frames)")
    parser.add_argument("--max-frames", type=int, default=None,   help="Max frames per video")
    parser.add_argument("--max-videos", type=int, default=None,   help="Max videos to process")
    parser.add_argument("--shard",      type=int, default=0,      help="Shard index (0-based, for job arrays)")
    parser.add_argument("--num-shards", type=int, default=1,      help="Total number of shards")

    args = parser.parse_args()

    # Load and filter JSON
    with open(args.json) as f:
        data = json.load(f)

    entries = [d for d in data if d["model"] == args.model]
    if not entries:
        available = sorted(set(d["model"] for d in data))
        print(f"Model '{args.model}' not found. Available: {available}")
        return

    if args.max_videos:
        entries = entries[:args.max_videos]

    # Shard: slice entries for this job array task
    if args.num_shards > 1:
        total = len(entries)
        shard_size = (total + args.num_shards - 1) // args.num_shards
        start_idx = args.shard * shard_size
        end_idx = min(start_idx + shard_size, total)
        entries = entries[start_idx:end_idx]
        print(f"Shard {args.shard}/{args.num_shards}: entries {start_idx}-{end_idx-1} ({len(entries)} videos)")

    # If --output given explicitly (e.g. from worker.sh), use it as-is.
    # Otherwise auto-generate with shard suffix.
    if args.output:
        output_path = Path(args.output)
    else:
        base = f"{args.model.replace(' ', '_')}_metrics"
        if args.num_shards > 1:
            output_path = Path(f"{base}_shard{args.shard:04d}.json")
        else:
            output_path = Path(f"{base}.json")
    video_dir = Path(args.video_dir)

    device = get_device()
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Videos: {len(entries)}")
    print(f"Video dir: {video_dir}")
    print(f"Output: {output_path}")
    print("=" * 80)

    # Load all models once — reused across every video in this shard
    print("\nLoading metric models...")
    metrics = load_metrics()
    print()

    results = []
    start_total = time.time()
    not_found = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for entry in tqdm(entries, desc=f"{args.model}"):
        result = process_entry(entry, video_dir, args.frame_skip, args.max_frames, metrics)
        if result:
            results.append(result)
            # Incremental save — crash-safe: write after each video
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
        else:
            not_found += 1

    total_time = time.time() - start_total

    print(f"\n\nFinal results saved to {output_path}")
    print(f"Processed: {len(results)} | Skipped (not found): {not_found}")
    print(f"Total time: {total_time:.2f}s | Avg per video: {total_time / max(len(results), 1):.2f}s")

    if not results:
        return

    # Summary
    print("\n" + "=" * 80)
    print(f"SUMMARY — {args.model}")
    print("=" * 80)

    def mean_std(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        if vals:
            print(f"{key:<40} mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, n={len(vals)}")

    mean_std("subjective_quality_image")
    mean_std("subjective_quality_aesthetic")
    mean_std("motion_magnitude")
    mean_std("photometric_consistency")
    mean_std("style_consistency")
    mean_std("motion_smoothness_mse")
    mean_std("motion_smoothness_ssim")
    mean_std("motion_smoothness_lpips")

    print()
    print("=" * 80)
    print("WORLDSCORE (normalized 0-1)")
    print("=" * 80)
    mean_std("worldscore_static")
    mean_std("worldscore_dynamic")
    mean_std("worldscore")

    # Breakdown by glitch category
    ws_vals = [r for r in results if r.get("worldscore") is not None]
    if ws_vals:
        print()
        print("WorldScore by glitch presence:")
        glitched     = [r["worldscore"] for r in ws_vals if r["has_glitches"] == 1]
        not_glitched = [r["worldscore"] for r in ws_vals if r["has_glitches"] == 0]
        if glitched:
            print(f"  has_glitches=1: mean={np.mean(glitched):.4f}, n={len(glitched)}")
        if not_glitched:
            print(f"  has_glitches=0: mean={np.mean(not_glitched):.4f}, n={len(not_glitched)}")


if __name__ == "__main__":
    main()
