#!/usr/bin/env python3
"""
Merge per-shard JSON results into a single file and print summary.

Usage:
    python merge_results.py --pattern "results/Sora_2_metrics_shard*.json" --output results/Sora_2_metrics.json
    python merge_results.py --pattern "results/*_shard*.json" --output results/all_models.json
"""

import json
import glob
import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge shard results into single JSON")
    parser.add_argument("--pattern", type=str, required=True, help="Glob pattern for shard files")
    parser.add_argument("--output",  type=str, required=True, help="Output merged JSON path")
    args = parser.parse_args()

    shard_files = sorted(glob.glob(args.pattern))
    if not shard_files:
        print(f"No files matched: {args.pattern}")
        return

    print(f"Merging {len(shard_files)} shard files...")

    all_results = []
    for f in shard_files:
        with open(f) as fh:
            data = json.load(fh)
        all_results.extend(data)
        print(f"  {f}: {len(data)} entries")

    # Sort by id
    all_results.sort(key=lambda r: r.get("id", ""))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nMerged {len(all_results)} entries → {output_path}")

    if not all_results:
        return

    # Summary per model
    from collections import defaultdict
    by_model = defaultdict(list)
    for r in all_results:
        by_model[r.get("model", "unknown")].append(r)

    for model, entries in sorted(by_model.items()):
        print(f"\n{'='*60}")
        print(f"Model: {model} ({len(entries)} videos)")
        print(f"{'='*60}")

        def stat(key):
            vals = [e[key] for e in entries if e.get(key) is not None]
            if vals:
                print(f"  {key:<40} mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  n={len(vals)}")

        stat("subjective_quality_image")
        stat("subjective_quality_aesthetic")
        stat("motion_magnitude")
        stat("photometric_consistency")
        stat("style_consistency")
        stat("motion_smoothness_mse")
        stat("motion_smoothness_ssim")
        stat("motion_smoothness_lpips")
        print()
        stat("worldscore_static")
        stat("worldscore_dynamic")
        stat("worldscore")

        ws = [e["worldscore"] for e in entries if e.get("worldscore") is not None]
        if ws:
            glitched     = [e["worldscore"] for e in entries if e.get("worldscore") and e.get("has_glitches") == 1]
            not_glitched = [e["worldscore"] for e in entries if e.get("worldscore") and e.get("has_glitches") == 0]
            print(f"\n  WorldScore by glitch:")
            if glitched:
                print(f"    has_glitches=1: mean={np.mean(glitched):.4f}  n={len(glitched)}")
            if not_glitched:
                print(f"    has_glitches=0: mean={np.mean(not_glitched):.4f}  n={len(not_glitched)}")


if __name__ == "__main__":
    main()
