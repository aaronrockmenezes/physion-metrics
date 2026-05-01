#!/usr/bin/env python3
"""
Summary report from merged results JSON.

Usage:
    python scripts/summarize_results.py --input logs/Sora_2/.../merged_Sora_2.json
    python scripts/summarize_results.py --input merged.json --output summary.txt
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


# ── helpers ───────────────────────────────────────────────────────────────────

def vals(entries, key):
    return [e[key] for e in entries if e.get(key) is not None]

def stat(entries, key):
    v = vals(entries, key)
    if not v:
        return None
    return {"mean": np.mean(v), "std": np.std(v), "min": np.min(v), "max": np.max(v), "n": len(v)}

def fmt_stat(s):
    if s is None:
        return "  n/a"
    return f"  mean={s['mean']:.2f}  std={s['std']:.2f}  min={s['min']:.2f}  max={s['max']:.2f}  n={s['n']}"

def sep(char="=", width=70):
    return char * width

def pct_zero(entries, key, tol=0.01):
    v = vals(entries, key)
    if not v:
        return 0
    return 100 * sum(1 for x in v if abs(x) < tol) / len(v)


# ── main ──────────────────────────────────────────────────────────────────────

def summarize(data: list, out):
    def p(*args, **kwargs):
        print(*args, **kwargs)
        if out:
            print(*args, **kwargs, file=out)

    # ── per-model split ───────────────────────────────────────────────────────
    by_model = defaultdict(list)
    for r in data:
        by_model[r.get("model", "unknown")].append(r)

    models = sorted(by_model.keys())

    for model in models:
        entries = by_model[model]
        p(sep())
        p(f"MODEL: {model}   ({len(entries)} videos)")
        p(sep())

        # ── overall WorldScore ────────────────────────────────────────────────
        p("\n── WorldScore ──────────────────────────────────────────────────────")
        for key in ["worldscore", "worldscore_static", "worldscore_dynamic"]:
            s = stat(entries, key)
            p(f"  {key:<25}{fmt_stat(s)}")

        # ── per-metric (normalized) ───────────────────────────────────────────
        p("\n── Normalized aspects (0–100) ──────────────────────────────────────")
        for key in ["subjective_quality", "photometric_consistency",
                    "style_consistency", "motion_magnitude", "motion_smoothness"]:
            s = stat(entries, key)
            p(f"  {key:<25}{fmt_stat(s)}")

        # ── raw metrics ───────────────────────────────────────────────────────
        p("\n── Raw metrics ─────────────────────────────────────────────────────")
        raw_keys = [
            ("subjective_quality_image",    "CLIP-IQA+"),
            ("subjective_quality_aesthetic","CLIP Aesthetic"),
            ("motion_magnitude",            "Optical Flow (raw)"),
            ("photometric_consistency",     "AEPE (raw)"),
            ("style_consistency",           "Gram Matrix (raw)"),
            ("motion_smoothness_mse",       "MS-MSE"),
            ("motion_smoothness_ssim",      "MS-SSIM"),
            ("motion_smoothness_lpips",     "MS-LPIPS"),
        ]
        for key, label in raw_keys:
            s = stat(entries, key)
            p(f"  {label:<25}{fmt_stat(s)}")

        # ── glitch vs clean ───────────────────────────────────────────────────
        p("\n── Glitch vs Clean ─────────────────────────────────────────────────")
        glitched     = [e for e in entries if e.get("has_glitches") == 1]
        not_glitched = [e for e in entries if e.get("has_glitches") == 0]
        for label, grp in [("Clean  (has_glitches=0)", not_glitched),
                           ("Glitchy (has_glitches=1)", glitched)]:
            s = stat(grp, "worldscore")
            p(f"  {label:<28} n={len(grp):<5}{fmt_stat(s)}")

        # ── by severity ───────────────────────────────────────────────────────
        p("\n── By glitch severity ──────────────────────────────────────────────")
        by_sev = defaultdict(list)
        for e in entries:
            by_sev[e.get("glitch_severity", "?")].append(e)
        for sev in sorted(by_sev.keys()):
            grp = by_sev[sev]
            s = stat(grp, "worldscore")
            p(f"  severity={sev}  n={len(grp):<5}{fmt_stat(s)}")

        # ── by glitch category ────────────────────────────────────────────────
        p("\n── By glitch category ──────────────────────────────────────────────")
        by_cat = defaultdict(list)
        for e in entries:
            by_cat[e.get("glitch_category", "Unknown")].append(e)
        rows = []
        for cat, grp in by_cat.items():
            s = stat(grp, "worldscore")
            rows.append((s["mean"] if s else 0, cat, len(grp), s))
        for _, cat, n, s in sorted(rows):   # sort ascending mean (worst first)
            p(f"  {cat:<38} n={n:<5}{fmt_stat(s)}")

        # ── anomalies ─────────────────────────────────────────────────────────
        p("\n── Anomalies ───────────────────────────────────────────────────────")

        # metrics zeroed out (clamped to floor)
        for key in ["photometric_consistency", "subjective_quality", "motion_smoothness"]:
            pz = pct_zero(entries, key)
            if pz > 0:
                n_zero = sum(1 for e in entries if abs(e.get(key) or 1) < 0.01)
                p(f"  {key} = 0.0 in {n_zero} videos ({pz:.1f}%)")

        # data inconsistencies: has_glitches vs glitch_category
        bad = [e for e in entries
               if e.get("has_glitches") == 1 and e.get("glitch_category") == "No Issue"]
        if bad:
            p(f"  has_glitches=1 but glitch_category='No Issue': {len(bad)} videos")
            for e in bad[:5]:
                p(f"    {e['id']}  severity={e.get('glitch_severity')}")

        bad2 = [e for e in entries
                if e.get("has_glitches") == 0 and e.get("glitch_category") not in ("No Issue", "", None)]
        if bad2:
            p(f"  has_glitches=0 but non-null category: {len(bad2)} videos")

        # timing
        times = vals(entries, "processing_time_seconds")
        if times:
            total_h = sum(times) / 3600
            p(f"\n── Timing ──────────────────────────────────────────────────────────")
            p(f"  avg {np.mean(times):.1f}s/video  |  total compute {total_h:.1f}h  |  n={len(times)}")

        p("")

    p(sep())
    p(f"Total entries: {len(data)}")
    p(sep())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Merged results JSON")
    parser.add_argument("--output", default=None,  help="Save summary to text file")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    if not data:
        print("Empty JSON.")
        return

    out_f = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(out_path, "w")

    try:
        summarize(data, out_f)
    finally:
        if out_f:
            out_f.close()
            print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
