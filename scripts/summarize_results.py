#!/usr/bin/env python3
"""
Summary report from merged results JSON.

Usage:
    python scripts/summarize_results.py --input merged_Sora_2.json
    python scripts/summarize_results.py --input merged.json --output summary.txt --output-json summary.json
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
    return {
        "mean": float(np.mean(v)),
        "std":  float(np.std(v)),
        "min":  float(np.min(v)),
        "max":  float(np.max(v)),
        "n":    int(len(v)),
    }

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


# ── core ──────────────────────────────────────────────────────────────────────

def build_model_summary(entries: list) -> dict:
    """Build structured summary dict for one model's entries."""
    s = {}

    # WorldScore
    s["worldscore"] = {k: stat(entries, k)
                       for k in ["worldscore", "worldscore_static", "worldscore_dynamic"]}

    # Normalized aspects
    s["aspects"] = {k: stat(entries, k) for k in [
        "3d_consistency", "subjective_quality", "photometric_consistency",
        "style_consistency", "motion_magnitude", "motion_smoothness",
    ]}

    # Raw metrics
    s["raw"] = {k: stat(entries, k) for k in [
        "subjective_quality_image", "subjective_quality_aesthetic",
        "motion_magnitude", "photometric_consistency", "style_consistency",
        "motion_smoothness_mse", "motion_smoothness_ssim", "motion_smoothness_lpips",
        "3d_consistency",
    ]}

    # Glitch vs clean
    glitched     = [e for e in entries if e.get("has_glitches") == 1]
    not_glitched = [e for e in entries if e.get("has_glitches") == 0]
    s["glitch_split"] = {
        "clean":   {**stat(glitched,     "worldscore"), "n": len(glitched)}     if glitched     else None,
        "glitchy": {**stat(not_glitched, "worldscore"), "n": len(not_glitched)} if not_glitched else None,
    }

    # By severity
    by_sev = defaultdict(list)
    for e in entries:
        by_sev[e.get("glitch_severity", "unknown")].append(e)
    s["by_severity"] = {
        str(sev): stat(grp, "worldscore")
        for sev, grp in sorted(by_sev.items())
    }

    # By category (sorted worst→best mean worldscore)
    by_cat = defaultdict(list)
    for e in entries:
        by_cat[e.get("glitch_category", "Unknown")].append(e)
    cat_stats = {cat: stat(grp, "worldscore") for cat, grp in by_cat.items()}
    s["by_category"] = dict(sorted(
        cat_stats.items(),
        key=lambda x: x[1]["mean"] if x[1] else float("inf")
    ))

    # Anomalies
    anomalies = {}
    for key in ["photometric_consistency", "subjective_quality", "motion_smoothness", "3d_consistency"]:
        n_zero = sum(1 for e in entries if abs(e.get(key) or 1) < 0.01)
        if n_zero:
            anomalies[f"{key}_zeroed"] = {"count": n_zero, "pct": round(100 * n_zero / len(entries), 1)}
    bad_labels = [e["id"] for e in entries
                  if e.get("has_glitches") == 1 and e.get("glitch_category") == "No Issue"]
    if bad_labels:
        anomalies["label_mismatch_glitch1_no_issue"] = {"count": len(bad_labels), "ids": bad_labels[:10]}
    s["anomalies"] = anomalies

    # Timing
    times = vals(entries, "processing_time_seconds")
    if times:
        s["timing"] = {
            "avg_seconds": float(np.mean(times)),
            "total_hours": round(sum(times) / 3600, 2),
            "n": len(times),
        }

    s["n_videos"] = len(entries)
    return s


def print_model_summary(model: str, entries: list, s: dict, out):
    def p(*args, **kwargs):
        print(*args, **kwargs)
        if out:
            print(*args, **kwargs, file=out)

    p(sep())
    p(f"MODEL: {model}   ({len(entries)} videos)")
    p(sep())

    p("\n── WorldScore ──────────────────────────────────────────────────────")
    for key in ["worldscore", "worldscore_static", "worldscore_dynamic"]:
        p(f"  {key:<25}{fmt_stat(s['worldscore'][key])}")

    p("\n── Normalized aspects (0–100) ──────────────────────────────────────")
    for key in ["3d_consistency", "subjective_quality", "photometric_consistency",
                "style_consistency", "motion_magnitude", "motion_smoothness"]:
        p(f"  {key:<25}{fmt_stat(s['aspects'][key])}")

    p("\n── Raw metrics ─────────────────────────────────────────────────────")
    raw_labels = [
        ("subjective_quality_image",    "CLIP-IQA+"),
        ("subjective_quality_aesthetic","CLIP Aesthetic"),
        ("3d_consistency",              "Reprojection Error"),
        ("motion_magnitude",            "Optical Flow (raw)"),
        ("photometric_consistency",     "AEPE (raw)"),
        ("style_consistency",           "Gram Matrix (raw)"),
        ("motion_smoothness_mse",       "MS-MSE"),
        ("motion_smoothness_ssim",      "MS-SSIM"),
        ("motion_smoothness_lpips",     "MS-LPIPS"),
    ]
    for key, label in raw_labels:
        p(f"  {label:<25}{fmt_stat(s['raw'].get(key))}")

    p("\n── Glitch vs Clean ─────────────────────────────────────────────────")
    gs = s["glitch_split"]
    for label, key in [("Clean  (has_glitches=0)", "clean"), ("Glitchy (has_glitches=1)", "glitchy")]:
        st = gs[key]
        p(f"  {label:<28}{fmt_stat(st)}")

    p("\n── By glitch severity ──────────────────────────────────────────────")
    for sev, st in s["by_severity"].items():
        p(f"  severity={sev}  {fmt_stat(st)}")

    p("\n── By glitch category (worst → best) ───────────────────────────────")
    for cat, st in s["by_category"].items():
        p(f"  {cat:<38}{fmt_stat(st)}")

    if s["anomalies"]:
        p("\n── Anomalies ───────────────────────────────────────────────────────")
        for k, v in s["anomalies"].items():
            if "ids" in v:
                p(f"  {k}: {v['count']} videos")
            else:
                p(f"  {k}: {v['count']} videos ({v['pct']}%)")

    if "timing" in s:
        t = s["timing"]
        p(f"\n── Timing ──────────────────────────────────────────────────────────")
        p(f"  avg {t['avg_seconds']:.1f}s/video  |  total compute {t['total_hours']}h  |  n={t['n']}")

    p("")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       required=True, help="Merged results JSON")
    parser.add_argument("--output",      default=None,  help="Save text summary (.txt)")
    parser.add_argument("--output-json", default=None,  help="Save JSON summary (.json)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    if not data:
        print("Empty JSON.")
        return

    by_model = defaultdict(list)
    for r in data:
        by_model[r.get("model", "unknown")].append(r)

    all_summaries = {}
    out_f = None

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out_f = open(args.output, "w")

    try:
        for model in sorted(by_model.keys()):
            entries = by_model[model]
            s = build_model_summary(entries)
            all_summaries[model] = s
            print_model_summary(model, entries, s, out_f)

        total_line = sep() + f"\nTotal entries: {len(data)}\n" + sep()
        print(total_line)
        if out_f:
            out_f.write(total_line + "\n")
    finally:
        if out_f:
            out_f.close()
            print(f"Text  → {args.output}")

    # JSON summary
    json_path = args.output_json
    if not json_path and args.output:
        json_path = str(Path(args.output).with_suffix(".json"))

    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"JSON  → {json_path}")


if __name__ == "__main__":
    main()
