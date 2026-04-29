"""
WorldScore normalization and aggregation.
Ported from WorldScore/worldscore/benchmark/helpers/evaluator.py + utils/utils.py
Paper: arXiv:2504.00983
"""

# Normalization params from worldscore/benchmark/utils/utils.py aspect_info
METRIC_INFO = {
    # Z-score normalized metrics
    "clip_iqa+": {
        "avg": 0.5842, "std": 0.0441,
        "z_max": 1.8703, "z_min": -1.8342,
        "range": [0.25, 0.75], "higher_is_better": True,
    },
    "clip_aesthetic": {
        "avg": 5.5952, "std": 0.2561,
        "z_max": 1.9741, "z_min": -2.9342,
        "range": [0.25, 0.75], "higher_is_better": True,
    },
    "optical_flow": {
        "avg": 3.2425, "std": 3.4505,
        "z_max": 2.7498, "z_min": -0.8638,
        "range": [0.25, 0.75], "higher_is_better": True,
    },
    # Empirical single-value metrics
    "optical_flow_aepe": {
        "empirical_max": 1.1920, "empirical_min": 0,
        "higher_is_better": False,
    },
    "gram_matrix": {
        "empirical_max": 0.0070, "empirical_min": 0,
        "higher_is_better": False,
    },
    # Empirical multi-component: [MSE, SSIM, LPIPS]
    "motion_smoothness": {
        "empirical_max": [82.4014, 1.0, 0.0228],
        "empirical_min": [0.0, 0.9224, 0.0],
        "higher_is_better": [False, True, False],
    },
}

# Which aspect each metric belongs to, and static vs dynamic
WORLDSCORE_LAYOUT = {
    "static": {
        "photometric_consistency": "optical_flow_aepe",
        "style_consistency":       "gram_matrix",
        "subjective_quality":      ["clip_iqa+", "clip_aesthetic"],
    },
    "dynamic": {
        "motion_magnitude":   "optical_flow",
        "motion_smoothness":  "motion_smoothness",
    },
}


def normalize_metric(name: str, score) -> float:
    """Normalize a single raw metric score to [0, 1]."""
    info = METRIC_INFO[name]

    if "avg" in info:
        # Z-score normalization
        z = (score - info["avg"]) / info["std"]
        x = (z - info["z_min"]) / (info["z_max"] - info["z_min"])
        if not info["higher_is_better"]:
            x = 1.0 - x
        lo, hi = info["range"]
        return max(0.0, min(1.0, lo + (hi - lo) * x))

    elif isinstance(info["empirical_max"], list):
        # Multi-component empirical (motion_smoothness)
        parts = []
        for s, emax, emin, hib in zip(
            score, info["empirical_max"], info["empirical_min"], info["higher_is_better"]
        ):
            s = max(emin, min(emax, s))
            n = (s - emin) / (emax - emin)
            if not hib:
                n = 1.0 - n
            parts.append(n)
        return sum(parts) / len(parts)

    else:
        # Single empirical
        emax, emin = info["empirical_max"], info["empirical_min"]
        score = max(emin, min(emax, score))
        n = (score - emin) / (emax - emin)
        if not info["higher_is_better"]:
            n = 1.0 - n
        return n


def compute_worldscore(result: dict) -> dict:
    """
    Compute normalized WorldScore from raw metric results dict.

    Args:
        result: dict with keys: subjective_quality_image, subjective_quality_aesthetic,
                motion_magnitude, photometric_consistency, style_consistency,
                motion_smoothness_mse, motion_smoothness_ssim, motion_smoothness_lpips

    Returns:
        dict with normalized per-aspect scores and WorldScore-Static / WorldScore-Dynamic
    """
    scores = {}

    # --- Static aspects ---

    # Subjective Quality: mean of CLIP-IQA+ and CLIP Aesthetic
    sq_scores = []
    if result.get("subjective_quality_image") is not None:
        sq_scores.append(normalize_metric("clip_iqa+", result["subjective_quality_image"]))
    if result.get("subjective_quality_aesthetic") is not None:
        sq_scores.append(normalize_metric("clip_aesthetic", result["subjective_quality_aesthetic"]))
    if sq_scores:
        scores["subjective_quality"] = sum(sq_scores) / len(sq_scores)

    # Photometric Consistency
    if result.get("photometric_consistency") is not None:
        scores["photometric_consistency"] = normalize_metric(
            "optical_flow_aepe", result["photometric_consistency"]
        )

    # Style Consistency
    if result.get("style_consistency") is not None:
        scores["style_consistency"] = normalize_metric(
            "gram_matrix", result["style_consistency"]
        )

    # --- Dynamic aspects ---

    # Motion Magnitude
    if result.get("motion_magnitude") is not None:
        scores["motion_magnitude"] = normalize_metric(
            "optical_flow", result["motion_magnitude"]
        )

    # Motion Smoothness: (MSE, SSIM, LPIPS) tuple
    mse   = result.get("motion_smoothness_mse")
    ssim  = result.get("motion_smoothness_ssim")
    lpips = result.get("motion_smoothness_lpips")
    if all(v is not None for v in [mse, ssim, lpips]):
        scores["motion_smoothness"] = normalize_metric(
            "motion_smoothness", (mse, ssim, lpips)
        )

    # --- Aggregate ---
    static_keys  = ["subjective_quality", "photometric_consistency", "style_consistency"]
    dynamic_keys = ["motion_magnitude", "motion_smoothness"]

    static_vals  = [scores[k] for k in static_keys  if k in scores]
    dynamic_vals = [scores[k] for k in dynamic_keys if k in scores]

    if static_vals:
        scores["worldscore_static"] = sum(static_vals) / len(static_vals)
    if dynamic_vals:
        scores["worldscore_dynamic"] = sum(dynamic_vals) / len(dynamic_vals)

    all_vals = static_vals + dynamic_vals
    if all_vals:
        scores["worldscore"] = sum(all_vals) / len(all_vals)

    return scores
