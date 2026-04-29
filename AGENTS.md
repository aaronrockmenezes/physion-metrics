# AGENTS.md — physion-metrics

Context for AI agents working on this repo.

## What This Repo Does

Computes 5 video-only metrics from the WorldScore paper (arXiv:2504.00983) on the Physion Eval dataset.
Input: MP4 videos. Output: JSON with per-video metric scores.

## Repo Structure

```
physion-metrics/
├── metrics_wrapper.py       # All 6 metric classes (edit this for metric changes)
├── video_utils.py           # Frame extraction from video (PIL Image output)
├── compute_metrics.py       # General-purpose runner
├── compute_metrics_hpc.py   # HPC/CUDA runner (requires CUDA)
├── compute_metrics_full.py  # All-platform runner with graceful fallback
├── test_mac.py              # Mac M4 test (CLIP only, no WorldScore deps)
└── AGENTS.md                # This file
```

## Sibling Dependency

WorldScore repo MUST be locatable. Priority order in `_get_worldscore_root()`:
1. `WORLDSCORE_ROOT` env var (explicit override, preferred on HPC)
2. `../WorldScore` sibling dir (local dev)
3. pip site-packages (last resort — **no checkpoints, will fail for flow/smoothness metrics**)

**HPC:** physion-metrics and WorldScore may be on different paths (e.g. `/oscar/data/...` vs `/users/arock3/scratch/...`). Always set env var:
```bash
export WORLDSCORE_ROOT=/users/arock3/scratch/physion_worldscore/WorldScore
```

WorldScore is also pip-installed in the `worldscore` conda env.

## Key Gotchas (don't break these)

1. **CWD sensitivity**: SEA-RAFT and VFIMamba use hardcoded CWD-relative paths
   (`worldscore/benchmark/metrics/third_party/SEA-RAFT/...`).
   All calls to these metrics are wrapped in `_worldscore_cwd()` context manager
   which temporarily chdirs to the worldscore package root.

2. **Abstract base classes**: WorldScore's `BaseMetric` and `IQAPytorchMetric` declare
   `_compute_scores()` as abstract. Our classes do NOT inherit from them — they are
   standalone classes that delegate to WorldScore metric instances via `_compute_scores()`.

3. **third_party sys.path**: `third_party/__init__.py` uses CWD-relative paths.
   `_inject_third_party_paths()` resolves these to absolute paths using
   `importlib.util.find_spec("worldscore")` before any imports.

4. **Import order**: `_inject_third_party_paths()` and `_worldscore_cwd()` must run
   BEFORE any WorldScore metric imports or the droid/SEA-RAFT/VFIMamba modules won't load.

## Metrics (Paper → Implementation)

| Paper Metric | Class | Backend |
|---|---|---|
| Subjective Quality (Image) | `CLIPIQAPlusMetric` | pyiqa `clipiqa+` |
| Subjective Quality (Aesthetic) | `CLIPAestheticMetric` | pyiqa `laion_aes` |
| Motion Magnitude | `OpticalFlowMetric` | WorldScore → SEA-RAFT |
| Photometric Consistency | `OpticalFlowAEPEMetric` | WorldScore → SEA-RAFT |
| Style Consistency | `StyleConsistencyMetric` | WorldScore → VGG19 Gram Matrix |
| Motion Smoothness | `MotionSmoothnessMetric` | WorldScore → VFIMamba |

## Not Computable (video-only)

- Object Controllability (needs prompts)
- Camera Rotation/Translation Error (needs GT camera poses)
- Reprojection Error (needs GT depth)

## Run Commands

```bash
# Mac M4 (CLIP only)
python test_mac.py --video example.mp4

# HPC (all metrics, CUDA required)
python compute_metrics_hpc.py --video-dir ./videos --output results.json --frame-skip 2
```

## Environment

Conda env: `worldscore`
Key deps: torch, torchvision, pyiqa, opencv-python, Pillow, tqdm, numpy
