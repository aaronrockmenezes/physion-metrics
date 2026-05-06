# AGENTS.md — physion-metrics

Context for AI agents working on this repo. Read this before touching anything.

## What This Repo Does

Computes 6 video-only WorldScore metrics (arXiv:2504.00983) on the Physion Eval dataset using SLURM job arrays. Input: MP4 videos + Physion JSON metadata. Output: per-video JSON with raw metrics, normalized WorldScore (0–100) aspects, and aggregated WorldScore-Static/Dynamic.

## Repo Structure

```
physion-metrics/
├── config.yaml                      # Single source of truth for all paths + SLURM config
├── physion_metrics/                 # Library package
│   ├── metrics_wrapper.py           # All 6 metric classes — edit here for metric changes
│   ├── score_utils.py               # Normalization params + compute_worldscore()
│   ├── video_utils.py               # extract_frames_from_video(), frames_to_file_paths()
│   └── __init__.py
├── scripts/
│   ├── compute_metrics_model.py     # Main runner: reads Physion JSON, filters by model, shards
│   ├── merge_results.py             # Glob shard JSONs → single merged JSON
│   ├── summarize_results.py         # Summary report (txt + json) from merged results
│   ├── launch.sh                    # Reads config.yaml, submits array + merge job
│   ├── worker.sh                    # SLURM worker: loads models once, runs shard
│   └── merge_worker.sh              # SLURM merge job: auto-queued via --dependency=afterany
├── tests/
│   └── test_single_video.py         # Runs all 6 metrics on one video, prints results
└── logs/                            # Created at runtime
    └── <Model>/<timestamp>/
        ├── logs/
        │   ├── nodes/               # node0.out/err, ... (GPU assignments, wait/done)
        │   └── shards/              # shard0.out/err, ... (Python per-shard output)
        ├── harvest/                 # Sora_2_shard0000.json, ... (incremental writes)
        ├── merged_<Model>.json      # Final merged results
        ├── merged_<Model>_summary.txt
        ├── merged_<Model>_summary.json
        ├── run_status.log           # Live per-shard status (flock-protected)
        └── .retries/                # Attempt counters per shard (for retry detection)
```

## Critical Gotchas

### 1. CWD Sensitivity
SEA-RAFT, VFIMamba, and DROID-SLAM all use hardcoded CWD-relative paths:
```
./worldscore/benchmark/metrics/third_party/SEA-RAFT/config/eval/spring-M.json
./worldscore/benchmark/metrics/checkpoints/VFIMamba.pkl
./worldscore/benchmark/metrics/checkpoints/droid.pth
```
All calls to these metrics are wrapped in `_worldscore_cwd()` which temporarily chdirs to the WorldScore repo root. Never call `_compute_scores()` on these metrics outside this context manager.

### 2. WORLDSCORE_ROOT env var
`_get_worldscore_root()` priority:
1. `WORLDSCORE_ROOT` env var ← **always set this on HPC**
2. `../WorldScore` sibling dir (local dev)
3. pip site-packages (last resort — no checkpoints, will fail)

```bash
export WORLDSCORE_ROOT=/users/arock3/scratch/physion_worldscore/WorldScore
```

### 3. Abstract Base Classes
WorldScore's `BaseMetric` declares `_compute_scores()` as abstract. Our classes do NOT inherit from it — they are standalone classes that hold a WorldScore metric instance and delegate to `_compute_scores()`. Never add BaseMetric inheritance.

### 4. third_party sys.path
`_inject_third_party_paths()` must run before any WorldScore imports. It adds these subdirs of `worldscore/benchmark/metrics/third_party/` to sys.path:
`droid_slam`, `groundingdino`, `sam2`, `VFIMamba`, `SEA-RAFT`

### 5. Input format mismatch
- Most metrics: accept `List[PIL.Image]`
- ReprojectionErrorMetric (DROID-SLAM): accepts `List[str]` (file paths)
- `frames_to_file_paths()` converts PIL frames → temp file paths for metrics that need paths
- Temp dirs: `/tmp/physion_3d_consistency`, `/tmp/physion_optical_flow`, etc.

### 6. DROID-SLAM is CUDA-only
`ThreeDConsistencyMetric` hardcodes `.to("cuda:0")`. Will raise RuntimeError gracefully if no CUDA. Do not add MPS support — DROID-SLAM's Lie group ops (lietorch) don't support MPS.

### 7. Output path bug (fixed)
`compute_metrics_model.py` previously stripped directory from `--output` when `num_shards > 1`. Fixed: when `--output` is given explicitly (as it is from worker.sh), use it as-is. Only auto-generate path when `--output` is absent.

### 8. Model loading
All 6 metrics are loaded once per shard at startup via `load_metrics()`, then reused for every video. Do not instantiate metric classes inside the per-video loop.

### 9. Incremental saves
Results are written to JSON after every video (not at end of shard). A crashed shard preserves all completed videos.

## Metrics

| Class | Paper Metric | Backend | CUDA-only | Normalization |
|-------|-------------|---------|-----------|---------------|
| `ThreeDConsistencyMetric` | 3D Consistency | DROID-SLAM | Yes | empirical, lower=better, max=1.0719 |
| `CLIPIQAPlusMetric` | Subjective Quality (image) | pyiqa `clipiqa+` | No | z-score |
| `CLIPAestheticMetric` | Subjective Quality (aesthetic) | pyiqa `laion_aes` | No | z-score |
| `OpticalFlowAEPEMetric` | Photometric Consistency | SEA-RAFT AEPE | No | empirical, lower=better, max=1.1920 |
| `StyleConsistencyMetric` | Style Consistency | VGG19 Gram matrix | No | empirical, lower=better, max=0.0070 |
| `OpticalFlowMetric` | Motion Magnitude | SEA-RAFT | No | z-score |
| `MotionSmoothnessMetric` | Motion Smoothness | VFIMamba | No | multi-component empirical |

## WorldScore Normalization

All aspects normalized to 0–100 per WorldScore paper (run_evaluate.py:55):
- **WorldScore-Static** = mean of static aspects (3D consistency, subjective quality, photometric consistency, style consistency)
- **WorldScore-Dynamic** = mean of ALL aspects (static + dynamic)
- `worldscore` = `worldscore_dynamic`

Normalization params live in `physion_metrics/score_utils.py` `METRIC_INFO` dict. Source: `WorldScore/worldscore/benchmark/utils/utils.py` `aspect_info`.

## SLURM Pattern

```
launch.sh → sbatch array → worker.sh × N shards
                        ↓ --dependency=afterany
                     merge_worker.sh → merge + summarize
```

- `worker.sh` has `--requeue` — SLURM auto-moves to new node on hardware failure
- Retry detection: `.retries/shardN` counter file, logs `RETRY` / `REPEATED FAILURE`
- `run_status.log` uses `flock` so all 40+ shards write safely in parallel

## Required Checkpoints

All in `$WORLDSCORE_ROOT/worldscore/benchmark/metrics/checkpoints/`:
- `droid.pth` — DROID-SLAM
- `VFIMamba.pkl` — motion smoothness
- `Tartan-C-T-TSKH-spring540x960-M.pth` — SEA-RAFT

## Not Computable (video-only)

- Camera Controllability — needs GT camera poses
- Object Controllability — needs text prompts
- Content Alignment — needs text prompts
- Motion Accuracy — needs prompts with motion specs

## Environment

```bash
conda activate worldscore
export WORLDSCORE_ROOT=/users/arock3/scratch/physion_worldscore/WorldScore
export PYTHONPATH=/users/arock3/scratch/physion_worldscore/WorldScore:$PYTHONPATH
export LD_LIBRARY_PATH=/users/arock3/scratch/conda/envs/worldscore/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
```

### Oscar HPC Gotchas
- `worldscore` pip package was uninstalled — use `PYTHONPATH` to load local repo
- `conda activate` doesn't set `LD_LIBRARY_PATH` on compute nodes — set explicitly
- `conda_base` in config.yaml must match actual conda install path (`/users/arock3/scratch/conda`)
- Setting `LD_LIBRARY_PATH` globally in shell breaks `clear` (ncurses conflict) — worker.sh scopes it to subprocess only
- Checkpoint downloads: `droid.pth` via `gdown 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh`, `Tartan-C-T-TSKH-spring540x960-M.pth` via `gdown 1a0C5FTdhjM4rKrfXiGhec7eq2YM141lu`

Key deps: `torch`, `torchvision`, `pyiqa`, `opencv-python`, `Pillow`, `tqdm`, `numpy`, `lietorch`, `torch_scatter`
