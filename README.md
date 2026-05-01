# physion-metrics

Compute **6 video-only metrics** from [WorldScore: A Unified Evaluation Benchmark for World Generation](https://arxiv.org/abs/2504.00983) (arXiv:2504.00983) on the Physion Eval dataset. Outputs normalized WorldScore (0–100) per video with full per-metric breakdown.

## Metrics

WorldScore has 10 metrics total. 6 are computable from video only:

| # | Metric | Aspect | Backend | Video-only |
|---|--------|--------|---------|------------|
| 1 | **3D Consistency** | Quality | DROID-SLAM reprojection error | ✅ |
| 2 | **Subjective Quality** | Quality | CLIP-IQA+ + CLIP Aesthetic | ✅ |
| 3 | **Photometric Consistency** | Quality | SEA-RAFT optical flow AEPE | ✅ |
| 4 | **Style Consistency** | Quality | VGG19 Gram matrix | ✅ |
| 5 | **Motion Magnitude** | Dynamics | SEA-RAFT optical flow | ✅ |
| 6 | **Motion Smoothness** | Dynamics | VFIMamba MSE/SSIM/LPIPS | ✅ |
| — | Camera Controllability | Controllability | DROID-SLAM | ❌ needs camera GT |
| — | Object Controllability | Controllability | Grounding DINO | ❌ needs prompts |
| — | Content Alignment | Controllability | CLIPScore | ❌ needs prompts |
| — | Motion Accuracy | Dynamics | Optical flow + masks | ❌ needs prompts |

All scores normalized to **0–100** using WorldScore paper normalization params.

## Repo Structure

```
physion-metrics/
├── config.yaml                          # All paths and SLURM config — edit this
├── physion_metrics/                     # Library
│   ├── metrics_wrapper.py               # All 6 metric classes
│   ├── score_utils.py                   # WorldScore normalization + aggregation
│   ├── video_utils.py                   # Frame extraction (PIL output)
│   └── __init__.py
├── scripts/
│   ├── compute_metrics_model.py         # Per-model runner (reads Physion JSON)
│   ├── merge_results.py                 # Merge shard JSONs into one
│   ├── summarize_results.py             # Summary report (txt + json) from merged
│   ├── launch.sh                        # SLURM launcher (reads config.yaml)
│   ├── worker.sh                        # SLURM array worker
│   └── merge_worker.sh                  # SLURM merge job (auto-queued by launch.sh)
└── tests/
    └── test_single_video.py             # Sanity test on one video
```

## Quick Start

### 1. Edit config
```yaml
# config.yaml
paths:
  physion_metrics_dir: /path/to/physion-metrics
  video_dir:           /path/to/videos
  json_path:           /path/to/Physion_Eval.json
  worldscore_root:     /path/to/WorldScore
```

### 2. Test on single video (interactive GPU node)
```bash
export WORLDSCORE_ROOT=/path/to/WorldScore
python tests/test_single_video.py --video path/to/video.mp4 --max-frames 50
```

### 3. Launch full run (SLURM job array)
```bash
bash scripts/launch.sh "Sora 2" 40
```
Submits 40 shards as a job array + auto-queues a merge job when all shards finish.

## Output Structure

Each run creates:
```
logs/
  Sora_2/
    20260501_120000/
      logs/                            # Per-shard stdout/stderr + merge.out
      harvest/                         # Per-shard JSONs (written incrementally)
      merged_Sora_2.json               # All results merged and sorted by id
      merged_Sora_2_summary.txt        # Human-readable summary report
      merged_Sora_2_summary.json       # Structured summary (programmatic)
      run_status.log                   # Live status of all shards
```

### Per-video JSON entry
```json
{
  "id": "T1_00001",
  "filename": "P06_108_67_..._processed_s",
  "model": "Sora 2",
  "has_glitches": 1,
  "glitch_severity": 3,
  "glitch_category": "Force & Motion Inconsistency",
  "frames": 235,

  "subjective_quality_image": 0.547,
  "subjective_quality_aesthetic": 4.491,
  "motion_smoothness_mse": 7.40,
  "motion_smoothness_ssim": 0.978,
  "motion_smoothness_lpips": 0.013,

  "3d_consistency": 95.45,
  "subjective_quality": 24.68,
  "photometric_consistency": 82.87,
  "style_consistency": 96.25,
  "motion_magnitude": 39.16,
  "motion_smoothness": 68.95,
  "worldscore_static": 67.93,
  "worldscore_dynamic": 62.38,
  "worldscore": 62.38,

  "processing_time_seconds": 308.5
}
```
> Note: `photometric_consistency`, `style_consistency`, `motion_magnitude` appear in both raw and normalized forms — the normalized (0–100) value overwrites the raw value in the output.

## SLURM Configuration

All SLURM settings live in two places (keep in sync):
- `config.yaml` — `slurm:` section (read by launch.sh)
- `scripts/worker.sh` — `#SBATCH` directives (SLURM reads these directly)

Current defaults: 12 CPU, 40G RAM, 1 GPU, 8h wall time, `gpu` partition, `carney-tserre-condo` account, `l40s|a6000` constraint.

### How launch works
1. `launch.sh` reads `config.yaml`, creates run folder, submits array job
2. `worker.sh` runs on each GPU node: loads all 6 models once, processes its shard, writes JSON incrementally
3. After all shards finish, `merge_worker.sh` auto-runs: merges JSONs, generates summary txt + json
4. `run_status.log` updated live by every shard (flock-protected)

### Shard sizing guide (at ~250s/video)
| Videos | Shards | Time/shard | Batches (cap=20) | Total time |
|--------|--------|------------|------------------|------------|
| 2206   | 40     | ~3.8h      | 2                | ~8h        |
| 1912   | 40     | ~3.3h      | 2                | ~7h        |
| 1759   | 36     | ~3.1h      | 2                | ~6h        |

## Monitoring

```bash
# Live shard status
tail -f logs/Sora_2/*/run_status.log

# Count completed shards
grep "OK.*Shard" logs/Sora_2/*/run_status.log | wc -l

# Check harvest folder
ls logs/Sora_2/*/harvest/*.json | wc -l

# Manual merge (if merge job failed)
python scripts/merge_results.py \
  --pattern "logs/Sora_2/*/harvest/Sora_2_shard*.json" \
  --output  "logs/Sora_2/*/merged_Sora_2.json"

# Manual summary
python scripts/summarize_results.py \
  --input  merged_Sora_2.json \
  --output merged_Sora_2_summary.txt
```

## Environment

```bash
conda activate worldscore
export WORLDSCORE_ROOT=/path/to/WorldScore
```

Required checkpoints in `$WORLDSCORE_ROOT/worldscore/benchmark/metrics/checkpoints/`:
- `droid.pth` — DROID-SLAM (3D consistency)
- `VFIMamba.pkl` — motion smoothness
- `Tartan-C-T-TSKH-spring540x960-M.pth` — SEA-RAFT (optical flow)

## References

- Paper: [WorldScore arXiv:2504.00983](https://arxiv.org/abs/2504.00983)
- WorldScore repo: [haoyi-duan/WorldScore](https://github.com/haoyi-duan/WorldScore)
- Dataset: Physion Eval (video-only, no prompts or GT poses used)
