# Physion Metrics

Compute 5 video-only metrics from **WorldScore: A Unified Evaluation Benchmark for World Generation** (arXiv:2504.00983).

## Metrics (Paper: 10 total, Video-only: 5 computable)

### Quality Aspect (2 metrics)
1. **Subjective Quality (Image)** — CLIP-IQA+ image quality (range 0-1, higher is better)
2. **Subjective Quality (Aesthetic)** — CLIP Aesthetic score (range 0-10, higher is better)

### Dynamics Aspect (3 metrics)
3. **Motion Magnitude** — Optical flow magnitude between frames (range 0+, higher = more motion)
4. **Photometric Consistency** — Optical flow AEPE bidirectional warping (range 0+, lower is better)
5. **Motion Smoothness** — Frame interpolation quality (MSE, SSIM, LPIPS)

### Bonus (video-only metric)
6. **Style Consistency** — Gram matrix texture features (VGG19-based)

### Not Computable (need external data)
- **Object Controllability** — requires prompts
- **Camera Rotation/Translation Error** — requires GT camera poses
- **Reprojection Error** — requires GT depth maps

## Usage

### Mac M4 (CLIP metrics only)
```bash
python test_mac.py --video example.mp4
python test_mac.py --video-dir ./videos --output test_results_mac.json
```

### Full Pipeline (all 5 metrics)
```bash
python compute_metrics.py --video-dir ./videos --output physion_metrics.json
python compute_metrics.py --video example.mp4
```

### HPC with GPU (H100/B200/A5000/etc)
```bash
python compute_metrics_hpc.py --video-dir ./videos --output physion_metrics_hpc.json --max-videos 1000
python compute_metrics_hpc.py --video example.mp4
```

### Full Version (graceful fallback for all platforms)
```bash
python compute_metrics_full.py --video-dir ./videos --output physion_metrics_full.json
```

## Options

- `--video` — Single video file
- `--video-dir` — Directory containing videos (MP4/AVI/MOV)
- `--output` — Output JSON file (default: depends on script)
- `--frame-skip` — Skip frames (1=all, 2=every 2nd, etc.) (default: 1)
- `--max-frames` — Max frames per video (default: None = all)
- `--max-videos` — Process only first N videos (default: None = all)

## Output Format

```json
{
  "videos": [
    {
      "video": "example.mp4",
      "frames": 120,
      "subjective_quality_image": 0.523,
      "subjective_quality_aesthetic": 6.245,
      "motion_magnitude": 3.142,
      "photometric_consistency": 0.087,
      "style_consistency": 0.156,
      "motion_smoothness_mse": 12.34,
      "motion_smoothness_ssim": 0.892,
      "motion_smoothness_lpips": 0.156,
      "processing_time_seconds": 45.2
    }
  ]
}
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Mac users with OpenMP issues:
export KMP_DUPLICATE_LIB_OK=TRUE
```

## Technical Details

- Frames resized to 512×512 for metric computation
- CLIP metrics use pyiqa library (CLIP-IQA+ via pyiqa, CLIP Aesthetic via LAION)
- Optical flow uses SEA-RAFT/FlowFormer++ models
- Motion smoothness uses VFIMamba (requires CUDA or MPS)
- Style consistency uses VGG19 features with gram matrix comparison
- GPU acceleration available for CUDA/MPS devices

## Scripts

| Script | Target | Features | Dependencies |
|--------|--------|----------|--------------|
| `test_mac.py` | Mac M4 | CLIP metrics only | pyiqa, torch |
| `compute_metrics.py` | General | All 5 metrics | WorldScore, optical flow |
| `compute_metrics_hpc.py` | GPU (HPC) | All 5 metrics + timing | CUDA required |
| `compute_metrics_full.py` | Any platform | All 5 metrics + fallback | WorldScore, all models |

## Performance

- Mac M4: ~30-60s per video (CLIP metrics only)
- HPC H100: ~5-15s per video (all metrics)
- Memory: 8GB+ recommended

## References

- Paper: [WorldScore: A Unified Evaluation Benchmark for World Generation](https://arxiv.org/abs/2504.00983)
- Code: WorldScore repository (included)
- Dataset: Physion Eval (videos only, no prompts/GT poses)
