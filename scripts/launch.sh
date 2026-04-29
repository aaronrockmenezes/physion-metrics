#!/bin/bash
# Launch a single-GPU job for one model.
# Usage: sbatch scripts/launch.sh "Sora 2"
# For multi-GPU array: sbatch --array=0-9 scripts/run_slurm.sh "Sora 2" 10

#SBATCH --job-name=physion_metrics
#SBATCH --output=/tmp/physion_metrics_slurm_%j.out
#SBATCH --error=/tmp/physion_metrics_slurm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --account=carney-tserre-condo

MODEL="${1:-Sora 2}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SLUG="${MODEL// /_}"
RUN_NAME="${MODEL_SLUG}_${TIMESTAMP}"

# Paths
PHYSION_METRICS_DIR="/oscar/data/tserre/arock3/physion_worldscore/physion-metrics"
VIDEO_DIR="/path/to/physion/videos"
JSON_PATH="/path/to/Physion_Eval_20260322.json"
WORLDSCORE_ROOT="/users/arock3/scratch/physion_worldscore/WorldScore"

# Per-run log dir: logs/Sora_2_20260429_143022/
LOG_DIR="${PHYSION_METRICS_DIR}/logs/${MODEL_SLUG}/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

# Move SLURM stdout/stderr into the run log dir
exec > >(tee -a "${LOG_DIR}/slurm.out") 2> >(tee -a "${LOG_DIR}/slurm.err" >&2)

source ~/.bashrc
conda activate worldscore
export WORLDSCORE_ROOT="${WORLDSCORE_ROOT}"

echo "=========================================="
echo "Run:    ${RUN_NAME}"
echo "Job:    ${SLURM_JOB_ID}"
echo "Model:  ${MODEL}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Log:    ${LOG_DIR}"
echo "=========================================="

cd "${PHYSION_METRICS_DIR}"

python scripts/compute_metrics_model.py \
    --json      "${JSON_PATH}" \
    --video-dir "${VIDEO_DIR}" \
    --model     "${MODEL}" \
    --output    "${LOG_DIR}/${MODEL_SLUG}_metrics.json"

echo "Done: exited with code $?"
