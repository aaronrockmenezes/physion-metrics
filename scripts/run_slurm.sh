#!/bin/bash
# SLURM job array for Physion-Eval metric computation.
# Each array task processes one shard of a model's videos on its own GPU.
#
# Usage:
#   sbatch --array=0-9 scripts/run_slurm.sh "Sora 2" 10 <run_name>
#   sbatch --array=0-4 scripts/run_slurm.sh "Kling 2.5" 5
#
# Args: $1=model name, $2=num_shards, $3=optional run name (default: model+timestamp)

#SBATCH --job-name=physion_metrics
#SBATCH --output=/tmp/physion_metrics_slurm_%A_%a.out
#SBATCH --error=/tmp/physion_metrics_slurm_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --account=carney-tserre-condo

MODEL="${1:-Sora 2}"
NUM_SHARDS="${2:-10}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SLUG="${MODEL// /_}"
RUN_NAME="${3:-${MODEL_SLUG}_${TIMESTAMP}}"

# Paths
PHYSION_METRICS_DIR="/oscar/data/tserre/arock3/physion_worldscore/physion-metrics"
VIDEO_DIR="/path/to/physion/videos"
JSON_PATH="/path/to/Physion_Eval_20260322.json"
WORLDSCORE_ROOT="/users/arock3/scratch/physion_worldscore/WorldScore"

# Per-run log dir: logs/Sora_2_20260429_143022/
# All shards of the same run write into same dir (RUN_NAME must be passed explicitly for arrays)
LOG_DIR="${PHYSION_METRICS_DIR}/logs/${MODEL_SLUG}/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

# Tee stdout/stderr into run log dir, shard-specific file
exec > >(tee -a "${LOG_DIR}/shard${SHARD}.out") 2> >(tee -a "${LOG_DIR}/shard${SHARD}.err" >&2)

source ~/.bashrc
conda activate worldscore
export WORLDSCORE_ROOT="${WORLDSCORE_ROOT}"

echo "=========================================="
echo "Run:    ${RUN_NAME}"
echo "Job:    ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Model:  ${MODEL}"
echo "Shard:  ${SHARD}/${NUM_SHARDS}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Log:    ${LOG_DIR}/shard${SHARD}.out"
echo "=========================================="

cd "${PHYSION_METRICS_DIR}"

python scripts/compute_metrics_model.py \
    --json       "${JSON_PATH}" \
    --video-dir  "${VIDEO_DIR}" \
    --model      "${MODEL}" \
    --output     "${LOG_DIR}/${MODEL_SLUG}_metrics.json" \
    --shard      "${SHARD}" \
    --num-shards "${NUM_SHARDS}"

echo "Done: shard ${SHARD} exited with code $?"
