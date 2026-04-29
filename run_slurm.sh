#!/bin/bash
# SLURM job array for Physion-Eval metric computation.
# Each array task processes one shard of a model's videos on its own GPU.
#
# Usage:
#   sbatch --array=0-9 run_slurm.sh "Sora 2" 10
#   sbatch --array=0-4 run_slurm.sh "Kling 2.5" 5
#
# Args: $1 = model name, $2 = num_shards (must match --array upper bound + 1)

#SBATCH --job-name=physion_metrics
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

MODEL="${1:-Sora 2}"
NUM_SHARDS="${2:-10}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

# Paths — update these for your cluster
PHYSION_METRICS_DIR="/oscar/data/tserre/arock3/physion_worldscore/physion-metrics"
VIDEO_DIR="/path/to/physion/videos"
JSON_PATH="/path/to/Physion_Eval_20260322.json"
OUTPUT_DIR="${PHYSION_METRICS_DIR}/results"
WORLDSCORE_ROOT="/users/arock3/scratch/physion_worldscore/WorldScore"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PHYSION_METRICS_DIR}/logs"

# Activate conda env
source ~/.bashrc
conda activate worldscore

export WORLDSCORE_ROOT="${WORLDSCORE_ROOT}"

echo "=========================================="
echo "Job:      ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Model:    ${MODEL}"
echo "Shard:    ${SHARD}/${NUM_SHARDS}"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=========================================="

cd "${PHYSION_METRICS_DIR}"

python compute_metrics_model.py \
    --json      "${JSON_PATH}" \
    --video-dir "${VIDEO_DIR}" \
    --model     "${MODEL}" \
    --output    "${OUTPUT_DIR}/${MODEL// /_}_metrics.json" \
    --shard     "${SHARD}" \
    --num-shards "${NUM_SHARDS}"

echo "Done: shard ${SHARD} exited with code $?"
