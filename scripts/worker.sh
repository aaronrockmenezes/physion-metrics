#!/bin/bash
# Worker — DO NOT EDIT config here. All config in config.yaml.
# Called by launch.sh via sbatch. Receives env vars from launcher.

#SBATCH --job-name=physion_metrics
#SBATCH --output=/tmp/physion_%A_%a.out
#SBATCH --error=/tmp/physion_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --account=carney-tserre-condo
#SBATCH --constraint="l40s|a6000"

# Note: #SBATCH directives can't be dynamic — if you change slurm settings
# in config.yaml, update the matching lines above too.

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_SLUG="${MODEL// /_}"

mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_DIR}/shard${SHARD}.out") 2> >(tee -a "${LOG_DIR}/shard${SHARD}.err" >&2)

# Load CUDA module
module load cuda 2>/dev/null || true

# Read conda env from config.yaml
CFG="${PHYSION_METRICS_DIR}/config.yaml"
CONDA_ENV=$(grep "^  conda_env:" "${CFG}" | awk '{print $2}')

source ~/.bashrc
conda activate "${CONDA_ENV}"

export WORLDSCORE_ROOT="${WORLDSCORE_ROOT}"

echo "=========================================="
echo "Job:    ${SLURM_JOB_ID}_${SHARD}"
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
    --output     "${LOG_DIR}/${MODEL_SLUG}_shard${SHARD}.json" \
    --shard      "${SHARD}" \
    --num-shards "${NUM_SHARDS}"

echo "Done: shard ${SHARD} exited with code $?"
