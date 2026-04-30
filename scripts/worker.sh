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
#SBATCH --requeue

# Note: #SBATCH directives can't be dynamic — if you change slurm settings
# in config.yaml, update the matching lines above too.

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_SLUG="${MODEL// /_}"
RUN_DIR="$(dirname "${LOG_DIR}")"       # LOG_DIR = RUN_DIR/logs
STATUS_LOG="${RUN_DIR}/run_status.log"
LOCK_FILE="${RUN_DIR}/.status.lock"
RETRY_DIR="${RUN_DIR}/.retries"

mkdir -p "${LOG_DIR}" "${RETRY_DIR}"

exec > >(tee -a "${LOG_DIR}/shard${SHARD}.out") 2> >(tee -a "${LOG_DIR}/shard${SHARD}.err" >&2)

# ── Attempt tracking ──────────────────────────────────────────────────────────
ATTEMPT_FILE="${RETRY_DIR}/shard${SHARD}"
if [ -f "${ATTEMPT_FILE}" ]; then
    ATTEMPT=$(( $(cat "${ATTEMPT_FILE}") + 1 ))
else
    ATTEMPT=1
fi
echo "${ATTEMPT}" > "${ATTEMPT_FILE}"

# ── Central status log ────────────────────────────────────────────────────────
log_status() {
    local level="$1"
    local msg="$2"
    local suffix=""
    if [ "${ATTEMPT}" -gt 1 ]; then
        case "${level}" in
            START) suffix=" ← RETRY (attempt ${ATTEMPT})" ;;
            FAIL)  suffix=" ← !! REPEATED FAILURE (attempt ${ATTEMPT})" ;;
        esac
    fi
    (
        flock -x 200
        printf "[%s] [%-5s] Shard %-3s %s%s\n" \
            "$(date '+%Y-%m-%d %H:%M:%S')" "${level}" "${SHARD}" "${msg}" "${suffix}" \
            >> "${STATUS_LOG}"
    ) 200>"${LOCK_FILE}"
}

# ── Trap: log exit status ─────────────────────────────────────────────────────
trap 'CODE=$?; if [ $CODE -eq 0 ]; then log_status "OK" "exit 0 — $(date +%H:%M:%S)"; else log_status "FAIL" "exit code ${CODE} — $(date +%H:%M:%S)"; fi' EXIT

# ── Start ─────────────────────────────────────────────────────────────────────
log_status "START" "job ${SLURM_JOB_ID}_${SHARD} on $(hostname) — $(date +%H:%M:%S)"

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
echo "Attempt:${ATTEMPT}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Log:    ${LOG_DIR}/shard${SHARD}.out"
echo "=========================================="

cd "${PHYSION_METRICS_DIR}"

python scripts/compute_metrics_model.py \
    --json       "${JSON_PATH}" \
    --video-dir  "${VIDEO_DIR}" \
    --model      "${MODEL}" \
    --output     "${HARVEST_DIR}/${MODEL_SLUG}_shard${SHARD}.json" \
    --shard      "${SHARD}" \
    --num-shards "${NUM_SHARDS}"

echo "Done: shard ${SHARD} exited with code $?"
