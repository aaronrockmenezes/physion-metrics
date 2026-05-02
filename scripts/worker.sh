#!/bin/bash
# Worker — DO NOT EDIT config here. All config in config.yaml.
# Called by launch.sh via sbatch. Receives env vars from launcher.
# Spawns one Python subprocess per GPU; each handles its own shard.

#SBATCH --job-name=physion_metrics
#SBATCH --output=/tmp/physion_%A_%a.out
#SBATCH --error=/tmp/physion_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --account=carney-tserre-condo
#SBATCH --constraint=l40s|a6000
#SBATCH --requeue

# Note: resource #SBATCH directives above are overridden by launch.sh args.
# Update them to match config.yaml for direct sbatch use.

NODE_IDX="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_SLUG="${MODEL// /_}"
RUN_DIR="$(dirname "${LOG_DIR}")"
STATUS_LOG="${RUN_DIR}/run_status.log"
LOCK_FILE="${RUN_DIR}/.status.lock"
RETRY_DIR="${RUN_DIR}/.retries"
NODE_LOG="${LOG_DIR}/node${NODE_IDX}.out"
NODE_ERR="${LOG_DIR}/node${NODE_IDX}.err"

mkdir -p "${LOG_DIR}" "${RETRY_DIR}"

# Node-level messages go to node log only (shard logs are separate)
nlog() { echo "$@" | tee -a "${NODE_LOG}"; }
nerr() { echo "$@" | tee -a "${NODE_ERR}" >&2; }

# Load CUDA module
module load cuda 2>/dev/null || true

# Activate conda env
CFG="${PHYSION_METRICS_DIR}/config.yaml"
CONDA_ENV=$(grep "^  conda_env:" "${CFG}" | awk '{print $2}')
source ~/.bashrc
conda activate "${CONDA_ENV}"

export WORLDSCORE_ROOT="${WORLDSCORE_ROOT}"

CPUS_PER_GPU=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))

nlog "=========================================="
nlog "Node job: ${SLURM_JOB_ID}_${NODE_IDX}"
nlog "Model:    ${MODEL}"
nlog "GPUs:     ${GPUS_PER_NODE}  (${CPUS_PER_GPU} CPUs each)"
nlog "Host:     $(hostname)"
nlog "=========================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader >> "${NODE_LOG}"
nlog "=========================================="

# ── Central status log helper (flock) ────────────────────────────────────────
log_status() {
    local shard="$1" level="$2" msg="$3" suffix="${4:-}"
    (
        flock -x 200
        printf "[%s] [%-5s] Shard %-3s %s%s\n" \
            "$(date '+%Y-%m-%d %H:%M:%S')" "${level}" "${shard}" "${msg}" "${suffix}" \
            >> "${STATUS_LOG}"
    ) 200>"${LOCK_FILE}"
}

# ── Spawn one subprocess per GPU ─────────────────────────────────────────────
PIDS=()

for GPU_ID in $(seq 0 $((GPUS_PER_NODE - 1))); do
    SHARD=$(( NODE_IDX * GPUS_PER_NODE + GPU_ID ))

    # Skip if shard index exceeds total shards
    if [ "${SHARD}" -ge "${NUM_SHARDS}" ]; then
        nlog "GPU ${GPU_ID}: no shard (shard ${SHARD} >= ${NUM_SHARDS}), skipping."
        continue
    fi

    # Attempt tracking
    ATTEMPT_FILE="${RETRY_DIR}/shard${SHARD}"
    if [ -f "${ATTEMPT_FILE}" ]; then
        ATTEMPT=$(( $(cat "${ATTEMPT_FILE}") + 1 ))
    else
        ATTEMPT=1
    fi
    echo "${ATTEMPT}" > "${ATTEMPT_FILE}"

    SUFFIX=""
    [ "${ATTEMPT}" -gt 1 ] && SUFFIX=" ← RETRY (attempt ${ATTEMPT})"
    log_status "${SHARD}" "START" "node ${SLURM_JOB_ID}_${NODE_IDX} GPU${GPU_ID} on $(hostname)" "${SUFFIX}"

    # Launch subprocess — stdout/stderr go only to shard logs (not node log)
    (
        export CUDA_VISIBLE_DEVICES="${GPU_ID}"
        export OMP_NUM_THREADS="${CPUS_PER_GPU}"
        export MKL_NUM_THREADS="${CPUS_PER_GPU}"
        export OPENBLAS_NUM_THREADS="${CPUS_PER_GPU}"
        export NUMEXPR_NUM_THREADS="${CPUS_PER_GPU}"

        cd "${PHYSION_METRICS_DIR}"

        python scripts/compute_metrics_model.py \
            --json       "${JSON_PATH}" \
            --video-dir  "${VIDEO_DIR}" \
            --model      "${MODEL}" \
            --output     "${HARVEST_DIR}/${MODEL_SLUG}_shard${SHARD}.json" \
            --shard      "${SHARD}" \
            --num-shards "${NUM_SHARDS}" \
            >> "${LOG_DIR}/shard${SHARD}.out" \
            2>> "${LOG_DIR}/shard${SHARD}.err"

        CODE=$?
        ATTEMPT_VAL=$(cat "${RETRY_DIR}/shard${SHARD}" 2>/dev/null || echo 1)
        FAIL_SUFFIX=""
        [ "${ATTEMPT_VAL}" -gt 1 ] && FAIL_SUFFIX=" ← !! REPEATED FAILURE (attempt ${ATTEMPT_VAL})"

        if [ $CODE -eq 0 ]; then
            log_status "${SHARD}" "OK"   "exit 0 — $(date +%H:%M:%S)" ""
        else
            log_status "${SHARD}" "FAIL" "exit code ${CODE} — $(date +%H:%M:%S)" "${FAIL_SUFFIX}"
        fi
    ) &

    PIDS+=($!)
    nlog "GPU ${GPU_ID} → shard ${SHARD} (PID $!)"
done

nlog "Waiting for ${#PIDS[@]} GPU processes..."
wait "${PIDS[@]}"
nlog "Node ${NODE_IDX} done."
