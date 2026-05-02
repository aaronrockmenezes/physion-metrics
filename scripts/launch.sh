#!/bin/bash
# Launcher — config lives in config.yaml at repo root.
# Usage:
#   bash scripts/launch.sh "Sora 2"        # single job, all videos
#   bash scripts/launch.sh "Sora 2" 40     # 40 shards across N nodes (gpus_per_node per node)

MODEL="${1:-Sora 2}"
NUM_SHARDS="${2:-1}"

# ── Read config.yaml ──────────────────────────────────────────────────────────
CFG="$(dirname "$0")/../config.yaml"
cfg() { grep "^  $1:" "${CFG}" | awk '{print $2}'; }

PHYSION_METRICS_DIR=$(cfg physion_metrics_dir)
VIDEO_DIR=$(cfg video_dir)
JSON_PATH=$(cfg json_path)
WORLDSCORE_ROOT=$(cfg worldscore_root)
CONDA_ENV=$(cfg conda_env)
GPUS_PER_NODE=$(cfg gpus_per_node)
CPUS_PER_TASK=$(cfg cpus_per_task)
MEM=$(cfg mem)
TIME=$(cfg time)
PARTITION=$(cfg partition)
ACCOUNT=$(cfg account)
CONSTRAINT=$(cfg constraint)
# ─────────────────────────────────────────────────────────────────────────────

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SLUG="${MODEL// /_}"

RUN_DIR="${PHYSION_METRICS_DIR}/logs/${MODEL_SLUG}/${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
HARVEST_DIR="${RUN_DIR}/harvest"
MERGED_OUTPUT="${RUN_DIR}/merged_${MODEL_SLUG}.json"

WORKER="${PHYSION_METRICS_DIR}/scripts/worker.sh"
MERGE_WORKER="${PHYSION_METRICS_DIR}/scripts/merge_worker.sh"

mkdir -p "${LOG_DIR}" "${HARVEST_DIR}"

# ── Node count: ceil(NUM_SHARDS / GPUS_PER_NODE) ─────────────────────────────
NUM_NODES=$(( (NUM_SHARDS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))

echo "Model:         ${MODEL}"
echo "Shards:        ${NUM_SHARDS}"
echo "GPUs/node:     ${GPUS_PER_NODE}"
echo "Nodes:         ${NUM_NODES}"
echo "Run        →   ${RUN_DIR}"
echo ""

EXPORTS="ALL,MODEL=${MODEL},NUM_SHARDS=${NUM_SHARDS},GPUS_PER_NODE=${GPUS_PER_NODE},LOG_DIR=${LOG_DIR},HARVEST_DIR=${HARVEST_DIR},PHYSION_METRICS_DIR=${PHYSION_METRICS_DIR},VIDEO_DIR=${VIDEO_DIR},JSON_PATH=${JSON_PATH},WORLDSCORE_ROOT=${WORLDSCORE_ROOT}"

SBATCH_ARGS="--gres=gpu:${GPUS_PER_NODE} --cpus-per-task=${CPUS_PER_TASK} --mem=${MEM} --time=${TIME} --partition=${PARTITION} --account=${ACCOUNT} --constraint=${CONSTRAINT}"

if [ "${NUM_NODES}" -gt 1 ]; then
    JOB=$(sbatch ${SBATCH_ARGS} --array=0-$((NUM_NODES - 1)) --export="${EXPORTS}" "${WORKER}")
else
    JOB=$(sbatch ${SBATCH_ARGS} --export="${EXPORTS}" "${WORKER}")
fi

echo "Submitted: ${JOB}"
JOB_ID=$(echo "${JOB}" | awk '{print $NF}')

# ── Queue merge — fires after all nodes finish (any exit code) ────────────────
MERGE_EXPORTS="ALL,HARVEST_DIR=${HARVEST_DIR},MERGED_OUTPUT=${MERGED_OUTPUT},MODEL_SLUG=${MODEL_SLUG},NUM_SHARDS=${NUM_SHARDS},PHYSION_METRICS_DIR=${PHYSION_METRICS_DIR},CONDA_ENV=${CONDA_ENV},LOG_DIR=${LOG_DIR}"

MERGE_JOB=$(sbatch \
    --dependency=afterany:${JOB_ID} \
    --export="${MERGE_EXPORTS}" \
    "${MERGE_WORKER}")

echo "Merge job: ${MERGE_JOB} (waits on ${JOB_ID})"
echo ""
echo "Results → ${MERGED_OUTPUT}"
