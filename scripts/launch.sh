#!/bin/bash
# Launcher — config lives in config.yaml at repo root.
# Usage:
#   bash scripts/launch.sh "Sora 2"        # single job, all videos
#   bash scripts/launch.sh "Sora 2" 10     # job array, 10 shards (10 GPUs)

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

echo "Model:    ${MODEL}"
echo "Shards:   ${NUM_SHARDS}"
echo "Run    →  ${RUN_DIR}"
echo "Logs   →  ${LOG_DIR}"
echo "Harvest→  ${HARVEST_DIR}"
echo ""

EXPORTS="ALL,MODEL=${MODEL},NUM_SHARDS=${NUM_SHARDS},LOG_DIR=${LOG_DIR},HARVEST_DIR=${HARVEST_DIR},PHYSION_METRICS_DIR=${PHYSION_METRICS_DIR},VIDEO_DIR=${VIDEO_DIR},JSON_PATH=${JSON_PATH},WORLDSCORE_ROOT=${WORLDSCORE_ROOT}"

if [ "${NUM_SHARDS}" -gt 1 ]; then
    JOB=$(sbatch --array=0-$((NUM_SHARDS - 1)) --export="${EXPORTS}" "${WORKER}")
else
    JOB=$(sbatch --export="${EXPORTS}" "${WORKER}")
fi

echo "Submitted: ${JOB}"
JOB_ID=$(echo "${JOB}" | awk '{print $NF}')

# ── Queue merge — fires after all shards finish (any exit code) ───────────────
MERGE_EXPORTS="ALL,HARVEST_DIR=${HARVEST_DIR},MERGED_OUTPUT=${MERGED_OUTPUT},MODEL_SLUG=${MODEL_SLUG},NUM_SHARDS=${NUM_SHARDS},PHYSION_METRICS_DIR=${PHYSION_METRICS_DIR},CONDA_ENV=${CONDA_ENV},LOG_DIR=${LOG_DIR}"

MERGE_JOB=$(sbatch \
    --dependency=afterany:${JOB_ID} \
    --export="${MERGE_EXPORTS}" \
    "${MERGE_WORKER}")

echo "Merge job: ${MERGE_JOB} (waits on ${JOB_ID})"
echo ""
echo "Results → ${MERGED_OUTPUT}"
