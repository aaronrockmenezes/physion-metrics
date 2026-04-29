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
# ─────────────────────────────────────────────────────────────────────────────

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SLUG="${MODEL// /_}"
LOG_DIR="${PHYSION_METRICS_DIR}/logs/${MODEL_SLUG}/${TIMESTAMP}"
WORKER="${PHYSION_METRICS_DIR}/scripts/worker.sh"

mkdir -p "${LOG_DIR}"

echo "Model:   ${MODEL}"
echo "Shards:  ${NUM_SHARDS}"
echo "Logs  →  ${LOG_DIR}"
echo ""

EXPORTS="ALL,MODEL=${MODEL},NUM_SHARDS=${NUM_SHARDS},LOG_DIR=${LOG_DIR},PHYSION_METRICS_DIR=${PHYSION_METRICS_DIR},VIDEO_DIR=${VIDEO_DIR},JSON_PATH=${JSON_PATH},WORLDSCORE_ROOT=${WORLDSCORE_ROOT}"

if [ "${NUM_SHARDS}" -gt 1 ]; then
    JOB=$(sbatch --array=0-$((NUM_SHARDS - 1)) --export="${EXPORTS}" "${WORKER}")
else
    JOB=$(sbatch --export="${EXPORTS}" "${WORKER}")
fi

echo "Submitted: ${JOB}"
echo ""
echo "Merge when done:"
echo "  python scripts/merge_results.py --pattern '${LOG_DIR}/*.json' --output '${LOG_DIR}/${MODEL_SLUG}_merged.json'"
