#!/bin/bash
# Merge worker — auto-queued by launch.sh after all shards finish.
# DO NOT EDIT config here — receives env vars from launch.sh.

#SBATCH --job-name=physion_merge
#SBATCH --output=/tmp/physion_merge_%j.out
#SBATCH --error=/tmp/physion_merge_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=batch

source ~/.bashrc
conda activate "${CONDA_ENV}"

exec > >(tee -a "${LOG_DIR}/merge.out") 2> >(tee -a "${LOG_DIR}/merge.err" >&2)

RUN_DIR="$(dirname "${LOG_DIR}")"
STATUS_LOG="${RUN_DIR}/run_status.log"
LOCK_FILE="${RUN_DIR}/.status.lock"

log_status() {
    local level="$1"
    local msg="$2"
    (
        flock -x 200
        printf "[%s] [%-5s] MERGE %s\n" \
            "$(date '+%Y-%m-%d %H:%M:%S')" "${level}" "${msg}" \
            >> "${STATUS_LOG}"
    ) 200>"${LOCK_FILE}"
}

log_status "START" "job ${SLURM_JOB_ID} on $(hostname) — $(date +%H:%M:%S)"

echo "=========================================="
echo "Merge job: ${SLURM_JOB_ID}"
echo "Harvest:   ${HARVEST_DIR}"
echo "Output:    ${MERGED_OUTPUT}"
echo "=========================================="

cd "${PHYSION_METRICS_DIR}"

# ── Check for missing shard JSONs ─────────────────────────────────────────────
echo ""
echo "Checking shard completeness..."
MISSING=()
for i in $(seq 0 $(( NUM_SHARDS - 1 )) ); do
    SHARD_FILE="${HARVEST_DIR}/${MODEL_SLUG}_shard${i}.json"
    if [ ! -f "${SHARD_FILE}" ]; then
        MISSING+=("${i}")
        log_status "WARN" "missing shard ${i} — no JSON at ${SHARD_FILE}"
        echo "  [MISSING] shard ${i}"
    fi
done

if [ ${#MISSING[@]} -eq 0 ]; then
    echo "  All ${NUM_SHARDS} shards present."
    log_status "INFO" "all ${NUM_SHARDS} shards present"
else
    echo "  ${#MISSING[@]} missing shard(s): ${MISSING[*]}"
    log_status "WARN" "${#MISSING[@]} shard(s) missing: ${MISSING[*]}"
fi

# ── Merge ─────────────────────────────────────────────────────────────────────
echo ""
python scripts/merge_results.py \
    --pattern "${HARVEST_DIR}/${MODEL_SLUG}_shard*.json" \
    --output  "${MERGED_OUTPUT}"

CODE=$?
if [ $CODE -eq 0 ]; then
    log_status "OK" "merged → ${MERGED_OUTPUT} — $(date +%H:%M:%S)"

    # ── Summary ───────────────────────────────────────────────────────────────
    SUMMARY_TXT="${MERGED_OUTPUT%.json}_summary.txt"
    SUMMARY_JSON="${MERGED_OUTPUT%.json}_summary.json"
    echo ""
    echo "Running summary..."
    python scripts/summarize_results.py \
        --input       "${MERGED_OUTPUT}" \
        --output      "${SUMMARY_TXT}" \
        --output-json "${SUMMARY_JSON}"
    echo "Summary txt  → ${SUMMARY_TXT}"
    echo "Summary json → ${SUMMARY_JSON}"
    log_status "OK" "summary → ${SUMMARY_TXT} + ${SUMMARY_JSON}"
else
    log_status "FAIL" "merge exit code ${CODE}"
fi

echo "Done: merge exited with code ${CODE}"
