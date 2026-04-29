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

echo "=========================================="
echo "Merge job: ${SLURM_JOB_ID}"
echo "Harvest:   ${HARVEST_DIR}"
echo "Output:    ${MERGED_OUTPUT}"
echo "=========================================="

cd "${PHYSION_METRICS_DIR}"

python scripts/merge_results.py \
    --pattern "${HARVEST_DIR}/${MODEL_SLUG}_shard*.json" \
    --output  "${MERGED_OUTPUT}"

echo "Done: merge exited with code $?"
