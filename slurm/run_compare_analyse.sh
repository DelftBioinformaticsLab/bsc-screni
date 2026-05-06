#!/bin/bash
#SBATCH --job-name=compare-analyse
#SBATCH --output=slurm/out/%j_compare_analyse.out
#SBATCH --error=slurm/out/%j_compare_analyse.out
#SBATCH --time=00:40:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Stage 2 of 2: Load cached networks, compute clustering ARI, precision/recall,
# and generate all comparison figures.
#
# Runtime estimate (8 CPUs):
#   Load cached networks    < 1 min
#   Degree clustering       ~ 3 min
#   Precision/recall (P/R)  ~ 3 min  (parallelised, ~10 000 calls × 20 ms)
#   Figures + CSV output    < 1 min
#   Total                   ~ 8 min
#
# Prerequisite: run_compare_infer.sh must have completed successfully.
#   The cache is at output/comparison/cache/*.npz
#
# Usage:
#   sbatch slurm/run_compare_analyse.sh
#
# To submit stage 2 automatically after stage 1 finishes:
#   INFER_JOB=$(sbatch --parsable slurm/run_compare_infer.sh)
#   sbatch --dependency=afterok:$INFER_JOB slurm/run_compare_analyse.sh

set -euo pipefail

CONTAINER="/tudelft.net/staff-umbrella/ScReNI/bsc-screni/container_0-1-3.sif"

if [[ ! -f "$CONTAINER" ]]; then
    echo "ERROR: container not found: $CONTAINER"
    exit 1
fi

# Verify cache exists before submitting
CACHE_DIR="output/comparison/cache"
if [[ ! -d "$CACHE_DIR" ]] || [[ -z "$(ls -A "$CACHE_DIR" 2>/dev/null)" ]]; then
    echo "ERROR: Cache directory empty or missing: $CACHE_DIR"
    echo "  Run run_compare_infer.sh first."
    exit 1
fi

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Container   : $CONTAINER"
echo "Working dir : $(pwd)"
echo "Stage       : analyse"
echo "Cache dir   : $CACHE_DIR"
echo "Cached files: $(ls "$CACHE_DIR" | tr '\n' ' ')"
echo "Started     : $(date)"
echo

apptainer exec \
    --writable-tmpfs \
    --pwd /opt/app \
    --containall \
    --bind src/:/opt/app/src/ \
    --bind data/:/opt/app/data/ \
    --bind output/:/opt/app/output/ \
    --bind compare_with_r.py:/opt/app/compare_with_r.py \
    --bind ../data/:/opt/app/ScReNI-master/data/ \
    --bind ../refer/:/opt/app/ScReNI-master/refer/ \
    --env PYTHONPATH=/opt/app/src \
    --env SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1} \
    "$CONTAINER" \
    pixi run --manifest-path /opt/app/pixi.toml \
    python -u /opt/app/compare_with_r.py --stage analyse

echo
echo "Finished : $(date)"
echo "Results saved to: output/comparison/"
