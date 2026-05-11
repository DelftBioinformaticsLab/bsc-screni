#!/bin/bash
#SBATCH --job-name=compare-infer
#SBATCH --output=slurm/out/%j_compare_infer.out
#SBATCH --error=slurm/out/%j_compare_infer.out
#SBATCH --time=04:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Stage 1 of 2: Infer all networks and cache them to output/comparison/cache/.
#
# Runtime estimate (8 CPUs):
#   CSN     ~  2 min
#   kScReNI ~ 53 min  (GENIE3, parallelised over cells)
#   wScReNI ~ 34 min  (random-forest, parallelised over genes per cell)
#   LIONESS ~ disabled by default (set RUN_LIONESS=True in compare_with_r.py)
#   Total   ~ 90 min
#
# Networks are written to output/comparison/cache/ as compressed .npz files.
# After this job finishes, submit run_compare_analyse.sh to compute P/R,
# clustering ARI and generate figures (~10 min).
#
# Usage:
#   mkdir -p slurm/out
#   sbatch slurm/run_compare_infer.sh

set -euo pipefail

CONTAINER="/tudelft.net/staff-umbrella/ScReNI/bsc-screni/container_0-1-3.sif"

if [[ ! -f "$CONTAINER" ]]; then
    echo "ERROR: container not found: $CONTAINER"
    echo "  See docs/using_containers.md for how to build or copy the SIF."
    exit 1
fi

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Container   : $CONTAINER"
echo "Working dir : $(pwd)"
echo "Stage       : infer"
echo "Started     : $(date)"
echo

mkdir -p slurm/out output/comparison/cache

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
    python -u /opt/app/compare_with_r.py --stage infer

echo
echo "Finished : $(date)"
echo "Cached networks: output/comparison/cache/"
echo
echo "Next step:"
echo "  sbatch slurm/run_compare_analyse.sh"
