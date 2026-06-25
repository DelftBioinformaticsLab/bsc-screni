#!/bin/bash
#SBATCH --job-name=compare-nets
#SBATCH --output=slurm/out/%j_compare_nets.out
#SBATCH --error=slurm/out/%j_compare_nets.out
#SBATCH --time=04:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

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
echo "Stage       : Nets"
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
    --bind compare_nets.py:/opt/app/compare_nets.py \
    --bind ../data/:/opt/app/ScReNI-master/data/ \
    --bind ../refer/:/opt/app/ScReNI-master/refer/ \
    --env PYTHONPATH=/opt/app/src \
    --env SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1} \
    "$CONTAINER" \
    pixi run --manifest-path /opt/app/pixi.toml \
    python -u /opt/app/compare_nets.py

echo
echo "Finished : $(date)"
echo "saved files: output/comparison/"