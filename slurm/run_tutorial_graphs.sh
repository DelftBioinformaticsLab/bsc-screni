#!/bin/bash
#SBATCH --job-name=tutorial-graphs
#SBATCH --output=slurm/out/%j_tutorial_graphs.out
#SBATCH --error=slurm/out/%j_tutorial_graphs.out
#SBATCH --time=00:30:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Reproduce all ScReNI tutorial graphs (Python equivalents of the R tutorial).
#
# Produces up to 6 PNGs in output/graphs/:
#   graph1_precision_recall_boxplots.png
#   graph2_kmeans_heatmap.png
#   graph3_integration_umap.png
#   graph4_degree_heatmap_umap.png   (CSN inference, ~3 min)
#   graph5_pseudotime.png
#   graph6_go_enrichment_bars.png    (requires gprofiler-official + internet)
#
# Prerequisites:
#   - Run from inside bsc-screni-main/
#   - ScReNI-master/ must be the parent folder (the default zip layout)
#   - slurm/out/ directory must exist:  mkdir -p slurm/out
#   - A container SIF must exist in the project root (see docs/using_containers.md)
#
# Usage:
#   mkdir -p slurm/out
#   sbatch slurm/run_tutorial_graphs.sh

set -euo pipefail

# The latest container — update this if you rebuild
CONTAINER="/tudelft.net/staff-umbrella/ScReNI/bsc-screni/container_0-1-3.sif"

if [[ ! -f "$CONTAINER" ]]; then
    echo "ERROR: container not found: $CONTAINER"
    echo "  Copy it to $(pwd)/ — see docs/using_containers.md"
    exit 1
fi

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Container   : $CONTAINER"
echo "Working dir : $(pwd)"
echo "Started     : $(date)"
echo

apptainer exec \
    --writable-tmpfs \
    --pwd /opt/app \
    --containall \
    --bind src/:/opt/app/src/ \
    --bind data/:/opt/app/data/ \
    --bind output/:/opt/app/output/ \
    --bind reproduce_tutorial_graphs.py:/opt/app/reproduce_tutorial_graphs.py \
    --bind ../data/:/opt/app/ScReNI-master/data/ \
    --env PYTHONPATH=/opt/app/src \
    "$CONTAINER" \
    pixi run --manifest-path /opt/app/pixi.toml \
    bash -c "
        # gprofiler-official is not in pixi.toml — install it on the fly for graph 6.
        # The --writable-tmpfs flag makes this work without modifying the SIF.
        pip install gprofiler-official --quiet --disable-pip-version-check 2>/dev/null || true

        python -u /opt/app/reproduce_tutorial_graphs.py
    "

echo
echo "Finished : $(date)"
echo "Graphs saved to: output/graphs/"
