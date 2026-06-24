#!/bin/bash
#SBATCH --job-name=jaccard-kij
#SBATCH --output=slurm/out/%j_jaccard_kij.out
#SBATCH --error=slurm/out/%j_jaccard_kij.out
#SBATCH --time=00:20:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# Compute Jaccard similarity to k_ij for retina and SEA-AD, then create the
# two-bar-per-method comparison chart.

set -euo pipefail

CONTAINER="/tudelft.net/staff-umbrella/ScReNI/bsc-screni/container_0-1-3.sif"
if [[ ! -f "$CONTAINER" ]]; then
    echo "ERROR: container not found: $CONTAINER"
    echo "  See docs/using_containers.md for how to build or copy the SIF."
    exit 1
fi

for F in \
    "output/weight_decomposition_retina/decomposed_sample.npz" \
    "output/weight_decomposition_retina/cell_meta.csv" \
    "output/seaad_grn/weight_decomposition/decomposed_sample.npz" \
    "output/seaad_grn/weight_decomposition/cell_meta.csv"; do
    if [[ ! -f "$F" ]]; then
        echo "ERROR: prerequisite not found: $F"
        echo "  Run the retina and SEA-AD weight decomposition jobs first."
        exit 1
    fi
done

EXTRA_ARGS=("$@")

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Container   : $CONTAINER"
echo "Working dir : $(pwd)"
if (($#)); then
    echo "Extra args  : ${EXTRA_ARGS[*]}"
else
    echo "Extra args  : none"
fi
echo "Started     : $(date)"
echo

mkdir -p slurm/out \
    output/weight_decomposition_retina/jaccard_variants \
    output/seaad_grn/weight_decomposition/jaccard_variants

apptainer exec \
    --writable-tmpfs --pwd /opt/app --containall \
    --bind output/:/opt/app/output/ \
    --bind scripts/:/opt/app/scripts/ \
    "$CONTAINER" \
    pixi run --manifest-path /opt/app/pixi.toml \
    python -u /opt/app/scripts/compute_jaccard_variants_seaad.py \
        --data-dir output/weight_decomposition_retina \
        --dataset-name retina \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

apptainer exec \
    --writable-tmpfs --pwd /opt/app --containall \
    --bind output/:/opt/app/output/ \
    --bind scripts/:/opt/app/scripts/ \
    "$CONTAINER" \
    pixi run --manifest-path /opt/app/pixi.toml \
    python -u /opt/app/scripts/compute_jaccard_variants_seaad.py \
        --data-dir output/seaad_grn/weight_decomposition \
        --dataset-name seaad \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

apptainer exec \
    --writable-tmpfs --pwd /opt/app --containall \
    --bind output/:/opt/app/output/ \
    --bind scripts/:/opt/app/scripts/ \
    "$CONTAINER" \
    pixi run --manifest-path /opt/app/pixi.toml \
    python -u /opt/app/scripts/plot_jaccard_variants_retina_seaad.py

echo
echo "Finished : $(date)"
echo "Results  : output/weight_decomposition_retina/jaccard_variants/"
