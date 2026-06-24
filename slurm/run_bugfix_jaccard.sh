#!/bin/bash
#SBATCH --job-name=bugfix-jaccard
#SBATCH --output=slurm/out/%j_bugfix_jaccard.out
#SBATCH --error=slurm/out/%j_bugfix_jaccard.out
#SBATCH --time=00:20:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# Jaccard similarity between normal R-compatible wScReNI and the per-TF ATAC
# bugfix previously implemented as the default in inference.py. This reads
# decomposed_sample.npz; no RF re-run is needed.
#
# Prerequisite:
#   output/weight_decomposition_retina/decomposed_sample.npz must contain either
#   w_matrices_normal + w_matrices_bugfix, or their legacy aliases
#   w_matrices_paper + w_matrices.
#
# Usage:
#   sbatch slurm/run_bugfix_jaccard.sh
#   sbatch slurm/run_bugfix_jaccard.sh --top-k 1000

set -euo pipefail

CONTAINER="/tudelft.net/staff-umbrella/ScReNI/bsc-screni/container_0-1-3.sif"

if [[ ! -f "$CONTAINER" ]]; then
    echo "ERROR: container not found: $CONTAINER"
    echo "  See docs/using_containers.md for how to build or copy the SIF."
    exit 1
fi

for F in \
    "output/weight_decomposition_retina/decomposed_sample.npz" \
    "output/weight_decomposition_retina/cell_meta.csv"; do
    if [[ ! -f "$F" ]]; then
        echo "ERROR: prerequisite not found: $F"
        echo "  Run slurm/run_weight_decomposition_retina.sh or slurm/run_decomp_retina_n400.sh first."
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

mkdir -p slurm/out output/weight_decomposition_retina/bugfix_jaccard

apptainer exec \
    --writable-tmpfs \
    --pwd /opt/app \
    --containall \
    --bind output/:/opt/app/output/ \
    --bind scripts/:/opt/app/scripts/ \
    "$CONTAINER" \
    pixi run --manifest-path /opt/app/pixi.toml \
    python -u /opt/app/scripts/analyze_bugfix_jaccard.py \
        --data-dir output/weight_decomposition_retina \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo
echo "Finished : $(date)"
echo "Results  : output/weight_decomposition_retina/bugfix_jaccard/"
