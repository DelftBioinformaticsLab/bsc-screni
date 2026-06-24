#!/bin/bash
#SBATCH --job-name=weight-composition
#SBATCH --output=slurm/out/%j_weight_composition.out
#SBATCH --error=slurm/out/%j_weight_composition.out
#SBATCH --time=00:15:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G

# Compute average wScReNI weight composition by cell type and render a compact
# stacked bar plot for both saved decomposition datasets.
#
# Inputs:
#   output/weight_decomposition_retina/decomposed_sample.npz
#   output/weight_decomposition_retina/cell_meta.csv
#   output/seaad_grn/weight_decomposition/decomposed_sample.npz
#   output/seaad_grn/weight_decomposition/cell_meta.csv
#
# Outputs:
#   output/weight_decomposition_retina/weight_composition/
#   output/seaad_grn/weight_decomposition/weight_composition/
#
# Usage:
#   mkdir -p slurm/out
#   sbatch final/weight_composition/run_weight_composition.sh
#   # or, from bsc-screni after copying/linking this script:
#   sbatch slurm/run_weight_composition.sh

set -euo pipefail

CONTAINER="/tudelft.net/staff-umbrella/ScReNI/bsc-screni/container_0-1-3.sif"

if [[ -d "output" && -f "pixi.toml" ]]; then
    PROJECT_DIR="$(pwd)"
elif [[ -d "bsc-screni/output" ]]; then
    PROJECT_DIR="$(pwd)/bsc-screni"
else
    PROJECT_DIR="$(pwd)"
fi
SCRIPT_DIR="$PROJECT_DIR/scripts"

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
    if [[ ! -f "$PROJECT_DIR/$F" ]]; then
        echo "ERROR: required input not found: $F"
        echo "  Run the corresponding weight decomposition job first."
        exit 1
    fi
done

for F in \
    "scripts/analyze_weight_composition.py" \
    "scripts/plot_weight_composition.py"; do
    if [[ ! -f "$PROJECT_DIR/$F" ]]; then
        echo "ERROR: required script not found: $F"
        echo "  Place the Python scripts under bsc-screni/scripts/."
        exit 1
    fi
done

echo "Job ID      : ${SLURM_JOB_ID:-local}"
echo "Node        : $(hostname)"
echo "Container   : $CONTAINER"
echo "Working dir : $(pwd)"
echo "Project dir : $PROJECT_DIR"
echo "Started     : $(date)"
echo

mkdir -p "$PROJECT_DIR/slurm/out" \
         "$PROJECT_DIR/output/weight_decomposition_retina/weight_composition" \
         "$PROJECT_DIR/output/seaad_grn/weight_decomposition/weight_composition"

run_dataset() {
    local label="$1"
    local data_dir="$2"
    local sort_by="$3"

    echo ">>> ${label}"
    echo

    apptainer exec \
        --writable-tmpfs \
        --pwd /opt/app \
        --containall \
        --bind "$PROJECT_DIR/output/:/opt/app/output/" \
        --bind "$SCRIPT_DIR/:/opt/app/scripts/" \
        --env SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1} \
        "$CONTAINER" \
        pixi run --manifest-path /opt/app/pixi.toml \
        python -u /opt/app/scripts/analyze_weight_composition.py \
            --data-dir "${data_dir}" \
            --dataset-name "${label}"

    apptainer exec \
        --writable-tmpfs \
        --pwd /opt/app \
        --containall \
        --bind "$PROJECT_DIR/output/:/opt/app/output/" \
        --bind "$SCRIPT_DIR/:/opt/app/scripts/" \
        "$CONTAINER" \
        pixi run --manifest-path /opt/app/pixi.toml \
        python -u /opt/app/scripts/plot_weight_composition.py \
            --data-dir "${data_dir}/weight_composition" \
            --sort-by "${sort_by}" \
            --title "${label} weight composition"
}

run_dataset "Mouse retina" "output/weight_decomposition_retina" "cell_type"
echo
run_dataset "SEA-AD" "output/seaad_grn/weight_decomposition" "peak_j_coef"

echo
echo "Finished : $(date)"
echo "Results  : output/.../weight_composition/"
