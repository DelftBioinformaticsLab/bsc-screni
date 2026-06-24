#!/bin/bash
#SBATCH --job-name=impact-iij
#SBATCH --output=slurm/out/%j_impact_iij.out
#SBATCH --error=slurm/out/%j_impact_iij.out
#SBATCH --time=00:20:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# Compute and plot weight composition only on edges where I_ij=1 and the
# saved per-cell ATAC boost is strictly positive, for mouse retina and SEA-AD.
#
# Copy before submitting:
#   analyze_impact_i_ij.py -> bsc-screni/scripts/
#   plot_impact_i_ij.py    -> bsc-screni/scripts/
#   run_impact_i_ij.sh     -> bsc-screni/slurm/
#
# Usage:
#   sbatch slurm/run_impact_i_ij.sh
#   sbatch slurm/run_impact_i_ij.sh --no-seaad
#   sbatch slurm/run_impact_i_ij.sh --seaad-prefix sub42

set -euo pipefail

CONTAINER="/tudelft.net/staff-umbrella/ScReNI/bsc-screni/container_0-1-3.sif"
SEAAD_PREFIX="sub42"
SKIP_RETINA=false
SKIP_SEAAD=false

while (($#)); do
    case "$1" in
        --no-retina)
            SKIP_RETINA=true
            shift
            ;;
        --no-seaad)
            SKIP_SEAAD=true
            shift
            ;;
        --seaad-prefix)
            if (($# < 2)); then
                echo "ERROR: --seaad-prefix requires a value"
                exit 1
            fi
            SEAAD_PREFIX="$2"
            shift 2
            ;;
        --seaad-prefix=*)
            SEAAD_PREFIX="${1#*=}"
            shift
            ;;
        *)
            echo "ERROR: unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ ! -f "$CONTAINER" ]]; then
    echo "ERROR: container not found: $CONTAINER"
    exit 1
fi

for F in scripts/analyze_impact_i_ij.py scripts/plot_impact_i_ij.py; do
    if [[ ! -f "$F" ]]; then
        echo "ERROR: required script not found: $F"
        exit 1
    fi
done

if [[ "$SKIP_RETINA" == false ]]; then
    for F in \
        output/weight_decomposition_retina/decomposed_sample.npz \
        output/weight_decomposition_retina/cell_meta.csv \
        data/processed/retinal_gene_labels.csv \
        data/processed/retinal_triplets.csv; do
        [[ -f "$F" ]] || { echo "ERROR: required retina input not found: $F"; exit 1; }
    done
fi

if [[ "$SKIP_SEAAD" == false ]]; then
    for F in \
        output/seaad_grn/weight_decomposition/decomposed_sample.npz \
        output/seaad_grn/weight_decomposition/cell_meta.csv \
        data/processed/seaad/seaad_paired_${SEAAD_PREFIX}_gene_labels.csv \
        data/processed/seaad/seaad_paired_${SEAAD_PREFIX}_triplets.csv; do
        [[ -f "$F" ]] || { echo "ERROR: required SEA-AD input not found: $F"; exit 1; }
    done
fi

echo "Job ID      : ${SLURM_JOB_ID:-local}"
echo "Node        : $(hostname)"
echo "Working dir : $(pwd)"
echo "SEA-AD pref : $SEAAD_PREFIX"
echo "Started     : $(date)"
echo

mkdir -p slurm/out \
    output/weight_decomposition_retina/impact_i_ij \
    output/seaad_grn/weight_decomposition/impact_i_ij

run_dataset() {
    local label="$1"
    local data_dir="$2"
    local gene_labels="$3"
    local triplets="$4"
    local sort_by="$5"
    local gene_h5ad="${6:-}"

    echo ">>> $label"
    analysis_args=(
        --data-dir "$data_dir"
        --gene-labels "$gene_labels"
        --triplets "$triplets"
        --dataset-name "$label"
    )
    if [[ -n "$gene_h5ad" && -f "$gene_h5ad" ]]; then
        analysis_args+=(--gene-h5ad "$gene_h5ad")
    fi

    apptainer exec \
        --writable-tmpfs --pwd /opt/app --containall \
        --bind data/:/opt/app/data/ \
        --bind output/:/opt/app/output/ \
        --bind scripts/:/opt/app/scripts/ \
        "$CONTAINER" \
        pixi run --manifest-path /opt/app/pixi.toml \
        python -u /opt/app/scripts/analyze_impact_i_ij.py \
            "${analysis_args[@]}"

    apptainer exec \
        --writable-tmpfs --pwd /opt/app --containall \
        --bind output/:/opt/app/output/ \
        --bind scripts/:/opt/app/scripts/ \
        "$CONTAINER" \
        pixi run --manifest-path /opt/app/pixi.toml \
        python -u /opt/app/scripts/plot_impact_i_ij.py \
            --data-dir "$data_dir/impact_i_ij" \
            --sort-by "$sort_by" \
            --title "$label: composition of practically boosted I_ij edges"
}

if [[ "$SKIP_RETINA" == false ]]; then
    run_dataset \
        "Mouse retina" \
        "output/weight_decomposition_retina" \
        "data/processed/retinal_gene_labels.csv" \
        "data/processed/retinal_triplets.csv" \
        "cell_type" \
        "data/processed/retinal_rna_sub.h5ad"
fi

if [[ "$SKIP_SEAAD" == false ]]; then
    run_dataset \
        "SEA-AD" \
        "output/seaad_grn/weight_decomposition" \
        "data/processed/seaad/seaad_paired_${SEAAD_PREFIX}_gene_labels.csv" \
        "data/processed/seaad/seaad_paired_${SEAAD_PREFIX}_triplets.csv" \
        "atac_boost"
fi

echo
echo "Finished : $(date)"
echo "Results  : output/.../impact_i_ij/"
