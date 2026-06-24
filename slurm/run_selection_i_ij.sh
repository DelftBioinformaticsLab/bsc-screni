#!/bin/bash
#SBATCH --job-name=selection-iij
#SBATCH --output=slurm/out/%j_selection_iij.out
#SBATCH --error=slurm/out/%j_selection_iij.out
#SBATCH --time=00:10:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Count the I_ij selection funnel for mouse retina and SEA-AD:
#   1. all directed non-self gene-gene pairs
#   2. pairs where regulator gene j is a TF
#   3. pairs where I_ij=1 (motif in correlated peak)
#
# Run from bsc-screni after copying:
#   analyze_selection_i_ij.py -> scripts/
#   run_selection_i_ij.sh     -> slurm/
#
# Usage:
#   sbatch slurm/run_selection_i_ij.sh
#   sbatch slurm/run_selection_i_ij.sh --no-seaad
#   sbatch slurm/run_selection_i_ij.sh --seaad-prefix sub42

set -euo pipefail

CONTAINER="/tudelft.net/staff-umbrella/ScReNI/bsc-screni/container_0-1-3.sif"

if [[ ! -f "$CONTAINER" ]]; then
    echo "ERROR: container not found: $CONTAINER"
    echo "  See docs/using_containers.md for how to build or copy the SIF."
    exit 1
fi

SKIP_RETINA=false
SKIP_SEAAD=false
SEAAD_PREFIX="sub42"
ORIGINAL_ARGC=$#
EXTRA_ARGS=("$@")
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
            shift
            ;;
    esac
done

if [[ "$SKIP_RETINA" == false ]]; then
    for F in \
        "data/processed/retinal_gene_labels.csv" \
        "data/processed/retinal_triplets.csv"; do
        if [[ ! -f "$F" ]]; then
            echo "ERROR: required retina input not found: $F"
            exit 1
        fi
    done
fi

if [[ "$SKIP_SEAAD" == false ]]; then
    for F in \
        "data/processed/seaad/seaad_paired_${SEAAD_PREFIX}_gene_labels.csv" \
        "data/processed/seaad/seaad_paired_${SEAAD_PREFIX}_triplets.csv"; do
        if [[ ! -f "$F" ]]; then
            echo "ERROR: required SEA-AD input not found: $F"
            echo "  Use --seaad-prefix if you need a prefix other than '${SEAAD_PREFIX}'."
            exit 1
        fi
    done
fi

echo "Job ID      : ${SLURM_JOB_ID:-local}"
echo "Node        : $(hostname)"
echo "Container   : $CONTAINER"
echo "Working dir : $(pwd)"
if ((ORIGINAL_ARGC)); then
    echo "Extra args  : ${EXTRA_ARGS[*]}"
else
    echo "Extra args  : none"
fi
echo "Started     : $(date)"
echo

mkdir -p slurm/out output/iij_selection

apptainer exec \
    --writable-tmpfs \
    --pwd /opt/app \
    --containall \
    --bind data/:/opt/app/data/ \
    --bind output/:/opt/app/output/ \
    --bind scripts/:/opt/app/scripts/ \
    "$CONTAINER" \
    pixi run --manifest-path /opt/app/pixi.toml \
    python -u /opt/app/scripts/analyze_selection_i_ij.py \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo
echo "Finished : $(date)"
echo "Results  : output/iij_selection/"
