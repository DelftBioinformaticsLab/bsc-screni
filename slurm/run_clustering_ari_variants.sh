#!/bin/bash
#SBATCH --job-name=cluster-ari
#SBATCH --output=slurm/out/%j_clustering_ari.out
#SBATCH --error=slurm/out/%j_clustering_ari.out
#SBATCH --time=00:30:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Network-based cell clustering ARI for all six wScReNI formula variants.
# Replicates the exact R Fig 3B evaluation path from Xu et al. 2025 and
# extends it to the novel variants from this thesis.
#
# VARIANTS EVALUATED
# ------------------
#   w_ij       dw + pk + bo_paper   original wScReNI  (R shared-boost formula)
#   k_ij       dk                   original kScReNI  (RNA-only)
#   w_refined  dw + bo              proposed: removes peak_j_coef
#   w_hybrid   dk + bo              kScReNI expression + corrected per-TF boost
#   z_ij       dw + pk              wScReNI with peak_j_coef, no boost
#   dw         dw                   wScReNI RF expression importances only
#
# R EVALUATION PATH (reproduced exactly)
# ---------------------------------------
#   1. Rank weight matrix in column-major (Fortran) order, self-edges included.
#   2. Keep top-k entries → binary indicator matrix.
#   3. colSums → out-degree per gene per cell  (n_cells × n_genes matrix).
#   4. dist(cor(log(degree + 1))), complete linkage.
#   5. Cut to n_cell_types clusters, compare to known labels via ARI.
#
# PAPER REFERENCE  (n=400, top-500, complete linkage)
#   kScReNI = 0.388   wScReNI = 0.481
#
# READS
#   output/weight_decomposition_retina/decomposed_sample.npz
#   output/weight_decomposition_retina/cell_meta.csv
#
# WRITES
#   output/weight_decomposition_retina/clustering_ari/
#     clustering_ari.csv            one row per variant, ARI at each k
#     clustering_ari_summary.json   metadata + paper reference
#     clustering_ari_plot.pdf       line plot (one line per variant)
#     clustering_ari_plot.png       same at 300 dpi
#
# USAGE
#   mkdir -p slurm/out
#   sbatch slurm/run_clustering_ari_variants.sh
#   sbatch slurm/run_clustering_ari_variants.sh --top-ks 400 500 600 700 800 900 1000 1100 1200 1300 1400
#   Default: top-k 400 through 1400 in steps of 100.

set -eo pipefail
export PYTHONHASHSEED=42

CONTAINER="/tudelft.net/staff-umbrella/ScReNI/bsc-screni/container_0-1-3.sif"
if [[ ! -f "$CONTAINER" ]]; then
    echo "ERROR: container not found: $CONTAINER"; exit 1
fi
for F in "output/weight_decomposition_retina/decomposed_sample.npz" \
         "output/weight_decomposition_retina/cell_meta.csv"; do
    if [[ ! -f "$F" ]]; then
        echo "ERROR: prerequisite not found: $F"
        echo "  Run slurm/run_decomp_retina_n400.sh first."; exit 1
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

mkdir -p slurm/out output/weight_decomposition_retina/clustering_ari

# ── Step 1: compute ARI ───────────────────────────────────────────────────────
apptainer exec \
    --writable-tmpfs --pwd /opt/app --containall \
    --bind src/:/opt/app/src/ \
    --bind data/:/opt/app/data/ \
    --bind output/:/opt/app/output/ \
    --bind scripts/:/opt/app/scripts/ \
    --env PYTHONPATH=/opt/app/src \
    --env SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1} \
    "$CONTAINER" \
    pixi run --manifest-path /opt/app/pixi.toml \
    python -u /opt/app/scripts/analyze_clustering_ari_variants.py \
        --data-dir output/weight_decomposition_retina \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

# ── Step 2: generate line plot ────────────────────────────────────────────────
apptainer exec \
    --writable-tmpfs --pwd /opt/app --containall \
    --bind src/:/opt/app/src/ \
    --bind output/:/opt/app/output/ \
    --bind scripts/:/opt/app/scripts/ \
    --env PYTHONPATH=/opt/app/src \
    "$CONTAINER" \
    pixi run --manifest-path /opt/app/pixi.toml \
    python -u /opt/app/scripts/plot_clustering_ari.py \
        --data-dir output/weight_decomposition_retina

echo
echo "Finished : $(date)"
echo "Results  : output/weight_decomposition_retina/clustering_ari/"
