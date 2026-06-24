#!/bin/bash
#SBATCH --job-name=pr-celltype
#SBATCH --output=slurm/out/%j_precision_recall_celltype.out
#SBATCH --error=slurm/out/%j_precision_recall_celltype.out
#SBATCH --time=00:30:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# Evaluate ChIP-seq precision/recall by retinal cell type for:
#   w_ij, k_ij, w_refined, w_hybrid, z_ij, and dw
# and generate the grouped precision bar chart.
#
# Run from bsc-screni/ after the current retinal weight decomposition:
#   sbatch slurm/run_precision_recall_celltype.sh

set -euo pipefail

CONTAINER="/tudelft.net/staff-umbrella/ScReNI/bsc-screni/container_0-1-3.sif"
CHIP_PATH="../refer/mmp9.TSV.5kb_TF_target.df.txt"
OUT_DIR="output/weight_decomposition_retina/precision_recall_celltype"

for F in \
    "$CONTAINER" \
    "output/weight_decomposition_retina/decomposed_sample.npz" \
    "output/weight_decomposition_retina/cell_meta.csv" \
    "data/processed/retinal_gene_labels.csv" \
    "$CHIP_PATH"; do
    if [[ ! -f "$F" ]]; then
        echo "ERROR: prerequisite not found: $F"
        echo "Re-run the current retinal weight decomposition before this job."
        exit 1
    fi
done

EXTRA_ARGS=("$@")
mkdir -p slurm/out "$OUT_DIR"

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Started     : $(date)"
if (($#)); then
    echo "Extra args  : ${EXTRA_ARGS[*]}"
else
    echo "Extra args  : none"
fi

apptainer exec \
    --writable-tmpfs --pwd /opt/app --containall \
    --bind data/:/opt/app/data/ \
    --bind output/:/opt/app/output/ \
    --bind ../refer/:/opt/refer/ \
    --bind scripts/:/opt/app/scripts/ \
    "$CONTAINER" \
    pixi run --manifest-path /opt/app/pixi.toml \
    python -u /opt/app/scripts/analyze_all_variants_chipseq.py \
        --data-dir output/weight_decomposition_retina \
        --gene-labels data/processed/retinal_gene_labels.csv \
        --chip-path /opt/refer/mmp9.TSV.5kb_TF_target.df.txt \
        --out-dir "$OUT_DIR" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

apptainer exec \
    --writable-tmpfs --pwd /opt/app --containall \
    --bind output/:/opt/app/output/ \
    --bind scripts/:/opt/app/scripts/ \
    "$CONTAINER" \
    pixi run --manifest-path /opt/app/pixi.toml \
    python -u /opt/app/scripts/plot_all_variants_chipseq.py --data-dir "$OUT_DIR"

echo "Finished : $(date)"
echo "Results  : $OUT_DIR"
