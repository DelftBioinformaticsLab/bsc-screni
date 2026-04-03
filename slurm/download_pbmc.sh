#!/bin/bash
#SBATCH --job-name=dl_pbmc
#SBATCH --output=slurm/out/%j_download_pbmc.out
#SBATCH --error=slurm/out/%j_download_pbmc.out
#SBATCH --time=02:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

# Download PBMC 10X Multiome dataset (paired scRNA-seq + scATAC-seq)
# Source: https://www.10xgenomics.com/datasets/pbmc-from-a-healthy-donor-no-cell-sorting-10-k-1-standard-2-0-0
# Used in ScReNI paper for paired data benchmarking.

set -euo pipefail

OUTDIR="${1:-data/pbmc_unsorted_10k}"
BASE_URL="https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_unsorted_10k"

mkdir -p "$OUTDIR"
cd "$OUTDIR"

echo "Downloading PBMC 10X Multiome data to $(pwd)"
echo "Started: $(date)"

# --- Essential files for scReNI ---

# Filtered feature barcode matrix (HDF5) - 152 MB
# Contains both GEX and ATAC count matrices for filtered (cell-containing) barcodes
curl -OLC - "${BASE_URL}/pbmc_unsorted_10k_filtered_feature_bc_matrix.h5"

# ATAC fragments file - 2.92 GB
# Per-fragment information needed for chromatin accessibility analysis
curl -OLC - "${BASE_URL}/pbmc_unsorted_10k_atac_fragments.tsv.gz"

# ATAC fragments index - 1.3 MB
curl -OLC - "${BASE_URL}/pbmc_unsorted_10k_atac_fragments.tsv.gz.tbi"

# ATAC peak locations - 2.67 MB
curl -OLC - "${BASE_URL}/pbmc_unsorted_10k_atac_peaks.bed"

# ATAC peak annotations (gene-peak associations) - 6.2 MB
curl -OLC - "${BASE_URL}/pbmc_unsorted_10k_atac_peak_annotation.tsv"

# --- Metadata & QC files ---

# Run summary metrics - 1.83 KB
curl -OLC - "${BASE_URL}/pbmc_unsorted_10k_summary.csv"

# Per barcode metrics - 89.5 MB
curl -OLC - "${BASE_URL}/pbmc_unsorted_10k_per_barcode_metrics.csv"

# Run summary report - 6.26 MB
curl -OLC - "${BASE_URL}/pbmc_unsorted_10k_web_summary.html"

# Secondary analysis outputs (clustering, t-SNE, etc.) - 367 MB
curl -OLC - "${BASE_URL}/pbmc_unsorted_10k_analysis.tar.gz"

# --- Optional: MEX format matrix (uncomment if needed) ---
# Filtered feature barcode matrix (MEX directory) - 375 MB
# curl -OLC - "${BASE_URL}/pbmc_unsorted_10k_filtered_feature_bc_matrix.tar.gz"

# Raw feature barcode matrix (HDF5) - 192 MB
# curl -OLC - "${BASE_URL}/pbmc_unsorted_10k_raw_feature_bc_matrix.h5"

# --- Skipped (too large / not needed for scReNI) ---
# GEX BAM         - 49.3 GB
# ATAC BAM        - 44.5 GB
# CLOUPE          -  1.3 GB
# BIGWIG          -  2.5 GB
# GEX molecule h5 -  276 MB

echo ""
echo "Verifying checksums..."

# md5 checksums from 10X Genomics
md5sum -c - <<'EOF'
72e727ca260df989cf7167ebc65b4f7e  pbmc_unsorted_10k_filtered_feature_bc_matrix.h5
5004a145c897a09511216a181bb07518  pbmc_unsorted_10k_atac_fragments.tsv.gz
70ab63c3ac73f6dfa1805187785de189  pbmc_unsorted_10k_atac_fragments.tsv.gz.tbi
cbb711c98a8baaf06f9975eccaba04fc  pbmc_unsorted_10k_atac_peaks.bed
2aeb596083be72d180b93e9a882c4772  pbmc_unsorted_10k_atac_peak_annotation.tsv
b35ce93f9b09efcf1c5adc8979427b01  pbmc_unsorted_10k_summary.csv
228130d85158837cf359882e09032807  pbmc_unsorted_10k_per_barcode_metrics.csv
77428554c245c57fe2ea32edff8ce0fa  pbmc_unsorted_10k_web_summary.html
24e86893e48dfa4711f1aad275b18df7  pbmc_unsorted_10k_analysis.tar.gz
EOF

echo ""
echo "Done: $(date)"
echo "Files downloaded to: $(pwd)"
ls -lh
