#!/bin/bash
#SBATCH --job-name=dl_retinal
#SBATCH --output=slurm/out/%j_download_retinal.out
#SBATCH --error=slurm/out/%j_download_retinal.out
#SBATCH --time=02:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

# Download mouse retinal development dataset (unpaired scRNA-seq + scATAC-seq)
# Source: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE181251
# Reference: Lyu et al. 2021, Cell Reports (doi:10.1016/j.celrep.2021.109994)
# Used in ScReNI paper for unpaired data benchmarking.

set -euo pipefail

OUTDIR="${1:-data/retinal_GSE181251}"
BASE_URL="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE181nnn/GSE181251/suppl"

mkdir -p "$OUTDIR"
cd "$OUTDIR"

echo "Downloading retinal development data to $(pwd)"
echo "Started: $(date)"

# --- Essential: developmental scATAC-seq (used by ScReNI) ---

# Cell annotations (cell types, timepoints) - 2.9 MB
wget -nc "${BASE_URL}/GSE181251_Single_Cell_ATACseq_cell_annotation.txt.gz"

# Peak annotations - 2.2 MB
wget -nc "${BASE_URL}/GSE181251_Single_Cell_ATACseq_peak_annotation.txt.gz"

# Count matrix (cells x peaks) - 1.0 GB
wget -nc "${BASE_URL}/GSE181251_Single_Cell_ATACseq_raw_matrix.mtx.gz"

# --- Perturbation experiments (may be useful for sub-projects) ---

# E14-E16 overexpression scATAC-seq - ~367 MB
wget -nc "${BASE_URL}/GSE181251_E14E16_overexpression_scATACseq_cell_annotation.txt.gz"
wget -nc "${BASE_URL}/GSE181251_E14E16_overexpression_scATACseq_peak_annotation.txt.gz"
wget -nc "${BASE_URL}/GSE181251_E14E16_overexpression_scATACseq_raw_matrix.mtx.gz"

# E14-E16 NFI overexpression scRNA-seq - ~111 MB
wget -nc "${BASE_URL}/GSE181251_E14E16_NFI_overexpression_scRNAseq_cell_annotation.txt.gz"
wget -nc "${BASE_URL}/GSE181251_E14E16_NFI_overexpression_scRNAseq_gene_annotation.txt.gz"
wget -nc "${BASE_URL}/GSE181251_E14E16_NFI_overexpression_scRNAseq_raw_matrix.mtx.gz"

# E14-P0 NFI overexpression scRNA-seq - ~245 MB
wget -nc "${BASE_URL}/GSE181251_E14P0_NFI_overexpression_scRNAseq_cell_annotation.txt.gz"
wget -nc "${BASE_URL}/GSE181251_E14P0_NFI_overexpression_scRNAseq_gene_annotation.txt.gz"
wget -nc "${BASE_URL}/GSE181251_E14P0_NFI_overexpression_scRNAseq_raw_matrix.mtx.gz"

# P0-P5 knockout scRNA-seq - ~127 MB
wget -nc "${BASE_URL}/GSE181251_P0P5_KO_cell_annotation.txt.gz"
wget -nc "${BASE_URL}/GSE181251_P0P5_KO_gene_annotation.txt.gz"
wget -nc "${BASE_URL}/GSE181251_P0P5_KO_raw_matrix.mtx.gz"

# P0-P5 overexpression scRNA-seq - ~389 MB
wget -nc "${BASE_URL}/GSE181251_P0P5_Overexpression_cell_annotation.txt.gz"
wget -nc "${BASE_URL}/GSE181251_P0P5_Overexpression_gene_annotation.txt.gz"
wget -nc "${BASE_URL}/GSE181251_P0P5_Overexpression_raw_matrix.mtx.gz"

# P2 Nfi knockout scRNA-seq - ~176 MB
wget -nc "${BASE_URL}/GSE181251_P2_Nfi_KO_scRNAseq_cell_annotation.txt.gz"
wget -nc "${BASE_URL}/GSE181251_P2_Nfi_KO_scRNAseq_gene_annotation.txt.gz"
wget -nc "${BASE_URL}/GSE181251_P2_Nfi_KO_scRNAseq_raw_matrix.mtx.gz"

# P2 knockout scATAC-seq - ~627 MB
wget -nc "${BASE_URL}/GSE181251_P2_KO_scATACseq_cell_annotation.txt.gz"
wget -nc "${BASE_URL}/GSE181251_P2_KO_scATACseq_peak_annotation.txt.gz"
wget -nc "${BASE_URL}/GSE181251_P2_KO_scATACseq_raw_matrix.mtx.gz"

# P14 knockout scATAC-seq - ~1.1 GB
wget -nc "${BASE_URL}/GSE181251_P14_KO_scATACseq_cell_annotation.txt.gz"
wget -nc "${BASE_URL}/GSE181251_P14_KO_scATACseq_peak_annotation.txt.gz"
wget -nc "${BASE_URL}/GSE181251_P14_KO_scATACseq_raw_matrix.mtx.gz"

# P2 Nfi ChIP-seq peaks - 171 KB
wget -nc "${BASE_URL}/GSE181251_P2_Nfi_peaks.txt.gz"

echo ""
echo "Done: $(date)"
echo "Files downloaded to: $(pwd)"
ls -lh
