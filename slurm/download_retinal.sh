#!/bin/bash
#SBATCH --job-name=dl_retinal
#SBATCH --output=slurm/out/%j_download_retinal.out
#SBATCH --error=slurm/out/%j_download_retinal.out
#SBATCH --time=02:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

# Download mouse retinal development dataset (unpaired scRNA-seq + scATAC-seq)
#
# scATAC-seq: GSE181251 (Lyu et al. 2021, Cell Reports)
# scRNA-seq:  GSE118614 (Clark et al. 2019, Neuron)
#   Processed data + annotations from: https://github.com/gofflab/developing_mouse_retina_scRNASeq
#
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

# ===========================================================================
#  Developmental scRNA-seq (Clark et al. 2019, Neuron)
#  Source: https://github.com/gofflab/developing_mouse_retina_scRNASeq
#  GEO: GSE118614
#  120,804 cells, 10 timepoints (E11-P14), 10X Chromium 3' v2
# ===========================================================================

SCRNA_DIR="scRNAseq_clark2019"
mkdir -p "$SCRNA_DIR"

# Count matrix (MTX format)
wget -nc -O "$SCRNA_DIR/10x_mouse_retina_development.mtx" \
  "https://www.dropbox.com/s/6d76z4grcnaxgcg/10x_mouse_retina_development.mtx?dl=1"

# Cell annotations with cell types, UMAP coords, timepoints (updated 3/11/21, includes horizontal cells)
wget -nc -O "$SCRNA_DIR/10x_mouse_retina_pData_umap2_CellType_annot_w_horiz.csv" \
  "https://www.dropbox.com/s/q5apkp52t0vy7lo/10x_Mouse_retina_pData_umap2_CellType_annot_w_horiz.csv?dl=1"

# Original cell annotations
wget -nc -O "$SCRNA_DIR/10x_mouse_retina_development_phenoData.csv" \
  "https://www.dropbox.com/s/y5lho9ifzoktjcs/10x_mouse_retina_development_phenotype.csv?dl=1"

# Gene/feature annotations
wget -nc -O "$SCRNA_DIR/10x_mouse_retina_development_featureData.csv" \
  "https://www.dropbox.com/s/1mc4geu3hixrxhj/10x_mouse_retina_development_feature.csv?dl=1"

echo ""
echo "Done: $(date)"
echo "Files downloaded to: $(pwd)"
ls -lh
echo ""
echo "scRNA-seq files:"
ls -lh "$SCRNA_DIR"
