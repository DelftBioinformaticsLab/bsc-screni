#!/bin/bash
#SBATCH --job-name=dl_seaad
#SBATCH --output=slurm/out/%j_download_seaad.out
#SBATCH --error=slurm/out/%j_download_seaad.out
#SBATCH --time=04:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# Download SEA-AD (Seattle Alzheimer's Disease) MTG single-cell data.
# Source: Allen Brain Map / SEA-AD portal
# S3 bucket: s3://sea-ad-single-cell-profiling (public, no credentials needed)
#
# Two h5ad files:
#   - RNA (~20 GB): pools singleome snRNA-seq (84 donors) + multiome RNA (28 donors)
#   - ATAC (~10 GB): pools singleome snATAC-seq + multiome ATAC
#
# A metadata column in each file distinguishes multiome from singleome nuclei.
# Splitting into paired/unpaired branches happens in the processing step.

set -euo pipefail

OUTDIR="${1:-data/seaad}"
BUCKET="s3://sea-ad-single-cell-profiling"
CONTAINER="container_0-1-3.sif"

AWS="apptainer exec --writable-tmpfs --pwd /opt/app --containall \
  --bind data/:/opt/app/data/ \
  ${CONTAINER} pixi run --manifest-path /opt/app/pixi.toml aws"

mkdir -p "$OUTDIR"

echo "Downloading SEA-AD MTG data to $(pwd)/${OUTDIR}"
echo "Started: $(date)"

# ===========================================================================
#  List bucket contents to verify exact filenames
# ===========================================================================

echo ""
echo "=== Listing MTG/RNAseq/ ==="
${AWS} s3 ls --no-sign-request "${BUCKET}/MTG/RNAseq/" | head -20

echo ""
echo "=== Listing MTG/ATACseq/ ==="
${AWS} s3 ls --no-sign-request "${BUCKET}/MTG/ATACseq/" | head -20

# ===========================================================================
#  Download RNA h5ad (~20 GB)
# ===========================================================================

echo ""
echo "=== Downloading RNA h5ad ==="

RNA_FILE="SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad"
if [ ! -f "${OUTDIR}/${RNA_FILE}" ]; then
    ${AWS} s3 cp --no-sign-request \
        "${BUCKET}/MTG/RNAseq/${RNA_FILE}" \
        "${OUTDIR}/${RNA_FILE}"
else
    echo "RNA file already exists, skipping"
fi

# ===========================================================================
#  Download ATAC h5ad (~10 GB)
#  Exact filename may vary — check bucket listing above and adjust if needed
# ===========================================================================

echo ""
echo "=== Downloading ATAC h5ad ==="

# Try the most likely filename; adjust based on bucket listing output
ATAC_FILE="SEAAD_MTG_ATACseq_final-nuclei.2024-12-06.h5ad"
if [ ! -f "${OUTDIR}/${ATAC_FILE}" ]; then
    ${AWS} s3 cp --no-sign-request \
        "${BUCKET}/MTG/ATACseq/${ATAC_FILE}" \
        "${OUTDIR}/${ATAC_FILE}" \
    || {
        echo "Exact ATAC filename not found. Check bucket listing above."
        echo "You may need to update ATAC_FILE in this script."
        exit 1
    }
else
    echo "ATAC file already exists, skipping"
fi

echo ""
echo "Done: $(date)"
echo "Files:"
ls -lh "${OUTDIR}"
