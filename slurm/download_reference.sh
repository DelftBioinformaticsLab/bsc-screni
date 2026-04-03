#!/bin/bash
#SBATCH --job-name=dl_ref
#SBATCH --output=slurm/out/%j_download_reference.out
#SBATCH --error=slurm/out/%j_download_reference.out
#SBATCH --time=02:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# Download reference genome data needed for Phase 3 (gene-peak-TF relationships):
#   - Gene annotations (GTF) for TSS coordinates
#   - Genome FASTA for motif matching
#
# Mouse (mm10): retinal dataset
# Human (hg38): PBMC dataset

set -euo pipefail

OUTDIR="${1:-data/reference}"
mkdir -p "$OUTDIR"
cd "$OUTDIR"

echo "Downloading reference data to $(pwd)"
echo "Started: $(date)"

# ===========================================================================
#  Mouse mm10 (Ensembl release 79, matching EnsDb.Mmusculus.v79 in R)
# ===========================================================================

echo ""
echo "=== Mouse mm10 (Ensembl 79) ==="

# Gene annotations GTF (~25 MB compressed)
if [ ! -f mm10.ensembl79.gtf.gz ]; then
    echo "Downloading mm10 GTF..."
    wget -O mm10.ensembl79.gtf.gz \
        "https://ftp.ensembl.org/pub/release-79/gtf/mus_musculus/Mus_musculus.GRCm38.79.gtf.gz"
else
    echo "mm10 GTF already exists, skipping"
fi

# Genome FASTA (~800 MB compressed, ~2.7 GB uncompressed)
if [ ! -f mm10.fa ]; then
    echo "Downloading mm10 genome FASTA..."
    wget -O mm10.fa.gz \
        "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz"
    echo "Decompressing..."
    gunzip mm10.fa.gz
else
    echo "mm10 FASTA already exists, skipping"
fi

# Index FASTA (needed by pyfaidx)
if [ ! -f mm10.fa.fai ]; then
    echo "Indexing mm10 FASTA..."
    # samtools faidx if available, otherwise pyfaidx will auto-index on first use
    if command -v samtools &> /dev/null; then
        samtools faidx mm10.fa
    else
        echo "  samtools not found; pyfaidx will index on first use"
    fi
fi

# ===========================================================================
#  Human hg38 / GRCh38 (Ensembl release 98)
# ===========================================================================

echo ""
echo "=== Human hg38 (Ensembl 98) ==="

# Gene annotations GTF (~50 MB compressed)
if [ ! -f hg38.ensembl98.gtf.gz ]; then
    echo "Downloading hg38 GTF..."
    wget -O hg38.ensembl98.gtf.gz \
        "https://ftp.ensembl.org/pub/release-98/gtf/homo_sapiens/Homo_sapiens.GRCh38.98.gtf.gz"
else
    echo "hg38 GTF already exists, skipping"
fi

# Genome FASTA (~900 MB compressed, ~3.1 GB uncompressed)
if [ ! -f hg38.fa ]; then
    echo "Downloading hg38 genome FASTA..."
    wget -O hg38.fa.gz \
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
    echo "Decompressing..."
    gunzip hg38.fa.gz
else
    echo "hg38 FASTA already exists, skipping"
fi

# Index FASTA
if [ ! -f hg38.fa.fai ]; then
    echo "Indexing hg38 FASTA..."
    if command -v samtools &> /dev/null; then
        samtools faidx hg38.fa
    else
        echo "  samtools not found; pyfaidx will index on first use"
    fi
fi

# ===========================================================================
#  JASPAR 2024 vertebrate motifs (for TF motif matching)
# ===========================================================================

echo ""
echo "=== JASPAR 2024 vertebrate motifs ==="

if [ ! -f JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt ]; then
    echo "Downloading JASPAR 2024 vertebrate PFMs..."
    wget -O JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt \
        "https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt"
else
    echo "JASPAR motifs already exist, skipping"
fi

echo ""
echo "Done: $(date)"
echo "Files:"
ls -lh
