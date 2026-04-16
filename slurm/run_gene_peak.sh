#!/bin/bash
#SBATCH --job-name=gene-peak
#SBATCH --output=slurm/out/%j_gene_peak.out
#SBATCH --error=slurm/out/%j_gene_peak.out
#SBATCH --time=04:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Phase 3: Gene-peak-TF relationships for all datasets.
#
# Runs on whichever subsampled files exist in data/processed/:
#   - PBMC (pbmc_rna_sub.h5ad + pbmc_atac_sub.h5ad)
#   - Retinal (retinal_rna_sub.h5ad + retinal_atac_sub.h5ad)
#   - SEA-AD paired (data/processed/seaad/seaad_paired_*_sub.h5ad)
#   - SEA-AD unpaired (data/processed/seaad/seaad_unpaired_*_sub.h5ad)
#
# Prerequisite: Phase 2 (feature selection) must have run.
# Also requires reference files: hg38.fa, mm10.fa, GTFs, JASPAR motifs.
#
# Usage:
#   sbatch slurm/run_gene_peak.sh

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
  --bind src/:/opt/app/src/ \
  --bind data/:/opt/app/data/ \
  --bind output/:/opt/app/output/ \
  --env PYTHONPATH=/opt/app/src \
  container_0-1-3.sif pixi run --manifest-path /opt/app/pixi.toml \
  python -m screni.data.gene_peak_relations
