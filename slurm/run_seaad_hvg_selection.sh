#!/bin/bash
#SBATCH --job-name=seaad-hvg
#SBATCH --output=slurm/out/%j_seaad_hvg_selection.out
#SBATCH --error=slurm/out/%j_seaad_hvg_selection.out
#SBATCH --time=04:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=400G

# HVG/HVP selection for SEA-AD paired data (Phase 2 equivalent).
# Selects 500 HVGs (RNA) + 10k HVPs (ATAC) via Seurat v3 VST on the FULL
# cell set (~138k cells each modality, no subsampling — that's left to
# each sub-question via scripts/subsample_seaad_paired.py). Writes
# seaad_paired_{rna_hvg,atac_hvp}.h5ad with full SEA-AD obs preserved
# (object cols coerced to str), plus the joint WNN-input embedding
# (obsm["X_pca"]) and full-set WNN k=20 nearest neighbor indices
# (uns["wnn_neighbor_indices"]).
#
# Prerequisite:
#   data/processed/seaad/seaad_paired_integrated.h5mu
# (produced by `pixi run integrate-seaad`; ~91 GB)
#
# Usage:
#   sbatch slurm/run_seaad_hvg_selection.sh

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
  --bind src/:/opt/app/src/ \
  --bind data/:/opt/app/data/ \
  --bind scripts/:/opt/app/scripts/ \
  --env PYTHONPATH=/opt/app/src \
  container_0-1-3.sif pixi run --manifest-path /opt/app/pixi.toml \
  python scripts/run_seaad_hvg_selection.py
