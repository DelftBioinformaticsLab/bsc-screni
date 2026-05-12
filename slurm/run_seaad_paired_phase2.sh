#!/bin/bash
#SBATCH --job-name=phase2-paired
#SBATCH --output=slurm/out/%j_seaad_paired_phase2.out
#SBATCH --error=slurm/out/%j_seaad_paired_phase2.out
#SBATCH --time=04:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=400G

# Phase 2 for SEA-AD PAIRED data only — independent of the unpaired job.
# Subsamples 50 cells per subclass, selects 500 HVGs + 10k HVPs (Seurat v3
# VST), writes seaad_paired_{rna,atac}_sub.h5ad.
#
# Prerequisite:
#   data/processed/seaad/seaad_paired_rna.h5ad
#   data/processed/seaad/seaad_paired_atac.h5ad
# (both produced by `pixi run load-seaad --process`)
#
# Usage:
#   sbatch slurm/run_seaad_paired_phase2.sh

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
  --bind src/:/opt/app/src/ \
  --bind data/:/opt/app/data/ \
  --bind scripts/:/opt/app/scripts/ \
  --env PYTHONPATH=/opt/app/src \
  container_0-1-3.sif pixi run --manifest-path /opt/app/pixi.toml \
  python scripts/run_seaad_paired_phase2.py
