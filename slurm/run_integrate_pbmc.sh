#!/bin/bash
#SBATCH --job-name=integrate-pbmc
#SBATCH --output=slurm/out/%j_integrate_pbmc.out
#SBATCH --error=slurm/out/%j_integrate_pbmc.out
#SBATCH --time=01:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Phase 1a: PBMC paired integration (WNN).
#
# Prerequisite: Phase 0 must have run (pbmc_rna.h5ad, pbmc_atac.h5ad in data/processed/).
#
# Usage:
#   sbatch slurm/run_integrate_pbmc.sh

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
  --bind src/:/opt/app/src/ \
  --bind data/:/opt/app/data/ \
  --bind output/:/opt/app/output/ \
  --env PYTHONPATH=/opt/app/src \
  container_0-1-2.sif pixi run --manifest-path /opt/app/pixi.toml \
  python -m screni.data.integration
