#!/bin/bash
#SBATCH --job-name=integrate-retinal
#SBATCH --output=slurm/out/%j_integrate_retinal.out
#SBATCH --error=slurm/out/%j_integrate_retinal.out
#SBATCH --time=02:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Phase 1b: Retinal unpaired integration (Harmony + cross-modality pairing).
#
# Prerequisite: Phase 0 must have run (retinal_rna.h5ad, retinal_atac.h5ad in data/processed/).
#
# Usage:
#   sbatch slurm/run_integrate_retinal.sh

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
  --bind src/:/opt/app/src/ \
  --bind data/:/opt/app/data/ \
  --bind output/:/opt/app/output/ \
  --env PYTHONPATH=/opt/app/src \
  container_0-1-2.sif pixi run --manifest-path /opt/app/pixi.toml \
  python -m screni.data.integration_retinal
