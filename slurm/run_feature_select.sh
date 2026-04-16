#!/bin/bash
#SBATCH --job-name=feature-select
#SBATCH --output=slurm/out/%j_feature_select.out
#SBATCH --error=slurm/out/%j_feature_select.out
#SBATCH --time=02:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Phase 2: Cell subsampling and feature selection for all datasets.
#
# Runs on whichever integrated files exist:
#   - PBMC (data/processed/pbmc_integrated.h5mu)
#   - Retinal (data/processed/retinal_rna.h5ad + retinal_nn_pairs.csv)
#   - SEA-AD paired (data/processed/seaad/seaad_paired_integrated.h5mu)
#   - SEA-AD unpaired (data/processed/seaad/seaad_unpaired_nn_pairs.csv)
#
# Prerequisite: Phase 1 (integration) must have run.
#
# Usage:
#   sbatch slurm/run_feature_select.sh

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
  --bind src/:/opt/app/src/ \
  --bind data/:/opt/app/data/ \
  --bind output/:/opt/app/output/ \
  --env PYTHONPATH=/opt/app/src \
  container_0-1-3.sif pixi run --manifest-path /opt/app/pixi.toml \
  python -m screni.data.feature_selection
