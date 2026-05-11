#!/bin/bash
#SBATCH --job-name=inspect-seaad-int
#SBATCH --output=slurm/out/%j_inspect_seaad_integration.out
#SBATCH --error=slurm/out/%j_inspect_seaad_integration.out
#SBATCH --time=00:30:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G

# Phase 1 QC: produces UMAP figures + donor-summary printout for the
# SEA-AD integration outputs. Reads the h5mu in full (paired branch
# is ~91 GB on disk but the obs/obsm are small) and the unpaired h5ad
# in backed mode.
#
# Prerequisite: integrate-seaad has produced
#   data/processed/seaad/seaad_paired_integrated.h5mu
#   data/processed/seaad/seaad_unpaired_integrated.h5ad
#   data/processed/seaad/seaad_unpaired_donor_summary.csv
#   data/processed/seaad/seaad_unpaired_nn_pairs.csv
#
# Usage:
#   sbatch slurm/run_inspect_seaad_integration.sh

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
  --bind src/:/opt/app/src/ \
  --bind data/:/opt/app/data/ \
  --bind output/:/opt/app/output/ \
  --bind scripts/:/opt/app/scripts/ \
  --env PYTHONPATH=/opt/app/src \
  container_0-1-3.sif pixi run --manifest-path /opt/app/pixi.toml \
  python scripts/inspect_seaad_integration.py
