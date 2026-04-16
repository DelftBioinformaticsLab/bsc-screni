#!/bin/bash
#SBATCH --job-name=load-seaad
#SBATCH --output=slurm/out/%j_load_seaad.out
#SBATCH --error=slurm/out/%j_load_seaad.out
#SBATCH --time=02:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

# Phase 0: Load, inspect, and split SEA-AD MTG data.
#
# Prerequisite: SEA-AD h5ad files downloaded to data/seaad/
#   (run slurm/download_seaad.sh first)
#
# First run (inspect only): produces schema dump + pairing audit.
# Second run (with --process): loads, filters, splits, saves.
#
# Usage:
#   sbatch slurm/run_load_seaad.sh                 # inspect only
#   sbatch slurm/run_load_seaad.sh --process       # full processing

EXTRA_ARGS="${@}"

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
  --bind src/:/opt/app/src/ \
  --bind data/:/opt/app/data/ \
  --bind output/:/opt/app/output/ \
  --env PYTHONPATH=/opt/app/src \
  container_0-1-3.sif pixi run --manifest-path /opt/app/pixi.toml \
  python -m screni.data.loading_seaad ${EXTRA_ARGS}
