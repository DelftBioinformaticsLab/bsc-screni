#!/bin/bash
#SBATCH --job-name=integrate-seaad
#SBATCH --output=slurm/out/%j_integrate_seaad.out
#SBATCH --error=slurm/out/%j_integrate_seaad.out
#SBATCH --time=08:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# Phase 1: SEA-AD integration.
#   - Paired branch: WNN on multiome donors (if pairing worked)
#   - Unpaired branch: per-donor Harmony alignment of singleome donors
#
# Prerequisite: Phase 0 must have run with --process
#   (seaad_paired_rna.h5ad, seaad_unpaired_rna.h5ad etc. in data/processed/seaad/)
#
# Usage:
#   sbatch slurm/run_integrate_seaad.sh

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
  --bind src/:/opt/app/src/ \
  --bind data/:/opt/app/data/ \
  --bind output/:/opt/app/output/ \
  --env PYTHONPATH=/opt/app/src \
  container_0-1-3.sif pixi run --manifest-path /opt/app/pixi.toml \
  python -m screni.data.integration_seaad
