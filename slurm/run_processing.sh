#!/bin/bash
#SBATCH --job-name=process
#SBATCH --output=slurm/out/%j_run_processing.out
#SBATCH --error=slurm/out/%j_run_processing.out
#SBATCH --time=04:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

# Run Phases 0-3 of the ScReNI data processing pipeline.
#
# Prerequisite: data must be downloaded (download_pbmc.sh, download_retinal.sh,
# download_reference.sh), and container.sif must be present.
#
# Usage:
#   sbatch slurm/run_processing.sh

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
  --bind src/:/opt/app/src/ \
  --bind data/:/opt/app/data/ \
  --bind output/:/opt/app/output/ \
  --env PYTHONPATH=/opt/app/src \
  container_0-1-1.sif pixi run --manifest-path /opt/app/pixi.toml \
  python -m screni.data.loading
