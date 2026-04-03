#!/bin/bash
#SBATCH --job-name=inspect
#SBATCH --output=slurm/out/%j_inspect_data.out
#SBATCH --error=slurm/out/%j_inspect_data.out
#SBATCH --time=00:30:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
  --bind src/:/opt/app/src/ \
  --bind data/:/opt/app/data/ \
  --bind output/:/opt/app/output/ \
  --env PYTHONPATH=/opt/app/src \
  container.sif pixi run --manifest-path /opt/app/pixi.toml \
  python src/inspect_data.py
