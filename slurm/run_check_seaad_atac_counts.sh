#!/bin/bash
#SBATCH --job-name=check-atac
#SBATCH --output=slurm/out/%j_check_seaad_atac_counts.out
#SBATCH --error=slurm/out/%j_check_seaad_atac_counts.out
#SBATCH --time=00:10:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# Quick one-shot check: are SEA-AD ATAC .X values raw counts, normalized,
# or binarized? The integration pipeline warns "no raw count layer found"
# but doesn't actually inspect .X — this script does.
#
# Usage:
#   sbatch slurm/run_check_seaad_atac_counts.sh
#
# Output: slurm/out/<jobid>_check_seaad_atac_counts.out

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
  --bind src/:/opt/app/src/ \
  --bind data/:/opt/app/data/ \
  --bind scripts/:/opt/app/scripts/ \
  --env PYTHONPATH=/opt/app/src \
  container_0-1-3.sif pixi run --manifest-path /opt/app/pixi.toml \
  python scripts/check_seaad_atac_counts.py
