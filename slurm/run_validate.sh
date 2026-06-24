#!/bin/bash
#SBATCH --job-name=seaad-validate
#SBATCH --output=slurm/out/%j_validate_p3.out
#SBATCH --error=slurm/out/%j_validate_p3.out
#SBATCH --time=04:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=300G

ARGS="$@"
if [ -z "$ARGS" ]; then
    ARGS="--prefix seaad_paired_sub77"
fi

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
  --bind src/:/opt/app/src/ \
  --bind data/:/opt/app/data/ \
  --bind scripts/:/opt/app/scripts/ \
  --env PYTHONPATH=/opt/app/src \
  container_0-1-3.sif pixi run --manifest-path /opt/app/pixi.toml \
  python scripts/validate_phase3_outputs.py $ARGS
