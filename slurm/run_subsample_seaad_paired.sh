#!/bin/bash
#SBATCH --job-name=seaad-subsample
#SBATCH --output=slurm/out/%j_subsample_seaad_paired.out
#SBATCH --error=slurm/out/%j_subsample_seaad_paired.out
#SBATCH --time=02:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=300G

# Subsample the SEA-AD paired HVG/HVP files into Phase 3 input files.
# Writes data/processed/seaad/seaad_paired_{rna,atac}_sub{seed}.h5ad.
#
# Prerequisite (one-off, already done on cluster):
#   data/processed/seaad/seaad_paired_rna_hvg.h5ad
#   data/processed/seaad/seaad_paired_atac_hvp.h5ad
# (produced by `sbatch slurm/run_seaad_hvg_selection.sh`)
#
# Usage:
#   sbatch slurm/run_subsample_seaad_paired.sh                 # default: --seed 42
#   sbatch slurm/run_subsample_seaad_paired.sh --seed 7        # custom seed
#   sbatch slurm/run_subsample_seaad_paired.sh --seed 99 --n-per-type 100
#   sbatch slurm/run_subsample_seaad_paired.sh --seed 11 \
#       --cell-types Microglia-PVM Astrocyte Oligodendrocyte
#
# If no args are passed, runs with `--seed 42` to (re)produce the default
# subsample.

ARGS="$@"
if [ -z "$ARGS" ]; then
    ARGS="--seed 42"
fi

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
  --bind src/:/opt/app/src/ \
  --bind data/:/opt/app/data/ \
  --bind scripts/:/opt/app/scripts/ \
  --env PYTHONPATH=/opt/app/src \
  container_0-1-3.sif pixi run --manifest-path /opt/app/pixi.toml \
  python scripts/subsample_seaad_paired.py $ARGS
