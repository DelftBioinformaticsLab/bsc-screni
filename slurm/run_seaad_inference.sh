#!/bin/bash
#SBATCH --job-name=seaad-wscreni
# TODO: re-add --account=Education-EEMCS-Courses-CSE3000 once admin
#       onboards iharsani to that SLURM account (currently rejected as
#       "Invalid account or account/partition combination specified").
#       See progress_log.md (2026-05-09) for details.
#SBATCH --output=slurm/out/%j_seaad_wscreni.out
#SBATCH --error=slurm/out/%j_seaad_wscreni.out
#SBATCH --time=16:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

# wScReNI inference on a SEA-AD paired-multiome cell-type slice.
#
# Usage:
#   # Pilot (5 donors x ~30 cells, projects full runtime):
#   sbatch slurm/run_seaad_inference.sh \
#     --cell-type "Microglia-PVM" \
#     --donors H21.33.003,H20.33.002,H21.33.019,H20.33.004,H20.33.008 \
#     --cells-per-donor 30 \
#     --output-dir output/seaad_pilot \
#     --pilot
#
#   # Full SQ1 run (Microglia, all eligible donors):
#   sbatch slurm/run_seaad_inference.sh \
#     --cell-type "Microglia-PVM" \
#     --cells-per-donor 50 \
#     --output-dir output/seaad_sq1
#
# Prerequisites:
#   - Phases 0+1 already run (data/processed/seaad/seaad_paired_*.h5ad,
#     seaad_paired_integrated.h5mu)
#   - container_0-1-3.sif present in the project root
#   - data/paper/reference/ contains GRCh38 GTF + TRANSFAC files
#   - data/reference/hg38.fa present

set -euo pipefail

EXTRA_ARGS="${@}"

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
  --bind src/:/opt/app/src/ \
  --bind scripts/:/opt/app/scripts/ \
  --bind data/:/opt/app/data/ \
  --bind output/:/opt/app/output/ \
  --env PYTHONPATH=/opt/app/src \
  --env OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" \
  --env MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" \
  container_0-1-3.sif pixi run --manifest-path /opt/app/pixi.toml \
  python /opt/app/scripts/run_seaad_inference.py ${EXTRA_ARGS}
