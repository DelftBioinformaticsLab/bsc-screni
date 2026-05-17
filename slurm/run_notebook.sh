#!/bin/bash
#SBATCH --job-name=seaad-nb
# TODO: re-add --account=Education-EEMCS-Courses-CSE3000 once admin onboards
#       iharsani to that SLURM account (currently rejected as "Invalid account
#       or account/partition combination specified"). Other slurm/*.sh on main
#       run without --account on the default ewi-insy-prb account.
#SBATCH --output=slurm/out/%j_%x.out
#SBATCH --error=slurm/out/%j_%x.out
#SBATCH --time=08:00:00
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# Generic SLURM wrapper that executes a Jupyter notebook end-to-end inside
# the project's pixi container.
#
# Usage:
#   sbatch slurm/run_notebook.sh <relative-path-to-notebook> [extra nbconvert args]
#
# Example:
#   sbatch slurm/run_notebook.sh src/screni/data/prep_seaad_sq1.ipynb
#
# The notebook is executed in place (cell outputs are written back into the
# .ipynb on disk), so the resulting file is itself a run record.
#
# Lessons baked in (see progress_log.md):
#   - --bind /tudelft.net so the data symlinks in iharsani/bsc-screni/data
#     resolve inside the container (otherwise FileNotFoundError on h5ad open).
#   - --containall + --writable-tmpfs is the team's convention.
#   - Account directive commented out: iharsani not onboarded to CSE3000.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 <relative-path-to-notebook> [extra nbconvert args]" >&2
    exit 1
fi

NOTEBOOK_REL="$1"
shift
EXTRA_ARGS="${@}"

NOTEBOOK_ABS="/opt/app/${NOTEBOOK_REL}"

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
    --bind /tudelft.net:/tudelft.net \
    --bind src/:/opt/app/src/ \
    --bind data/:/opt/app/data/ \
    --bind output/:/opt/app/output/ \
    --env PYTHONPATH=/opt/app/src \
    --env OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" \
    --env MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" \
    container_0-1-3.sif pixi run --manifest-path /opt/app/pixi.toml \
    jupyter nbconvert \
        --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=28800 \
        ${EXTRA_ARGS} \
        "${NOTEBOOK_ABS}"
