#!/bin/bash
#SBATCH --job-name=seaad-sq1
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

# Generic SLURM wrapper that executes a Python script inside the project's
# pixi container.  The container's pixi env does not include nbformat /
# nbconvert, so cluster-runnable scripts must be plain .py files
# (notebook counterparts under src/screni/data/*.ipynb stay around for
# interactive use only — keep them in sync with their .py twins).
#
# Usage:
#   sbatch slurm/run_notebook.sh <path-to-python-script> [extra script args]
#
# Example:
#   sbatch slurm/run_notebook.sh scripts/prep_seaad_sq1.py
#
# Lessons baked in (see progress_log.md):
#   - --bind /tudelft.net so the data symlinks in iharsani/bsc-screni/data
#     resolve inside the container (otherwise FileNotFoundError on h5ad open).
#   - --containall + --writable-tmpfs is the team's convention.
#   - --qos=medium so we can request >4 h (short QOS caps wall time).
#   - Account directive commented out: iharsani not onboarded to CSE3000.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 <path-to-python-script> [extra args]" >&2
    exit 1
fi

SCRIPT_REL="$1"
shift
EXTRA_ARGS="${@}"

SCRIPT_ABS="/opt/app/${SCRIPT_REL}"

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
    --bind /tudelft.net:/tudelft.net \
    --bind src/:/opt/app/src/ \
    --bind scripts/:/opt/app/scripts/ \
    --bind data/:/opt/app/data/ \
    --bind output/:/opt/app/output/ \
    --env PYTHONPATH=/opt/app/src \
    --env OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" \
    --env MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" \
    container_0-1-3.sif pixi run --manifest-path /opt/app/pixi.toml \
    python "${SCRIPT_ABS}" ${EXTRA_ARGS}
