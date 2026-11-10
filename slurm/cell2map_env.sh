#!/bin/bash
# Sourced by the cell2map_*.sh jobs. Runs commands in the `cell2map` pixi
# environment through the project container (pixi is not on DAIC's PATH).
# Submit jobs from the repo root: sbatch slurm/cell2map_01_pilot.sh
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
SIF="${SIF:-/tudelft.net/staff-umbrella/ScReNI/bsc-screni/container_0-1-3.sif}"
cd "$REPO" || exit 1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# CPU-only jobs (prepare, merge) still use the CUDA environment; tell pixi
# the CUDA virtual package exists so it does not refuse to run there.
export CONDA_OVERRIDE_CUDA=12

c2m() {
  apptainer exec --nv --bind /tudelft.net:/tudelft.net \
    --env PYTHONPATH="$REPO/src" \
    "$SIF" pixi run --frozen -e cell2map --manifest-path "$REPO/pixi.toml" "$@"
}

echo "Job $SLURM_JOB_ID on $(hostname), start $(date), repo $REPO @ $(git -C "$REPO" rev-parse --short HEAD)"
