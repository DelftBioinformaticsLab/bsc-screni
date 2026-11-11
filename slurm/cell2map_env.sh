#!/bin/bash
# Sourced by the cell2map_*.sh jobs. Runs commands in the `cell2map` pixi
# environment inside the project container (DAIC's glibc 2.17 is too old for
# the CUDA packages; the container has 2.35). The container's own pixi (0.40)
# cannot read this pixi.toml, so a pixi >= 0.49 binary from $HOME is used:
#   mkdir -p ~/.pixi/bin && cd ~/.pixi/bin && curl -fsSL https://github.com/prefix-dev/pixi/releases/download/v0.49.0/pixi-x86_64-unknown-linux-musl.tar.gz | tar xz
# Submit jobs from the repo root: sbatch slurm/cell2map_01_pilot.sh
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
SIF="${SIF:-/tudelft.net/staff-umbrella/ScReNI/bsc-screni/container_0-1-3.sif}"
PIXI="${PIXI:-$HOME/.pixi/bin/pixi}"
cd "$REPO" || exit 1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# CPU-only jobs (prepare, merge) still use the CUDA environment; tell pixi
# the CUDA virtual package exists so it does not refuse to run there.
export CONDA_OVERRIDE_CUDA=12

c2m() {
  apptainer exec --nv --bind /tudelft.net:/tudelft.net \
    --env PYTHONPATH="$REPO/src" \
    "$SIF" "$PIXI" run --frozen -e cell2map --manifest-path "$REPO/pixi.toml" "$@"
}

echo "Job $SLURM_JOB_ID on $(hostname), start $(date), repo $REPO @ $(cd "$REPO" && git rev-parse --short HEAD)"
