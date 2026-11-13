#!/bin/bash
# Sourced by the svgrn_*.sh jobs. Runs commands in the `svgrn` pixi environment
# inside the project container (DAIC's glibc 2.17 is too old for the CUDA
# packages; the container has 2.35). Needs a pixi >= 0.49 binary in $HOME:
#   mkdir -p ~/.pixi/bin && cd ~/.pixi/bin && curl -fsSL https://github.com/prefix-dev/pixi/releases/download/v0.49.0/pixi-x86_64-unknown-linux-musl.tar.gz | tar xz
# Submit jobs from the repo root. SIM points at the unzipped SVGRN simulations
# (Google Drive link in external/SVGRN/README.md); OUT is where results go.
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
SIF="${SIF:-/tudelft.net/staff-umbrella/ScReNI/bsc-screni/container_0-1-3.sif}"
PIXI="${PIXI:-$HOME/.pixi/bin/pixi}"
SIM="${SIM:-/tudelft.net/staff-umbrella/ScReNI/dflam/svgrn_sim}"
DS="${DS:-g110_c2k_n01}"
OUT="${OUT:-output/svgrn_eval/$DS}"
cd "$REPO" || exit 1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CONDA_OVERRIDE_CUDA=12

sv() {
  apptainer exec --nv --bind /tudelft.net:/tudelft.net "$SIF" \
    "$PIXI" run --frozen -e svgrn --manifest-path "$REPO/pixi.toml" "$@"
}

echo "Job $SLURM_JOB_ID on $(hostname), start $(date), repo $REPO @ $(cd "$REPO" && git rev-parse --short HEAD), dataset $DS"
