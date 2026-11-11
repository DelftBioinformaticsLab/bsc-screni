#!/bin/bash
#SBATCH --job-name=c2m-setup
#SBATCH --output=slurm/out/%j_cell2map_setup.out
#SBATCH --error=slurm/out/%j_cell2map_setup.out
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Solve and install the `cell2map` pixi environment into $REPO/.pixi, then
# check that torch sees the GPU and that Cell2Map and Tangram import.
# Run once (and again after pixi.toml changes): sbatch slurm/cell2map_setup_env.sh
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
SIF="${SIF:-/tudelft.net/staff-umbrella/ScReNI/bsc-screni/container_0-1-3.sif}"
PIXI="${PIXI:-$HOME/.pixi/bin/pixi}"
cd "$REPO" || exit 1
git submodule update --init external/Cell2Map
# The umbrella share refuses pixi's file copies ("Operation not permitted"), so
# the environment lives next to the package cache in $HOME (~8 GB).
mkdir -p .pixi "$HOME/.pixi/envs-detached"
[ -f .pixi/config.toml ] || printf 'detached-environments = "%s"\n' "$HOME/.pixi/envs-detached" > .pixi/config.toml
CONDA_OVERRIDE_CUDA=12 apptainer exec --bind /tudelft.net:/tudelft.net "$SIF" \
  "$PIXI" install -e cell2map --manifest-path "$REPO/pixi.toml" || exit 1
source slurm/cell2map_env.sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
c2m python scripts/cell2map/check_env.py
