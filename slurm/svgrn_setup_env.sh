#!/bin/bash
#SBATCH --job-name=svgrn-setup
#SBATCH --output=slurm/out/%j_svgrn_setup.out
#SBATCH --error=slurm/out/%j_svgrn_setup.out
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Install the `svgrn` pixi environment and check torch sees the GPU.
# The umbrella share refuses pixi's file copies, so the environment lives in
# $HOME/.pixi/envs-detached (see .pixi/config.toml, written here if missing).
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
SIF="${SIF:-/tudelft.net/staff-umbrella/ScReNI/bsc-screni/container_0-1-3.sif}"
PIXI="${PIXI:-$HOME/.pixi/bin/pixi}"
cd "$REPO" || exit 1
mkdir -p .pixi "$HOME/.pixi/envs-detached"
[ -f .pixi/config.toml ] || printf 'detached-environments = "%s"\n' "$HOME/.pixi/envs-detached" > .pixi/config.toml
CONDA_OVERRIDE_CUDA=12 apptainer exec --bind /tudelft.net:/tudelft.net "$SIF" \
  "$PIXI" install -e svgrn --manifest-path "$REPO/pixi.toml" || exit 1
source slurm/svgrn_env.sh
sv python -c "import torch, sklearn, pandas; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
