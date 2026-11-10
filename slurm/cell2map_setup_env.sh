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
CONDA_OVERRIDE_CUDA=12 apptainer exec --bind /tudelft.net:/tudelft.net "$SIF" \
  "$PIXI" install -e cell2map --manifest-path "$REPO/pixi.toml" || exit 1
source slurm/cell2map_env.sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
c2m python -c "
import torch, torch_geometric, tangram, scanpy
from screni.cell2map import mapping as mp
mp._cell2map_modules()
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
print('pyg', torch_geometric.__version__, 'tangram', getattr(tangram, '__version__', '?'), 'scanpy', scanpy.__version__)
print('Cell2Map import OK')
"
