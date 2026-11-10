#!/bin/bash
#SBATCH --job-name=c2m-validate
#SBATCH --output=slurm/out/%j_cell2map_validate.out
#SBATCH --error=slurm/out/%j_cell2map_validate.out
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# 5% MERFISH hold-out validation, Cell2Map vs Tangram, section and donor
# scope. Default donor: the pilot donor. Override: DONOR=H21.33.019 sbatch ...
source slurm/cell2map_env.sh
DONOR="${DONOR:-$(cat data/cell2map/pilot_donor.txt)}"
c2m python -u scripts/cell2map/validate_holdout.py --donor "$DONOR" \
  --methods cell2map tangram --out output/cell2map/validation
