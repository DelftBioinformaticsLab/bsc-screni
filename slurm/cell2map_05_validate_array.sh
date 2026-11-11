#!/bin/bash
#SBATCH --job-name=c2m-val-all
#SBATCH --output=slurm/out/%A_%a_cell2map_validate.out
#SBATCH --error=slurm/out/%A_%a_cell2map_validate.out
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0-26

# Step 3: the 5% MERFISH hold-out validation for every donor (index into
# data/cell2map/donors.txt), Cell2Map vs Tangram vs random, section and donor
# scope. Writes output/cell2map/validation_all/<donor>_{cells,summary}.csv.
source slurm/cell2map_env.sh
c2m python -u scripts/cell2map/validate_holdout.py --donor-index "$SLURM_ARRAY_TASK_ID" \
  --methods cell2map tangram random --out output/cell2map/validation_all
