#!/bin/bash
#SBATCH --job-name=c2m-map
#SBATCH --output=slurm/out/%A_%a_cell2map_map.out
#SBATCH --error=slurm/out/%A_%a_cell2map_map.out
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0-26

# Step 2: one donor per task (index into data/cell2map/donors.txt), all
# nuclei, all subclasses. Only after the pilot passes its decision gate.
source slurm/cell2map_env.sh
c2m python -u scripts/cell2map/map_donor.py --donor-index "$SLURM_ARRAY_TASK_ID" \
  --methods cell2map --out output/cell2map/maps
