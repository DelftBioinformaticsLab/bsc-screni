#!/bin/bash
#SBATCH --job-name=c2m-prepare
#SBATCH --output=slurm/out/%j_cell2map_prepare.out
#SBATCH --error=slurm/out/%j_cell2map_prepare.out
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

# Step 0: per-donor MERFISH + snRNA inputs (140 genes), counts per donor x
# subclass, donors.txt and pilot_donor.txt in data/cell2map/.
source slurm/cell2map_env.sh
c2m python -u scripts/cell2map/prepare_inputs.py --out data/cell2map
