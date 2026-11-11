#!/bin/bash
#SBATCH --job-name=c2m-merge
#SBATCH --output=slurm/out/%j_cell2map_merge.out
#SBATCH --error=slurm/out/%j_cell2map_merge.out
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

# Join the per-donor maps into output/cell2map/maps/cell2map_all_donors.csv.
source slurm/cell2map_env.sh
c2m python -u scripts/cell2map/merge_maps.py --maps output/cell2map/maps
