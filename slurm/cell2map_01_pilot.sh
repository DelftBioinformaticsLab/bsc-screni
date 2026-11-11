#!/bin/bash
#SBATCH --job-name=c2m-pilot
#SBATCH --output=slurm/out/%j_cell2map_pilot.out
#SBATCH --error=slurm/out/%j_cell2map_pilot.out
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Step 1 pilot: the donor in data/cell2map/pilot_donor.txt, all subclasses,
# at most 300 nuclei per subclass, Cell2Map and Tangram on the same input.
source slurm/cell2map_env.sh
DONOR="${DONOR:-$(cat data/cell2map/pilot_donor.txt)}"
c2m python -u scripts/cell2map/map_donor.py --donor "$DONOR" \
  --methods cell2map tangram --max-nuclei 300 --out output/cell2map/pilot/maps
