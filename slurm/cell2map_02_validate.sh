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

# 5% MERFISH hold-out validation, Cell2Map vs Tangram vs random, section and donor
# scope. Default donor: the pilot donor. Overrides via environment, e.g.
#   DONOR=H21.33.019 MATCH=none METHODS="cell2map nearest" OUT=output/x sbatch ...
source slurm/cell2map_env.sh
DONOR="${DONOR:-$(cat data/cell2map/pilot_donor.txt)}"
c2m python -u scripts/cell2map/validate_holdout.py --donor "$DONOR" \
  --methods ${METHODS:-cell2map tangram random} --match "${MATCH:-subclass}" \
  --out "${OUT:-output/cell2map/validation}"
