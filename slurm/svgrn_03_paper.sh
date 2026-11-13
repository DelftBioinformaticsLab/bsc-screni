#!/bin/bash
#SBATCH --job-name=svgrn-paper
#SBATCH --output=slurm/out/%A_%a_svgrn_paper.out
#SBATCH --error=slurm/out/%A_%a_svgrn_paper.out
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-9

# "Paper-faithful" SVGRN: published settings with the supplement's bandwidth
# W = 4, reusing the published stage-1 model (W only affects stage 2), on
# N_PAPER target cells split over the array tasks. Score afterwards with
# eval_paper.py (SVGRN's own eval.py).
#   DS=g110_c2k_n01 N_PAPER=300 sbatch slurm/svgrn_03_paper.sh
source slurm/svgrn_env.sh
N_PAPER="${N_PAPER:-300}"
T="$OUT/targets_paper.txt"
[ -s "$T" ] || { echo "make $T first (make_targets.py --n $N_PAPER --seed 1)"; exit 1; }
sv python -u scripts/svgrn_eval/run_svgrn.py --data "$SIM/$DS" --targets "$T" --out "$OUT" \
  --variant published --W 4 --name paper_w4 --chunk "$SLURM_ARRAY_TASK_ID" --n-chunks "${SLURM_ARRAY_TASK_COUNT:-10}" \
  2>&1 | grep -E "^stage|Traceback|Error"
