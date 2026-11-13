#!/bin/bash
#SBATCH --job-name=svgrn-paper-cpu
#SBATCH --output=slurm/out/%A_%a_svgrn_paper_cpu.out
#SBATCH --error=slurm/out/%A_%a_svgrn_paper_cpu.out
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-99

# CPU version of svgrn_03_paper.sh (GPU queue was ~16 h): same run
# (W = 4, published stage-1 model reused), targets split over 100 tasks.
#   DS=g110_c2k_n01 sbatch slurm/svgrn_03_paper_cpu.sh
source slurm/svgrn_env.sh
T="$OUT/targets_paper.txt"
[ -s "$T" ] || { echo "no $T"; exit 1; }
sv python -u scripts/svgrn_eval/run_svgrn.py --data "$SIM/$DS" --targets "$T" --out "$OUT" \
  --variant published --W 4 --name paper_w4 --chunk "$SLURM_ARRAY_TASK_ID" --n-chunks "${SLURM_ARRAY_TASK_COUNT:-100}" \
  2>&1 | grep -E "^stage|Traceback|Error"
