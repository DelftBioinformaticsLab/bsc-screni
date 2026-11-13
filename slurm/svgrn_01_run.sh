#!/bin/bash
#SBATCH --job-name=svgrn-run
#SBATCH --output=slurm/out/%j_svgrn_run.out
#SBATCH --error=slurm/out/%j_svgrn_run.out
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# One dataset: pick target cells, run SVGRN (published, then the controls),
# the simple baselines, and score everything.
#   DS=g110_c2k_n01 N_TARGETS=40 VARIANTS="published shuffle_pos shuffle_rna" sbatch slurm/svgrn_01_run.sh
source slurm/svgrn_env.sh
N_TARGETS="${N_TARGETS:-40}"
VARIANTS="${VARIANTS:-published shuffle_pos shuffle_rna}"
mkdir -p "$OUT"
[ -f "$OUT/targets.txt" ] || sv python -c "
import sys; sys.path.insert(0, 'scripts/svgrn_eval'); import common as c
from pathlib import Path
_, meta, _, _ = c.load(Path('$SIM/$DS'))
Path('$OUT/targets.txt').write_text('\n'.join(c.sample_targets(meta, $N_TARGETS)) + '\n')"
echo "targets: $(wc -l < "$OUT/targets.txt")"
sv python -u scripts/svgrn_eval/baselines.py --data "$SIM/$DS" --targets "$OUT/targets.txt" --out "$OUT"
for v in $VARIANTS; do
  sv python -u scripts/svgrn_eval/run_svgrn.py --data "$SIM/$DS" --targets "$OUT/targets.txt" \
    --out "$OUT" --variant "$v" > "$OUT/log_$v.txt" 2>&1 || { echo "variant $v failed:"; tail -20 "$OUT/log_$v.txt"; }
  grep -E "^stage|Traceback|Error" "$OUT/log_$v.txt" | tail -5
done
sv python -u scripts/svgrn_eval/score.py --data "$SIM/$DS" --out "$OUT"
