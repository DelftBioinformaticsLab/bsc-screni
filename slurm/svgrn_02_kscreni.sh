#!/bin/bash
#SBATCH --job-name=kscreni
#SBATCH --output=slurm/out/%j_kscreni.out
#SBATCH --error=slurm/out/%j_kscreni.out
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

# ScReNI's kScReNI on one simulated dataset (same targets as svgrn_01_run.sh),
# then rescore everything in $OUT.
#   DS=g110_c2k_n01 sbatch slurm/svgrn_02_kscreni.sh
source slurm/svgrn_env.sh
[ -s "$OUT/targets.txt" ] || { echo "no $OUT/targets.txt: run svgrn_01_run.sh first"; exit 1; }
ks() {
  apptainer exec --bind /tudelft.net:/tudelft.net "$SIF" \
    "$PIXI" run --frozen -e kscreni --manifest-path "$REPO/pixi.toml" "$@"
}
ks Rscript scripts/svgrn_eval/run_kscreni.R "$SIM/$DS" "$OUT/targets.txt" "$OUT" "${SLURM_CPUS_PER_TASK:-8}" || exit 1
ks python scripts/svgrn_eval/collect_csv_preds.py --data "$SIM/$DS" --cells "$OUT/kscreni/cells" --out "$OUT/kscreni/preds_kscreni.npz"
sv python -u scripts/svgrn_eval/score.py --data "$SIM/$DS" --out "$OUT"
