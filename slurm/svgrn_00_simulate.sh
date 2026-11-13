#!/bin/bash
#SBATCH --job-name=svgrn-sim
#SBATCH --output=slurm/out/%A_%a_svgrn_sim.out
#SBATCH --error=slurm/out/%A_%a_svgrn_sim.out
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=0-5

# Regenerate SVGRN's simulated datasets with scMultiSim (scripts/svgrn_eval/simulate.R),
# one task per setting: genes {110, 160} x intrinsic noise {0.1, 0.5, 1.0}.
# SEED=0 matches SVGRN's scripts (rand.seed = 0); other seeds give replicates.
#   SEED=1 sbatch slurm/svgrn_00_simulate.sh
# Output: $SIM_GEN/g<genes>_c2k_n<noise>_s<seed>/ in SVGRN's input format.
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
SIF="${SIF:-/tudelft.net/staff-umbrella/ScReNI/bsc-screni/container_0-1-3.sif}"
PIXI="${PIXI:-$HOME/.pixi/bin/pixi}"
SIM_GEN="${SIM_GEN:-/tudelft.net/staff-umbrella/ScReNI/dflam/svgrn_sim_gen}"
SEED="${SEED:-0}"
cd "$REPO" || exit 1
GENES=(110 110 110 160 160 160); NOISE=(0.1 0.5 1.0 0.1 0.5 1.0)
i="${SLURM_ARRAY_TASK_ID:-0}"
G="${GENES[$i]}"; N="${NOISE[$i]}"
D="$SIM_GEN/g${G}_c2k_n${N}_s${SEED}"
echo "Job ${SLURM_ARRAY_JOB_ID}_${i} on $(hostname), start $(date), repo @ $(git rev-parse --short HEAD): genes $G, noise $N, seed $SEED -> $D"
sm() {
  apptainer exec --bind /tudelft.net:/tudelft.net "$SIF" \
    "$PIXI" run --frozen -e scmultisim --manifest-path "$REPO/pixi.toml" "$@"
}
mkdir -p "$D"
sm Rscript scripts/svgrn_eval/simulate.R "$G" "$N" "$SEED" "$D" || exit 1
sm python scripts/svgrn_eval/prepare_sim.py "$D"
