#!/bin/bash
#SBATCH --job-name=mg-hvg-v2
#SBATCH --output=slurm/out/%j_mg_hvg_selection_v2.out
#SBATCH --error=slurm/out/%j_mg_hvg_selection_v2.out
#SBATCH --time=04:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G

# Microglia HVG/HVP selection — v2 with three fixes:
#
#   FIX 1 — Raw counts
#     Tries seaad_paired_rna.h5ad raw-count layers first (saves RAM).
#     Falls back to seaad_paired_integrated.h5mu (91 GB) if needed.
#     300 G is requested to safely handle the h5mu fallback.  If
#     quick_inspect_rna_counts.py confirmed that seaad_paired_rna.h5ad has
#     a raw-count layer, you can reduce to --mem=120G --time=02:00:00.
#
#   FIX 2 — 2000 HVGs  (was 500)
#     Higher ceiling makes it more likely that microglia-variable TFs
#     survive VST selection.  Also increases gene-peak proximity pairs
#     available for Phase 3 correlation.
#
#   FIX 3 — All microglia cells retained  (~3,129 cells)
#     The subsample step is called with --n-per-type 9999 so the
#     correlator sees all available cells, lifting the 96.8% ATAC
#     filtering rate seen in v1 (200 cells).
#
# RUNTIME ESTIMATE  (4 CPUs, 300 G)
#   If using seaad_paired_rna.h5ad (raw layer exists):
#     RNA load + filter             ~ 15 min
#     VST 2000 HVGs on 3129 cells  ~  5 min
#     ATAC load + filter            ~ 20 min
#     VST 10000 HVPs on 3129 cells  ~  5 min
#     Total                         ~ 45 min
#
#   If h5mu fallback is used:
#     h5mu load (91 GB)             ~ 30 min
#     rest                          ~ 30 min
#     Total                         ~ 60 min
#
# PREREQUISITES
#   Run quick_inspect_rna_counts.py on the login node first to confirm
#   which data source will be used and whether --mem can be reduced.
#
# USAGE
#   mkdir -p slurm/out
#   sbatch slurm/run_seaad_hvg_selection_microglia_v2.sh
#
# OUTPUT
#   data/processed/seaad/seaad_paired_rna_mg2k_hvg.h5ad
#   data/processed/seaad/seaad_paired_atac_mg10k_hvp.h5ad
#
# NEXT STEPS (run after this job completes)
#   See the printed instructions at the end of the job log, or run:
#
#   # Step 1 — Subsample (all microglia cells)
#   apptainer exec --writable-tmpfs --pwd /opt/app --containall \
#     --bind src/:/opt/app/src/ --bind data/:/opt/app/data/ \
#     --bind scripts/:/opt/app/scripts/ --env PYTHONPATH=/opt/app/src \
#     container_0-1-3.sif pixi run --manifest-path /opt/app/pixi.toml \
#     python scripts/subsample_seaad_paired.py \
#       --seed 88 --n-per-type 9999 \
#       --cell-types Microglia-PVM \
#       --rna  data/processed/seaad/seaad_paired_rna_mg2k_hvg.h5ad \
#       --atac data/processed/seaad/seaad_paired_atac_mg10k_hvp.h5ad
#
#   # Step 2 — Gene-peak (picks up sub88 via glob automatically)
#   sbatch slurm/run_gene_peak.sh
#
#   # Step 3 — Validate
#   python scripts/validate_phase3_outputs.py --prefix seaad_paired_sub88

set -euo pipefail

CONTAINER="/tudelft.net/staff-umbrella/ScReNI/bsc-screni/container_0-1-3.sif"

if [[ ! -f "$CONTAINER" ]]; then
    echo "ERROR: container not found: $CONTAINER"
    exit 1
fi

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Container   : $CONTAINER"
echo "Memory      : ${SLURM_MEM_PER_NODE:-300G}"
echo "CPUs        : ${SLURM_CPUS_PER_TASK:-4}"
echo "Working dir : $(pwd)"
echo "Started     : $(date)"
echo

mkdir -p slurm/out data/processed/seaad

apptainer exec \
    --writable-tmpfs \
    --pwd /opt/app \
    --containall \
    --bind src/:/opt/app/src/ \
    --bind data/:/opt/app/data/ \
    --bind scripts/:/opt/app/scripts/ \
    --bind run_seaad_hvg_selection_microglia_v2.py:/opt/app/run_seaad_hvg_selection_microglia_v2.py \
    --env PYTHONPATH=/opt/app/src \
    --env SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-4} \
    "$CONTAINER" \
    pixi run --manifest-path /opt/app/pixi.toml \
    python -u /opt/app/run_seaad_hvg_selection_microglia_v2.py

echo
echo "Finished : $(date)"
echo
echo "Check the AD gene presence list above, then run the subsample step."
echo "If SPI1 is still absent, note it and proceed anyway — 2000 HVGs"
echo "and all 3129 cells will still produce a meaningful microglia GRN"
echo "for the genes that DO vary within the population."
