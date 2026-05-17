"""SQ1 preprocessing — donor-stratified Phase 2 + Phase 3 on SeaAD paired.

Script counterpart to ``src/screni/data/prep_seaad_sq1.ipynb`` for cluster
execution.  Same logic, runnable via ``python scripts/prep_seaad_sq1.py``.

The team's pixi env doesn't include ``nbformat``/``nbconvert``, so we ship
the cluster-runnable form as a plain .py.  Keep this file and the notebook
in sync when changing one.

Produces (under ``data/processed/seaad/``):

* ``seaad_paired_rna_sq1.h5ad`` — donor-stratified subsample with 500 HVGs,
  KNN in ``.uns['knn_indices']``.
* ``seaad_paired_atac_sq1.h5ad`` — same cells, 10k HVPs.
* ``seaad_paired_sq1_triplets.csv`` and the rest of Phase 3.

After this runs, set ``SUB_TAG = "sq1"`` in ``run_infer_wscreni_seaad.ipynb``
and execute inference.
"""

from __future__ import annotations

import logging
import os
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd

# Make the local screni package importable regardless of where we're invoked.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from screni.data.feature_selection import prepare_subsample
from screni.data.gene_peak_relations import load_transfac_motifs, run_phase3
from screni.data.loading_seaad import (
    add_condition_column,
    add_copathology_columns,
    select_eligible_donors,
    subsample_cells_per_donor,
)
from screni.data.utils import load_gene_annotations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ---- Configuration ----------------------------------------------------------

DATASET = "seaad_paired"
SUB_TAG = "sq1"
TARGET_CELL_TYPES = ["Microglia-PVM", "L2/3 IT"]
N_CELLS_PER_DONOR = 50
MIN_CELLS_PER_DONOR = 50
N_GENES = 500
N_PEAKS = 10000
KNN_K = 20
SEED = 42

DATA_DIR = os.path.join(REPO_ROOT, "data", "processed", "seaad")
REF_DIR = os.path.join(REPO_ROOT, "data", "paper", "reference")
GENOME_FA = os.path.join(REPO_ROOT, "data", "reference", "hg38.fa")

RNA_FULL = os.path.join(DATA_DIR, f"{DATASET}_rna.h5ad")
ATAC_FULL = os.path.join(DATA_DIR, f"{DATASET}_atac.h5ad")
INTEGRATED = os.path.join(DATA_DIR, f"{DATASET}_integrated.h5mu")
GTF_PATH = os.path.join(REF_DIR, "gencode.v38.annotation.gtf")
MOTIF_TXT = os.path.join(REF_DIR, "Tranfac201803_Hs_MotifTFsFinal")
MOTIF_RDS = os.path.join(REF_DIR, "all_motif_pwm.rds")


def main() -> int:
    logger.info("=" * 70)
    logger.info(f"SQ1 preprocessing | DATASET={DATASET} | SUB_TAG={SUB_TAG}")
    logger.info(f"target cell types : {TARGET_CELL_TYPES}")
    logger.info(f"cells per donor   : {N_CELLS_PER_DONOR}")
    logger.info("=" * 70)

    # Verify inputs exist before doing expensive loads.
    for label, path in [
        ("RNA full", RNA_FULL), ("ATAC full", ATAC_FULL),
        ("integrated h5mu", INTEGRATED), ("GTF", GTF_PATH),
        ("motif TSV", MOTIF_TXT), ("motif RDS", MOTIF_RDS),
        ("genome FA", GENOME_FA),
    ]:
        ok = "OK" if os.path.exists(path) else "MISSING"
        logger.info(f"  {ok:7s} {label}: {path}")
        if not os.path.exists(path):
            logger.error(f"Required input missing: {path}")
            return 2

    # ---- Step 1: load + tag --------------------------------------------------
    logger.info("\n=== Step 1: load full RNA + ATAC and tag condition/co-pathology ===")
    rna = ad.read_h5ad(RNA_FULL)
    logger.info(f"  RNA loaded: {rna.shape}")
    atac = ad.read_h5ad(ATAC_FULL)
    logger.info(f"  ATAC loaded: {atac.shape}")

    if "cell_type" not in rna.obs.columns:
        rna.obs["cell_type"] = rna.obs["Subclass"].astype(str)
    if "cell_type" not in atac.obs.columns:
        atac.obs["cell_type"] = atac.obs["Subclass"].astype(str)

    add_condition_column(rna)
    add_copathology_columns(rna)
    add_condition_column(atac)
    add_copathology_columns(atac)

    ct_mask = rna.obs["cell_type"].isin(TARGET_CELL_TYPES) & rna.obs["condition"].notna()
    rna_f = rna[ct_mask].copy()
    atac_f = atac[atac.obs["cell_type"].isin(TARGET_CELL_TYPES) & atac.obs["condition"].notna()].copy()
    logger.info(
        f"  after target-cell-type + condition filter: "
        f"RNA={rna_f.shape}, ATAC={atac_f.shape}"
    )
    del rna, atac

    # ---- Step 2: eligible donors --------------------------------------------
    logger.info("\n=== Step 2: identify eligible donors per cell type ===")
    elig_tables = {}
    for ct in TARGET_CELL_TYPES:
        elig_tables[ct] = select_eligible_donors(
            rna_f, cell_type=ct,
            min_cells_per_donor=MIN_CELLS_PER_DONOR,
            require_condition=True,
        )
        logger.info(f"\n  {ct}: {len(elig_tables[ct])} eligible donors")
        logger.info("\n" + elig_tables[ct].to_string(index=False))

    eligible_donors = sorted(set().union(*[set(t["donor_id"]) for t in elig_tables.values()]))
    logger.info(f"\n  union of eligible donors: {len(eligible_donors)}")

    # ---- Step 3: donor-stratified subsample + pairing ------------------------
    logger.info("\n=== Step 3: donor-stratified subsample + pair RNA<->ATAC ===")
    rna_e = rna_f[rna_f.obs["Donor ID"].isin(eligible_donors)].copy()
    atac_e = atac_f[atac_f.obs["Donor ID"].isin(eligible_donors)].copy()
    del rna_f, atac_f

    rna_sub = subsample_cells_per_donor(
        rna_e, n_per_donor=N_CELLS_PER_DONOR,
        donor_col="Donor ID", cell_type=None,
        cell_type_col="cell_type", seed=SEED,
    )

    shared = sorted(set(rna_sub.obs_names) & set(atac_e.obs_names))
    if len(shared) < rna_sub.n_obs:
        logger.warning(
            f"  lost {rna_sub.n_obs - len(shared)} cells in RNA<->ATAC pairing"
        )
    rna_sub = rna_sub[shared].copy()
    atac_sub = atac_e[shared].copy()
    del rna_e, atac_e
    logger.info(f"  final subsample: RNA={rna_sub.shape}, ATAC={atac_sub.shape}")
    logger.info("  (donor, cell_type) counts:")
    counts = rna_sub.obs.groupby(["Donor ID", "cell_type"], observed=True).size()
    logger.info("\n" + counts.to_string())

    # ---- Step 4: WNN embedding via h5py (avoid full 94 GB load) --------------
    logger.info("\n=== Step 4: pull joint embedding for the subsampled cells (h5py) ===")
    EMB_KEY = "X_pca"
    MOD_NAME = "rna"
    with h5py.File(INTEGRATED, "r") as f:
        full_obs_names = [
            n.decode() if isinstance(n, bytes) else n
            for n in f[f"mod/{MOD_NAME}/obs/_index"][:]
        ]
        name_to_row = {n: i for i, n in enumerate(full_obs_names)}
        rows = [name_to_row[c] for c in rna_sub.obs_names if c in name_to_row]
        if len(rows) != rna_sub.n_obs:
            missing = set(rna_sub.obs_names) - set(full_obs_names)
            raise RuntimeError(
                f"{len(missing)} subsampled cells absent from integrated h5mu "
                f"(e.g. {sorted(missing)[:3]})"
            )
        embedding = np.asarray(f[f"mod/{MOD_NAME}/obsm/{EMB_KEY}"][rows, :])
    embedding_cell_names = list(rna_sub.obs_names)
    logger.info(
        f"  embedding pulled: {embedding.shape} "
        f"(from mod/{MOD_NAME}/obsm/{EMB_KEY})"
    )

    # ---- Step 5: Phase 2 (HVG/HVP + KNN) -------------------------------------
    logger.info("\n=== Step 5: Phase 2 — HVG/HVP feature selection + KNN ===")
    phase2 = prepare_subsample(
        rna=rna_sub, atac=atac_sub,
        n_per_type=10**9,  # already pre-subsampled by donor x cell type
        n_genes=N_GENES, n_peaks=N_PEAKS, seed=SEED,
        embedding=embedding, embedding_cell_names=embedding_cell_names,
        knn_k=KNN_K,
    )
    rna_p2 = phase2["rna"]
    atac_p2 = phase2["atac"]
    knn = phase2["knn_indices"]
    rna_p2.uns["knn_indices"] = knn
    logger.info(
        f"  Phase 2 done: RNA={rna_p2.shape}, ATAC={atac_p2.shape}, "
        f"KNN={knn.shape}"
    )

    # ---- Step 6: save Phase 2 outputs ---------------------------------------
    rna_out = os.path.join(DATA_DIR, f"{DATASET}_rna_{SUB_TAG}.h5ad")
    atac_out = os.path.join(DATA_DIR, f"{DATASET}_atac_{SUB_TAG}.h5ad")
    rna_p2.write_h5ad(rna_out)
    atac_p2.write_h5ad(atac_out)
    logger.info(f"  wrote {rna_out}")
    logger.info(f"  wrote {atac_out}")

    # ---- Step 7: Phase 3 (triplets) -----------------------------------------
    logger.info("\n=== Step 7: Phase 3 — gene-peak-TF triplets ===")
    gene_ann = load_gene_annotations(GTF_PATH)
    logger.info(f"  gene_ann: {len(gene_ann)} records")

    pwm_dict, motif_db = load_transfac_motifs(
        MOTIF_RDS, MOTIF_TXT, gene_name_type="symbol"
    )
    logger.info(f"  motifs: {len(motif_db)}; PWMs: {len(pwm_dict)}")

    phase3 = run_phase3(
        rna_adata=rna_p2,
        atac_adata=atac_p2,
        gene_annotations=gene_ann,
        genome_fasta=GENOME_FA,
        pwm_dict=pwm_dict,
        motif_db=motif_db,
        gene_name_type="symbol",
        output_dir=DATA_DIR,
        prefix=f"{DATASET}_{SUB_TAG}",
    )
    logger.info("\nPhase 3 done:")
    logger.info(f"  triplets           : {phase3['triplets'].shape}")
    logger.info(f"  gene_labels        : {phase3['gene_labels'].shape}")
    logger.info(f"  peak_overlap_matrix: {phase3['peak_matrix'].shape}")

    logger.info("\nAll outputs written to %s with prefix %s", DATA_DIR, f"{DATASET}_{SUB_TAG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
