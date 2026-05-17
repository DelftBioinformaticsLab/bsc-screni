"""wScReNI inference + precision/recall on the SeaAD paired multiome.

Script counterpart to ``src/screni/data/run_infer_wscreni_seaad.ipynb`` for
cluster execution.  Same logic, runnable via:

    python scripts/run_infer_wscreni_seaad.py

The team's pixi container does not include ``nbformat`` / ``nbconvert``,
so cluster-runnable scripts must be plain ``.py`` files.  Keep this and
the notebook in sync when changing one.

Pipeline:

1. Load processed SeaAD paired inputs produced by ``prep_seaad_sq1.py``
   (or the team's ``_sub42`` variant; controlled by ``SUB_TAG``).
2. Build the ``GenePeakOverlapLabs`` object.
3. Run ``infer_wscreni_networks`` -> per-cell weighted gene x gene
   regulatory networks (writes per-cell .txt files + returns the dict).
4. Save an AnnData checkpoint carrying the gene order.
5. Evaluate against the **human** ChIP-Atlas
   (``data/chip_seq_5kb_TF_target.df.txt``) using precision/recall at
   several top-K cutoffs.

Output goes under ``src/screni/data/output/`` to match the notebook's
convention; per-cell network files live in
``output/networks_<DATASET>/wScReNI/``.
"""

from __future__ import annotations

import logging
import os
import sys

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from screni.data import (
    GenePeakOverlapLabs,
    calculate_precision_recall,
    infer_wscreni_networks,
    load_chip_atlas,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ---- Configuration ----------------------------------------------------------

DATASET = "seaad_paired"
SUB_TAG = os.environ.get("WSCRENI_SUB_TAG", "sq1")   # 'sq1' (our prep) or 'sub42' (team's)
DATA_DIR = os.path.join(REPO_ROOT, "data", "processed", "seaad")
REF_PATH = os.path.join(REPO_ROOT, "data", "chip_seq_5kb_TF_target.df.txt")
OUT_DIR = os.path.join(REPO_ROOT, "src", "screni", "data", "output")
NET_DIR = os.path.join(OUT_DIR, f"networks_{DATASET}")

RNA_PATH = os.path.join(DATA_DIR, f"{DATASET}_rna_{SUB_TAG}.h5ad")
ATAC_PATH = os.path.join(DATA_DIR, f"{DATASET}_atac_{SUB_TAG}.h5ad")
TRIPLETS_PATH = os.path.join(DATA_DIR, f"{DATASET}_{SUB_TAG}_triplets.csv")
LABELS_PATH = os.path.join(DATA_DIR, f"{DATASET}_{SUB_TAG}_gene_labels.csv")
PEAK_MAT_PATH = os.path.join(DATA_DIR, f"{DATASET}_{SUB_TAG}_peak_overlap_matrix.npz")

N_JOBS = int(os.environ.get("WSCRENI_N_JOBS", "32"))
MAX_CELLS_PER_BATCH = 10
TOP_KS = [100, 250, 500, 1000, 2000, 5000, 10000, 20000, 35000]


def main() -> int:
    logger.info("=" * 70)
    logger.info(f"wScReNI inference | DATASET={DATASET} | SUB_TAG={SUB_TAG}")
    logger.info(f"n_jobs={N_JOBS}, output dir={NET_DIR}")
    logger.info("=" * 70)

    for label, path in [
        ("RNA", RNA_PATH), ("ATAC", ATAC_PATH), ("triplets", TRIPLETS_PATH),
        ("gene_labels", LABELS_PATH), ("peak_matrix", PEAK_MAT_PATH),
        ("ChIP-Atlas", REF_PATH),
    ]:
        ok = "OK" if os.path.exists(path) else "MISSING"
        logger.info(f"  {ok:7s} {label}: {path}")
        if not os.path.exists(path):
            logger.error(f"Required input missing: {path}")
            return 2
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Step 1: load processed inputs --------------------------------------
    logger.info("\n=== Step 1: load processed inputs ===")
    rna = ad.read_h5ad(RNA_PATH)
    triplets = pd.read_csv(TRIPLETS_PATH)
    labels_df = pd.read_csv(LABELS_PATH)
    peak_matrix = np.load(PEAK_MAT_PATH)["peak_matrix"]
    peak_names = triplets["peak"].unique().tolist()

    if "knn_indices" not in rna.uns:
        raise KeyError(
            f"{RNA_PATH} has no uns['knn_indices']. "
            f"Re-run Phase 2 with KNN export enabled."
        )
    knn = np.asarray(rna.uns["knn_indices"])

    logger.info(f"  RNA: {rna.shape}, peak_matrix: {peak_matrix.shape}, KNN: {knn.shape}")
    logger.info(f"  triplets: {triplets.shape}, gene_labels: {labels_df.shape}")
    logger.info(f"  unique peaks across triplets: {len(peak_names)}")

    # ---- Step 2: build GenePeakOverlapLabs -----------------------------------
    logger.info("\n=== Step 2: build GenePeakOverlapLabs ===")
    rows = []
    for _, row in labels_df.iterrows():
        if pd.isna(row["associated_peaks"]):
            continue
        peaks_for_gene = str(row["associated_peaks"]).split(",")
        tf_str = (
            ";".join(str(row["associated_TFs"]).split(","))
            if pd.notna(row["associated_TFs"]) else ""
        )
        for peak in peaks_for_gene:
            rows.append({
                "gene": row["gene"],
                "peak": peak.strip(),
                "TF": tf_str,
                "label": row["type"],
            })
    labs_df = pd.DataFrame(rows)
    labs = GenePeakOverlapLabs(
        genes=labs_df["gene"].tolist(),
        peaks=labs_df["peak"].tolist(),
        tfs=labs_df["TF"].tolist(),
        labels=labs_df["label"].tolist(),
    )
    logger.info(
        f"  labs entries: {len(labs.genes)} "
        f"(TFs: {(labels_df['type'] == 'TF').sum()}, "
        f"targets: {(labels_df['type'] == 'target').sum()})"
    )

    # ---- Step 3: run wScReNI inference --------------------------------------
    logger.info("\n=== Step 3: run wScReNI inference (this is the long step) ===")
    networks = infer_wscreni_networks(
        expr=rna,
        peak_mat=peak_matrix,
        labs=labs,
        nearest_neighbors_idx=knn,
        network_path=NET_DIR,
        data_name=DATASET,
        cell_index=None,
        n_jobs=N_JOBS,
        max_cells_per_batch=MAX_CELLS_PER_BATCH,
        peak_names=peak_names,
        cell_names=rna.obs_names.tolist(),
    )
    first_cell = next(iter(networks))
    logger.info(
        f"  inferred {len(networks)} networks; "
        f"first matrix {networks[first_cell].shape}, "
        f"non-zero {int(np.count_nonzero(networks[first_cell]))}"
    )

    # ---- Step 4: AnnData checkpoint ----------------------------------------
    rna.uns["wScReNI_gene_order"] = list(networks.gene_names)
    chk_path = os.path.join(OUT_DIR, f"{DATASET}_{SUB_TAG}_with_networks.h5ad")
    rna.write_h5ad(chk_path)
    logger.info(f"  checkpoint written: {chk_path}")

    # ---- Step 5: ChIP-Atlas precision/recall --------------------------------
    logger.info("\n=== Step 5: precision/recall vs human ChIP-Atlas ===")
    tf_pairs = load_chip_atlas(REF_PATH)
    gene_list = list(networks.gene_names)
    gene_set = set(gene_list)
    reachable = {
        p for p in tf_pairs
        if len(p.split("_", 1)) == 2
        and p.split("_", 1)[0] in gene_set
        and p.split("_", 1)[1] in gene_set
    }
    logger.info(
        f"  ChIP-Atlas total: {len(tf_pairs):,}; "
        f"reachable in 500-gene set: {len(reachable):,}"
    )

    records = []
    for cell, mat in networks.items():
        for k in TOP_KS:
            p, r = calculate_precision_recall(
                scnetwork_weights=mat,
                tf_target_pair=tf_pairs,
                top_number=k,
                gene_names=gene_list,
            )
            records.append({"cell": cell, "K": k, "precision": p, "recall": r})
    pr_df = pd.DataFrame(records)
    pr_csv = os.path.join(OUT_DIR, f"{DATASET}_{SUB_TAG}_wscreni_precision_recall.csv")
    pr_df.to_csv(pr_csv, index=False)
    logger.info(f"  wrote {len(pr_df)} (cell, K) rows -> {pr_csv}")

    summary = (
        pr_df.groupby("K")[["precision", "recall"]]
        .agg(["mean", "std"])
        .round(4)
    )
    sum_csv = os.path.join(OUT_DIR, f"{DATASET}_{SUB_TAG}_wscreni_pr_summary.csv")
    summary.to_csv(sum_csv)
    logger.info(f"\nPR summary across cells:\n{summary}")

    # ---- Step 6: PR curve plot ---------------------------------------------
    mean_pr = pr_df.groupby("K")[["precision", "recall"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(mean_pr["recall"], mean_pr["precision"], marker="o",
            label=f"wScReNI ({DATASET}, {SUB_TAG})")
    for _, row in mean_pr.iterrows():
        ax.annotate(f"K={int(row['K'])}",
                    (row["recall"], row["precision"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    random_p = len(reachable) / (len(gene_list) ** 2)
    ax.axhline(random_p, ls="--", color="gray", alpha=0.6,
               label=f"Random baseline ({random_p:.4f})")
    ax.set_xlabel("Recall (mean across cells)")
    ax.set_ylabel("Precision (mean across cells)")
    ax.set_title(f"wScReNI vs human ChIP-Atlas - {DATASET} ({SUB_TAG})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, f"{DATASET}_{SUB_TAG}_wscreni_pr_curve.png")
    fig.savefig(plot_path, dpi=120)
    logger.info(f"  PR curve plot: {plot_path}")

    logger.info("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
