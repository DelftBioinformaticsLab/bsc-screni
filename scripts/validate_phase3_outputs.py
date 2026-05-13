"""Sanity-check Phase 3 outputs before downstream wScReNI consumes them.

Substitutes for a wScReNI smoke test (the random-forest step lives outside
this repo). Loads the six Phase 3 artifacts for a given prefix and verifies:

  - peak_matrix has the right shape and contains noise-floor non-zero entries
  - triplets reference only TFs / targets / peaks that exist in the labels
    and peak_info files
  - peak_info covers exactly the peaks in correlated_pairs
  - gene_labels has every input HVG exactly once, labelled TF or target
  - peak_overlap_matrix rows correspond to cells in the matching _sub h5ads
  - WNN KNN, if present in the rna_sub file, is well-formed

Usage:
    pixi run python scripts/validate_phase3_outputs.py --prefix seaad_paired_sub42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA = Path("data/processed/seaad")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--prefix", required=True,
        help="Filename prefix used by run_phase3 (e.g. 'seaad_paired_sub42').",
    )
    p.add_argument(
        "--data-dir", type=Path, default=DATA,
        help=f"Directory containing the Phase 3 outputs (default: {DATA}).",
    )
    return p.parse_args()


def _require(cond: bool, msg: str) -> None:
    if not cond:
        logger.error(f"  FAIL: {msg}")
        global _n_fail
        _n_fail += 1
    else:
        logger.info(f"  OK  : {msg}")
        global _n_pass
        _n_pass += 1


_n_pass = 0
_n_fail = 0


def main() -> None:
    args = parse_args()
    d = args.data_dir
    p = args.prefix

    paths = {
        "peak_gene_pairs": d / f"{p}_peak_gene_pairs.csv",
        "motif_peak_pairs": d / f"{p}_motif_peak_pairs.csv",
        "triplets": d / f"{p}_triplets.csv",
        "gene_labels": d / f"{p}_gene_labels.csv",
        "peak_info": d / f"{p}_peak_info.csv",
        "peak_matrix": d / f"{p}_peak_overlap_matrix.npz",
    }
    # The matching sub-input files (use the seed suffix in the prefix)
    suffix = p.split("_sub", 1)[1] if "_sub" in p else ""
    branch = "paired" if "paired" in p else "unpaired"
    sub_rna = d / f"seaad_{branch}_rna_sub{suffix}.h5ad"
    sub_atac = d / f"seaad_{branch}_atac_sub{suffix}.h5ad"

    logger.info(f"\n=== Validating Phase 3 outputs (prefix={p!r}) ===\n")
    for name, path in paths.items():
        _require(path.exists(), f"output exists: {path.name}")
    if _n_fail:
        logger.error("\nMissing outputs — abort.")
        sys.exit(1)

    pgp = pd.read_csv(paths["peak_gene_pairs"])
    mpp = pd.read_csv(paths["motif_peak_pairs"])
    trip = pd.read_csv(paths["triplets"])
    gl = pd.read_csv(paths["gene_labels"])
    pi = pd.read_csv(paths["peak_info"])
    pm = np.load(paths["peak_matrix"])["peak_matrix"]

    logger.info("\n--- File shapes ---")
    for name, df in (("peak_gene_pairs", pgp), ("motif_peak_pairs", mpp),
                     ("triplets", trip), ("gene_labels", gl), ("peak_info", pi)):
        logger.info(f"  {name:18s}: {len(df):>7} rows, cols={list(df.columns)}")
    logger.info(f"  peak_matrix       : {pm.shape}")

    # ---- Inter-file consistency ----
    logger.info("\n--- Cross-references ---")

    _require(set(pgp.columns) >= {"gene", "peak", "spearman_r"},
             "peak_gene_pairs has columns gene/peak/spearman_r")
    _require(pgp["spearman_r"].abs().min() > 0.1 - 1e-6,
             "all correlated pairs have |r| > 0.1")

    _require(set(trip.columns) >= {"TF", "peak", "target_gene", "spearman_r"},
             "triplets has columns TF/peak/target_gene/spearman_r")
    _require((trip["TF"] != trip["target_gene"]).all(),
             "no self-regulation in triplets (TF != target_gene)")

    _require(set(trip["TF"]) <= set(gl.loc[gl["type"] == "TF", "gene"]),
             "every triplet TF is labelled as TF in gene_labels")
    _require(set(trip["target_gene"]) <= set(gl["gene"]),
             "every triplet target is in gene_labels")
    _require(set(trip["peak"]) <= set(pi["peak"]),
             "every triplet peak is in peak_info")
    _require(set(pi["peak"]) == set(pgp["peak"]),
             "peak_info covers exactly the peaks in correlated_pairs")
    _require(pm.shape[1] == len(pi),
             "peak_matrix column count matches peak_info row count")

    # ---- gene_labels ----
    logger.info("\n--- gene_labels ---")
    _require(gl["gene"].is_unique, "every HVG appears at most once in gene_labels")
    _require(set(gl["type"]) <= {"TF", "target"},
             "gene_labels.type is TF or target only")
    n_tf = (gl["type"] == "TF").sum()
    n_target = (gl["type"] == "target").sum()
    logger.info(f"  TFs       : {n_tf}")
    logger.info(f"  targets   : {n_target}")
    logger.info(f"  total HVGs: {len(gl)}")

    # ---- peak_matrix value sanity ----
    logger.info("\n--- peak_matrix values ---")
    logger.info(f"  shape   : {pm.shape}")
    logger.info(f"  min     : {pm.min():.4f}")
    logger.info(f"  max     : {pm.max():.1f}")
    logger.info(f"  mean    : {pm.mean():.3f}")
    _require(pm.min() < 0,
             "peak_matrix has negative values (i.e. Gaussian noise was added)")
    _require(pm.max() >= 1,
             "peak_matrix has integer-scale max (raw ATAC counts present)")

    # ---- Cell alignment with sub h5ads ----
    logger.info("\n--- Cell alignment with sub-input files ---")
    if sub_rna.exists() and sub_atac.exists():
        rna = ad.read_h5ad(sub_rna, backed="r")
        atac = ad.read_h5ad(sub_atac, backed="r")
        _require(pm.shape[0] == rna.n_obs,
                 f"peak_matrix rows ({pm.shape[0]}) == sub RNA cells ({rna.n_obs})")
        _require(pm.shape[0] == atac.n_obs,
                 f"peak_matrix rows ({pm.shape[0]}) == sub ATAC cells ({atac.n_obs})")
        _require(set(gl["gene"]) == set(rna.var_names),
                 "gene_labels covers exactly the HVGs in sub RNA")

        # WNN KNN (carried through from Phase 2 as global uns).
        # NOTE: uns is not sliced on subsample so the indices reference the
        # FULL ~138k-cell set, not the n_obs subsample. For per-cell-network
        # work students should recompute KNN on obsm["X_pca"][selected_cells].
        if "wnn_neighbor_indices" in rna.uns:
            knn = np.asarray(rna.uns["wnn_neighbor_indices"])
            logger.info(f"  WNN KNN : shape={knn.shape}, dtype={knn.dtype}")
            logger.info(
                f"            (references full-set positions, not subsample — "
                f"recompute on obsm['X_pca'] for subsample KNN)"
            )
            _require(knn.shape[1] == 20,
                     "WNN KNN has 20 neighbors per cell")
            _require(knn.min() >= 0 and knn.max() < knn.shape[0],
                     "WNN KNN indices are in [0, full_n_cells)")
        else:
            logger.info("  WNN KNN : not present in sub RNA (ok if subsampled manually)")

        if "X_pca" in rna.obsm:
            xp = rna.obsm["X_pca"]
            logger.info(f"  X_pca   : shape={xp.shape} (joint WNN-input embedding for subsample)")
            _require(xp.shape[0] == rna.n_obs,
                     "X_pca row count matches sub n_obs (ready for student KNN)")
        rna.file.close(); atac.file.close()
    else:
        logger.warning(
            f"  skipped — sub input files not found "
            f"({sub_rna.name}, {sub_atac.name})"
        )

    # ---- Summary ----
    logger.info(f"\n=== {_n_pass} checks passed, {_n_fail} failed ===")
    if _n_fail:
        sys.exit(2)


if __name__ == "__main__":
    main()
