"""HVG / HVP selection for SEA-AD paired data (Phase 2 equivalent).

Standalone counterpart to ``screni.data.feature_selection.__main__`` that
processes the SEA-AD paired branch and nothing else.

Selection order vs the original pipeline:

The paper's `prepare_subsample()` subsamples ~200 cells first, then picks
HVGs/HVPs from that small subsample — VST on so few cells is noisy. SEA-AD
also has multi-donor structure: a pooled "50 cells per cell type" sample
across 28 multiome donors mixes samples in a way that breaks downstream
per-donor analyses. So this script doesn't subsample at all. It picks
HVGs/HVPs on the FULL cell set (~138k cells, much more stable) and leaves
cell selection to each sub-question (see scripts/subsample_seaad_paired.py).

Outputs (both written under data/processed/seaad/):

  seaad_paired_rna_hvg.h5ad   — full cells × 500 HVGs    (raw counts)
  seaad_paired_atac_hvp.h5ad  — full cells × 10000 HVPs  (raw counts)

Each output AnnData carries:
  - .X                              raw counts of the selected features
  - .obs                            full SEA-AD obs (object cols → str)
  - .obsm['X_pca']                  joint RNA-PCA + ATAC-LSI embedding
                                    (the WNN input space, 138k × 40)
  - .uns['wnn_neighbor_indices']    full-set WNN KNN, shape (138k, 20)

For wScReNI / downstream cell-specific networks, students should compute
KNN on their own cell selection from `obsm['X_pca']` — the full-set
indices in `uns['wnn_neighbor_indices']` reference positions in the full
138k array and aren't directly usable on a subsample.

Input:
  data/processed/seaad/seaad_paired_integrated.h5mu

(Switched from the separate seaad_paired_{rna,atac}.h5ad files to the
integrated h5mu so the WNN embedding + KNN come along for free. With
400 GB allocated the 91 GB load is comfortable.)

Run via slurm/run_seaad_hvg_selection.sh or:
  pixi run python scripts/run_seaad_hvg_selection.py
"""

import gc
import logging
from pathlib import Path

import anndata as ad
import muon as mu
import pandas as pd

from screni.data.feature_selection import (
    filter_chr_peaks,
    select_variable_features,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SEAAD_DIR = Path("data/processed/seaad")
H5MU_PATH = SEAAD_DIR / "seaad_paired_integrated.h5mu"

N_HVGS = 500
N_HVPS = 10_000


def _pull_obs(adata: ad.AnnData) -> pd.DataFrame:
    """Return the full obs, with cell_type mirrored from Subclass if needed
    and object-dtype columns coerced to str so h5py can write them.

    SEA-AD has ~100 obs columns and some carry mixed Python types (str+NaN
    +bool) that h5py rejects with "Can't implicitly convert non-string
    objects to strings". Coercing to str is the cheapest safe fix.

    Caveat for students:
        astype(str) on an object column stringifies every value, including
        missing ones — NaN becomes ``"nan"``, None becomes ``"None"``,
        pd.NA becomes ``"<NA>"``. Downstream ``.isna()`` will return False
        on these. To detect missingness, compare to the string explicitly,
        e.g. ``df["Cognitive Status"].isin(["nan", "None", "<NA>"])``.
        Categorical columns are unaffected and keep their NaN handling.
    """
    obs = adata.obs.copy()
    if "cell_type" not in obs.columns:
        if "Subclass" not in obs.columns:
            raise KeyError(
                f"Neither 'cell_type' nor 'Subclass' in obs (have: "
                f"{list(obs.columns)[:10]}...)"
            )
        obs["cell_type"] = obs["Subclass"]
    for c in obs.columns:
        if obs[c].dtype == "object":
            obs[c] = obs[c].astype(str)
    return obs


def _select_and_save(
    mod: ad.AnnData,
    joint_emb,
    wnn_knn,
    n_features: int,
    span: float,
    filter_chr: bool,
    out_path: Path,
    label: str,
) -> None:
    """Build a slim AnnData from a MuData mod, select features, save.

    The slim AnnData keeps raw counts in .X, the full obs (sanitized),
    and carries the joint WNN-input embedding + full-set WNN KNN as
    obsm / uns so downstream sub-questions can compute their own KNN.
    """
    logger.info(f"\n{'=' * 60}\n  {label}\n{'=' * 60}")
    logger.info(f"  input shape: {mod.shape}")

    if "counts" not in mod.layers:
        raise KeyError(
            f"{label}: missing layers['counts'] (have: {list(mod.layers.keys())}). "
            f"The h5mu must come from integrate_paired() which copies raw counts there."
        )

    adata = ad.AnnData(
        X=mod.layers["counts"].copy(),
        obs=_pull_obs(mod),
        var=mod.var.copy(),
    )
    # Attach WNN-derived obsm + uns up front so they survive the var-slicing
    # inside select_variable_features (which preserves obs/obsm/uns).
    if joint_emb is not None:
        adata.obsm["X_pca"] = joint_emb.copy()
    if wnn_knn is not None:
        adata.uns["wnn_neighbor_indices"] = wnn_knn.copy()

    logger.info(f"  cell types: {adata.obs['cell_type'].value_counts().to_dict()}")

    if filter_chr:
        logger.info("  filtering chr-prefixed peaks ...")
        adata = filter_chr_peaks(adata)

    logger.info(
        f"  selecting {n_features} variable features (Seurat v3 VST, span={span}) ..."
    )
    selected = select_variable_features(adata, n_features=n_features, span=span)
    logger.info(f"  selected shape: {selected.shape}")
    del adata
    gc.collect()

    selected.write_h5ad(out_path)
    logger.info(
        f"  wrote {out_path}: shape={selected.shape}, "
        f"obsm={list(selected.obsm.keys())}, uns={list(selected.uns.keys())}"
    )


def main() -> None:
    if not H5MU_PATH.exists():
        raise FileNotFoundError(f"Missing input: {H5MU_PATH}")

    logger.info(f"Loading {H5MU_PATH} (this is the 91 GB integrated file) ...")
    mdata = mu.read(str(H5MU_PATH))
    logger.info(
        f"  loaded: {mdata.shape}, modalities={list(mdata.mod.keys())}, "
        f"obsm={list(mdata.obsm.keys())}, uns keys={list(mdata.uns.keys())[:8]}..."
    )

    # WNN-derived inputs for downstream KNN. The joint embedding is the
    # input space the WNN graph was built in; wnn_neighbor_indices is the
    # k=20 KNN of that graph for all cells.
    joint_emb = mdata.obsm.get("X_pca")
    wnn_knn = mdata.uns.get("wnn_neighbor_indices")
    if joint_emb is None:
        logger.warning("  mdata.obsm['X_pca'] missing — outputs will lack the joint embedding")
    else:
        logger.info(f"  joint embedding: {joint_emb.shape}")
    if wnn_knn is None:
        logger.warning("  mdata.uns['wnn_neighbor_indices'] missing — outputs will lack WNN KNN")
    else:
        logger.info(f"  WNN neighbor indices: {wnn_knn.shape}")

    _select_and_save(
        mod=mdata.mod["rna"],
        joint_emb=joint_emb,
        wnn_knn=wnn_knn,
        n_features=N_HVGS,
        span=0.3,
        filter_chr=False,
        out_path=SEAAD_DIR / "seaad_paired_rna_hvg.h5ad",
        label="RNA",
    )
    gc.collect()

    _select_and_save(
        mod=mdata.mod["atac"],
        joint_emb=joint_emb,
        wnn_knn=wnn_knn,
        n_features=N_HVPS,
        span=0.5,   # wider LOESS span for ATAC's near-degenerate mean-variance
        filter_chr=True,
        out_path=SEAAD_DIR / "seaad_paired_atac_hvp.h5ad",
        label="ATAC",
    )

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
