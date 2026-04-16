"""Phase 2: Cell subsampling and feature selection.

Matches the original R ``select_features()`` and
``Select_partial_cells_for_scNewtorks()`` functions.

Key parameters from the paper:
    - 100 cells per cell type (retinal, 4 types = 400 total)
    - 50 cells per cell type (PBMC, 8 types = 400 total)
    - 500 HVGs for network inference (main benchmark)
    - 2000 HVGs for clustering benchmark
    - 10,000 HV peaks for ATAC
    - Feature selection uses Seurat v3 VST
    - Returns RAW COUNTS (not normalized) for selected features
"""

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def subsample_cells(
    adata: ad.AnnData,
    n_per_type: int,
    cell_type_col: str = "cell_type",
    seed: int = 42,
) -> ad.AnnData:
    """Randomly subsample a fixed number of cells per cell type.

    Matches ``Select_partial_cells_for_scNewtorks()`` from the original R code.

    Parameters
    ----------
    adata
        Input AnnData.
    n_per_type
        Number of cells to sample from each cell type.
    cell_type_col
        Column in ``.obs`` containing cell type labels.
    seed
        Random seed for reproducibility.

    Returns
    -------
    Subsampled AnnData with exactly ``n_per_type * n_types`` cells.
    """
    rng = np.random.RandomState(seed)
    indices = []

    for ct in sorted(adata.obs[cell_type_col].unique()):
        ct_idx = np.where(adata.obs[cell_type_col] == ct)[0]
        n_available = len(ct_idx)
        if n_available < n_per_type:
            logger.warning(
                f"  Cell type '{ct}' has only {n_available} cells, "
                f"requested {n_per_type}. Using all available."
            )
            indices.extend(ct_idx.tolist())
        else:
            sampled = rng.choice(ct_idx, size=n_per_type, replace=False)
            indices.extend(sorted(sampled.tolist()))

    result = adata[indices].copy()
    logger.info(
        f"  Subsampled {adata.n_obs} -> {result.n_obs} cells "
        f"({n_per_type} per type x {adata.obs[cell_type_col].nunique()} types)"
    )
    logger.info(f"  Cell type counts: {result.obs[cell_type_col].value_counts().to_dict()}")
    return result


def select_variable_features(
    adata: ad.AnnData,
    n_features: int,
    flavor: str = "seurat_v3",
    span: float = 0.3,
) -> ad.AnnData:
    """Select highly variable features using VST.

    Matches ``select_features()`` from the original R code:
    1. Remove zero-sum features
    2. LogNormalize (scale_factor=10000) -- only for HVG computation
    3. FindVariableFeatures with VST
    4. Return RAW COUNTS for selected features

    Parameters
    ----------
    adata
        Input AnnData with raw counts in ``.X``.
    n_features
        Number of highly variable features to select.
    flavor
        HVG selection method. Use 'seurat_v3' to match original.
    span
        LOESS span for seurat_v3 VST (default 0.3, matching Seurat).
        May need to be increased for small or binary data (e.g. ATAC).

    Returns
    -------
    AnnData subset with ``n_features`` columns, raw counts in ``.X``.
    """
    # Step 1: Remove zero-sum features (matches R: scrna[rowSums(scrna)>0, ])
    if sp.issparse(adata.X):
        feature_sums = np.array(adata.X.sum(axis=0)).flatten()
    else:
        feature_sums = adata.X.sum(axis=0)
    nonzero_mask = feature_sums > 0
    n_removed = (~nonzero_mask).sum()
    if n_removed > 0:
        logger.info(f"  Removed {n_removed} zero-sum features")
        adata = adata[:, nonzero_mask].copy()

    # Step 1b: Filter to features expressed in >= 3 cells.
    # Seurat's VST handles near-degenerate genes internally, but scanpy's
    # skmisc LOESS can become numerically singular on small cell counts
    # (e.g. 400 cells) when many genes are expressed in only 1-2 cells.
    n_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=3)
    n_filtered = n_before - adata.n_vars
    if n_filtered > 0:
        logger.info(f"  Filtered {n_filtered} features expressed in <3 cells")

    # Step 2-3: Normalize and find HVGs
    # seurat_v3 flavor works on raw counts directly (no need to normalize first)
    if flavor == "seurat_v3":
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_features,
            flavor="seurat_v3",
            span=span,
        )
    else:
        # For other flavors, normalize first
        work = adata.copy()
        sc.pp.normalize_total(work, target_sum=1e4)
        sc.pp.log1p(work)
        sc.pp.highly_variable_genes(work, n_top_genes=n_features, flavor=flavor)
        adata.var["highly_variable"] = work.var["highly_variable"]

    # Step 4: Return raw counts for selected features
    hvg_mask = adata.var["highly_variable"]
    result = adata[:, hvg_mask].copy()

    # Verify we have raw counts
    if sp.issparse(result.X):
        max_val = result.X.max()
    else:
        max_val = result.X.max()

    logger.info(
        f"  Selected {result.n_vars} variable features "
        f"(max value: {max_val:.1f}, should be >> 1 if raw counts)"
    )

    return result


def filter_chr_peaks(adata: ad.AnnData) -> ad.AnnData:
    """Filter ATAC peaks to chr-prefixed only, removing scaffolds.

    Removes peaks on scaffolds like GL000194.1, KI270711.1, etc.
    """
    chr_mask = adata.var_names.str.startswith("chr")
    n_before = adata.n_vars
    result = adata[:, chr_mask].copy()
    n_removed = n_before - result.n_vars
    if n_removed > 0:
        logger.info(f"  Filtered peaks: {n_before} -> {result.n_vars} (removed {n_removed} non-chr)")
    return result


def subsample_pairs(
    rna: ad.AnnData,
    atac: ad.AnnData,
    pairs: pd.DataFrame,
    n_per_type: int,
    seed: int = 42,
) -> tuple[ad.AnnData, ad.AnnData]:
    """Subsample matched RNA-ATAC cell pairs for unpaired datasets.

    For unpaired data (e.g. retinal), integration produces a 1:1 NN pairing
    between RNA and ATAC cells.  Subsampling must operate on these pairs so
    the resulting RNA and ATAC matrices stay aligned.

    Parameters
    ----------
    rna
        Full RNA AnnData with raw counts in ``.X``.
    atac
        Full ATAC AnnData with raw counts in ``.X``.
    pairs
        DataFrame with columns ``rna_cell``, ``atac_cell``, ``cell_type``.
    n_per_type
        Number of pairs to sample per cell type.
    seed
        Random seed.

    Returns
    -------
    ``(rna_sub, atac_sub)`` with shared obs_names and ``cell_type`` in ``.obs``.
    """
    rng = np.random.RandomState(seed)
    sampled_indices = []
    for ct in sorted(pairs["cell_type"].unique()):
        ct_idx = pairs.index[pairs["cell_type"] == ct].tolist()
        n = min(len(ct_idx), n_per_type)
        sampled = rng.choice(ct_idx, size=n, replace=False)
        sampled_indices.extend(sorted(sampled.tolist()))

    pairs_sub = pairs.loc[sampled_indices]

    rna_sub = rna[pairs_sub["rna_cell"].values].copy()
    atac_sub = atac[pairs_sub["atac_cell"].values].copy()

    # Assign cell type from pairs (RNA may have original labels)
    rna_sub.obs["cell_type"] = pairs_sub["cell_type"].values
    atac_sub.obs["cell_type"] = pairs_sub["cell_type"].values

    # Align obs_names so downstream code can match cells across modalities
    shared_names = [f"cell_{i}" for i in range(len(pairs_sub))]
    rna_sub.obs_names = shared_names
    atac_sub.obs_names = shared_names

    logger.info(
        f"  Subsampled {len(pairs)} pairs -> {len(pairs_sub)} "
        f"({n_per_type} per type x {pairs['cell_type'].nunique()} types)"
    )
    logger.info(f"  Cell type counts: {rna_sub.obs['cell_type'].value_counts().to_dict()}")
    return rna_sub, atac_sub


def prepare_subsample(
    rna: ad.AnnData,
    atac: ad.AnnData,
    n_per_type: int = 100,
    n_genes: int = 500,
    n_peaks: int = 10000,
    seed: int = 42,
    pairs: pd.DataFrame | None = None,
) -> dict[str, ad.AnnData]:
    """Full Phase 2 pipeline: subsample cells and select features.

    Parameters
    ----------
    rna
        RNA AnnData with ``.obs['cell_type']`` and raw counts.
    atac
        ATAC AnnData with ``.obs['cell_type']`` and raw counts.
    n_per_type
        Cells per cell type to sample.
    n_genes
        Number of HVGs to select.
    n_peaks
        Number of HV peaks to select.
    seed
        Random seed.
    pairs
        For unpaired datasets: DataFrame with ``rna_cell``, ``atac_cell``,
        ``cell_type`` columns (from Phase 1 NN pairing).  When provided,
        subsampling operates on matched pairs instead of independent cells.

    Returns
    -------
    Dict with keys 'rna' and 'atac', each a subsampled + feature-selected AnnData.
    """
    logger.info("=== Phase 2: Cell Subsampling & Feature Selection ===")

    if pairs is not None:
        # Unpaired data: subsample matched pairs
        logger.info("Subsampling matched RNA-ATAC pairs (unpaired mode)...")
        rna_sub, atac_sub = subsample_pairs(
            rna, atac, pairs, n_per_type=n_per_type, seed=seed,
        )
    else:
        # Paired data: subsample cells independently
        logger.info("Subsampling RNA cells...")
        rna_sub = subsample_cells(rna, n_per_type=n_per_type, seed=seed)

        logger.info("Subsampling ATAC cells...")
        atac_sub = subsample_cells(atac, n_per_type=n_per_type, seed=seed)

    # Filter ATAC to chr peaks
    atac_sub = filter_chr_peaks(atac_sub)

    # Feature selection
    logger.info(f"Selecting {n_genes} HVGs from RNA...")
    rna_sub = select_variable_features(rna_sub, n_features=n_genes)

    logger.info(f"Selecting {n_peaks} HV peaks from ATAC...")
    # ATAC data is often binary/near-binary, making the mean-variance
    # relationship near-degenerate.  A wider LOESS span avoids numerical
    # singularity in skmisc that R's loess() handles gracefully.
    atac_sub = select_variable_features(atac_sub, n_features=n_peaks, span=0.5)

    # Final summary
    logger.info(
        f"Phase 2 complete:\n"
        f"  RNA:  {rna_sub.shape} (raw counts)\n"
        f"  ATAC: {atac_sub.shape} (raw counts)"
    )

    return {"rna": rna_sub, "atac": atac_sub}


if __name__ == "__main__":
    import logging
    from pathlib import Path

    import muon as mu

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data_dir = Path("data/processed")
    out_dir = Path("data/processed")

    # --- PBMC (paired) ---
    logger.info("\n" + "=" * 60)
    logger.info("PBMC (paired)")
    logger.info("=" * 60)

    mdata = mu.read(str(data_dir / "pbmc_integrated.h5mu"))
    rna_mod = mdata.mod["rna"]
    atac_mod = mdata.mod["atac"]

    # Extract raw counts (integration leaves normalized values in .X)
    pbmc_rna = ad.AnnData(
        X=rna_mod.layers["counts"].copy(),
        obs=rna_mod.obs[["cell_type"]].copy(),
        var=rna_mod.var.copy(),
    )
    pbmc_atac = ad.AnnData(
        X=atac_mod.layers["counts"].copy(),
        obs=atac_mod.obs[["cell_type"]].copy(),
        var=atac_mod.var.copy(),
    )
    del mdata

    pbmc = prepare_subsample(
        rna=pbmc_rna, atac=pbmc_atac,
        n_per_type=50, n_genes=500, n_peaks=10000, seed=42,
    )

    pbmc["rna"].write_h5ad(out_dir / "pbmc_rna_sub.h5ad")
    pbmc["atac"].write_h5ad(out_dir / "pbmc_atac_sub.h5ad")
    logger.info(f"Saved PBMC subsampled data to {out_dir}")

    del pbmc, pbmc_rna, pbmc_atac

    # --- Retinal (unpaired) ---
    logger.info("\n" + "=" * 60)
    logger.info("Retinal (unpaired)")
    logger.info("=" * 60)

    pairs = pd.read_csv(data_dir / "retinal_nn_pairs.csv")
    rna_full = ad.read_h5ad(data_dir / "retinal_rna.h5ad")
    atac_full = ad.read_h5ad(data_dir / "retinal_atac.h5ad")

    retinal = prepare_subsample(
        rna=rna_full, atac=atac_full,
        n_per_type=100, n_genes=500, n_peaks=10000, seed=42,
        pairs=pairs,
    )

    retinal["rna"].write_h5ad(out_dir / "retinal_rna_sub.h5ad")
    retinal["atac"].write_h5ad(out_dir / "retinal_atac_sub.h5ad")
    logger.info(f"Saved retinal subsampled data to {out_dir}")

    del retinal, rna_full, atac_full

    # --- SEA-AD (paired, multiome) ---
    seaad_dir = data_dir / "seaad"

    if (seaad_dir / "seaad_paired_integrated.h5mu").exists():
        logger.info("\n" + "=" * 60)
        logger.info("SEA-AD Paired (multiome)")
        logger.info("=" * 60)

        mdata = mu.read(str(seaad_dir / "seaad_paired_integrated.h5mu"))
        rna_mod = mdata.mod["rna"]
        atac_mod = mdata.mod["atac"]

        seaad_p_rna = ad.AnnData(
            X=rna_mod.layers["counts"].copy(),
            obs=rna_mod.obs[["cell_type"]].copy(),
            var=rna_mod.var.copy(),
        )
        seaad_p_atac = ad.AnnData(
            X=atac_mod.layers["counts"].copy(),
            obs=atac_mod.obs[["cell_type"]].copy(),
            var=atac_mod.var.copy(),
        )
        del mdata

        seaad_p = prepare_subsample(
            rna=seaad_p_rna, atac=seaad_p_atac,
            n_per_type=50, n_genes=500, n_peaks=10000, seed=42,
        )

        seaad_p["rna"].write_h5ad(seaad_dir / "seaad_paired_rna_sub.h5ad")
        seaad_p["atac"].write_h5ad(seaad_dir / "seaad_paired_atac_sub.h5ad")
        logger.info(f"Saved SEA-AD paired subsampled data to {seaad_dir}")

        del seaad_p, seaad_p_rna, seaad_p_atac

    # --- SEA-AD (unpaired, singleome) ---
    if (seaad_dir / "seaad_unpaired_nn_pairs.csv").exists():
        logger.info("\n" + "=" * 60)
        logger.info("SEA-AD Unpaired (singleome)")
        logger.info("=" * 60)

        seaad_pairs = pd.read_csv(seaad_dir / "seaad_unpaired_nn_pairs.csv")
        seaad_rna = ad.read_h5ad(seaad_dir / "seaad_unpaired_rna.h5ad")
        seaad_atac = ad.read_h5ad(seaad_dir / "seaad_unpaired_atac.h5ad")

        seaad_u = prepare_subsample(
            rna=seaad_rna, atac=seaad_atac,
            n_per_type=50, n_genes=500, n_peaks=10000, seed=42,
            pairs=seaad_pairs,
        )

        seaad_u["rna"].write_h5ad(seaad_dir / "seaad_unpaired_rna_sub.h5ad")
        seaad_u["atac"].write_h5ad(seaad_dir / "seaad_unpaired_atac_sub.h5ad")
        logger.info(f"Saved SEA-AD unpaired subsampled data to {seaad_dir}")

        del seaad_u, seaad_rna, seaad_atac

    logger.info("\nDone.")
