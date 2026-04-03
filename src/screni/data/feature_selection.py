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
        f"  Subsampled {adata.n_obs} → {result.n_obs} cells "
        f"({n_per_type} per type × {adata.obs[cell_type_col].nunique()} types)"
    )
    logger.info(f"  Cell type counts: {result.obs[cell_type_col].value_counts().to_dict()}")
    return result


def select_variable_features(
    adata: ad.AnnData,
    n_features: int,
    flavor: str = "seurat_v3",
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

    Returns
    -------
    AnnData subset with ``n_features`` columns, raw counts in ``.X``.
    """
    # Step 1: Remove zero-sum features
    if sp.issparse(adata.X):
        feature_sums = np.array(adata.X.sum(axis=0)).flatten()
    else:
        feature_sums = adata.X.sum(axis=0)
    nonzero_mask = feature_sums > 0
    n_removed = (~nonzero_mask).sum()
    if n_removed > 0:
        logger.info(f"  Removed {n_removed} zero-sum features")
        adata = adata[:, nonzero_mask].copy()

    # Step 2-3: Normalize and find HVGs
    # seurat_v3 flavor works on raw counts directly (no need to normalize first)
    if flavor == "seurat_v3":
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_features,
            flavor="seurat_v3",
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
        logger.info(f"  Filtered peaks: {n_before} → {result.n_vars} (removed {n_removed} non-chr)")
    return result


def prepare_subsample(
    rna: ad.AnnData,
    atac: ad.AnnData,
    n_per_type: int = 100,
    n_genes: int = 500,
    n_peaks: int = 10000,
    seed: int = 42,
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

    Returns
    -------
    Dict with keys 'rna' and 'atac', each a subsampled + feature-selected AnnData.
    """
    logger.info("=== Phase 2: Cell Subsampling & Feature Selection ===")

    # Subsample cells
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
    atac_sub = select_variable_features(atac_sub, n_features=n_peaks)

    # Final summary
    logger.info(
        f"Phase 2 complete:\n"
        f"  RNA:  {rna_sub.shape} (raw counts)\n"
        f"  ATAC: {atac_sub.shape} (raw counts)"
    )

    return {"rna": rna_sub, "atac": atac_sub}
