import logging
import anndata as ad
import numpy as np
import scipy.sparse as sp
import scanpy as sc

logger = logging.getLogger(__name__)

def select_variable_features_per_celltype(
    adata: ad.AnnData,
    n_features: int,
    cell_type_key: str = "cell_type",
    flavor: str = "seurat_v3",
    span: float = 0.3,
) -> ad.AnnData:
    """Select highly variable features per cell type using VST.

    1. Remove global zero-sum features.
    2. For each cell type:
       a. Subset to cells of that type.
       b. Filter features expressed in < 3 cells (within the subset).
       c. Find top `n_features` variable features.
    3. Take the union of selected features across all cell types.
    4. Return RAW COUNTS for the union of selected features.

    Parameters
    ----------
    adata
        Input AnnData with raw counts in ``.X``.
    n_features
        Number of highly variable features to select PER CELL TYPE.
    cell_type_key
        The column in ``adata.obs`` containing cell type annotations.
    flavor
        HVG selection method. Use 'seurat_v3' to match original.
    span
        LOESS span for seurat_v3 VST (default 0.3, matching Seurat).

    Returns
    -------
    AnnData subset containing the union of per-cell-type highly variable features.
    """
    if cell_type_key not in adata.obs:
        raise ValueError(f"Key '{cell_type_key}' not found in adata.obs")

    # Step 1: Remove global zero-sum features
    if sp.issparse(adata.X):
        feature_sums = np.array(adata.X.sum(axis=0)).flatten()
    else:
        feature_sums = adata.X.sum(axis=0)
        
    nonzero_mask = feature_sums > 0
    n_removed = (~nonzero_mask).sum()
    if n_removed > 0:
        logger.info(f"  Removed {n_removed} global zero-sum features")
        adata = adata[:, nonzero_mask].copy()

    unique_cell_types = adata.obs[cell_type_key].dropna().unique()
    all_selected_features = set()

    # Step 2: Iterate over cell types to compute HVGs locally
    for ct in unique_cell_types:
        # Subset to current cell type
        ct_mask = adata.obs[cell_type_key] == ct
        adata_ct = adata[ct_mask].copy()
        
        # Filter local zero/low-expression genes to avoid LOESS singularity errors
        sc.pp.filter_genes(adata_ct, min_cells=3)
        
        # Skip if subset is too small
        if adata_ct.n_obs < 3 or adata_ct.n_vars <= n_features:
            logger.warning(f"  Skipping {ct}: Not enough cells or features.")
            all_selected_features.update(adata_ct.var_names)
            continue

        if flavor == "seurat_v3":
            sc.pp.highly_variable_genes(
                adata_ct,
                n_top_genes=n_features,
                flavor="seurat_v3",
                span=span,
            )
        else:
            sc.pp.normalize_total(adata_ct, target_sum=1e4)
            sc.pp.log1p(adata_ct)
            sc.pp.highly_variable_genes(
                adata_ct, 
                n_top_genes=n_features, 
                flavor=flavor
            )

        # Get HVGs for this cell type and add to our union set
        ct_hvgs = adata_ct.var_names[adata_ct.var["highly_variable"]]
        all_selected_features.update(ct_hvgs)
        
        logger.info(f"  Selected {len(ct_hvgs)} features for {ct}")

    # Step 3 & 4: Subset original object with the union of all selected features
    final_features = list(all_selected_features)
    result = adata[:, final_features].copy()

    # Verify we have raw counts
    if sp.issparse(result.X):
        max_val = result.X.max()
    else:
        max_val = result.X.max()

    logger.info(
        f"  Total unique features selected across all cell types: {result.n_vars} "
        f"(max value: {max_val:.1f}, should be >> 1 if raw counts)"
    )

    return result

if __name__ == "__main__":
    rna_full = ad.read_h5ad(data_dir / "retinal_rna.h5ad")
    rna_sub = subsample_cells(rna, n_per_type=n_per_type, seed=seed)
    rna_sub = select_variable_features_per_celltype(rna_sub, n_features=20)
    