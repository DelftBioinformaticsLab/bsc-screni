"""
kScReNI — kernel Single-Cell Regulatory Network Inference
Python translation of the original R implementation.

Key differences from wScReNI:
- Uses a SNN (Shared Nearest Neighbor) graph instead of precomputed KNN indices
  to select neighbours for each cell
- Uses GENIE3 (random forest on expression only, no ATAC peaks) instead of
  gene_peak_randomForest — suitable for unpaired / RNA-only data
- Preprocessing (normalisation, scaling, PCA, SNN graph) is done internally

Dependencies:
    pip install numpy pandas scikit-learn joblib scanpy anndata
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
import scanpy as sc
import anndata as ad
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# GENIE3 — random forest gene regulatory network inference (RNA only)
# ---------------------------------------------------------------------------

def genie3(
    expr_matrix: np.ndarray,     # genes × cells
    gene_names: list[str],
    n_trees: int = 100,
    n_jobs: int = 1,
    seed: int = 100,
) -> np.ndarray:
    """
    GENIE3: infer a genes × genes regulatory weight matrix using Random Forests.
    For each target gene, predicts its expression from all other genes and
    records feature importances as regulatory weights.

    Mirrors R's GENIE3::GENIE3(snn.expr, nCores=1, nTrees=100).

    Parameters
    ----------
    expr_matrix : np.ndarray, shape (n_genes, n_cells)
    gene_names  : list of gene name strings
    n_trees     : number of trees per forest (matches R's nTrees=100)
    n_jobs      : parallel workers
    seed        : random seed (matches R's set.seed(100))

    Returns
    -------
    weights : np.ndarray, shape (n_genes, n_genes)
        weights[i, j] = importance of gene j regulating gene i
        Diagonal is 0 (gene excluded from its own prediction).
    """
    rng = np.random.default_rng(seed)

    # Transpose to (cells × genes) for sklearn
    X = expr_matrix.T.astype(float)   # (n_cells, n_genes)
    n_cells, n_genes = X.shape

    def _fit_one_gene(target_idx: int) -> np.ndarray:
        other_idxs  = [i for i in range(n_genes) if i != target_idx]
        X_input     = X[:, other_idxs]      # (n_cells, n_genes-1)
        y           = X[:, target_idx]      # (n_cells,)

        weight_vector = np.zeros(n_genes)

        if y.sum() == 0:
            return weight_vector

        rf = RandomForestRegressor(
            n_estimators=n_trees,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=1,
            random_state=int(rng.integers(0, 2**31)),
        )
        rf.fit(X_input, y)

        importances = rf.feature_importances_   # length n_genes-1
        for rank, orig_idx in enumerate(other_idxs):
            weight_vector[orig_idx] = importances[rank]

        return weight_vector

    columns = [_fit_one_gene(idx) for idx in range(n_genes)]
    weights = np.column_stack(columns)     # (n_genes, n_genes)

    # Replace any NaN with 0 — mirrors R's ifelse(sub_res=='NaN', 0, sub_res)
    weights = np.nan_to_num(weights, nan=0.0)

    return weights


# ---------------------------------------------------------------------------
# Main function — Infer_kScReNI_scNetworks
# ---------------------------------------------------------------------------

def infer_kScReNI_sc_networks(
    expr_matrix,              # pd.DataFrame or np.ndarray, genes × cells
    n_features: int = 4000,   # HVGs for SNN graph construction
    knn: int = 20,            # k neighbours for SNN graph
    nthread: int = 20,        # parallel workers
    n_trees: int = 100,       # trees per GENIE3 forest
) -> dict[str, np.ndarray]:
    """
    Infer cell-specific regulatory networks using kScReNI.

    For each cell i:
      1. Build a SNN graph from HVG-PCA space (via scanpy).
      2. Select the top knn+1 neighbours by SNN weight.
      3. Run GENIE3 on the pooled expression of those neighbours.

    Mirrors R's Infer_kScReNI_scNetworks which uses:
      - Seurat: NormalizeData → ScaleData → FindVariableFeatures →
                RunPCA → FindNeighbors → RNA_snn graph
      - GENIE3 on the SNN-pooled expression

    Parameters
    ----------
    expr_matrix : genes × cells expression (raw counts)
    n_features  : number of HVGs for PCA + SNN (matches R's nfeatures=4000)
    knn         : number of neighbours (matches R's k.param=20)
    nthread     : parallel workers across cells
    n_trees     : trees per forest (matches R's nTrees=100)

    Returns
    -------
    dict: cell_name -> np.ndarray (n_genes, n_genes)
    """

    # 1. Convert to AnnData
    if isinstance(expr_matrix, pd.DataFrame):
        cell_names = list(expr_matrix.columns)
        gene_names = list(expr_matrix.index)
        X = expr_matrix.values.T.astype(float)   # → (cells, genes)
    else:
        X = expr_matrix.T.astype(float)
        cell_names = [f"cell_{i}" for i in range(X.shape[0])]
        gene_names = [f"gene_{i}" for i in range(X.shape[1])]

    adata = ad.AnnData(X=X)
    adata.obs_names  = pd.Index(cell_names)
    adata.var_names  = pd.Index(gene_names)
    n_cells, n_genes = adata.shape

    print(f"Total number of cells: {n_cells}")

    # 2. Preprocessing: mirrors Seurat pipeline
    # NormalizeData — log1p normalisation to 10k counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ScaleData — zero-mean unit-variance per gene
    sc.pp.scale(adata, zero_center=True)

    # FindVariableFeatures — HVGs via Seurat's VST method
    sc.pp.highly_variable_genes(adata, n_top_genes=n_features, flavor="seurat")
    hvg_mask = adata.var["highly_variable"].values

    # RunPCA on HVGs
    sc.tl.pca(adata, use_highly_variable=True)

    # FindNeighbors — builds KNN + SNN graph on PCA embedding
    # n_neighbors=knn matches Seurat's k.param
    sc.pp.neighbors(adata, n_neighbors=knn, use_rep="X_pca")

    # Extract SNN connectivity matrix (cells × cells)
    # Seurat's RNA_snn = scanpy's connectivities
    snn_mat = adata.obsp["connectivities"]   # sparse (n_cells, n_cells)
    if sp.issparse(snn_mat):
        snn_mat = snn_mat.toarray()          # → dense for easy row sorting

    print(f"SNN matrix shape: {snn_mat.shape}")

    # Raw counts for GENIE3 (use original counts, not normalised)
    # mirrors R: snn.expr <- exprMatrix[, neighbours]  (raw exprMatrix)
    expr_raw = expr_matrix.values.astype(float) if isinstance(expr_matrix, pd.DataFrame) \
               else expr_matrix.astype(float)    # (genes, cells)

    # 3. Per-cell GENIE3
    def _process_cell(i: int) -> np.ndarray:
        # Select top knn+1 neighbours by SNN weight (descending)
        # Mirrors R: order(mat[i,], decreasing=T)[1:(knn+1)]
        neighbour_idxs = np.argsort(snn_mat[i])[::-1][: knn + 1]

        snn_expr = expr_raw[:, neighbour_idxs]   # (genes, knn+1)

        weights = genie3(snn_expr, gene_names, n_trees=n_trees, seed=100)
        return weights

    # Batch parallel — same pattern as wScReNI
    max_per_batch = 10
    n_batches     = int(np.ceil(n_cells / max_per_batch))
    all_weights   = []

    for batch_idx in range(n_batches):
        start       = batch_idx * max_per_batch
        end         = min(start + max_per_batch, n_cells)
        batch_cells = list(range(start, end))
        print(f"Cell {batch_cells[0]} to cell {batch_cells[-1]}")

        n_workers = min(nthread, max(1, int(len(batch_cells) * 1.5)))
        results   = Parallel(n_jobs=n_workers)(
            delayed(_process_cell)(i) for i in batch_cells
        )
        all_weights.extend(results)

    # Return dict keyed by cell name — mirrors R's names(scNet_list) <- colnames(exprMatrix)
    return {cell_names[i]: all_weights[i] for i in range(n_cells)}


# ---------------------------------------------------------------------------
# Load preprocessed data and run kScReNI
# ---------------------------------------------------------------------------
# kScReNI is designed for UNPAIRED data — RNA only, no ATAC needed.
# It builds its own SNN graph internally from the expression matrix.
#
# Expected files:
#   *_rna_sub.h5ad   (400 cells, 500 HVGs, raw counts)

if __name__ == "__main__":

    DATASET = "retinal"
    BASE    = f"../../../data/processed/{DATASET}"

    # ── Load RNA
    rna = ad.read_h5ad(f"{BASE}_rna_sub.h5ad")   # (400, 500)
    print("RNA:", rna.shape)

    X = rna.X.toarray() if sp.issparse(rna.X) else rna.X   # (400, 500)
    expr_df = pd.DataFrame(
        X.T,                      # → (500 genes, 400 cells)
        index   = rna.var_names,
        columns = rna.obs_names,
    )

    # Run kScReNI
    networks = infer_kScReNI_sc_networks(
        expr_matrix = expr_df,
        n_features  = 500,    # only 500 HVGs in this subsampled data
        knn         = 20,
        nthread     = 8,
        n_trees     = 100,
    )

    # Output
    # networks is a dict: cell_name -> np.ndarray (500, 500)
    print(f"\nDone. {len(networks)} networks inferred.")
    first_key = list(networks.keys())[0]
    print(f"First cell : {first_key}")
    print(f"Network shape : {networks[first_key].shape}")