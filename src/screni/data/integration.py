"""Phase 1: Multi-modal integration of paired scRNA-seq and scATAC-seq (PBMC).

Implements the paired integration pipeline from ScReNI's
``Integrate_scRNA_scATAC(..., data.type='paired')``, following the
exact steps from the R source code (lines 62-111 of Integrate_scRNA_scATAC.R).

Pipeline:
    1. RNA: normalize → HVG → scale → PCA
    2. ATAC: TF-IDF → LSI (keeping component 1 for WNN)
    3. Per-modality neighbor graphs
    4. WNN integration (muon, same algorithm as Seurat v4)
    5. Joint Leiden clustering + UMAP
    6. Extract WNN neighbor indices for downstream network inference

Key parameters from ScReNI tutorial:
    - IntegratedDimensions = 20 (PCA/LSI dims used in WNN)
    - KNN = 20
    - knn.range = 100
    - FindClusters: algorithm 3 (SLM), default resolution 0.8
"""

import logging

import anndata as ad
import muon as mu
import numpy as np
import scanpy as sc
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

# Default parameters matching the ScReNI R code
DEFAULT_N_PCS = 20  # IntegratedDimensions in the tutorial
DEFAULT_KNN = 20
DEFAULT_LEIDEN_RESOLUTION = 0.8


def integrate_paired(
    rna: ad.AnnData,
    atac: ad.AnnData,
    n_hvgs: int = 2000,
    n_pcs: int = DEFAULT_N_PCS,
    n_neighbors: int = DEFAULT_KNN,
    leiden_resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    batch_key: str | None = None,
) -> mu.MuData:
    """Integrate paired scRNA-seq and scATAC-seq via WNN.

    Parameters
    ----------
    rna
        RNA AnnData with raw counts and ``.obs['cell_type']``.
        Same cells (barcodes) as ``atac``.
    atac
        ATAC AnnData with raw peak counts. Same cells as ``rna``.
    n_hvgs
        Number of highly variable genes for RNA.
    n_pcs
        Number of PCA/LSI dimensions to use in WNN.
    n_neighbors
        Number of nearest neighbors for WNN.
    leiden_resolution
        Resolution for Leiden clustering on the WNN graph.
    batch_key
        Optional obs column for batch-aware HVG selection (e.g. 'donor_id'
        for multi-donor datasets). When set, ``highly_variable_genes``
        selects genes that are variable across batches rather than
        dominated by a single batch.

    Returns
    -------
    MuData with joint WNN graph, UMAP, clusters, and neighbor indices.
    Access RNA/ATAC via ``mdata['rna']`` / ``mdata['atac']``.
    WNN neighbor indices in ``mdata.uns['wnn_neighbor_indices']``.
    """
    logger.info("=== Phase 1: Paired Integration (WNN) ===")

    # ---- Step 1: RNA preprocessing ----
    # SCTransform → normalize_total + log1p (pragmatic Python equivalent)
    logger.info("  Step 1: RNA preprocessing...")
    rna = rna.copy()
    rna.layers["counts"] = rna.X.copy()
    # Filter genes expressed in very few cells (avoids LOESS singularity
    # in seurat_v3 VST, especially with batch_key on many small batches)
    sc.pp.filter_genes(rna, min_cells=3)
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    # seurat_v3 (VST) uses LOESS which can be singular with many batches.
    # Fall back to seurat (dispersion-based) when batch_key is set —
    # it's more robust and the HVG overlap is typically >90%.
    if batch_key is not None:
        hvg_kw = dict(n_top_genes=n_hvgs, flavor="seurat",
                      batch_key=batch_key)
    else:
        hvg_kw = dict(n_top_genes=n_hvgs, flavor="seurat_v3", layer="counts",
                      span=0.3)
    if batch_key is not None:
        hvg_kw["batch_key"] = batch_key
    sc.pp.highly_variable_genes(rna, **hvg_kw)
    sc.pp.scale(rna)
    sc.tl.pca(rna, n_comps=50)
    logger.info(f"  RNA: {rna.shape}, {n_hvgs} HVGs, PCA done (50 comps)")

    # ---- Step 2: ATAC preprocessing ----
    # TF-IDF → LSI (keep component 1 for WNN, drop for standalone UMAP)
    logger.info("  Step 2: ATAC preprocessing...")
    atac = atac.copy()
    atac.layers["counts"] = atac.X.copy()
    # Filter peaks with zero counts to avoid divide-by-zero in TF-IDF
    peak_counts = np.asarray(atac.X.sum(axis=0)).flatten()
    atac = atac[:, peak_counts > 0].copy()
    mu.atac.pp.tfidf(atac)
    mu.atac.tl.lsi(atac, n_comps=50)
    logger.info(f"  ATAC: {atac.shape}, TF-IDF + LSI done (50 comps)")

    # ---- Step 3: Per-modality neighbor graphs ----
    # L2-normalize embeddings before WNN (Seurat does this by default) so that
    # neither modality dominates the WNN weighting due to scale differences.
    logger.info(f"  Step 3: Per-modality neighbors (n_pcs={n_pcs})...")
    rna.obsm["X_pca"] = normalize(rna.obsm["X_pca"], norm="l2")
    atac.obsm["X_lsi"] = normalize(atac.obsm["X_lsi"], norm="l2")
    sc.pp.neighbors(rna, use_rep="X_pca", n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.pp.neighbors(atac, use_rep="X_lsi", n_neighbors=n_neighbors, n_pcs=n_pcs)

    # ---- Step 4: WNN integration ----
    logger.info("  Step 4: WNN integration...")
    mdata = mu.MuData({"rna": rna, "atac": atac})
    mu.pp.neighbors(mdata)
    # Muon stores use_rep as a dict (one per modality), which breaks scanpy's
    # UMAP that tries to hash it. Remove and provide a joint embedding instead
    # so scanpy can initialize UMAP from the combined PCA/LSI space.
    # Remove all dict-valued params (use_rep, n_pcs, etc.) that break scanpy
    mdata.uns["neighbors"]["params"] = {
        k: v for k, v in mdata.uns["neighbors"]["params"].items()
        if not isinstance(v, dict)
    }
    mdata.obsm["X_pca"] = np.concatenate(
        [rna.obsm["X_pca"][:, :n_pcs], atac.obsm["X_lsi"][:, :n_pcs]], axis=1
    )
    logger.info("  WNN graph computed")

    # ---- Step 5: Joint clustering + UMAP ----
    logger.info(f"  Step 5: Leiden clustering (resolution={leiden_resolution}) + UMAP...")
    sc.tl.leiden(mdata, resolution=leiden_resolution)
    sc.tl.umap(mdata)
    logger.info(f"  Clusters: {mdata.obs['leiden'].nunique()}")

    # ---- Step 6: Extract WNN neighbor indices ----
    # These are needed downstream for ScReNI network inference: each cell's
    # network is inferred from the cell + its WNN neighbors.
    logger.info("  Step 6: Extracting WNN neighbor indices...")
    nn_indices = _extract_wnn_neighbor_indices(mdata, n_neighbors)
    mdata.uns["wnn_neighbor_indices"] = nn_indices
    logger.info(f"  WNN neighbor indices: {nn_indices.shape}")

    return mdata


def _extract_wnn_neighbor_indices(
    mdata: mu.MuData,
    k: int,
) -> np.ndarray:
    """Extract k-nearest neighbor indices from the WNN graph.

    Returns an (n_cells, k) integer array where row i contains the
    indices of cell i's k nearest WNN neighbors.
    """
    # The WNN graph is stored as a sparse connectivities matrix
    conn_key = "wnn_connectivities"
    if conn_key in mdata.obsp:
        conn = mdata.obsp[conn_key]
    elif "connectivities" in mdata.obsp:
        conn = mdata.obsp["connectivities"]
    else:
        raise KeyError(
            f"Cannot find WNN connectivities in mdata.obsp. "
            f"Available keys: {list(mdata.obsp.keys())}"
        )

    n_cells = conn.shape[0]
    nn_indices = np.zeros((n_cells, k), dtype=int)

    for i in range(n_cells):
        row = conn[i].toarray().flatten()
        # Get indices of nonzero entries, sorted by descending weight
        nonzero_idx = np.where(row > 0)[0]
        weights = row[nonzero_idx]
        sorted_order = np.argsort(-weights)
        neighbors = nonzero_idx[sorted_order]
        # Take top k (pad with -1 if fewer than k)
        n_found = min(len(neighbors), k)
        nn_indices[i, :n_found] = neighbors[:n_found]
        if n_found < k:
            nn_indices[i, n_found:] = -1

    return nn_indices


if __name__ == "__main__":
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out_dir = Path("data/processed")

    rna = ad.read_h5ad(out_dir / "pbmc_rna.h5ad")
    atac = ad.read_h5ad(out_dir / "pbmc_atac.h5ad")
    logger.info(f"Loaded RNA: {rna.shape}, ATAC: {atac.shape}")

    mdata = integrate_paired(rna, atac)

    mdata.write(out_dir / "pbmc_integrated.h5mu")
    logger.info(f"Saved pbmc_integrated.h5mu")

    # ---- Validation plots ----
    import matplotlib.pyplot as plt
    import pandas as pd

    fig_dir = Path("output/integration")
    fig_dir.mkdir(parents=True, exist_ok=True)
    rna_mod = mdata.mod["rna"]
    atac_mod = mdata.mod["atac"]

    # 1. Standalone RNA UMAP
    if "X_umap" in rna_mod.obsm:
        fig, ax = plt.subplots(figsize=(7, 5))
        sc.pl.umap(rna_mod, color="cell_type", ax=ax, show=False, title="RNA UMAP (standalone)")
        fig.savefig(fig_dir / "pbmc_rna_umap.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {fig_dir / 'pbmc_rna_umap.png'}")

    # 2. Standalone ATAC UMAP (LSI dims 2:N, dropping component 1)
    atac_mod.obsm["X_lsi_no1"] = atac_mod.obsm["X_lsi"][:, 1:21]
    sc.pp.neighbors(atac_mod, use_rep="X_lsi_no1", key_added="lsi_no1")
    sc.tl.umap(atac_mod, neighbors_key="lsi_no1")
    fig, ax = plt.subplots(figsize=(7, 5))
    sc.pl.umap(atac_mod, color="cell_type", ax=ax, show=False, title="ATAC UMAP (LSI dims 2:N)")
    fig.savefig(fig_dir / "pbmc_atac_umap.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {fig_dir / 'pbmc_atac_umap.png'}")

    # 3. WNN UMAP (clusters + cell type)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc.pl.umap(mdata, color="leiden", ax=axes[0], show=False, title="Leiden clusters")
    sc.pl.umap(mdata, color="rna:cell_type", ax=axes[1], show=False, title="Cell type")
    fig.tight_layout()
    fig.savefig(fig_dir / "pbmc_wnn_umap.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {fig_dir / 'pbmc_wnn_umap.png'}")

    # 4. Modality weights per cell type
    obs = mdata.obs
    if "rna:mod_weight" in obs.columns:
        weights = obs.groupby("rna:cell_type", observed=True)[
            ["rna:mod_weight", "atac:mod_weight"]
        ].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        weights["rna:mod_weight"].sort_values().plot.barh(ax=ax, color="steelblue")
        ax.set_xlabel("Mean RNA weight")
        ax.set_title("RNA modality weight by cell type")
        ax.axvline(0.5, color="grey", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(fig_dir / "pbmc_modality_weights.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {fig_dir / 'pbmc_modality_weights.png'}")

    # 5. Cell type composition per cluster
    ct = obs["rna:cell_type"]
    leiden = obs["leiden"]
    crosstab = pd.crosstab(leiden, ct)
    proportions = crosstab.div(crosstab.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    proportions.plot.bar(stacked=True, ax=ax, legend=True)
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Leiden cluster")
    ax.set_title("Cell type composition per cluster")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "pbmc_cluster_composition.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {fig_dir / 'pbmc_cluster_composition.png'}")
