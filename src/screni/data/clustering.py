"""Degree-based network clustering and functional enrichment for ScReNI.

Matches the original R functions:

    ``calculate_scNetwork_degree()``  →  :func:`calculate_scnetwork_degree`
      (``Calculate_scNetwork_degree.R``)

    ``clustering_Kmeans()``           →  :func:`clustering_kmeans`
      (``clustering_Kmeans.R``)

    ``enrich_module()``               →  :func:`enrich_module`
      (``enrich_module.R``)

Pipeline position
-----------------
These three functions sit immediately after network inference and feed into
Duco's regulator-enrichment analysis (rows 26–27):

    ScReniNetworks
        → :func:`calculate_scnetwork_degree`  (degree matrices + ARI)
        → :func:`clustering_kmeans`            (gene → module assignments)
        → :func:`enrich_module`                (module → pathway terms)
        → ``network_analysis`` / ``Identify_enriched_scRegulators``

Typical usage
-------------
::

    from screni.data.clustering import (
        calculate_scnetwork_degree,
        clustering_kmeans,
        enrich_module,
    )

    # Degree analysis
    degree_results = calculate_scnetwork_degree(
        sc_networks={"CSN": csn_nets, "wScReNI": w_nets},
        top=nonzero_counts,          # list of per-cell top-N values
        cell_type_annotation=labels,
        ntype=4,
    )

    # K-means module clustering (on smoothed pseudo-time expression)
    kmeans_result = clustering_kmeans(expr_matrix, k=6)

    # Functional enrichment of each module
    enrichment_df = enrich_module(
        kmeans_result, organism="mmusculus", enrich_db="GO"
    )
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from sklearn.metrics import adjusted_rand_score

from screni.data.combine import ScReniNetworks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NetworkDegreeResult dataclass-style container
# ---------------------------------------------------------------------------


class NetworkDegreeResult:
    """Container for degree analysis results for one network type.

    Attributes
    ----------
    indegree : np.ndarray
        ``(n_genes, n_cells)`` matrix of in-degree values.
    outdegree : np.ndarray
        ``(n_genes, n_cells)`` matrix of out-degree values.
    gene_names : list[str]
        Ordered gene labels (row labels for degree matrices).
    cell_names : list[str]
        Ordered cell labels (column labels).
    in_degree_umap : anndata.AnnData
        AnnData after scanpy UMAP pipeline on in-degree matrix.
    out_degree_umap : anndata.AnnData
        AnnData after scanpy UMAP pipeline on out-degree matrix.
    in_degree_umap_ari : float
        Adjusted Rand Index for in-degree Leiden clusters vs ground truth.
    in_degree_hclust_ari : float
        Adjusted Rand Index for in-degree hierarchical clusters vs ground truth.
    out_degree_umap_ari : float
        Adjusted Rand Index for out-degree Leiden clusters vs ground truth.
    out_degree_hclust_ari : float
        Adjusted Rand Index for out-degree hierarchical clusters vs ground truth.
    """

    def __init__(
        self,
        indegree: np.ndarray,
        outdegree: np.ndarray,
        gene_names: list[str],
        cell_names: list[str],
        in_degree_umap: ad.AnnData,
        out_degree_umap: ad.AnnData,
        in_degree_umap_ari: float,
        in_degree_hclust_ari: float,
        out_degree_umap_ari: float,
        out_degree_hclust_ari: float,
    ) -> None:
        self.indegree = indegree
        self.outdegree = outdegree
        self.gene_names = gene_names
        self.cell_names = cell_names
        self.in_degree_umap = in_degree_umap
        self.out_degree_umap = out_degree_umap
        self.in_degree_umap_ari = in_degree_umap_ari
        self.in_degree_hclust_ari = in_degree_hclust_ari
        self.out_degree_umap_ari = out_degree_umap_ari
        self.out_degree_hclust_ari = out_degree_hclust_ari

    def __repr__(self) -> str:
        return (
            f"NetworkDegreeResult("
            f"genes={len(self.gene_names)}, cells={len(self.cell_names)}, "
            f"in_umap_ARI={self.in_degree_umap_ari:.3f}, "
            f"out_umap_ARI={self.out_degree_umap_ari:.3f})"
        )


# ---------------------------------------------------------------------------
# calculate_scnetwork_degree
# ---------------------------------------------------------------------------


def calculate_scnetwork_degree(
    sc_networks: dict[str, ScReniNetworks],
    top: list[int],
    cell_type_annotation: "Union[list[str], pd.Series, np.ndarray]",
    ntype: Optional[int] = None,
    n_pcs: int = 10,
    n_neighbor_pcs: int = 20,
    n_neighbors: int = 20,  # matches R FindNeighbors k.param=20
    leiden_resolution: float = 0.5,
    umap_random_state: int = 42,
) -> dict[str, NetworkDegreeResult]:
    """Compute per-cell in/out degree and evaluate clustering quality.

    Python equivalent of ``calculate_scNetwork_degree()``
    (``Calculate_scNetwork_degree.R``).

    For each network type the function:

    1. Binarises continuous weight matrices by keeping only the top-N edges
       per cell (where N = ``top[j]`` for cell ``j``).  CSN matrices are
       already binary so they are used as-is.
    2. Computes gene in-degree (``rowSums``) and out-degree (``colSums``) for
       each cell, producing two ``(n_genes, n_cells)`` matrices.
    3. Runs a scanpy pipeline on each degree matrix (normalise → scale →
       HVG → PCA → UMAP → Leiden clustering) to get UMAP-based cluster labels.
    4. Runs hierarchical clustering on the log-transformed degree matrix
       (``hclust(dist(cor(log(degree + 1))))``) and cuts the dendrogram at
       ``ntype`` clusters.
    5. Evaluates both clustering methods against ``cell_type_annotation`` using
       the Adjusted Rand Index (ARI).

    R → Python translation notes
    ----------------------------
    * ``CreateSeuratObject`` / ``NormalizeData`` / ``ScaleData`` →
      ``scanpy`` pipeline on an ``AnnData`` transposed so cells are rows.
    * ``FindVariableFeatures(nfeatures=4000)`` → ``sc.pp.highly_variable_genes``
      with ``n_top_genes=4000``.
    * ``RunPCA`` / ``RunUMAP`` / ``FindNeighbors`` / ``FindClusters`` →
      ``sc.tl.pca`` / ``sc.pp.neighbors`` / ``sc.tl.umap`` / ``sc.tl.leiden``.
    * ``flexclust::randIndex(..., correct=TRUE)`` → ``adjusted_rand_score``.
    * ``hclust(dist(cor(...)))`` → ``scipy.cluster.hierarchy.linkage`` on
      ``1 - np.corrcoef(...)`` distance matrix.

    Parameters
    ----------
    sc_networks
        Mapping of network-type name → :class:`~screni.data.combine.ScReniNetworks`.
        CSN matrices are expected to be binary; all others are binarised using
        the ``top`` values.
    top
        Per-cell top-N values: ``top[j]`` is the number of top edges to keep
        when binarising cell ``j``'s weight matrix.  Must have the same length
        as the number of cells in each network.
    cell_type_annotation
        True cell-type labels in the same order as the cells in ``sc_networks``.
    ntype
        Number of clusters for hierarchical cutting.  Defaults to the number of
        unique values in ``cell_type_annotation``.
    n_pcs
        Number of PCs to use for UMAP (``dims=1:10`` in R).  Default: 10.
    n_neighbor_pcs
        Number of PCs to use for the neighbour graph (``dims=1:20`` in R).
        Default: 20.
    n_neighbors
        Number of neighbours for the scanpy neighbour graph.  Default: 15.
    leiden_resolution
        Leiden clustering resolution.  Default: 0.5.
    umap_random_state
        Random seed for UMAP reproducibility.  Default: 42.

    Returns
    -------
    dict mapping network-type name → :class:`NetworkDegreeResult`.
    """
    true_labels = np.asarray(cell_type_annotation)
    if ntype is None:
        ntype = len(np.unique(true_labels))

    results: dict[str, NetworkDegreeResult] = {}

    for net_type, net in sc_networks.items():
        logger.info(f"calculate_scnetwork_degree: processing '{net_type}' ...")
        cells = list(net.keys())
        n_cells = len(cells)
        gene_names = list(net.gene_names or [])

        if not gene_names:
            # Fallback: infer from first matrix shape
            first_mat = next(iter(net.values()))
            gene_names = [f"G{i}" for i in range(first_mat.shape[0])]
        n_genes = len(gene_names)

        # ---- build binary weight matrices and compute degree ----
        indegree = np.zeros((n_genes, n_cells), dtype=float)
        outdegree = np.zeros((n_genes, n_cells), dtype=float)

        for j, cell in enumerate(cells):
            mat = np.asarray(net[cell], dtype=float).copy()

            if net_type != "CSN":
                # Binarise: keep top top[j] entries by weight
                cell_top = top[j] if j < len(top) else top[-1]
                flat = mat.ravel()
                if cell_top < len(flat):
                    threshold_idx = np.argpartition(-flat, cell_top)[cell_top]
                    threshold_val = np.sort(-flat)[cell_top - 1] * -1
                    binary = (mat >= threshold_val).astype(float)
                    # If tie at boundary, zero out to exact top count
                    if binary.sum() > cell_top:
                        idx = np.unravel_index(
                            np.argsort(mat.ravel())[::-1][:cell_top], mat.shape
                        )
                        binary = np.zeros_like(mat)
                        binary[idx] = 1.0
                else:
                    binary = (mat != 0).astype(float)
            else:
                binary = mat  # CSN already binary

            # R: indegree[, j] = rowSums(weights1)
            #    outdegree[, j] = colSums(weights1)
            indegree[:, j] = binary.sum(axis=1)   # rowSums
            outdegree[:, j] = binary.sum(axis=0)  # colSums

        # ---- UMAP + hierarchical clustering for in- and out-degree ----
        in_adata, in_umap_ari, in_hclust_ari = _degree_clustering(
            degree_mat=indegree,
            gene_names=gene_names,
            cell_names=cells,
            true_labels=true_labels,
            ntype=ntype,
            n_pcs=n_pcs,
            n_neighbor_pcs=n_neighbor_pcs,
            n_neighbors=n_neighbors,
            leiden_resolution=leiden_resolution,
            umap_random_state=umap_random_state,
            label=f"{net_type}/in-degree",
        )
        out_adata, out_umap_ari, out_hclust_ari = _degree_clustering(
            degree_mat=outdegree,
            gene_names=gene_names,
            cell_names=cells,
            true_labels=true_labels,
            ntype=ntype,
            n_pcs=n_pcs,
            n_neighbor_pcs=n_neighbor_pcs,
            n_neighbors=n_neighbors,
            leiden_resolution=leiden_resolution,
            umap_random_state=umap_random_state,
            label=f"{net_type}/out-degree",
        )

        results[net_type] = NetworkDegreeResult(
            indegree=indegree,
            outdegree=outdegree,
            gene_names=gene_names,
            cell_names=cells,
            in_degree_umap=in_adata,
            out_degree_umap=out_adata,
            in_degree_umap_ari=in_umap_ari,
            in_degree_hclust_ari=in_hclust_ari,
            out_degree_umap_ari=out_umap_ari,
            out_degree_hclust_ari=out_hclust_ari,
        )
        logger.info(
            f"  {net_type}: in_umap_ARI={in_umap_ari:.3f}, "
            f"in_hclust_ARI={in_hclust_ari:.3f}, "
            f"out_umap_ARI={out_umap_ari:.3f}, "
            f"out_hclust_ARI={out_hclust_ari:.3f}"
        )

    return results


def _degree_clustering(
    degree_mat: np.ndarray,
    gene_names: list[str],
    cell_names: list[str],
    true_labels: np.ndarray,
    ntype: int,
    n_pcs: int,
    n_neighbor_pcs: int,
    n_neighbors: int,
    leiden_resolution: float,
    umap_random_state: int,
    label: str,
) -> tuple[ad.AnnData, float, float]:
    """Run UMAP + hierarchical clustering on one degree matrix.

    Internal helper for :func:`calculate_scnetwork_degree`.

    The degree matrix is (n_genes × n_cells); Seurat/scanpy treat cells as
    observations, so we transpose to (n_cells × n_genes) before building the
    AnnData.

    Matches R:
        ``degree_umap <- CreateSeuratObject(counts = degree_data + 1)``
        followed by NormalizeData → ScaleData → FindVariableFeatures(4000)
        → PCA → UMAP → FindNeighbors → FindClusters.

    Returns
    -------
    ``(adata, umap_ari, hclust_ari)``
    """
    n_genes, n_cells = degree_mat.shape
    n_pcs_actual = max(1, min(n_pcs, n_cells - 1, n_genes - 1))
    # R uses up to 20 PCs for neighbour graph, 10 for UMAP
    n_nbr_pcs_actual = max(1, min(n_neighbor_pcs, n_cells - 1, n_genes - 1))
    n_hvg = min(4000, n_genes)

    # ---- Guard: too few cells to run PCA/UMAP ----
    _MIN_CELLS = max(n_pcs_actual + 2, 4)
    if n_cells < _MIN_CELLS:
        logger.warning(
            f"  [{label}] Only {n_cells} cell(s) — too few for PCA/UMAP. "
            f"Returning trivial ARI=NaN."
        )
        import math
        adata = ad.AnnData(X=(degree_mat + 1).T.astype(float))
        adata.obs_names = list(cell_names)
        adata.var_names = list(gene_names)
        adata.obs["leiden"] = ["0"] * n_cells
        adata.obs["hclust"] = ["0"] * n_cells
        return adata, float("nan"), float("nan")

    # ---- Seurat-equivalent pipeline ----
    # Transpose: cells × genes, add 1 (matching R's degree_data + 1)
    X = (degree_mat + 1).T.astype(float)  # (n_cells, n_genes)

    adata = ad.AnnData(X=X.copy())
    adata.obs_names = list(cell_names)
    adata.var_names = list(gene_names)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # R: FindVariableFeatures runs on NORMALIZED (log1p) data, before ScaleData.
    # Seurat VST internally uses raw counts, but the scanpy 'seurat' flavor
    # (dispersion-based) uses the log-normalized data — so HVG selection must
    # happen here, BEFORE sc.pp.scale, to match Seurat's pipeline order.
    if n_genes > n_hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat")
    # R's ScaleData() uses clip.max=10 by default — truncates scaled values
    # outside [-10, 10].  Pass max_value=10 to match.
    sc.pp.scale(adata, max_value=10.0)
    # Add tiny noise to prevent zero-variance features from crashing ARPACK
    rng = np.random.RandomState(umap_random_state)
    adata.X = adata.X + rng.normal(0, 1e-5, adata.X.shape)

    if n_genes > n_hvg:
        sc.tl.pca(adata, n_comps=n_nbr_pcs_actual, use_highly_variable=True)
    else:
        sc.tl.pca(adata, n_comps=n_nbr_pcs_actual)

    # R uses two separate dimensionality levels:
    #   FindNeighbors(dims=1:20) → 20 PCs for the Leiden SNN graph
    #   RunUMAP(dims=1:10)       → 10 PCs for the UMAP layout
    #
    # In scanpy, sc.pp.neighbors computes ONE graph used by both UMAP and
    # Leiden.  To replicate R's split, we compute two separate neighbor
    # graphs with different keys and route each downstream step to the
    # correct one.
    #
    # Leiden clustering: 20 PCs (FindNeighbors default dims=1:20)
    sc.pp.neighbors(
        adata,
        n_pcs=n_nbr_pcs_actual,
        n_neighbors=n_neighbors,
        key_added="neighbors_leiden",
    )
    # Use igraph backend to match recommended practice and silence FutureWarning
    # n_iterations=-1: run until convergence, matching R's Seurat FindClusters
    # which defaults to n.iter=10 (enough to converge for these dataset sizes).
    sc.tl.leiden(
        adata,
        resolution=leiden_resolution,
        key_added="leiden",
        flavor="igraph",
        n_iterations=-1,
        directed=False,
        neighbors_key="neighbors_leiden",
    )
    # UMAP: 10 PCs (RunUMAP(dims=1:10))
    sc.pp.neighbors(
        adata,
        n_pcs=n_pcs_actual,
        n_neighbors=n_neighbors,
        key_added="neighbors_umap",
    )
    sc.tl.umap(adata, random_state=umap_random_state, neighbors_key="neighbors_umap")

    umap_clusters = adata.obs["leiden"].values
    umap_ari = adjusted_rand_score(true_labels, umap_clusters)
    logger.debug(f"  [{label}] UMAP ARI = {umap_ari:.4f}")

    # ---- hierarchical clustering ----
    # ---- hierarchical clustering ----
    # R: degree_hclust <- dist(cor(log(degree_data + 1)))
    #    hc1 <- hclust(degree_hclust)
    #    Cluster_hclust <- cutree(hc1, k=ntype)
    #
    # cor() in R on a matrix (genes × cells) correlates COLUMNS (cells).
    # This gives a (n_cells × n_cells) correlation matrix C where C[i,j]
    # is the Pearson correlation between cell i's and cell j's log-degree
    # vectors (across all 500 genes).
    #
    # CRITICAL: R's dist() applied to a matrix computes EUCLIDEAN DISTANCES
    # between ROWS — NOT "1 - correlation".
    #   dist(C)[i,j] = sqrt(sum((C[i,:] - C[j,:])^2))
    # i.e. Euclidean distance between cell i's and cell j's correlation-
    # profile vectors (each vector has length n_cells).
    # This is provably different from 1 - C[i,j]:
    #   dist(C)[1,2] ≈ 1.82 vs (1-C)[1,2] ≈ 1.25 (typical values).
    # Using 1-cor produces wrong clusters and ARIs far from R's paper values.
    log_deg = np.log(degree_mat + 1)  # (n_genes, n_cells)
    # Correlation matrix between cells: (n_cells × n_cells)
    # Suppress divide-by-zero for constant-valued genes.
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(log_deg.T)
    corr = np.nan_to_num(corr, nan=0.0)
    corr = (corr + corr.T) / 2.0  # symmetrize floating-point rounding
    np.fill_diagonal(corr, 1.0)
    # R's dist(C): Euclidean distance between rows of C (= pdist on C)
    dist_vec = ssd.pdist(corr, metric="euclidean")
    linkage = sch.linkage(dist_vec, method="complete")
    hclust_labels = sch.cut_tree(linkage, n_clusters=ntype).ravel()
    hclust_ari = adjusted_rand_score(true_labels, hclust_labels)
    logger.debug(f"  [{label}] hclust ARI = {hclust_ari:.4f}")

    # Store hclust labels on the adata for downstream use
    adata.obs["hclust"] = hclust_labels.astype(str)

    return adata, umap_ari, hclust_ari


# ---------------------------------------------------------------------------
# clustering_kmeans
# ---------------------------------------------------------------------------


def clustering_kmeans(
    expr_matrix: "Union[np.ndarray, pd.DataFrame]",
    k: int = 1,
    column_group: "Optional[list]" = None,
    scale: str = "row",
    value_range: tuple = (-np.inf, np.inf),
    reorder: bool = True,
    rev_order: "Union[int, list[int]]" = -1,
    na_column: "Optional[list[int]]" = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """K-means clustering with row-scaling, hierarchical reordering, and clipping.

    Python equivalent of the anonymous ``clustering_Kmeans`` function
    (``clustering_Kmeans.R``), which is used in IReNA's ``get_Kmeans`` pipeline.

    Steps
    -----
    1. **Row-scale** the expression matrix (z-score each gene across cells) when
       ``scale='row'``.  NaN / Inf values are replaced with 0 after scaling,
       matching R's ``RNA1[is.na(RNA1)] = 0; RNA1[is.nan(RNA1)] = 0``.
    2. **K-means** clustering into ``k`` groups (skipped when ``k=1`` — all genes
       are assigned cluster 1, matching R's special case).
    3. **Hierarchical reordering** of genes within each cluster using
       ``1 - cor / 2`` distance (matching R's within-cluster hclust).
       Can optionally reverse the order for specified cluster indices
       (``rev_order`` parameter).
    4. **Outlier clipping** to the ``value_range`` bounds.
    5. Returns a DataFrame with a ``KmeansGroup`` column prepended and genes
       sorted by cluster, ready for downstream heatmap / enrichment use.

    Parameters
    ----------
    expr_matrix
        Gene × cell expression matrix (genes as rows, cells as columns).
        Can be a numpy array or a pandas DataFrame with gene names as index.
    k
        Number of k-means clusters.  When ``k=1`` all genes are assigned
        cluster 1 without running k-means (matching R's special case).
    column_group
        Optional grouping vector for columns.  When provided, a column-blank
        separator is inserted between groups in the returned matrix.  Set to
        ``None`` (default) to skip.
    scale
        ``'row'`` (default) to z-score each gene, or ``None`` / ``''`` to skip.
    value_range
        ``(min, max)`` clipping bounds applied after scaling.
        Defaults to ``(-inf, inf)`` (no clipping).
    reorder
        If ``True`` (default), apply hierarchical reordering within each cluster.
    rev_order
        Cluster index (1-based) or list of cluster indices whose within-cluster
        order should be reversed.  ``-1`` means no reversal (R default).
    na_column
        0-based column indices to exclude from k-means (but keep in output).
    random_state
        Random seed for k-means reproducibility.

    Returns
    -------
    DataFrame with:

    * ``KmeansGroup`` — integer cluster assignment (1-based, matching R).
    * One column per cell, containing the (scaled, clipped) expression values.
    * Genes sorted by cluster, then hierarchically within each cluster.
    * Index: gene names.

    Notes
    -----
    The gene names are preserved from the DataFrame index when ``expr_matrix``
    is a DataFrame, or generated as ``G0, G1, ...`` for numpy arrays.

    Examples
    --------
    >>> result = clustering_kmeans(smooth_expr, k=6)
    >>> result["KmeansGroup"].value_counts()
    """
    from sklearn.cluster import KMeans

    # ---- coerce to numpy + capture gene/cell names ----
    if isinstance(expr_matrix, pd.DataFrame):
        gene_names = list(expr_matrix.index)
        cell_names = list(expr_matrix.columns)
        mat = expr_matrix.to_numpy(dtype=float).copy()
    else:
        mat = np.asarray(expr_matrix, dtype=float).copy()
        gene_names = [f"G{i}" for i in range(mat.shape[0])]
        cell_names = [f"C{j}" for j in range(mat.shape[1])]

    n_genes, n_cells = mat.shape

    # ---- Step 1: row scaling ----
    if scale == "row":
        means = mat.mean(axis=1, keepdims=True)
        stds = mat.std(axis=1, ddof=1, keepdims=True)
        stds[stds == 0] = 1.0  # avoid division by zero for constant genes
        mat = (mat - means) / stds
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- Step 2: k-means ----
    if na_column is not None:
        cols_for_kmeans = [c for c in range(n_cells) if c not in na_column]
        mat_for_kmeans = mat[:, cols_for_kmeans]
    else:
        mat_for_kmeans = mat

    if k == 1:
        cluster_labels = np.ones(n_genes, dtype=int)
    else:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        cluster_labels = km.fit_predict(mat_for_kmeans) + 1  # 1-based to match R

    # ---- Step 3: hierarchical reordering within clusters ----
    if isinstance(rev_order, int):
        rev_order_set: set[int] = set() if rev_order == -1 else {rev_order}
    else:
        rev_order_set = set(rev_order)

    ordered_indices: list[int] = []

    for ci in range(1, k + 1):
        cluster_idx = np.where(cluster_labels == ci)[0]
        if len(cluster_idx) == 0:
            continue
        if reorder and len(cluster_idx) > 1:
            sub = mat_for_kmeans[cluster_idx]
            # R: Hier1 <- hclust(as.dist((1 - cor(t(RNA03[, 2:ncol])))/2))
            # cor(t(sub)) = correlation between genes → distance = (1 - corr) / 2
            # Suppress divide-by-zero from constant-valued genes.
            with np.errstate(invalid="ignore"):
                corr = np.corrcoef(sub)
            corr = np.nan_to_num(corr, nan=0.0)  # constant genes → NaN → 0
            corr = (corr + corr.T) / 2.0  # symmetrize floating-point rounding
            np.fill_diagonal(corr, 1.0)
            dist_vec = ssd.squareform(np.clip((1.0 - corr) / 2, 0, None))
            linkage = sch.linkage(dist_vec, method="complete")
            order = sch.leaves_list(linkage)
            if ci in rev_order_set:
                order = order[::-1]
            ordered_indices.extend(cluster_idx[order])
        else:
            ordered_indices.extend(cluster_idx)

    # ---- Step 4: outlier clipping ----
    mat_clipped = mat.copy()
    lo, hi = value_range
    if not np.isinf(lo):
        mat_clipped = np.clip(mat_clipped, lo, None)
    if not np.isinf(hi):
        mat_clipped = np.clip(mat_clipped, None, hi)

    # ---- Step 5: assemble output DataFrame ----
    sorted_genes = [gene_names[i] for i in ordered_indices]
    sorted_clusters = cluster_labels[ordered_indices]
    sorted_mat = mat_clipped[ordered_indices]

    df = pd.DataFrame(sorted_mat, index=sorted_genes, columns=cell_names)
    df.insert(0, "KmeansGroup", sorted_clusters)
    df.index.name = None

    logger.info(
        f"clustering_kmeans: k={k}, {n_genes} genes → "
        + ", ".join(
            f"cluster {ci}: {(sorted_clusters == ci).sum()}"
            for ci in range(1, k + 1)
        )
    )
    return df


# ---------------------------------------------------------------------------
# enrich_module
# ---------------------------------------------------------------------------


def enrich_module(
    kmeans_result: pd.DataFrame,
    organism: str,
    enrich_db: str = "GO",
    fun_num: int = 5,
    pvalue_cutoff: float = 0.05,
    gene_col: str = "Symbol",
) -> pd.DataFrame:
    """Run GO or KEGG enrichment for each k-means gene module.

    Python equivalent of ``enrich_module()`` (``enrich_module.R``).

    Uses the `gprofiler2 <https://pypi.org/project/gprofiler-official/>`_ API
    (``g:Profiler``) as a drop-in replacement for R's ``clusterProfiler``:

    * ``enrich_db='GO'``   → ``sources=["GO:BP"]`` (Biological Process)
    * ``enrich_db='KEGG'`` → ``sources=["KEGG"]``

    For each cluster, the top ``fun_num`` terms sorted by ``-log10(p_value)``
    are retained, matching R's ``-log10(q-value)`` sort.

    Parameters
    ----------
    kmeans_result
        DataFrame returned by :func:`clustering_kmeans`.  Must have a
        ``KmeansGroup`` column and a column containing gene symbols.
        The gene symbol column is specified by ``gene_col``.
    organism
        g:Profiler organism code, e.g. ``"mmusculus"`` (mouse) or
        ``"hsapiens"`` (human).  Equivalent to R's ``organism`` param.
    enrich_db
        ``"GO"`` (default, Biological Process) or ``"KEGG"``.
    fun_num
        Number of top enriched terms to keep per module (default 5).
    pvalue_cutoff
        Significance threshold for enrichment (default 0.05).
    gene_col
        Name of the column in ``kmeans_result`` holding gene symbols.
        Defaults to ``"Symbol"`` (as used in the IReNA pipeline).  If the
        column is absent, the DataFrame index is used instead.

    Returns
    -------
    DataFrame with one row per (module, term), columns:

    * ``module`` — cluster index (1-based integer)
    * ``term_id`` — pathway / GO term ID
    * ``term_name`` — human-readable term name
    * ``p_value`` — enrichment p-value
    * ``p_value_log10`` — ``-log10(p_value)``
    * ``intersection_size`` — number of query genes annotated to the term
    * ``term_size`` — total genes annotated to the term in the background
    * ``query_size`` — number of genes in the query

    Sorted descending by ``p_value_log10`` within each module (matching R's
    ``order(-acc2$'-log10(q-value)')``.

    Notes
    -----
    g:Profiler queries the EBI REST API and therefore requires internet access.
    Tests that call this function should either mock the API or be marked
    ``@pytest.mark.integration``.

    Examples
    --------
    >>> result = enrich_module(kmeans_df, organism="mmusculus", enrich_db="GO")
    >>> result.groupby("module")["term_name"].first()
    """
    from gprofiler import GProfiler

    gp = GProfiler(return_dataframe=True)

    # Determine enrichment sources
    if enrich_db == "GO":
        sources = ["GO:BP"]
    elif enrich_db == "KEGG":
        sources = ["KEGG"]
    else:
        raise ValueError(f"enrich_db must be 'GO' or 'KEGG', got {enrich_db!r}")

    # Sort by module (matching R's all_gene[order(all_gene$KmeansGroup),])
    df = kmeans_result.sort_values("KmeansGroup").copy()
    modules = sorted(df["KmeansGroup"].unique())

    all_results: list[pd.DataFrame] = []

    for module_id in modules:
        module_df = df[df["KmeansGroup"] == module_id]

        # Extract gene symbols
        if gene_col in module_df.columns:
            genes = module_df[gene_col].dropna().astype(str).tolist()
        else:
            genes = module_df.index.dropna().astype(str).tolist()

        if not genes:
            logger.warning(f"  enrich_module: module {module_id} has no genes, skipping")
            continue

        logger.info(
            f"  enrich_module: module {module_id}, {len(genes)} genes, "
            f"db={enrich_db}"
        )

        try:
            result = gp.profile(
                organism=organism,
                query=genes,
                sources=sources,
                significance_threshold_method="fdr",
                user_threshold=pvalue_cutoff,
            )
        except Exception as exc:
            logger.warning(f"  enrich_module: g:Profiler query failed for module {module_id}: {exc}")
            continue

        if result.empty:
            logger.info(f"  enrich_module: no significant terms for module {module_id}")
            continue

        # Compute -log10(p_value) and sort descending (matching R's q-value sort)
        result = result.copy()
        result["p_value_log10"] = -np.log10(result["p_value"].clip(lower=1e-300))
        result = result.sort_values("p_value_log10", ascending=False)

        # Keep top fun_num terms
        top_terms = result.head(fun_num).copy()
        top_terms["module"] = module_id

        # Select and rename columns to match R output structure
        keep_cols = {
            "native": "term_id",
            "name": "term_name",
            "p_value": "p_value",
            "p_value_log10": "p_value_log10",
            "intersection_size": "intersection_size",
            "term_size": "term_size",
            "query_size": "query_size",
            "module": "module",
        }
        available = {k: v for k, v in keep_cols.items() if k in top_terms.columns}
        top_terms = top_terms.rename(columns=available)[
            ["module"] + [available[k] for k in keep_cols if k in available and k != "module"]
        ]

        all_results.append(top_terms)

    if not all_results:
        logger.warning("enrich_module: no enrichment results found for any module")
        return pd.DataFrame(
            columns=["module", "term_id", "term_name", "p_value",
                     "p_value_log10", "intersection_size", "term_size", "query_size"]
        )

    combined = pd.concat(all_results, ignore_index=True)
    logger.info(
        f"enrich_module: {len(combined)} rows across {len(modules)} modules"
    )
    return combined
