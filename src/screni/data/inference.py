"""Cell-specific regulatory network inference methods.

Translates four R functions from ScReNI:

    ``Infer_kScReNI_scNetworks()``  →  :func:`infer_kscreni_networks`
      (``Infer_kScReNI_scNetworks.R``) — Eduard

    ``Infer_wScReNI_scNetworks()``  →  :func:`infer_wscreni_networks`
      (``Infer_wScReNI_scNetworks.R``) — Eduard

    ``Infer_CSN_scNetworks()``      →  :func:`infer_csn_networks`
      (``Infer_CSN_scNetworks.R``) — Duco

    ``Infer_LIONESS_scNetworks()``  →  :func:`infer_lioness_networks`
      (``Infer_LIONESS_scNetworks.R``) — Duco

Pipeline position
-----------------
These functions sit at Step 4 of the ScReNI pipeline — they consume the
subsampled expression matrix (from Phase 2 feature selection) and produce
a dict of per-cell regulatory weight matrices fed into evaluation::

    expr_matrix (cells × genes)
        → infer_*_networks()
            → ScReniNetworks {cell_name: (n_genes × n_genes) weight matrix}
                → combine_wscreni_networks / calculate_network_precision_recall

Method overview
---------------
kScReNI
    RNA-only approach. Builds a Shared Nearest Neighbour (SNN) graph in
    HVG space (mirroring Seurat's FindNeighbors), then for each cell runs
    GENIE3 (random-forest feature importance) on the k+1 nearest cells in
    the raw expression matrix.

CSN (Cell-Specific Networks)
    Purely statistical. For each cell k, the expression of every other cell
    is classified as inside/outside a sliding box around cell k's values,
    then a z-score co-occurrence statistic is computed for every gene pair.
    No ML required; runs in pure NumPy.

LIONESS (Linear Interpolation to Obtain Network Estimates for Single Samples)
    Leave-one-out approach. Runs GENIE3 on the full dataset and on the
    leave-one-out sub-datasets, then linearly interpolates to estimate the
    contribution of each single cell.

GENIE3 implementation note
--------------------------
The original R GENIE3 package uses random forests where, for each target
gene j, all other genes are used as features to predict the expression of
gene j across the training cells.  The feature importances of those random
forests become the row entries in column j of the weight matrix.

Here we implement this with ``sklearn.ensemble.RandomForestRegressor``,
which produces identical results in expectation.  The ``seed`` parameter
maps to R's ``set.seed(100)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors

from screni.data.combine import ScReniNetworks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_input(
    expr: Union[np.ndarray, pd.DataFrame, ad.AnnData],
    gene_names: Optional[list[str]],
    cell_names: Optional[list[str]],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Resolve expression input to (raw_mat, gene_names, cell_names).

    raw_mat is always (n_cells, n_genes) float64 with dense layout.
    """
    if isinstance(expr, ad.AnnData):
        raw_mat = expr.X.toarray() if sp.issparse(expr.X) else np.asarray(expr.X)
        gnames = expr.var_names.tolist()
        cnames = expr.obs_names.tolist()
    elif isinstance(expr, pd.DataFrame):
        raw_mat = expr.to_numpy(dtype=np.float64)
        gnames = expr.columns.tolist()
        cnames = expr.index.tolist()
    else:
        raw_mat = np.asarray(expr, dtype=np.float64)
        if gene_names is None:
            raise ValueError("gene_names must be provided when expr is a numpy array")
        if cell_names is None:
            raise ValueError("cell_names must be provided when expr is a numpy array")
        gnames = list(gene_names)
        cnames = list(cell_names)

    return raw_mat.astype(np.float64), gnames, cnames


def _normalize_and_scale(raw_mat: np.ndarray) -> np.ndarray:
    """Log-normalise and z-score scale an expression matrix.

    Matches Seurat's NormalizeData (library-size normalisation to 10 000,
    then log1p) followed by ScaleData (zero-centred z-score per gene).

    Parameters
    ----------
    raw_mat
        (n_cells, n_genes) raw count matrix.

    Returns
    -------
    (n_cells, n_genes) scaled matrix — used for SNN computation only.
    """
    # NormalizeData: counts / library_size * 10_000, then log1p
    lib_sizes = raw_mat.sum(axis=1, keepdims=True)
    lib_sizes[lib_sizes == 0] = 1.0  # guard against empty cells
    norm_mat = np.log1p(raw_mat / lib_sizes * 1e4)

    # ScaleData: zero-centre and unit-variance per gene, then clip to [-10, 10].
    # R's ScaleData() has clip.max=10 by default, which truncates scaled values
    # exceeding 10 in absolute value.  On the retinal 500-gene dataset 161 of
    # 200 000 scaled values exceed 10 (max ≈ 14.6), causing 193/400 cells to
    # receive different kNN neighbours without clipping.
    gene_mean = norm_mat.mean(axis=0)
    gene_std = norm_mat.std(axis=0)
    gene_std[gene_std == 0] = 1.0  # guard against constant genes
    scaled_mat = (norm_mat - gene_mean) / gene_std
    # R default clip.max=10 (ScaleData argument); equivalent to np.clip(…, -10, 10)
    scaled_mat = np.clip(scaled_mat, -10.0, 10.0)

    return scaled_mat


def _select_hvgs(raw_mat: np.ndarray, gene_names: list[str], n_features: int) -> np.ndarray:
    """Return a boolean mask for the top n_features highly variable genes.

    Attempts Seurat v3 VST (matching R's ``FindVariableFeatures``).
    Falls back to dispersion-based selection when the LOESS fit fails —
    this can happen on very small matrices where the mean-variance curve
    has too few points for the SVD decomposition inside skmisc LOESS.

    Parameters
    ----------
    raw_mat
        (n_cells, n_genes) raw counts.
    gene_names
        Ordered gene labels.
    n_features
        Number of HVGs to select.

    Returns
    -------
    Boolean mask of shape (n_genes,).
    """
    n_genes = raw_mat.shape[1]
    # When n_features >= n_genes just select all genes (no selection needed)
    if n_features >= n_genes:
        logger.info(
            f"  _select_hvgs: n_features={n_features} >= n_genes={n_genes}, "
            f"selecting all genes"
        )
        return np.ones(n_genes, dtype=bool)

    tmp = ad.AnnData(X=raw_mat.astype(np.float32))
    tmp.var_names = gene_names

    try:
        sc.pp.highly_variable_genes(tmp, n_top_genes=n_features, flavor="seurat_v3")
        logger.debug("  HVG selection: seurat_v3 VST succeeded")
    except (ValueError, Exception) as exc:
        # LOESS can fail with "svddc failed in l2fit." on very small matrices.
        # Fall back to dispersion-based selection (seurat flavor), which uses
        # a binned normalised dispersion and is more robust for small datasets.
        logger.warning(
            f"  seurat_v3 HVG failed ({exc}); "
            f"falling back to dispersion-based selection (seurat flavor)"
        )
        tmp2 = ad.AnnData(X=raw_mat.astype(np.float32))
        tmp2.var_names = gene_names
        sc.pp.normalize_total(tmp2, target_sum=1e4)
        sc.pp.log1p(tmp2)
        sc.pp.highly_variable_genes(tmp2, n_top_genes=n_features, flavor="seurat")
        tmp.var["highly_variable"] = tmp2.var["highly_variable"]

    return tmp.var["highly_variable"].values.astype(bool)


def _compute_snn(scaled_hvg: np.ndarray, k: int) -> np.ndarray:
    """Compute the Shared Nearest Neighbour (SNN) Jaccard similarity matrix.

    Matches Seurat's ``FindNeighbors()`` when called with
    ``features = VariableFeatures(...)``.  The SNN weight between cells i
    and j is the Jaccard similarity of their k-nearest-neighbour sets
    (each of size k+1, including the query cell itself).

    Parameters
    ----------
    scaled_hvg
        (n_cells, n_hvg) scaled expression matrix in HVG space.
    k
        Number of nearest neighbours (k.param in R).

    Returns
    -------
    (n_cells, n_cells) float32 Jaccard similarity matrix.  Diagonal is 1.
    """
    n_cells = scaled_hvg.shape[0]

    # k+1 because sklearn includes the query cell as its own nearest neighbour
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="auto")
    nn.fit(scaled_hvg)
    # neighbors[i] = indices of the k+1 nearest cells of cell i (including i)
    neighbors = nn.kneighbors(return_distance=False)  # (n_cells, k+1)

    # Binary adjacency: adj[i, j] = 1 if j ∈ kNN(i)
    adj = np.zeros((n_cells, n_cells), dtype=np.float32)
    row_idx = np.repeat(np.arange(n_cells), k + 1)
    col_idx = neighbors.ravel()
    adj[row_idx, col_idx] = 1.0

    # Jaccard: SNN[i,j] = |N(i) ∩ N(j)| / |N(i) ∪ N(j)|
    #   intersection = (adj @ adj.T)[i,j]  (dot product of neighbour sets)
    #   union = (k+1) + (k+1) - intersection  (inclusion-exclusion)
    intersection = adj @ adj.T  # (n_cells, n_cells)
    union = 2.0 * (k + 1) - intersection
    snn = intersection / (union + 1e-12)
    np.fill_diagonal(snn, 1.0)  # each cell is perfectly similar to itself

    return snn


def _genie3_single_target(
    expr_neighborhood_T: np.ndarray,
    target_idx: int,
    n_trees: int,
    seed: int,
) -> np.ndarray:
    """Random forest for one target gene in a cell neighbourhood.

    Parameters
    ----------
    expr_neighborhood_T
        (n_samples, n_genes) — the k+1 neighbourhood cells × all genes,
        transposed from R's (n_genes, k+1) convention.
    target_idx
        Column index of the target gene in expr_neighborhood_T.
    n_trees
        Number of trees in the random forest.
    seed
        Random seed.

    Returns
    -------
    (n_genes,) importance vector.  Entry target_idx is 0 (a gene cannot
    be its own regulator).
    """
    n_genes = expr_neighborhood_T.shape[1]

    # All genes except the target are potential regulators (features)
    regulator_mask = np.ones(n_genes, dtype=bool)
    regulator_mask[target_idx] = False

    X = expr_neighborhood_T[:, regulator_mask]  # (n_samples, n_genes-1)
    y = expr_neighborhood_T[:, target_idx]       # (n_samples,)

    # ── R GENIE3 1.22.0 parameter equivalence ──────────────────────────────
    # R: GENIE3::GENIE3(snn.expr, nCores=1, nTrees=100)
    #    uses K="sqrt" (default) → mtry = round(sqrt(n_regulators))
    #    GENIE3 internally checks whether 'nodesize' was passed by the caller.
    #    kScReNI passes NO nodesize → GENIE3 takes the else-branch and explicitly
    #    calls randomForest(..., nodesize=1)  — "grow fully developed trees".
    #    See Infer_wScReNI_scNetworks.R lines 82-93 for the identical pattern
    #    (that code is a direct copy of GENIE3's internal helper).
    #
    # sklearn equivalents:
    #   max_features=mtry  (= round(sqrt(n_regulators))) matches R's K="sqrt"
    #   min_samples_leaf=1                               matches R's nodesize=1
    n_regulators = X.shape[1]  # n_genes - 1
    mtry = round(np.sqrt(n_regulators))  # R: K="sqrt", round(sqrt(p))

    rf = RandomForestRegressor(
        n_estimators=n_trees,
        max_features=mtry,        # R GENIE3: K="sqrt" → round(sqrt(n_regulators))
        min_samples_leaf=1,       # R GENIE3: nodesize=1 ("grow fully developed trees")
        random_state=seed,
        n_jobs=1,
    )
    rf.fit(X, y)

    importances = np.zeros(n_genes)
    importances[regulator_mask] = rf.feature_importances_
    return importances


def _genie3(
    neighborhood_expr: np.ndarray,
    n_trees: int = 100,
    seed: int = 100,
    n_jobs: int = 1,
) -> np.ndarray:
    """Run GENIE3 on a neighbourhood expression matrix.

    Matches ``GENIE3::GENIE3()`` from the R package.

    Parameters
    ----------
    neighborhood_expr
        (n_genes, n_cells_in_neighbourhood) matrix — R convention, rows=genes,
        columns=cells.  This is the direct output of
        ``exprMatrix[, neighbour_indices]`` in R.
    n_trees
        Number of trees per random forest.
    seed
        Random seed (R: set.seed(100)).
    n_jobs
        Parallel workers for the gene loop (joblib).

    Returns
    -------
    (n_genes, n_genes) weight matrix where ``result[i, j]`` is the
    importance of gene i as a regulator of gene j — identical to R's
    GENIE3 output orientation.
    """
    n_genes = neighborhood_expr.shape[0]

    # Transpose to (n_cells, n_genes) for sklearn
    expr_T = neighborhood_expr.T

    # Run one RF per target gene; parallelise over genes if n_jobs > 1
    cols = Parallel(n_jobs=n_jobs)(
        delayed(_genie3_single_target)(expr_T, j, n_trees, seed)
        for j in range(n_genes)
    )

    weights = np.column_stack(cols)  # (n_genes, n_genes)
    np.nan_to_num(weights, nan=0.0, copy=False)
    return weights


# ---------------------------------------------------------------------------
# kScReNI
# ---------------------------------------------------------------------------


def infer_kscreni_networks(
    expr: Union[np.ndarray, pd.DataFrame, ad.AnnData],
    *,
    n_features: int = 4000,
    k: int = 20,
    n_trees: int = 100,
    n_jobs: int = 1,
    seed: int = 100,
    gene_names: Optional[list[str]] = None,
    cell_names: Optional[list[str]] = None,
) -> ScReniNetworks:
    """Infer cell-specific regulatory networks using kScReNI.

    Matches ``Infer_kScReNI_scNetworks()`` from ``Infer_kScReNI_scNetworks.R``.

    Algorithm
    ---------
    1.  Normalise expression (library-size + log1p) and z-score scale per
        gene — used only for SNN computation.
    2.  Select the top ``n_features`` highly variable genes (Seurat v3 VST).
    3.  Compute the SNN Jaccard similarity graph in HVG space
        (matches Seurat's ``FindNeighbors(features=VariableFeatures(...)``).
    4.  For each cell i, select the k+1 cells with the highest SNN similarity
        (R: ``order(mat[i,], decreasing=T)[1:(knn+1)]``).
    5.  Extract those cells' columns from the **raw** expression matrix
        (R: ``snn.expr <- exprMatrix[, neighbour_indices]``).
    6.  Run GENIE3 on the (n_genes × k+1) neighbourhood matrix.
    7.  Replace NaN weights with 0 (R: ``ifelse(sub_res=='NaN', 0, sub_res)``).

    Important: GENIE3 at step 6 always runs on **all genes**, not just HVGs.
    HVGs are used solely to define the cell neighbourhood at steps 2-4.

    Parameters
    ----------
    expr
        Expression matrix.  Accepted formats:

        - ``AnnData``: cells × genes, raw counts in ``.X``
        - ``DataFrame``: cells × genes (index=cell names, columns=gene names)
        - ``ndarray``: cells × genes (provide ``gene_names`` and ``cell_names``)
    n_features
        HVGs used for SNN computation.  Default 4000 (matches R).
    k
        Number of nearest neighbours.  Default 20 (matches R's knn=20).
    n_trees
        Trees per random forest.  Default 100 (matches R's nTrees=100).
    n_jobs
        Parallel workers for the per-gene GENIE3 loop (joblib).  Default 1
        (sequential).  Cell-level parallelism (R's ``%dopar%``) can be added
        externally by chunking cells.
    seed
        Random seed.  Default 100 (matches R's ``set.seed(100)``).
    gene_names
        Gene labels when ``expr`` is a plain ndarray.
    cell_names
        Cell labels when ``expr`` is a plain ndarray.

    Returns
    -------
    ScReniNetworks
        ``{cell_name: (n_genes, n_genes) weight_matrix}`` dict.
        ``.gene_names`` contains ordered gene labels.
        ``.cell_names`` keys preserve the input cell order.

    Examples
    --------
    >>> networks = infer_kscreni_networks(rna_adata, k=20, n_trees=100)
    >>> weight_matrix = networks["AAACCTGAGAAACCAT-1"]  # shape (n_genes, n_genes)
    """
    raw_mat, _gene_names, _cell_names = _resolve_input(expr, gene_names, cell_names)
    n_cells, n_genes = raw_mat.shape

    logger.info(
        f"kScReNI: {n_cells} cells × {n_genes} genes "
        f"| n_features={n_features}, k={k}, n_trees={n_trees}, seed={seed}"
    )

    # Steps 1-2: normalise, scale, select HVGs
    logger.info("  Normalising and scaling...")
    scaled_mat = _normalize_and_scale(raw_mat)  # (n_cells, n_genes)

    logger.info(f"  Selecting {n_features} HVGs for SNN computation...")
    hvg_mask = _select_hvgs(raw_mat, _gene_names, n_features)
    n_hvg = hvg_mask.sum()
    logger.info(f"  {n_hvg} HVGs selected")

    # Step 3: SNN graph in HVG scaled space
    logger.info(f"  Building SNN graph (k={k}) in {n_hvg}-dim HVG space...")
    scaled_hvg = scaled_mat[:, hvg_mask]
    snn = _compute_snn(scaled_hvg, k=k)  # (n_cells, n_cells)

    # Steps 4-7: per-cell GENIE3
    # R stores exprMatrix as (genes × cells); we use raw_mat.T to match
    raw_mat_T = raw_mat.T  # (n_genes, n_cells) — R orientation

    logger.info(f"  Running GENIE3 for each of {n_cells} cells...")
    networks = ScReniNetworks(gene_names=_gene_names)

    for i, cell_name in enumerate(_cell_names):
        # Top k+1 cells by SNN similarity — R: order(mat[i,], decreasing=T)[1:(knn+1)]
        # R's order() is stable: ties broken by ascending cell index.
        # np.argsort default (quicksort) is unstable — must use kind='stable'.
        # With only 22 possible Jaccard values (integer/integer fractions), 330/400
        # cells have ties at the boundary, causing 303/400 cells to get wrong
        # neighbour sets with the unstable sort.
        neighbour_indices = np.argsort(-snn[i], kind="stable")[: k + 1]

        # snn.expr = exprMatrix[, neighbour_indices] — (n_genes, k+1) RAW counts
        neighbourhood = raw_mat_T[:, neighbour_indices]

        # GENIE3 → (n_genes, n_genes) weight matrix
        weights = _genie3(neighbourhood, n_trees=n_trees, seed=seed, n_jobs=n_jobs)
        # NaN → 0: R does ifelse(sub_res=='NaN', 0, sub_res)
        np.nan_to_num(weights, nan=0.0, copy=False)

        networks[cell_name] = weights

        if (i + 1) % 50 == 0 or i == 0 or i == n_cells - 1:
            logger.info(f"  [{i + 1}/{n_cells}] {cell_name}")

    logger.info(f"kScReNI complete: {n_cells} networks, shape {(n_genes, n_genes)}")
    return networks


# ---------------------------------------------------------------------------
# CSN
# ---------------------------------------------------------------------------


def infer_csn_networks(
    expr: Union[np.ndarray, pd.DataFrame, ad.AnnData],
    *,
    alpha: float = 0.01,
    boxsize: float = 0.1,
    weighted: bool = False,
    gene_names: Optional[list[str]] = None,
    cell_names: Optional[list[str]] = None,
) -> ScReniNetworks:
    """Infer cell-specific networks using the CSN method.

    Matches ``Infer_CSN_scNetworks()`` from ``Infer_CSN_scNetworks.R``.

    Algorithm (per cell k)
    ----------------------
    For each cell k, define a "box" around its expression value for each gene:

    1.  Sort each gene's expression across all cells.
    2.  Compute a half-box width h proportional to the fraction of non-zero
        cells (R: ``h = round(boxsize/2 * sum(sign(s1)))``).
    3.  Slide the box along the sorted values, assigning upper/lower bounds
        that group cells into windows of width 2h.
    4.  Build B: B[g, j] = 1 if cell j's expression of gene g falls in cell
        k's box for gene g.
    5.  Compute a z-score co-occurrence statistic for every gene pair
        (hypergeometric-style):
        ``d = (B @ B.T * m - a @ a.T) / sqrt(a @ a.T * (m-a) @ (m-a).T / (m-1))``
    6.  If ``weighted=True``: retain positive z-scores (weighted edges).
        If ``weighted=False`` (default): threshold at the 1-alpha normal quantile
        (binary adjacency matrix).

    Parameters
    ----------
    expr
        Expression matrix.  Same formats as :func:`infer_kscreni_networks`.
        R convention: rows=genes, columns=cells.
        Python/AnnData convention: rows=cells, columns=genes.
    alpha
        Significance threshold for the z-score test.  Default 0.01 (R default).
    boxsize
        Fraction of non-zero cells used as the box half-width.
        Default 0.1 (R default).
    weighted
        If ``True``, return z-scores clipped to positives (``weighted=1`` in R).
        If ``False`` (default), return a binary adjacency matrix.
    gene_names
        Gene labels when ``expr`` is a plain ndarray.
    cell_names
        Cell labels when ``expr`` is a plain ndarray.

    Returns
    -------
    ScReniNetworks
        ``{cell_name: (n_genes, n_genes) matrix}`` — either binary adjacency
        (weighted=False) or positive z-scores (weighted=True).

    Notes
    -----
    The inner loop over cells is O(n_cells²) in memory for the ``B @ B.T``
    products.  For large datasets (>500 cells), consider chunking.

    Examples
    --------
    >>> csn = infer_csn_networks(rna_adata, alpha=0.01, boxsize=0.1)
    >>> adjacency = csn["AAACCTGAGAAACCAT-1"]  # binary (n_genes × n_genes)
    """
    raw_mat, _gene_names, _cell_names = _resolve_input(expr, gene_names, cell_names)
    n_cells, n_genes = raw_mat.shape

    logger.info(
        f"CSN: {n_cells} cells × {n_genes} genes "
        f"| alpha={alpha}, boxsize={boxsize}, weighted={weighted}"
    )

    # R orientation: data is (n_genes × n_cells); n=nrow=genes, m=ncol=cells
    # We keep internal variable names matching R for clarity.
    data = raw_mat.T  # (n_genes, n_cells)
    n = n_genes  # rows in R
    m = n_cells  # columns in R

    # ------------------------------------------------------------------
    # Pre-compute upper and lower bounds for every gene across all cells
    # R: for each gene i, slide a box along the sorted expression values.
    # upper[i, j] and lower[i, j] give the box bounds for cell j, gene i.
    # ------------------------------------------------------------------
    upper = np.zeros((n, m), dtype=np.float64)
    lower = np.zeros((n, m), dtype=np.float64)

    for i in range(n):
        # R: sorted <- sort(data[i,]); s1=sorted values, s2=sort order (1-based)
        s2 = np.argsort(data[i])          # s2[rank] = cell index (0-based)
        s1 = data[i, s2]                  # sorted expression values

        # R: n3 = m - sum(sign(s1))  (number of zero-expression cells)
        n3 = m - int(np.sign(s1).sum())

        # R: h = round(boxsize/2 * sum(sign(s1)))
        h = int(np.round(boxsize / 2.0 * np.sign(s1).sum()))
        h = max(h, 0)

        k_pos = 0
        while k_pos < m:
            # Find run of equal values starting at k_pos
            s = 0
            while k_pos + s + 1 < m and s1[k_pos + s + 1] == s1[k_pos]:
                s += 1
            # Indices of cells in this tied block (0-based)
            block = s2[k_pos : k_pos + s + 1]

            if s >= h:
                # Tie block wider than box half-width: box collapses to the tie value
                upper[i, block] = data[i, s2[k_pos]]
                lower[i, block] = data[i, s2[k_pos]]
            else:
                # R: upper[i, block] = data[i, s2[min(m, k_pos+s+h)]]  (1-based → clip at m)
                # R: lower[i, block] = data[i, s2[max(n3*(n3>h)+1, k_pos-h)]]  (1-based → clip at 1)
                # Python: adjust to 0-based indexing
                upper_rank = min(m - 1, k_pos + s + h)
                # R: max(n3*(n3>h)+1, k_pos-h) is 1-based; n3>h is logical (0/1)
                # → 0-based: max(n3*(n3>h), k_pos-h)
                lower_rank = max(n3 * int(n3 > h), k_pos - h)
                upper[i, block] = data[i, s2[upper_rank]]
                lower[i, block] = data[i, s2[lower_rank]]

            k_pos += s + 1

    # ------------------------------------------------------------------
    # For each cell k, build B and compute the z-score co-occurrence matrix
    # ------------------------------------------------------------------
    # Normal quantile for threshold (R: p <- -qnorm(alpha, 0, 1))
    from scipy.stats import norm

    p_threshold = norm.ppf(1.0 - alpha)  # -qnorm(alpha) = qnorm(1-alpha)

    networks = ScReniNetworks(gene_names=_gene_names)
    B = np.zeros((n, m), dtype=np.float64)

    for k_cell in range(m):
        # B[g, j] = 1 if data[g, j] is in [lower[g, k_cell], upper[g, k_cell]]
        # R inner loop: for j in 1:m: B[,j] <- data[,j] <= upper[,k] & >= lower[,k]
        B = (data <= upper[:, k_cell : k_cell + 1]) & (
            data >= lower[:, k_cell : k_cell + 1]
        )
        B = B.astype(np.float64)

        # a[g] = number of cells whose gene g expression is in the box for cell k
        a = B.sum(axis=1)  # (n_genes,)

        # Co-occurrence: (B @ B.T)[g1,g2] = cells where BOTH genes are in box
        BtB = B @ B.T  # (n_genes, n_genes)

        # Outer products for the z-score denominator
        a_outer = np.outer(a, a)           # a @ a.T
        ma_outer = np.outer(m - a, m - a) # (m-a) @ (m-a).T

        # Z-score: d = (B@B.T * m - a@a.T) / sqrt(a@a.T * (m-a)@(m-a).T / (m-1) + eps)
        numerator = BtB * m - a_outer
        denominator = np.sqrt(
            a_outer * ma_outer / (m - 1) + np.finfo(np.float64).tiny
        )
        d = numerator / denominator
        np.fill_diagonal(d, 0.0)

        if weighted:
            # weighted=1 in R: retain positive z-scores
            cell_net = d * (d > 0)
        else:
            # weighted=0 in R: binary threshold at p
            cell_net = (d > p_threshold).astype(np.float64)

        networks[_cell_names[k_cell]] = cell_net

        if (k_cell + 1) % 50 == 0 or k_cell == 0 or k_cell == m - 1:
            logger.info(f"  [{k_cell + 1}/{m}] Cell {_cell_names[k_cell]} done")

    logger.info(f"CSN complete: {m} networks, shape {(n_genes, n_genes)}")
    return networks


# ---------------------------------------------------------------------------
# LIONESS
# ---------------------------------------------------------------------------


def infer_lioness_networks(
    expr: Union[np.ndarray, pd.DataFrame, ad.AnnData],
    *,
    n_trees: int = 100,
    n_jobs: int = 1,
    seed: int = 100,
    gene_names: Optional[list[str]] = None,
    cell_names: Optional[list[str]] = None,
) -> ScReniNetworks:
    """Infer cell-specific networks using LIONESS + GENIE3.

    Matches ``Infer_LIONESS_scNetworks()`` from ``Infer_LIONESS_scNetworks.R``.

    Algorithm
    ---------
    LIONESS (Kuijjer et al. 2019) estimates a single-cell network by linearly
    interpolating between a global network and a leave-one-out network:

    1.  Run GENIE3 on the full expression matrix → global network G.
    2.  For each cell i:
        a. Run GENIE3 on the matrix **excluding** cell i → leave-one-out network L_i.
        b. Cell-specific network S_i = n_cells * (G - L_i) + L_i.

    The intuition: if removing cell i changes the global network a lot, cell i
    has a large cell-specific contribution.

    Note on R's ``genie3_res`` reordering
    --------------------------------------
    The R code performs::

        genie3_res <- rbind(genie3_res[nrow(genie3_res), ], genie3_res[-nrow(genie3_res), ])
        rownames(genie3_res) <- colnames(genie3_res)

    This shifts the last row to the first position so that row and column
    names align (GENIE3 in R omits the last gene from the regulator set by
    default, making the output non-square without this fix).  In our
    implementation, ``_genie3`` produces a fully square matrix with the same
    gene ordering, so this reordering is **not needed**.

    Parameters
    ----------
    expr
        Expression matrix.  Same formats as :func:`infer_kscreni_networks`.
    n_trees
        Trees per random forest.  Default 100 (matches R's nTrees=100).
    n_jobs
        Parallel workers for the per-gene GENIE3 loop (joblib).  Default 1.
    seed
        Random seed.  Default 100 (matches R's ``set.seed(100)``).
    gene_names
        Gene labels when ``expr`` is a plain ndarray.
    cell_names
        Cell labels when ``expr`` is a plain ndarray.

    Returns
    -------
    ScReniNetworks
        ``{cell_name: (n_genes, n_genes) weight_matrix}`` LIONESS networks.

    Warning
    -------
    LIONESS is O(n_cells) GENIE3 runs — each itself O(n_genes² × n_cells).
    For large datasets this is expensive.  Use small cell counts or a subset
    for benchmarking.

    Examples
    --------
    >>> networks = infer_lioness_networks(rna_adata, n_trees=100)
    >>> sc_network = networks["AAACCTGAGAAACCAT-1"]  # (n_genes, n_genes)
    """
    raw_mat, _gene_names, _cell_names = _resolve_input(expr, gene_names, cell_names)
    n_cells, n_genes = raw_mat.shape

    logger.info(
        f"LIONESS: {n_cells} cells × {n_genes} genes "
        f"| n_trees={n_trees}, seed={seed}"
    )
    logger.warning(
        f"LIONESS will run {n_cells + 1} GENIE3 passes — "
        f"this is O(n_cells) and can be slow for large datasets."
    )

    # R orientation: exprMatrix is (genes × cells)
    expr_R = raw_mat.T  # (n_genes, n_cells)

    # Step 1: global GENIE3 on full matrix
    logger.info("  Running global GENIE3 on full matrix...")
    global_net = _genie3(expr_R, n_trees=n_trees, seed=seed, n_jobs=n_jobs)
    # shape: (n_genes, n_genes), global_net[i, j] = importance of gene i for gene j

    # Note on R's row reordering:
    # R GENIE3 returns an n_genes×n_genes matrix where the row order is
    # [all regulators...] which in R's package omits the last gene as target
    # then restores ordering via rbind(last_row, all_other_rows).
    # Our _genie3 produces a proper square matrix with consistent ordering,
    # so no reordering is needed.

    # Step 2: leave-one-out GENIE3 + LIONESS interpolation
    networks = ScReniNetworks(gene_names=_gene_names)

    for i, cell_name in enumerate(_cell_names):
        logger.info(f"  [{i + 1}/{n_cells}] Leave-one-out GENIE3 for {cell_name}...")

        # Leave-one-out matrix: all cells except cell i
        # R: exprMatrix[, -i]  →  (n_genes, n_cells-1)
        loo_indices = [j for j in range(n_cells) if j != i]
        loo_mat = expr_R[:, loo_indices]  # (n_genes, n_cells-1)

        loo_net = _genie3(loo_mat, n_trees=n_trees, seed=seed, n_jobs=n_jobs)

        # LIONESS formula: S_i = n_cells * (G - L_i) + L_i
        # R: sc_res <- as.matrix(ncell * (genie3_res - sub_res) + sub_res)
        sc_net = n_cells * (global_net - loo_net) + loo_net

        networks[cell_name] = sc_net

    logger.info(f"LIONESS complete: {n_cells} networks, shape {(n_genes, n_genes)}")
    return networks


# ---------------------------------------------------------------------------
# wScReNI — Gene–peak–TF label object
# ---------------------------------------------------------------------------


@dataclass
class GenePeakOverlapLabs:
    """Gene–peak–TF label container; Python equivalent of the R S4 object.

    The R function ``peak_gene_TF_labs()`` (in ``wScReNI_affiliated_functions.R``)
    builds an S4 ``"information match"`` object with four parallel character
    vectors.  This dataclass mirrors those four slots and pre-builds lookup
    dicts so the inner per-gene loop in :func:`_gene_peak_random_forest` can
    query associations in O(1) instead of scanning the full arrays each time.

    Each of the four lists has the same length — one entry per row of the
    ``peak_gene_TF`` data frame produced by ``peak_gene_TF_match()``.

    Attributes
    ----------
    labels : list[str]
        ``'TF'`` or ``'target'`` for each entry.  A gene that appears as a TF
        anywhere in the ``peak_gene_TF`` table receives the ``'TF'`` label.
    genes : list[str]
        Gene name for each entry — the gene whose expression drives the peak.
    peaks : list[str]
        Peak name for each entry — the ATAC peak linked to the gene.
    tfs : list[str]
        Semicolon-separated TF names that regulate the gene via this peak.
        Mirrors the ``TF`` column of ``peak_gene_TF``.

    Examples
    --------
    Build from a ``peak_gene_TF`` DataFrame (mirrors R's ``peak_gene_TF_labs``):

    >>> labs = GenePeakOverlapLabs.from_dataframe(peak_gene_tf_df)
    >>> labs.peaks_for_gene("Nrl")         # list of associated peak names
    >>> labs.tfs_for_gene("Rho")           # set of TF names regulating Rho
    >>> labs.label_for_gene("Nrl")         # 'TF' or 'target'
    """

    labels: list[str]
    genes: list[str]
    peaks: list[str]
    tfs: list[str]

    def __post_init__(self) -> None:
        # Precompute O(1) lookup dicts keyed by gene name.
        from collections import defaultdict

        self._gene_to_peaks: dict[str, list[str]] = defaultdict(list)
        self._gene_to_labels: dict[str, list[str]] = defaultdict(list)
        self._gene_to_tfs: dict[str, set[str]] = defaultdict(set)

        for gene, peak, label, tf_str in zip(
            self.genes, self.peaks, self.labels, self.tfs
        ):
            self._gene_to_peaks[gene].append(peak)
            self._gene_to_labels[gene].append(label)
            for tf in tf_str.split(";"):
                if tf:
                    self._gene_to_tfs[gene].add(tf)

    # ------------------------------------------------------------------
    # Convenience accessors (mirroring R's @ slot-subsetting idiom)
    # ------------------------------------------------------------------

    def peaks_for_gene(self, gene: str) -> list[str]:
        """All peak names associated with *gene* (R: ``@peaks[@genes == gene]``)."""
        return self._gene_to_peaks.get(gene, [])

    def labels_for_gene(self, gene: str) -> list[str]:
        """All label entries for *gene* (R: ``@labels[@genes == gene]``)."""
        return self._gene_to_labels.get(gene, [])

    def label_for_gene(self, gene: str) -> Optional[str]:
        """Unique label for *gene*, or ``None`` if gene is absent.

        Mirrors R's ``unique(gene_peak_overlap_labs@labels[@genes == gene])``.
        Returns ``None`` when the gene has no entries (R: ``length(...) == 0``).
        """
        lbls = self._gene_to_labels.get(gene)
        if not lbls:
            return None
        unique_lbls = set(lbls)
        return next(iter(unique_lbls))  # typically a single value

    def tfs_for_gene(self, gene: str) -> set[str]:
        """Set of TF names that regulate *gene* (R: ``@TFs[@genes == gene]``, split by ``;``)."""
        return self._gene_to_tfs.get(gene, set())

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "GenePeakOverlapLabs":
        """Build from a ``peak_gene_TF`` DataFrame.

        Expects columns ``gene.name``, ``peak.name``, ``TF``
        (matching the output of ``peak_gene_TF_match()`` in R).
        Labels are assigned as ``'TF'`` for any gene that appears in the
        TF column (after splitting on ``;``), otherwise ``'target'``.

        Parameters
        ----------
        df
            DataFrame with columns ``gene.name``, ``peak.name``, ``TF``.

        Returns
        -------
        GenePeakOverlapLabs
        """
        all_tfs: set[str] = set()
        for tf_str in df["TF"].dropna():
            all_tfs.update(str(tf_str).split(";"))
        all_tfs.discard("")

        gene_col = df["gene.name"].tolist()
        peak_col = df["peak.name"].tolist()
        tf_col = df["TF"].fillna("").astype(str).tolist()
        label_col = [
            "TF" if g in all_tfs else "target" for g in gene_col
        ]

        return cls(labels=label_col, genes=gene_col, peaks=peak_col, tfs=tf_col)


# ---------------------------------------------------------------------------
# wScReNI — core random-forest helper
# ---------------------------------------------------------------------------


def _gene_peak_random_forest(
    expr_mat: np.ndarray,
    peak_mat: np.ndarray,
    labs: "GenePeakOverlapLabs",
    gene_names: list[str],
    peak_names: list[str],
    *,
    k: Union[str, int] = "sqrt",
    n_trees: int = 100,
    n_jobs: int = 1,
    importance_measure: str = "IncNodePurity",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Compute gene-regulatory weights using random forest feature importance.

    Python translation of the nested ``gene_peak_randomForest()`` function
    inside ``Infer_wScReNI_scNetworks.R``.

    For each target gene *j*, a ``RandomForestRegressor`` is trained on the
    combined feature matrix ``[other_genes | all_peaks]`` to predict the
    expression of gene *j* across the *k+1* neighbourhood cells.  The feature
    importances are then collapsed into a gene-level regulatory-weight vector
    using the gene–peak–TF linkage encoded in *labs*.

    Weight assignment rule (mirrors R implementation)
    --------------------------------------------------
    Let ``y_peaks`` = importances of peaks linked to the **target** gene,
    ``x_peaks_j`` = importances of peaks linked to **gene j** (regulator),
    ``direct_j`` = direct feature importance of gene j.

    - If gene j is a TF for the target:
      ``weight[j] = direct_j + y_peaks + x_peaks_j``
    - Otherwise (non-TF regulator):
      ``weight[j] = direct_j + x_peaks_j``
    - Self-weight (target → target):
      - If target is a TF: ``weight[target] = y_peaks``
      - Otherwise: ``weight[target] = 0``
    - If ``sum(y) == 0``: all weights are 0 (gene not expressed in neighbourhood).

    The returned matrix is divided by ``n_samples`` (number of cells in
    neighbourhood) to match R's final normalisation step.

    Parameters
    ----------
    expr_mat
        Expression neighbourhood — shape ``(n_cells, n_genes)`` with rows as
        cells and columns as genes.  Mirrors R's transposed ``exprMatrix``.
    peak_mat
        Peak-accessibility neighbourhood — shape ``(n_cells, n_peaks)``.
        Mirrors R's transposed ``gene_peak_overlap_matrix``.
    labs
        Gene–peak–TF label object (:class:`GenePeakOverlapLabs`).
    gene_names
        Ordered gene names (length ``n_genes``).
    peak_names
        Ordered peak names (length ``n_peaks``).
    k
        ``mtry`` strategy — ``'sqrt'`` (default), ``'all'``, or an integer.
        Maps to R's ``K`` parameter.
    n_trees
        Number of trees in each random forest.  Default 100.
    n_jobs
        Joblib parallel workers for the per-gene loop.  Default 1.
    importance_measure
        ``'IncNodePurity'`` (default, impurity-based; ``RandomForest
        Regressor.feature_importances_``) or ``'%IncMSE'`` (permutation
        importance via ``sklearn.inspection.permutation_importance``).
    seed
        Random seed passed to every ``RandomForestRegressor``.

    Returns
    -------
    weights : ndarray of shape (n_genes, n_genes)
        ``weights[i, j]`` = regulatory weight of gene *i* on target gene *j*.
        Follows R's ``cbind`` convention: each per-target column is stacked.
        Values are divided by the number of neighbourhood cells.
    """
    if importance_measure not in ("IncNodePurity", "%IncMSE"):
        raise ValueError(
            "importance_measure must be 'IncNodePurity' or '%IncMSE', "
            f"got {importance_measure!r}"
        )

    n_samples, n_genes = expr_mat.shape
    n_peaks = peak_mat.shape[1]

    # Pre-build gene-name → column index map for fast lookup
    gene_idx: dict[str, int] = {g: i for i, g in enumerate(gene_names)}
    peak_idx: dict[str, int] = {p: i for i, p in enumerate(peak_names)}

    # Precompute peak-index sets per gene to avoid repeated dict lookups
    target_peak_sets: dict[str, set[int]] = {
        g: {peak_idx[p] for p in labs.peaks_for_gene(g) if p in peak_idx}
        for g in gene_names
    }

    def _process_one_target(t_idx: int) -> np.ndarray:
        """Return the weight column for target gene at index *t_idx*."""
        target_name = gene_names[t_idx]
        y = expr_mat[:, t_idx].copy().astype(np.float64)

        weight_col = np.zeros(n_genes, dtype=np.float64)

        if y.sum() == 0.0:
            return weight_col

        # Build feature matrix: [other genes | all peaks]
        other_mask = np.ones(n_genes, dtype=bool)
        other_mask[t_idx] = False
        other_genes_idx = np.where(other_mask)[0]          # (n_genes-1,)
        X_genes = expr_mat[:, other_genes_idx]              # (n_samples, n_genes-1)
        X = np.hstack([X_genes, peak_mat])                  # (n_samples, n_genes-1+n_peaks)
        n_input_vars = X.shape[1]

        # Determine mtry
        if isinstance(k, int):
            mtry = k
        elif k == "sqrt":
            mtry = max(1, round(np.sqrt(n_input_vars)))
        elif k == "all":
            mtry = n_input_vars
        else:
            raise ValueError(f"k must be 'sqrt', 'all', or an int, got {k!r}")

        if importance_measure == "IncNodePurity":
            y_fit = y / (y.std() or 1.0)   # R: y <- y / sd(y); guard div-by-0
            rf = RandomForestRegressor(
                n_estimators=n_trees,
                max_features=mtry,
                min_samples_leaf=1,
                random_state=seed,
                n_jobs=1,  # outer joblib handles cell-level parallelism
            )
            rf.fit(X, y_fit)
            importances = rf.feature_importances_   # (n_genes-1+n_peaks,)
        else:
            # %IncMSE: permutation importance
            from sklearn.inspection import permutation_importance as _pi

            rf = RandomForestRegressor(
                n_estimators=n_trees,
                max_features=mtry,
                min_samples_leaf=1,
                random_state=seed,
                n_jobs=1,
            )
            rf.fit(X, y)
            perm = _pi(rf, X, y, n_repeats=5, random_state=seed, n_jobs=1)
            importances = np.maximum(perm.importances_mean, 0.0)

        # importances layout: [gene_0, ..., gene_{t-1}, gene_{t+1}, ..., gene_{G-1},
        #                       peak_0, ..., peak_{P-1}]
        #   — first (n_genes-1) entries correspond to other_genes_idx
        #   — last  n_peaks    entries correspond to peak order

        # Importance of peaks for the target gene
        t_peak_idxs = target_peak_sets.get(target_name, set())
        y_peak_coef = sum(
            importances[n_genes - 1 + pi] for pi in t_peak_idxs
        )

        # TFs that regulate the target gene (for weight boosting)
        target_tfs: set[str] = labs.tfs_for_gene(target_name)

        # Importance of peaks for each regulator gene j; direct gene importance
        for col_pos, g_idx in enumerate(other_genes_idx):
            gene_name_j = gene_names[g_idx]

            direct_j = float(importances[col_pos])

            j_peak_idxs = target_peak_sets.get(gene_name_j, set())
            peak_j_coef = sum(
                importances[n_genes - 1 + pi] for pi in j_peak_idxs
            )

            if gene_name_j in target_tfs:
                weight_col[g_idx] = direct_j + y_peak_coef + peak_j_coef
            else:
                weight_col[g_idx] = direct_j + peak_j_coef

        # Self-weight for the target gene
        target_label = labs.label_for_gene(target_name)
        if target_label == "TF":
            weight_col[t_idx] = y_peak_coef
        else:
            weight_col[t_idx] = 0.0

        return weight_col

    # Run per-target columns, optionally in parallel
    columns = Parallel(n_jobs=n_jobs)(
        delayed(_process_one_target)(t) for t in range(n_genes)
    )

    # Stack columns: weights[:, j] = weight column for target gene j
    weights = np.column_stack(columns)  # (n_genes, n_genes)

    # Normalise by number of neighbourhood cells (matches R: weights / num.samples)
    weights /= n_samples

    return weights


# ---------------------------------------------------------------------------
# wScReNI — public network inference
# ---------------------------------------------------------------------------


def infer_wscreni_networks(
    expr: Union[np.ndarray, pd.DataFrame, "ad.AnnData"],
    peak_mat: Union[np.ndarray, pd.DataFrame, "ad.AnnData"],
    labs: "GenePeakOverlapLabs",
    nearest_neighbors_idx: np.ndarray,
    network_path: Union[str, "Path"],
    *,
    data_name: str = "",
    cell_index: Optional[list[int]] = None,
    n_jobs: int = 1,
    max_cells_per_batch: int = 10,
    n_trees: int = 100,
    seed: int = 100,
    importance_measure: str = "IncNodePurity",
    gene_names: Optional[list[str]] = None,
    cell_names: Optional[list[str]] = None,
    peak_names: Optional[list[str]] = None,
) -> ScReniNetworks:
    """Infer cell-specific regulatory networks using wScReNI.

    Matches ``Infer_wScReNI_scNetworks()`` from ``Infer_wScReNI_scNetworks.R``.

    For each cell, this function:

    1. Collects the cell itself plus its WNN neighbours from both the
       expression matrix and the peak-accessibility matrix.
    2. Runs :func:`_gene_peak_random_forest` on that local neighbourhood
       to compute gene-regulatory weights incorporating peak accessibility.
    3. Writes the resulting ``(n_genes × n_genes)`` weight matrix to disk as
       ``<network_path>/wScReNI/<1-based-idx>.<cell_name>.network.txt``.
    4. Returns all per-cell networks as a :class:`ScReniNetworks` dict.

    The file format is identical to what ``combine_wscreni_networks`` expects,
    so that function can reload the saved networks from disk later.

    Parameters
    ----------
    expr
        Expression matrix.  Accepted formats:

        - ``AnnData``: cells × genes, raw counts in ``.X``
        - ``DataFrame``: cells × genes (index = cell names, columns = gene names)
        - ``ndarray``: cells × genes (provide *gene_names* / *cell_names*)
    peak_mat
        Peak-accessibility matrix.

        - ``AnnData``: cells × peaks in ``.X``
        - ``DataFrame``: cells × peaks (index = cell names, columns = peak names)
        - ``ndarray``: cells × peaks (provide *peak_names* / *cell_names*)

        Must have the same number of cells (rows) as *expr* and the same cell
        ordering.  Corresponds to ``gene_peak_overlap_matrix`` in the R code.
    labs
        Gene–peak–TF label object (:class:`GenePeakOverlapLabs`).  Typically
        built via :meth:`GenePeakOverlapLabs.from_dataframe`.
    nearest_neighbors_idx
        Integer array of shape ``(n_cells, k)`` with 0-based column indices
        giving the *k* WNN nearest neighbours for each cell.  Matches
        ``nearest.neighbors.idx`` in the R code (R is 1-based; Python 0-based).
    network_path
        Parent directory to write per-cell files into.  The sub-folder
        ``wScReNI/`` is created automatically.
    data_name
        Optional label prepended to log messages; mirrors R's ``data.name``.
    cell_index
        Optional 0-based list of cell indices to process.  When ``None``
        (default), all cells are processed.  Mirrors R's ``cell.index``.
    n_jobs
        Joblib parallel workers for the per-gene RF loop inside each cell's
        :func:`_gene_peak_random_forest` call.  Default 1 (sequential).
    max_cells_per_batch
        Number of cells processed per progress-reporting batch.  Default 10.
        Mirrors R's ``max.cell.per.batch``.
    n_trees
        Trees per random forest.  Default 100.
    seed
        Random seed.  Default 100.
    importance_measure
        ``'IncNodePurity'`` (default) or ``'%IncMSE'``.  Passed to
        :func:`_gene_peak_random_forest`.
    gene_names
        Gene labels when *expr* is a plain ndarray.
    cell_names
        Cell labels when *expr* is a plain ndarray.
    peak_names
        Peak labels when *peak_mat* is a plain ndarray.

    Returns
    -------
    ScReniNetworks
        ``{cell_name: (n_genes, n_genes) weight_matrix}`` for each processed
        cell.  Gene names are stored in ``.gene_names``.  Written files can
        be reloaded later with :func:`~screni.data.combine.combine_wscreni_networks`.

    Examples
    --------
    >>> labs = GenePeakOverlapLabs.from_dataframe(triplets_df)
    >>> knn = np.load("retinal_knn_indices.npy")   # (n_cells, k)
    >>> networks = infer_wscreni_networks(
    ...     rna_adata, atac_adata, labs, knn,
    ...     network_path="output/networks",
    ...     n_jobs=4,
    ... )
    >>> weight_mat = networks["AAACCTGAGAAACCAT-1"]  # shape (n_genes, n_genes)
    """
    network_path = Path(network_path)

    # ------------------------------------------------------------------
    # Resolve expression matrix → (n_cells, n_genes)
    # ------------------------------------------------------------------
    raw_mat, _gene_names, _cell_names = _resolve_input(expr, gene_names, cell_names)
    n_cells, n_genes = raw_mat.shape

    # ------------------------------------------------------------------
    # Resolve peak matrix → (n_cells, n_peaks)
    # ------------------------------------------------------------------
    if isinstance(peak_mat, ad.AnnData):
        peak_arr = (
            peak_mat.X.toarray() if sp.issparse(peak_mat.X) else np.asarray(peak_mat.X)
        ).astype(np.float64)
        _peak_names: list[str] = (
            peak_names if peak_names is not None
            else peak_mat.var_names.tolist()
        )
    elif isinstance(peak_mat, pd.DataFrame):
        peak_arr = peak_mat.to_numpy(dtype=np.float64)
        _peak_names = (
            peak_names if peak_names is not None else peak_mat.columns.tolist()
        )
    else:
        peak_arr = np.asarray(peak_mat, dtype=np.float64)
        _peak_names = peak_names if peak_names is not None else [
            f"peak_{i}" for i in range(peak_arr.shape[1])
        ]

    if peak_arr.shape[0] != n_cells:
        raise ValueError(
            f"peak_mat has {peak_arr.shape[0]} rows but expr has {n_cells} cells."
        )

    # ------------------------------------------------------------------
    # Determine which cells to process
    # ------------------------------------------------------------------
    if cell_index is None:
        indices_to_process = list(range(n_cells))
    else:
        indices_to_process = list(cell_index)

    n_to_process = len(indices_to_process)
    prefix = f"[{data_name}] " if data_name else ""

    logger.info(
        f"{prefix}wScReNI: {n_cells} total cells, processing {n_to_process} "
        f"| n_genes={n_genes}, n_peaks={len(_peak_names)}, "
        f"n_trees={n_trees}, seed={seed}"
    )

    # ------------------------------------------------------------------
    # Create output directories
    # ------------------------------------------------------------------
    wscreni_dir = network_path / "wScReNI"
    wscreni_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"{prefix}Writing per-cell network files to: {wscreni_dir}")

    # ------------------------------------------------------------------
    # Main loop: process cells in batches for progress reporting
    # ------------------------------------------------------------------
    networks = ScReniNetworks(gene_names=_gene_names)

    n_batches = (n_to_process + max_cells_per_batch - 1) // max_cells_per_batch

    for batch_idx in range(n_batches):
        batch_start = batch_idx * max_cells_per_batch
        batch_end = min(batch_start + max_cells_per_batch, n_to_process)
        batch = indices_to_process[batch_start:batch_end]

        first_cell = _cell_names[batch[0]]
        last_cell = _cell_names[batch[-1]]
        logger.info(
            f"{prefix}Batch {batch_idx + 1}/{n_batches}: "
            f"cells {batch[0]} – {batch[-1]} "
            f"({first_cell} → {last_cell})"
        )

        for cell_i in batch:
            cell_name = _cell_names[cell_i]

            # Gather neighbour indices for this cell
            # nearest_neighbors_idx[cell_i] is 0-based (Python convention)
            neighbour_idxs = nearest_neighbors_idx[cell_i]  # 1-D array of 0-based idxs
            neighborhood = np.concatenate([[cell_i], neighbour_idxs])

            # Extract neighbourhood sub-matrices
            wnn_expr = raw_mat[neighborhood, :]   # (k+1, n_genes)
            wnn_peak = peak_arr[neighborhood, :]  # (k+1, n_peaks)

            # Compute wScReNI weights via random forest
            sc_res = _gene_peak_random_forest(
                wnn_expr,
                wnn_peak,
                labs,
                _gene_names,
                _peak_names,
                n_trees=n_trees,
                n_jobs=n_jobs,
                importance_measure=importance_measure,
                seed=seed,
            )  # (n_genes, n_genes)

            # Write to disk using the same convention as combine_wscreni_networks
            # File: <1-based-idx>.<cell_name>.network.txt
            # R: write.table(sc_res, paste0(network.path, "wScReNI/", i, ".", tmp_name, ".network.txt"), sep="\t")
            file_num = cell_i + 1  # convert to 1-based index
            filename = wscreni_dir / f"{file_num}.{cell_name}.network.txt"
            df_out = pd.DataFrame(sc_res, index=_gene_names, columns=_gene_names)
            df_out.to_csv(filename, sep="\t")

            networks[cell_name] = sc_res

            logger.debug(f"{prefix}  [{cell_i + 1}/{n_cells}] {cell_name} written.")

    logger.info(
        f"{prefix}wScReNI complete: {len(networks)} networks, "
        f"shape {(n_genes, n_genes)}"
    )
    return networks
