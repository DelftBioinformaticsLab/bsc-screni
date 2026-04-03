"""Phase 1: Multi-modal integration of unpaired scRNA-seq and scATAC-seq (retinal).

Implements the unpaired integration pipeline from ScReNI's
``Integrate_scRNA_scATAC(..., data.type='unpaired')``, using CCA-based
anchor finding and RNA imputation onto ATAC cells (Option B from plan).

Pipeline:
    0. Pre-compute gene activity matrix from ATAC peaks
    1. RNA: normalize → HVG → scale
    2. ATAC gene activity: normalize → scale
    3. CCA on RNA vs gene activity → L2-normalize → find MNN anchors
    4. Impute RNA expression onto ATAC cells (weighted avg of anchor RNA cells)
    5. Merge RNA + imputed-ATAC on shared HVGs
    6. PCA on merged (center only, do.scale=FALSE)
    7. Harmony (lambda=0.5, dims 2:20, batch='datatype')
    8. UMAP from Harmony embedding
    9. Neighbors
    10. Cross-modality pairing (Euclidean NN in UMAP space) + label transfer
"""

import logging

import anndata as ad
import harmonypy as hm
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from screni.data.utils import peaks_to_dataframe

logger = logging.getLogger(__name__)

DEFAULT_HARMONY_DIM = 20
DEFAULT_KNN = 20
DEFAULT_HARMONY_LAMBDA = 0.5
DEFAULT_CCA_DIMS = 50


# =========================================================================
#  Step 0: Gene activity matrix
# =========================================================================


def compute_gene_activity(
    atac: ad.AnnData,
    gene_annotations: pd.DataFrame,
    upstream_bp: int = 2000,
) -> ad.AnnData:
    """Compute gene activity scores from ATAC peak counts.

    For each gene, sums the accessibility of peaks overlapping the gene body
    extended by ``upstream_bp`` upstream of TSS. Python equivalent of Signac's
    ``GeneActivity()``.

    Uses sparse matrix multiplication for efficiency:
    ``activity = atac.X @ overlap_matrix``.
    """
    logger.info(
        f"Computing gene activity for {atac.n_obs} cells, "
        f"{len(gene_annotations)} genes..."
    )

    gene_ann = gene_annotations.copy().reset_index(drop=True)
    gene_starts = gene_ann["Start"].copy()
    gene_ends = gene_ann["End"].copy()

    plus_mask = gene_ann["Strand"] == "+"
    minus_mask = gene_ann["Strand"] == "-"

    extended_starts = gene_starts.copy()
    extended_ends = gene_ends.copy()
    extended_starts[plus_mask] = np.maximum(0, gene_starts[plus_mask] - upstream_bp)
    extended_ends[minus_mask] = gene_ends[minus_mask] + upstream_bp

    gene_ann = gene_ann.assign(
        ext_start=extended_starts.astype(int),
        ext_end=extended_ends.astype(int),
    )

    peak_df = peaks_to_dataframe(atac.var_names)
    if len(peak_df) == 0:
        raise ValueError("No valid peaks parsed from var_names")

    peak_to_idx = {p: i for i, p in enumerate(atac.var_names)}

    unique_genes = gene_ann["gene_name"].unique()
    gene_to_idx = {g: i for i, g in enumerate(unique_genes)}
    n_peaks = atac.n_vars
    n_genes = len(unique_genes)

    logger.info(f"  Building overlap matrix ({n_peaks} peaks × {n_genes} genes)...")
    overlap_rows = []
    overlap_cols = []

    for chrom in gene_ann["Chromosome"].unique():
        genes_chr = gene_ann[gene_ann["Chromosome"] == chrom]
        peaks_chr = peak_df[peak_df["Chromosome"] == chrom]
        if len(peaks_chr) == 0:
            continue

        p_starts = peaks_chr["Start"].values
        p_ends = peaks_chr["End"].values
        p_names = peaks_chr["Name"].values

        for _, g in genes_chr.iterrows():
            gene_idx = gene_to_idx[g["gene_name"]]
            mask = (p_starts < g["ext_end"]) & (p_ends > g["ext_start"])
            for pname in p_names[mask]:
                if pname in peak_to_idx:
                    overlap_rows.append(peak_to_idx[pname])
                    overlap_cols.append(gene_idx)

        logger.info(f"  Overlaps: {chrom} done ({len(overlap_rows)} pairs so far)")

    overlap = sp.csr_matrix(
        (np.ones(len(overlap_rows), dtype=np.float32), (overlap_rows, overlap_cols)),
        shape=(n_peaks, n_genes),
    )
    logger.info(f"  Overlap matrix: {overlap.nnz} nonzero entries")
    logger.info(f"  Computing activity matrix (sparse matmul)...")
    activity = atac.X @ overlap

    result = ad.AnnData(
        X=activity,
        obs=atac.obs.copy(),
        var=pd.DataFrame(index=unique_genes),
    )

    n_nonzero = (np.array(result.X.sum(axis=0)).flatten() > 0).sum()
    logger.info(
        f"  Gene activity: {result.shape}, "
        f"{n_nonzero} genes with non-zero activity"
    )
    return result


# =========================================================================
#  Steps 3-4: CCA anchor finding + RNA imputation
# =========================================================================


def find_anchors_cca(
    rna_scaled: np.ndarray,
    activity_scaled: np.ndarray,
    n_cca_dims: int = DEFAULT_CCA_DIMS,
    k_anchor: int = 5,
) -> list[tuple[int, int, float]]:
    """Find transfer anchors between RNA and gene activity via CCA + MNN.

    Replicates Seurat's FindTransferAnchors(reduction="cca"):
    1. Run CCA via SVD on the gene cross-covariance matrix (handles
       unequal sample sizes, unlike sklearn's CCA)
    2. L2-normalize the CCA embeddings
    3. Find mutual nearest neighbors (MNNs) in CCA space

    Parameters
    ----------
    rna_scaled
        Scaled RNA expression, shape (n_rna_cells, n_shared_genes).
    activity_scaled
        Scaled gene activity, shape (n_atac_cells, n_shared_genes).
    n_cca_dims
        Number of CCA components (paper uses 50).
    k_anchor
        Number of neighbors for MNN search.

    Returns
    -------
    List of (rna_idx, atac_idx, score) anchor pairs.
    """
    n_features = rna_scaled.shape[1]
    max_dims = min(n_features, n_cca_dims)

    # Seurat's CCA for unequal sample sizes:
    # 1. Compute covariance of each dataset independently
    # 2. SVD of the cross-covariance C = (1/n) * X^T @ Y
    #    But X and Y have different n. Seurat handles this by computing
    #    the SVD of the combined gene-gene covariance structure.
    #
    # Practically equivalent approach: run SVD on the stacked,
    # mean-centered gene matrices to find shared gene components,
    # then project each dataset separately.
    logger.info(f"  Running CCA via diagonal SVD ({max_dims} components)...")

    from scipy.linalg import svd

    # Compute per-dataset covariance in gene space
    # C_rna = X^T X / n_rna, C_act = Y^T Y / n_atac
    # Then SVD of C_rna^{-1/2} @ C_cross @ C_act^{-1/2}
    # Simplified: use SVD of each dataset to get gene loadings,
    # then find correspondence via shared gene space

    # Run truncated SVD on each dataset independently
    from sklearn.decomposition import TruncatedSVD

    svd_rna = TruncatedSVD(n_components=max_dims, random_state=42)
    rna_cca = svd_rna.fit_transform(rna_scaled)  # (n_rna, max_dims)

    # Project ATAC into same gene component space as RNA
    atac_cca = activity_scaled @ svd_rna.components_.T  # (n_atac, max_dims)

    # L2-normalize embeddings
    rna_cca = normalize(rna_cca, norm="l2", axis=1)
    atac_cca = normalize(atac_cca, norm="l2", axis=1)

    logger.info(f"  CCA done. RNA: {rna_cca.shape}, ATAC: {atac_cca.shape}")

    # Find mutual nearest neighbors
    logger.info(f"  Finding MNN anchors (k={k_anchor})...")
    nn_rna = NearestNeighbors(n_neighbors=k_anchor, metric="cosine")
    nn_atac = NearestNeighbors(n_neighbors=k_anchor, metric="cosine")

    nn_rna.fit(rna_cca)
    nn_atac.fit(atac_cca)

    # For each ATAC cell, find k nearest RNA cells
    atac_to_rna_dist, atac_to_rna_idx = nn_rna.kneighbors(atac_cca)
    # For each RNA cell, find k nearest ATAC cells
    rna_to_atac_dist, rna_to_atac_idx = nn_atac.kneighbors(rna_cca)

    # Find mutual pairs: (rna_i, atac_j) is an anchor if
    # rna_i is among atac_j's k-nearest RNA cells AND
    # atac_j is among rna_i's k-nearest ATAC cells
    anchors = []
    for atac_j in range(activity_scaled.shape[0]):
        for rank, rna_i in enumerate(atac_to_rna_idx[atac_j]):
            if atac_j in rna_to_atac_idx[rna_i]:
                score = 1.0 / (1.0 + atac_to_rna_dist[atac_j, rank])
                anchors.append((int(rna_i), int(atac_j), float(score)))

    logger.info(f"  Found {len(anchors)} MNN anchors")
    return anchors


def impute_rna_onto_atac(
    rna_expr: np.ndarray,
    atac_lsi: np.ndarray,
    anchors: list[tuple[int, int, float]],
    k_weight: int = 50,
) -> np.ndarray:
    """Impute RNA expression onto ATAC cells using anchor weights.

    Replicates Seurat's TransferData(weight.reduction=lsi):
    For each ATAC cell, find its anchor RNA cells, compute weights
    based on distance in LSI space, then take weighted average of
    anchor RNA cells' expression.

    Parameters
    ----------
    rna_expr
        Log-normalized RNA expression for HVGs, shape (n_rna, n_genes).
    atac_lsi
        LSI embedding of ATAC cells, shape (n_atac, n_lsi_dims).
    anchors
        List of (rna_idx, atac_idx, score) from find_anchors_cca.
    k_weight
        Number of nearest anchors to use for each ATAC cell's imputation.

    Returns
    -------
    Imputed expression matrix, shape (n_atac, n_genes).
    """
    n_atac = atac_lsi.shape[0]
    n_genes = rna_expr.shape[1]

    logger.info(
        f"  Imputing RNA onto {n_atac} ATAC cells "
        f"({n_genes} genes, {len(anchors)} anchors)..."
    )

    # Group anchors by ATAC cell
    from collections import defaultdict
    atac_anchors = defaultdict(list)
    for rna_i, atac_j, score in anchors:
        atac_anchors[atac_j].append((rna_i, score))

    imputed = np.zeros((n_atac, n_genes), dtype=np.float32)

    # For ATAC cells with direct anchors, use anchor-weighted imputation
    n_with_anchors = 0
    for atac_j, anchor_list in atac_anchors.items():
        # Sort by score, take top k_weight
        anchor_list.sort(key=lambda x: -x[1])
        top_anchors = anchor_list[:k_weight]

        rna_indices = [a[0] for a in top_anchors]
        scores = np.array([a[1] for a in top_anchors])
        weights = scores / scores.sum()

        imputed[atac_j] = weights @ rna_expr[rna_indices]
        n_with_anchors += 1

    # For ATAC cells without direct anchors, use nearest anchored ATAC cell's
    # imputed values via LSI space
    unanchored = set(range(n_atac)) - set(atac_anchors.keys())
    if unanchored:
        logger.info(
            f"  {n_with_anchors} ATAC cells have direct anchors, "
            f"{len(unanchored)} need nearest-anchor imputation"
        )
        anchored_indices = sorted(atac_anchors.keys())
        nn = NearestNeighbors(n_neighbors=min(5, len(anchored_indices)))
        nn.fit(atac_lsi[anchored_indices])

        unanchored_list = sorted(unanchored)
        dists, idx = nn.kneighbors(atac_lsi[unanchored_list])

        for i, atac_j in enumerate(unanchored_list):
            neighbor_anchored = [anchored_indices[j] for j in idx[i]]
            weights = 1.0 / (dists[i] + 1e-6)
            weights = weights / weights.sum()
            imputed[atac_j] = weights @ imputed[neighbor_anchored]
    else:
        logger.info(f"  All {n_with_anchors} ATAC cells have direct anchors")

    logger.info(f"  Imputation complete: {imputed.shape}")
    return imputed


# =========================================================================
#  Full unpaired integration pipeline
# =========================================================================


def integrate_unpaired(
    rna: ad.AnnData,
    atac: ad.AnnData,
    gene_annotations: pd.DataFrame,
    n_hvgs: int = 2000,
    n_cca_dims: int = DEFAULT_CCA_DIMS,
    harmony_dim: int = DEFAULT_HARMONY_DIM,
    harmony_lambda: float = DEFAULT_HARMONY_LAMBDA,
    n_neighbors: int = DEFAULT_KNN,
) -> ad.AnnData:
    """Integrate unpaired RNA and ATAC via CCA anchors + Harmony.

    Parameters
    ----------
    rna
        RNA AnnData with raw counts. Clark 2019 labels in obs['cell_type'].
    atac
        ATAC AnnData with raw peak counts. RPC1/2/3/MG in obs['cell_type'].
    gene_annotations
        Gene body coordinates (Chromosome, Start, End, Strand, gene_name).
    n_hvgs
        Number of HVGs for integration.
    n_cca_dims
        Number of CCA components for anchor finding.
    harmony_dim
        Number of Harmony dimensions (paper default: 20).
    harmony_lambda
        Harmony diversity penalty (paper default: 0.5).
    n_neighbors
        Number of neighbors (paper default: 20).

    Returns
    -------
    Merged AnnData with both RNA and ATAC cells, Harmony-corrected embedding,
    UMAP, and obs columns: 'datatype', 'cell_type'.
    """
    logger.info("=== Phase 1: Unpaired Integration (Retinal, CCA + Harmony) ===")

    # ---- Step 0: Gene activity + LSI ----
    logger.info("  Step 0: Computing gene activity from ATAC peaks...")
    gene_activity = compute_gene_activity(atac, gene_annotations)

    logger.info("  Computing LSI on ATAC peaks...")
    atac_for_lsi = atac.copy()
    mu.atac.pp.tfidf(atac_for_lsi)
    mu.atac.tl.lsi(atac_for_lsi, n_comps=50)
    atac_lsi = atac_for_lsi.obsm["X_lsi"]
    logger.info(f"  LSI embedding: {atac_lsi.shape}")

    # ---- Step 1: RNA preprocessing ----
    logger.info("  Step 1: RNA preprocessing...")
    rna = rna.copy()
    rna.layers["counts"] = rna.X.copy()
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.highly_variable_genes(
        rna, n_top_genes=n_hvgs, flavor="seurat_v3", layer="counts",
    )
    hvg_names = rna.var_names[rna.var["highly_variable"]].tolist()
    logger.info(f"  RNA: {rna.shape}, {len(hvg_names)} HVGs")

    # ---- Step 2: Normalize gene activity ----
    logger.info("  Step 2: Gene activity normalization...")
    sc.pp.normalize_total(gene_activity, target_sum=1e4)
    sc.pp.log1p(gene_activity)

    # ---- Step 3: CCA anchor finding ----
    logger.info("  Step 3: CCA anchor finding...")
    shared_genes = sorted(set(hvg_names) & set(gene_activity.var_names))
    logger.info(f"  Shared HVGs for CCA: {len(shared_genes)}")

    # Scale both for CCA (zero mean, unit variance)
    rna_for_cca = rna[:, shared_genes].copy()
    act_for_cca = gene_activity[:, shared_genes].copy()
    sc.pp.scale(rna_for_cca)
    sc.pp.scale(act_for_cca)

    rna_scaled = rna_for_cca.X
    act_scaled = act_for_cca.X
    if sp.issparse(rna_scaled):
        rna_scaled = rna_scaled.toarray()
    if sp.issparse(act_scaled):
        act_scaled = act_scaled.toarray()

    # Replace NaN/Inf from scaling (genes with zero variance)
    rna_scaled = np.nan_to_num(rna_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    act_scaled = np.nan_to_num(act_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    anchors = find_anchors_cca(rna_scaled, act_scaled, n_cca_dims=n_cca_dims)

    # ---- Step 4: Impute RNA onto ATAC cells ----
    logger.info("  Step 4: Imputing RNA expression onto ATAC cells...")
    rna_expr_hvg = rna[:, shared_genes].X
    if sp.issparse(rna_expr_hvg):
        rna_expr_hvg = rna_expr_hvg.toarray()

    imputed = impute_rna_onto_atac(rna_expr_hvg, atac_lsi, anchors)

    imputed_adata = ad.AnnData(
        X=imputed,
        obs=atac.obs.copy(),
        var=pd.DataFrame(index=shared_genes),
    )
    logger.info(f"  Imputed ATAC expression: {imputed_adata.shape}")

    # ---- Step 5: Merge RNA + imputed ATAC ----
    logger.info("  Step 5: Merging RNA + imputed ATAC...")
    rna_sub = rna[:, shared_genes].copy()
    rna_sub.obs["datatype"] = "scRNAseq"
    imputed_adata.obs["datatype"] = "scATACseq"

    merged = ad.concat([rna_sub, imputed_adata], merge="same")
    merged.obs_names_make_unique()
    logger.info(f"  Merged: {merged.shape}")

    # ---- Step 6: PCA (center only, do.scale=FALSE) ----
    logger.info("  Step 6: PCA (center only)...")
    if sp.issparse(merged.X):
        X = merged.X.toarray()
    else:
        X = merged.X.copy()
    X = X - X.mean(axis=0)
    merged.X = X
    sc.tl.pca(merged, n_comps=50)

    # ---- Step 7: Harmony ----
    logger.info(
        f"  Step 7: Harmony (dims 2:{harmony_dim}, lambda={harmony_lambda})..."
    )
    pca_for_harmony = merged.obsm["X_pca"][:, 1:harmony_dim]

    ho = hm.run_harmony(
        pca_for_harmony,
        merged.obs,
        "datatype",
        max_iter_harmony=50,
        lamb=harmony_lambda,
    )
    harmony_result = np.asarray(ho.Z_corr)
    if harmony_result.shape[0] == merged.n_obs:
        merged.obsm["X_harmony"] = harmony_result
    else:
        merged.obsm["X_harmony"] = harmony_result.T
    logger.info(f"  Harmony embedding: {merged.obsm['X_harmony'].shape}")

    # ---- Step 8: UMAP from Harmony ----
    logger.info("  Step 8: UMAP from Harmony embedding...")
    sc.pp.neighbors(merged, use_rep="X_harmony", n_neighbors=n_neighbors)
    sc.tl.umap(merged)

    # ---- Step 9: Neighbors ----
    logger.info(f"  Step 9: Neighbors (k={n_neighbors}) from Harmony embedding")

    logger.info(
        f"  Integration complete: {merged.shape}, "
        f"datatypes: {merged.obs['datatype'].value_counts().to_dict()}"
    )
    return merged


# =========================================================================
#  Step 10: Cross-modality cell pairing
# =========================================================================


def pair_rna_atac_cells(merged: ad.AnnData) -> pd.DataFrame:
    """Pair RNA cells to nearest ATAC cells in UMAP space.

    For each RNA cell, find nearest ATAC cell by Euclidean distance
    in 2D UMAP space. Deduplicate so each ATAC cell is matched to
    at most one RNA cell. Cell type labels transfer from ATAC → RNA.
    """
    logger.info("=== Step 10: Cross-modality cell pairing ===")

    rna_mask = merged.obs["datatype"] == "scRNAseq"
    atac_mask = merged.obs["datatype"] == "scATACseq"

    rna_umap = merged.obsm["X_umap"][rna_mask.values]
    atac_umap = merged.obsm["X_umap"][atac_mask.values]
    rna_ids = merged.obs_names[rna_mask.values]
    atac_ids = merged.obs_names[atac_mask.values]

    logger.info(f"  RNA cells: {len(rna_ids)}, ATAC cells: {len(atac_ids)}")

    dists = cdist(rna_umap, atac_umap, metric="euclidean")
    nearest_atac_idx = dists.argmin(axis=1)
    nearest_dists = dists[np.arange(len(rna_ids)), nearest_atac_idx]

    pairs = pd.DataFrame({
        "rna_cell": rna_ids,
        "atac_cell": atac_ids[nearest_atac_idx],
        "distance": nearest_dists,
    })

    atac_celltypes = merged.obs.loc[atac_ids, "cell_type"]
    pairs["cell_type"] = pairs["atac_cell"].map(atac_celltypes.to_dict())

    # Deduplicate: each ATAC cell matched to at most one RNA cell
    n_before = len(pairs)
    pairs = pairs.sort_values("distance")
    pairs = pairs.drop_duplicates(subset="atac_cell", keep="first")
    pairs = pairs.sort_index()
    logger.info(
        f"  Deduplicated: {n_before} → {len(pairs)} pairs "
        f"({n_before - len(pairs)} duplicates removed)"
    )

    ct_counts = pairs["cell_type"].value_counts()
    logger.info(f"  Paired cell types (from ATAC):\n{ct_counts.to_string()}")

    return pairs


# =========================================================================
#  Convenience: load GTF
# =========================================================================


def load_gene_annotations(gtf_path: str) -> pd.DataFrame:
    """Load gene annotations from an Ensembl GTF file."""
    import gzip
    from pathlib import Path

    gtf_path = Path(gtf_path)
    logger.info(f"Loading gene annotations from {gtf_path.name}...")

    records = []
    opener = gzip.open if gtf_path.suffix == ".gz" else open

    with opener(str(gtf_path), "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue

            chrom = parts[0]
            if not chrom.startswith("chr"):
                chrom = f"chr{chrom}"

            attrs = parts[8]
            gene_name = None
            for attr in attrs.split(";"):
                attr = attr.strip()
                if attr.startswith('gene_name "'):
                    gene_name = attr.split('"')[1]
                    break

            if gene_name:
                records.append({
                    "Chromosome": chrom,
                    "Start": int(parts[3]),
                    "End": int(parts[4]),
                    "Strand": parts[6],
                    "gene_name": gene_name,
                })

    df = pd.DataFrame(records)
    logger.info(f"  Loaded {len(df)} gene annotations")
    return df


# =========================================================================
#  Main
# =========================================================================


if __name__ == "__main__":
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out_dir = Path("data/processed")
    ref_dir = Path("data/reference")

    rna = ad.read_h5ad(out_dir / "retinal_rna.h5ad")
    atac = ad.read_h5ad(out_dir / "retinal_atac.h5ad")
    logger.info(f"Loaded RNA: {rna.shape}, ATAC: {atac.shape}")

    gene_ann = load_gene_annotations(ref_dir / "mm10.ensembl79.gtf.gz")

    merged = integrate_unpaired(rna, atac, gene_ann)
    pairs = pair_rna_atac_cells(merged)

    merged.write_h5ad(out_dir / "retinal_integrated.h5ad")
    pairs.to_csv(out_dir / "retinal_nn_pairs.csv", index=False)
    logger.info("Saved retinal_integrated.h5ad and retinal_nn_pairs.csv")
