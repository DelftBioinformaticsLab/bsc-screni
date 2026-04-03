"""Phase 1: Multi-modal integration of scRNA-seq and scATAC-seq.

Unpaired data (retinal): GeneActivity → concatenation → Harmony
Paired data (PBMC): WNN via muon

Matches ``Integrate_scRNA_scATAC()`` and ``Get_scRNA_scATAC_neighbors()``
from the original R code, adapted for pure Python.

Key differences from R pipeline (documented):
    1. No CCA anchor transfer: we concatenate RNA + GeneActivity matrices
       directly and use Harmony for batch correction, instead of Seurat's
       FindTransferAnchors + TransferData.
    2. SCTransform → LogNormalize: scanpy's normalize_total + log1p replaces
       Seurat's SCTransform for RNA normalization.
"""

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from screni.data.utils import peaks_to_dataframe

logger = logging.getLogger(__name__)

# Default parameters matching the original R code
DEFAULT_KNN = 20
DEFAULT_HARMONY_LAMBDA = 0.5
DEFAULT_INTEGRATED_DIMS = 20


# =========================================================================
#  GeneActivity: count ATAC peaks overlapping gene body + 2kb upstream
# =========================================================================


def compute_gene_activity(
    atac: ad.AnnData,
    gene_annotations: pd.DataFrame,
    upstream_bp: int = 2000,
) -> ad.AnnData:
    """Compute gene activity scores from ATAC peak counts.

    For each gene, sums the accessibility of peaks overlapping the gene body
    extended by ``upstream_bp`` base pairs upstream.

    This is the Python equivalent of Signac's ``GeneActivity()``.

    Parameters
    ----------
    atac
        ATAC AnnData with peak names as var_names (format: chrX:start-end).
    gene_annotations
        DataFrame with columns: Chromosome, Start, End, Strand, gene_name.
        Gene body coordinates from a GTF file.
    upstream_bp
        How far upstream of TSS to extend (default 2000, matching Signac).

    Returns
    -------
    AnnData with shape (n_atac_cells, n_genes), containing activity scores.
    """
    logger.info(f"Computing GeneActivity for {atac.n_obs} cells, {len(gene_annotations)} genes...")

    # Build gene body + upstream regions
    gene_regions = gene_annotations.copy()
    gene_starts = gene_regions["Start"].copy()
    gene_ends = gene_regions["End"].copy()

    # For + strand: extend Start upstream
    # For - strand: extend End upstream (which is actually upstream for - strand)
    plus_mask = gene_regions["Strand"] == "+"
    minus_mask = gene_regions["Strand"] == "-"

    extended_starts = gene_starts.copy()
    extended_ends = gene_ends.copy()
    extended_starts[plus_mask] = np.maximum(0, gene_starts[plus_mask] - upstream_bp)
    extended_ends[minus_mask] = gene_ends[minus_mask] + upstream_bp

    gene_regions = gene_regions.assign(
        ext_start=extended_starts.astype(int),
        ext_end=extended_ends.astype(int),
    )

    # Parse peak coordinates
    peak_df = peaks_to_dataframe(atac.var_names)
    if len(peak_df) == 0:
        raise ValueError("No valid peaks could be parsed from var_names")

    # Find overlaps per chromosome using pandas
    overlap_records = []
    for chrom in gene_regions["Chromosome"].unique():
        genes_chr = gene_regions[gene_regions["Chromosome"] == chrom]
        peaks_chr = peak_df[peak_df["Chromosome"] == chrom]
        if len(peaks_chr) == 0:
            continue
        p_starts = peaks_chr["Start"].values
        p_ends = peaks_chr["End"].values
        p_names = peaks_chr["Name"].values
        for _, g in genes_chr.iterrows():
            mask = (p_starts < g["ext_end"]) & (p_ends > g["ext_start"])
            for pn in p_names[mask]:
                overlap_records.append({"gene_name": g["gene_name"], "Name": pn})

    overlap_df = pd.DataFrame(overlap_records)
    logger.info(f"  Found {len(overlap_df)} gene-peak overlaps")

    # Build activity matrix: for each gene, sum the peak counts
    unique_genes = gene_regions["gene_name"].unique()
    gene_to_idx = {g: i for i, g in enumerate(unique_genes)}
    peak_to_idx = {p: i for i, p in enumerate(atac.var_names)}

    # Create sparse activity matrix
    from scipy.sparse import lil_matrix
    activity = lil_matrix((atac.n_obs, len(unique_genes)), dtype=np.float32)

    # Group overlaps by gene
    for gene_name, group in overlap_df.groupby("gene_name"):
        if gene_name not in gene_to_idx:
            continue
        gene_idx = gene_to_idx[gene_name]
        # Sum all overlapping peak columns for this gene
        peak_names = group["Name"].values
        valid_peaks = [p for p in peak_names if p in peak_to_idx]
        if not valid_peaks:
            continue
        peak_indices = [peak_to_idx[p] for p in valid_peaks]

        if sp.issparse(atac.X):
            activity[:, gene_idx] = np.array(
                atac.X[:, peak_indices].sum(axis=1)
            ).flatten()
        else:
            activity[:, gene_idx] = atac.X[:, peak_indices].sum(axis=1)

    activity = sp.csr_matrix(activity)

    # Build AnnData
    obs = atac.obs.copy()
    var = pd.DataFrame(index=unique_genes)
    result = ad.AnnData(X=activity, obs=obs, var=var)

    n_nonzero_genes = (np.array(result.X.sum(axis=0)).flatten() > 0).sum()
    logger.info(
        f"  GeneActivity matrix: {result.shape}, "
        f"{n_nonzero_genes} genes with non-zero activity"
    )

    return result


# =========================================================================
#  Unpaired integration: GeneActivity → Harmony
# =========================================================================


def integrate_unpaired(
    rna: ad.AnnData,
    atac: ad.AnnData,
    gene_annotations: pd.DataFrame,
    n_hvgs: int = 2000,
    n_pcs: int = 30,
    n_harmony_dims: int = DEFAULT_INTEGRATED_DIMS,
    harmony_lambda: float = DEFAULT_HARMONY_LAMBDA,
    n_neighbors: int = DEFAULT_KNN,
) -> ad.AnnData:
    """Integrate unpaired scRNA-seq and scATAC-seq via GeneActivity + Harmony.

    Pure-Python equivalent of the original R pipeline:
    ``Integrate_scRNA_scATAC(data.type="unpaired")``.

    Steps:
    1. Compute GeneActivity from ATAC peaks
    2. Restrict to shared genes (intersection of RNA genes and activity genes)
    3. Normalize both modalities
    4. Find shared HVGs
    5. Concatenate
    6. PCA → Harmony (batch = data type) → UMAP → Neighbors

    Parameters
    ----------
    rna
        RNA AnnData with raw counts. For retinal data, ``.obs['cell_type']``
        contains Clark 2019 labels (e.g., ``Early RPCs``), NOT the final
        ScReNI labels. Final labels are assigned via ``match_rna_atac_neighbors()``.
    atac
        ATAC AnnData with raw peak counts and ``.obs['cell_type']``
        containing the authoritative cell type labels (e.g., RPC1/2/3/MG).
    gene_annotations
        Gene body coordinates for GeneActivity computation.
    n_hvgs
        Number of highly variable genes for integration.
    n_pcs
        Number of PCA components.
    n_harmony_dims
        Number of Harmony dimensions (paper default: 20).
    harmony_lambda
        Harmony diversity penalty (paper default: 0.5).
    n_neighbors
        Number of nearest neighbors (paper default: 20).

    Returns
    -------
    Integrated AnnData with ``.obs['cell_type']``, ``.obs['datatype']``,
    ``.obsm['X_harmony']``, ``.obsm['X_umap']``.
    """
    logger.info("=== Phase 1a: Unpaired Integration (GeneActivity + Harmony) ===")

    # Step 1: Compute GeneActivity
    activity = compute_gene_activity(atac, gene_annotations)

    # Step 2: Restrict to shared genes
    shared_genes = sorted(set(rna.var_names) & set(activity.var_names))
    logger.info(f"  Shared genes between RNA and GeneActivity: {len(shared_genes)}")

    rna_shared = rna[:, shared_genes].copy()
    activity_shared = activity[:, shared_genes].copy()

    # Step 3: Normalize both
    rna_norm = rna_shared.copy()
    sc.pp.normalize_total(rna_norm, target_sum=1e4)
    sc.pp.log1p(rna_norm)

    act_norm = activity_shared.copy()
    sc.pp.normalize_total(act_norm, target_sum=1e4)
    sc.pp.log1p(act_norm)

    # Step 4: Find HVGs on RNA
    sc.pp.highly_variable_genes(rna_norm, n_top_genes=n_hvgs, flavor="seurat_v3")
    hvg_mask = rna_norm.var["highly_variable"]
    hvg_genes = rna_norm.var_names[hvg_mask].tolist()
    logger.info(f"  Selected {len(hvg_genes)} HVGs from RNA")

    # Subset both to HVGs
    rna_hvg = rna_norm[:, hvg_genes].copy()
    act_hvg = act_norm[:, hvg_genes].copy()

    # Add datatype labels
    rna_hvg.obs["datatype"] = "RNA"
    act_hvg.obs["datatype"] = "ATAC"

    # Step 5: Concatenate
    combined = ad.concat([rna_hvg, act_hvg], merge="same")
    combined.obs_names_make_unique()
    logger.info(f"  Combined shape: {combined.shape}")

    # Step 6: Scale (center only, matching do.scale=FALSE)
    sc.pp.scale(combined, zero_center=True, max_value=None)

    # PCA
    sc.tl.pca(combined, n_comps=n_pcs)

    # Harmony: use dims 2:n_harmony_dims (skip first component)
    try:
        import harmonypy
    except ImportError:
        raise ImportError("harmonypy required. Install with: pip install harmonypy")

    # Run harmony on PCA embeddings (skip component 1, matching original)
    pca_for_harmony = combined.obsm["X_pca"][:, 1:n_harmony_dims]
    meta = combined.obs[["datatype"]].copy()

    logger.info(
        f"  Running Harmony on PCA dims 2:{n_harmony_dims}, "
        f"lambda={harmony_lambda}..."
    )
    ho = harmonypy.run_harmony(
        pca_for_harmony,
        meta,
        "datatype",
        max_iter_harmony=50,
        sigma=0.1,
        lamb=harmony_lambda,
    )
    combined.obsm["X_harmony"] = ho.Z_corr.T

    # UMAP from Harmony embedding
    sc.pp.neighbors(
        combined,
        n_neighbors=n_neighbors,
        use_rep="X_harmony",
    )
    sc.tl.umap(combined)

    logger.info(
        f"  Integration complete. Shape: {combined.shape}, "
        f"datatypes: {combined.obs['datatype'].value_counts().to_dict()}"
    )

    return combined


# =========================================================================
#  Nearest neighbor matching (unpaired → pseudo-paired)
# =========================================================================


def match_rna_atac_neighbors(
    integrated: ad.AnnData,
) -> pd.DataFrame:
    """Match RNA cells to nearest ATAC cells in UMAP space.

    Matches ``Get_scRNA_scATAC_neighbors()`` from the original R code.

    For each RNA cell, finds the nearest ATAC cell by Euclidean distance
    in UMAP space. Removes duplicate matches (where multiple RNA cells
    map to the same ATAC cell, keeping the closest).

    IMPORTANT: For the retinal dataset, this is where RNA cells receive
    their final cell type labels (RPC1/2/3/MG). The ``cell_type`` column
    in the returned DataFrame comes from the ATAC cell's label, which is
    the authoritative source. The Clark 2019 RNA labels (Early RPCs, etc.)
    are NOT used as the final cell type assignment.

    Parameters
    ----------
    integrated
        Integrated AnnData with ``.obsm['X_umap']``, ``.obs['datatype']``,
        and ``.obs['cell_type']`` (ATAC cells have authoritative labels,
        RNA cells have preliminary/Clark labels).

    Returns
    -------
    DataFrame with columns: rna_cell, atac_cell, cell_type, distance.
    The ``cell_type`` comes from the matched ATAC cell's label.
    """
    logger.info("=== Phase 1b: Nearest Neighbor Matching ===")

    umap = integrated.obsm["X_umap"]
    obs = integrated.obs

    rna_mask = obs["datatype"] == "RNA"
    atac_mask = obs["datatype"] == "ATAC"

    rna_umap = umap[rna_mask.values]
    atac_umap = umap[atac_mask.values]
    rna_names = obs.index[rna_mask.values]
    atac_names = obs.index[atac_mask.values]

    logger.info(f"  RNA cells: {len(rna_names)}, ATAC cells: {len(atac_names)}")

    # Find nearest ATAC cell for each RNA cell
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(atac_umap)
    distances, indices = nn.kneighbors(rna_umap)

    matched_atac_cells = atac_names[indices.flatten()]

    # Cell type comes from the ATAC cell (authoritative labels)
    pairs = pd.DataFrame({
        "rna_cell": rna_names,
        "atac_cell": matched_atac_cells,
        "distance": distances.flatten(),
        "cell_type": obs.loc[matched_atac_cells, "cell_type"].values,
    })

    # Also store the RNA cell's original label for diagnostics
    if "cell_type" in obs.columns:
        pairs["rna_cell_type_original"] = obs.loc[rna_names, "cell_type"].values

    # Remove duplicate ATAC matches: keep closest RNA cell for each ATAC cell
    n_before = len(pairs)
    pairs = pairs.sort_values("distance")
    pairs = pairs.drop_duplicates(subset="atac_cell", keep="first")
    pairs = pairs.sort_index()
    n_after = len(pairs)
    logger.info(f"  Removed {n_before - n_after} duplicate ATAC matches: {n_before} → {n_after} pairs")

    # Report cell type distribution (these are ATAC-derived labels)
    ct_counts = pairs["cell_type"].value_counts()
    logger.info(f"  Matched pair cell types (from ATAC):\n{ct_counts.to_string()}")

    return pairs


# =========================================================================
#  Paired integration: WNN via muon
# =========================================================================


def integrate_paired(
    rna: ad.AnnData,
    atac: ad.AnnData,
    n_hvgs: int = 2000,
    n_pcs: int = 50,
    n_lsi: int = 30,
    n_neighbors: int = DEFAULT_KNN,
    leiden_resolution: float = 0.5,
) -> ad.AnnData:
    """Integrate paired scRNA-seq and scATAC-seq via WNN.

    Pure-Python equivalent of the original R pipeline:
    ``Integrate_scRNA_scATAC(data.type="paired")``.

    Steps:
    1. RNA: normalize → HVG → PCA → UMAP
    2. ATAC: TF-IDF → LSI → UMAP
    3. WNN: combine both modalities → joint UMAP → Leiden

    Parameters
    ----------
    rna
        RNA AnnData with raw counts and ``.obs['cell_type']``.
    atac
        ATAC AnnData with raw peak counts, same cells as ``rna``.
    n_hvgs
        Number of HVGs for RNA.
    n_pcs
        Number of PCA components for RNA.
    n_lsi
        Number of LSI components for ATAC.
    n_neighbors
        Number of neighbors for WNN.
    leiden_resolution
        Resolution for Leiden clustering.

    Returns
    -------
    MuData or AnnData with joint UMAP, WNN graph, and cell type labels.
    """
    try:
        import muon as mu
    except ImportError:
        raise ImportError("muon is required for paired integration. Install with: pip install muon")

    logger.info("=== Phase 1c: Paired Integration (WNN via muon) ===")

    # --- RNA preprocessing ---
    logger.info("  RNA preprocessing...")
    rna_proc = rna.copy()
    rna_proc.layers["counts"] = rna_proc.X.copy()
    sc.pp.normalize_total(rna_proc, target_sum=1e4)
    sc.pp.log1p(rna_proc)
    sc.pp.highly_variable_genes(rna_proc, n_top_genes=n_hvgs, flavor="seurat_v3")
    rna_hvg = rna_proc[:, rna_proc.var["highly_variable"]].copy()
    sc.pp.scale(rna_hvg, max_value=10)
    sc.tl.pca(rna_hvg, n_comps=n_pcs)
    sc.tl.umap(rna_hvg)
    # Transfer PCA back
    rna_proc.obsm["X_pca"] = np.zeros((rna_proc.n_obs, n_pcs))
    rna_proc.obsm["X_pca"][:, :rna_hvg.obsm["X_pca"].shape[1]] = rna_hvg.obsm["X_pca"]

    # --- ATAC preprocessing ---
    logger.info("  ATAC preprocessing...")
    atac_proc = atac.copy()
    mu.atac.pp.tfidf(atac_proc)
    mu.atac.tl.lsi(atac_proc, n_comps=n_lsi + 1)
    # Remove first LSI component (depth-correlated)
    atac_proc.obsm["X_lsi"] = atac_proc.obsm["X_lsi"][:, 1:]
    sc.pp.neighbors(atac_proc, use_rep="X_lsi", n_neighbors=n_neighbors)
    sc.tl.umap(atac_proc)

    # --- WNN ---
    logger.info("  Computing WNN...")
    mdata = mu.MuData({"rna": rna_proc, "atac": atac_proc})
    mu.pp.neighbors(mdata, key_added="wnn", n_neighbors=n_neighbors)
    sc.tl.umap(mdata, neighbors_key="wnn")
    sc.tl.leiden(mdata, neighbors_key="wnn", resolution=leiden_resolution)

    logger.info(
        f"  WNN integration complete. "
        f"Clusters: {mdata.obs['leiden'].nunique()}"
    )

    # Extract joint result as AnnData
    result = ad.AnnData(
        X=rna.X.copy(),
        obs=rna.obs.copy(),
        var=rna.var.copy(),
    )
    result.obsm["X_umap"] = mdata.obsm["X_umap"]
    if "wnn_connectivities" in mdata.obsp:
        result.obsp["connectivities"] = mdata.obsp["wnn_connectivities"]
        result.obsp["distances"] = mdata.obsp["wnn_distances"]
    result.obs["leiden"] = mdata.obs["leiden"].values

    return result
