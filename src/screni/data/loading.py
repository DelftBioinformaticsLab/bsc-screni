"""Phase 0: Load retinal and PBMC datasets into AnnData objects.

Loads raw count matrices and cell annotations, applies QC filtering,
annotates cell types, and saves as .h5ad files with diagnostic plots.

Retinal (unpaired):
    scATAC-seq: GSE181251 (Lyu et al. 2021) - 94,318 cells, 283,847 peaks
    scRNA-seq:  GSE118614 (Clark et al. 2019) - ~120k cells

PBMC 10X Multiome (paired):
    12,012 cells with both Gene Expression (36,601 genes) and ATAC Peaks (111,857)

PBMC cell type annotation:
    The ScReNI paper uses Seurat reference-based label transfer for PBMC
    annotation, but provides no code for this step. We use CellTypist
    (Immune_All_Low.pkl model) as the Python equivalent, followed by a
    manual mapping from CellTypist fine-grained labels to the 8 ScReNI types.
    See CELLTYPIST_TO_SCRENI for the mapping.
"""

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# Cell types used in the ScReNI paper
RETINAL_CELL_TYPES = ["RPC1", "RPC2", "RPC3", "MG"]

# Mapping from Lyu et al. 2021 ATAC annotations to ScReNI paper names
RETINAL_ATAC_CELLTYPE_MAP = {
    "RPCs_S1": "RPC1",
    "RPCs_S2": "RPC2",
    "RPCs_S3": "RPC3",
    "MG": "MG",
}

# The paper uses 9 age-matched timepoints (visible in Figure 2A UMAP legend).
# P11 is absent because the RNA dataset (Clark 2019) doesn't have it.
# P14 is excluded by the authors' choice — it contains mostly mature cells
# (937 MG out of 940 total at P14) which are less relevant to the
# developmental trajectory analysis.
RETINAL_TIMEPOINTS = ["E11", "E12", "E14", "E16", "E18", "P0", "P2", "P5", "P8"]

PBMC_CELL_TYPES = [
    "CD14 monocyte",
    "CD16 monocyte",
    "CD4 naive cell",
    "CD8 naive cell",
    "cDC",
    "Memory B cell",
    "NK",
    "Treg",
]

# Mapping from CellTypist Immune_All_Low labels to ScReNI paper types.
# Determined by inspecting CellTypist output on UMAP and matching clusters
# to the 8 types from the paper. Only 1:1 mappings - subtypes not used
# by the paper (e.g., Tem/Temra, MAIT, Naive B, pDC) are dropped.
CELLTYPIST_TO_SCRENI = {
    "Classical monocytes": "CD14 monocyte",
    "Non-classical monocytes": "CD16 monocyte",
    "Tcm/Naive helper T cells": "CD4 naive cell",
    "Tcm/Naive cytotoxic T cells": "CD8 naive cell",
    "DC2": "cDC",
    "Memory B cells": "Memory B cell",
    "CD16+ NK cells": "NK",
    "Regulatory T cells": "Treg",
}

# Expected cell counts from the paper (Table / Results section)
EXPECTED_COUNTS = {
    "retinal_rna": {"RPC1": 7853, "RPC2": 16645, "RPC3": 22943, "MG": 936},
    "retinal_atac": {"RPC1": 6049, "RPC2": 10464, "RPC3": 11912, "MG": 768},
    "pbmc": {
        "CD14 monocyte": 2812,
        "CD16 monocyte": 514,
        "CD4 naive cell": 1419,
        "CD8 naive cell": 1410,
        "cDC": 198,
        "Memory B cell": 371,
        "NK": 468,
        "Treg": 162,
    },
}


def _check_counts(adata: ad.AnnData, expected: dict, name: str) -> None:
    """Compare observed vs expected cell counts per cell type."""
    observed = adata.obs["cell_type"].value_counts().to_dict()
    all_match = True
    for ct, exp in expected.items():
        obs = observed.get(ct, 0)
        status = "OK" if obs == exp else "MISMATCH"
        if status == "MISMATCH":
            all_match = False
        logger.info(f"  {name} {ct}: {obs} (expected {exp}) [{status}]")
    if not all_match:
        logger.warning(
            f"  {name}: cell counts do not match paper exactly. "
            "This may be due to annotation or timepoint differences."
        )


# =========================================================================
#  Retinal scATAC-seq
# =========================================================================


def load_retinal_atac(
    data_dir: Path,
    timepoints: list[str] | None = None,
) -> ad.AnnData:
    """Load retinal developmental scATAC-seq from GSE181251.

    Parameters
    ----------
    data_dir
        Path to ``data/retinal_GSE181251/``.
    timepoints
        If given, filter to these timepoints only (for age-matching with RNA).

    Returns
    -------
    AnnData with ``.obs['cell_type']``, ``.obs['timepoint']``, raw peak counts.
    """
    data_dir = Path(data_dir)
    prefix = "GSE181251_Single_Cell_ATACseq"

    # Load cell annotations
    cells = pd.read_csv(
        data_dir / f"{prefix}_cell_annotation.txt.gz",
        sep="\t",
        compression="gzip",
    )
    logger.info(f"Retinal ATAC cell annotations: {cells.shape[0]} cells")
    logger.info(f"  Columns: {list(cells.columns)}")

    # Load peak annotations
    peaks = pd.read_csv(
        data_dir / f"{prefix}_peak_annotation.txt.gz",
        sep="\t",
        compression="gzip",
    )
    logger.info(f"Retinal ATAC peak annotations: {peaks.shape[0]} peaks")

    # Load sparse matrix (features × cells in GEO format)
    logger.info("Loading ATAC count matrix (this may take a while)...")
    mat = sio.mmread(str(data_dir / f"{prefix}_raw_matrix.mtx.gz"))
    mat = sp.csr_matrix(mat)
    logger.info(f"  Raw shape: {mat.shape} (peaks × cells)")

    # Transpose to cells × peaks
    mat = mat.T.tocsr()

    # Build AnnData
    obs = pd.DataFrame(index=cells["cell_id"].values)
    obs["cell_type_original"] = cells["celltypes"].values
    obs["timepoint"] = cells["realtime"].values

    # Map cell type names to paper convention
    obs["cell_type"] = obs["cell_type_original"].map(RETINAL_ATAC_CELLTYPE_MAP)

    # Peak names as var index
    # The peak annotation file has a single column header that IS the first peak name
    peak_names = [peaks.columns[0]] + peaks.iloc[:, 0].tolist()
    # Standardize peak names: chr1-start-end → chr1:start-end
    standardized = []
    for p in peak_names:
        parts = p.split("-")
        if len(parts) == 3:
            standardized.append(f"{parts[0]}:{parts[1]}-{parts[2]}")
        else:
            standardized.append(p)
    peak_names = standardized

    var = pd.DataFrame(index=peak_names[: mat.shape[1]])

    adata = ad.AnnData(X=mat, obs=obs, var=var)
    adata.var_names_make_unique()
    logger.info(f"  AnnData shape: {adata.shape}")

    # Filter by timepoints if specified
    if timepoints is not None:
        mask = adata.obs["timepoint"].isin(timepoints)
        n_before = adata.n_obs
        adata = adata[mask].copy()
        logger.info(
            f"  Filtered to {len(timepoints)} timepoints: "
            f"{n_before} → {adata.n_obs} cells"
        )

    # Filter to ScReNI cell types (RPC1, RPC2, RPC3, MG)
    mask = adata.obs["cell_type"].isin(RETINAL_CELL_TYPES)
    n_before = adata.n_obs
    adata = adata[mask].copy()
    logger.info(f"  Filtered to {RETINAL_CELL_TYPES}: {n_before} → {adata.n_obs} cells")

    # Report counts
    _check_counts(adata, EXPECTED_COUNTS["retinal_atac"], "retinal_atac")

    return adata


# =========================================================================
#  Retinal scRNA-seq (Clark et al. 2019)
# =========================================================================


def load_retinal_rna(
    data_dir: Path,
    timepoints: list[str] | None = None,
    prefilter_types: list[str] | None = None,
) -> ad.AnnData:
    """Load retinal developmental scRNA-seq from Clark et al. 2019.

    NOTE: Clark 2019 uses different cell type names than the ScReNI paper.
    Clark has ``Early RPCs``, ``Late RPCs``, ``Muller Glia``, etc.
    The ScReNI paper's ``RPC1/RPC2/RPC3/MG`` labels come from the ATAC
    data (Lyu et al.) and are transferred to RNA cells AFTER integration.

    Therefore this function does NOT filter to the final ScReNI cell types.
    Use ``prefilter_types`` to do a coarse pre-filter on Clark labels to
    reduce computation before integration.

    Parameters
    ----------
    data_dir
        Path to ``data/retinal_GSE181251/scRNAseq_clark2019/``.
    timepoints
        If given, filter to these timepoints only.
    prefilter_types
        If given, keep only cells matching these Clark 2019 cell type names.
        Suggested: ``["Early RPCs", "Late RPCs", "Muller Glia",
        "Neurogenic Cells"]`` to roughly match the ScReNI subset.
        If None, keeps all cells.

    Returns
    -------
    AnnData with ``.obs['cell_type']`` (Clark labels), ``.obs['timepoint']``,
    raw counts. Final ScReNI labels (RPC1/2/3/MG) must be assigned after
    integration with ATAC data.
    """
    data_dir = Path(data_dir)

    # Load cell annotations (has cell types, UMAP coords, timepoints)
    annot_path = data_dir / "10x_mouse_retina_pData_umap2_CellType_annot_w_horiz.csv"
    cells = pd.read_csv(annot_path, index_col=0)
    logger.info(f"Retinal RNA cell annotations: {cells.shape[0]} cells")
    logger.info(f"  Columns: {list(cells.columns)}")

    # Load gene/feature annotations
    feat_path = data_dir / "10x_mouse_retina_development_featureData.csv"
    features = pd.read_csv(feat_path, index_col=0)
    logger.info(f"Retinal RNA feature annotations: {features.shape[0]} genes")

    # Load count matrix (genes × cells in this dataset)
    mtx_path = data_dir / "10x_mouse_retina_development.mtx"
    logger.info("Loading RNA count matrix (this may take a while)...")
    mat = sio.mmread(str(mtx_path))
    mat = sp.csr_matrix(mat)
    logger.info(f"  Raw shape: {mat.shape}")

    # Determine orientation: if shape matches (genes, cells), transpose
    if mat.shape[0] == features.shape[0] and mat.shape[1] == cells.shape[0]:
        mat = mat.T.tocsr()
        logger.info(f"  Transposed to cells × genes: {mat.shape}")
    elif mat.shape[0] == cells.shape[0] and mat.shape[1] == features.shape[0]:
        logger.info(f"  Already cells × genes: {mat.shape}")
    else:
        logger.warning(
            f"  Matrix shape {mat.shape} doesn't match cells ({cells.shape[0]}) "
            f"or genes ({features.shape[0]}) - attempting to proceed"
        )

    # Build AnnData
    obs = pd.DataFrame(index=cells.index[: mat.shape[0]])

    # Map cell type column - Clark 2019 uses 'umap2_CellType'
    if "umap2_CellType" in cells.columns:
        obs["cell_type"] = cells["umap2_CellType"].values[: mat.shape[0]]
    elif "CellType" in cells.columns:
        obs["cell_type"] = cells["CellType"].values[: mat.shape[0]]
    else:
        raise ValueError(
            f"Cannot find cell type column in annotations. "
            f"Available: {list(cells.columns)}"
        )

    # Timepoint column
    if "age" in cells.columns:
        obs["timepoint"] = cells["age"].values[: mat.shape[0]]
    elif "Stage" in cells.columns:
        obs["timepoint"] = cells["Stage"].values[: mat.shape[0]]

    # Gene names
    if "gene_short_name" in features.columns:
        gene_names = features["gene_short_name"].values[: mat.shape[1]]
    else:
        gene_names = features.index.values[: mat.shape[1]]

    var = pd.DataFrame(index=gene_names)

    adata = ad.AnnData(X=mat, obs=obs, var=var)
    adata.var_names_make_unique()
    logger.info(f"  AnnData shape: {adata.shape}")
    logger.info(
        f"  Cell types: {adata.obs['cell_type'].nunique()} unique, "
        f"top 5: {adata.obs['cell_type'].value_counts().head().to_dict()}"
    )

    # Filter by timepoints if specified
    if timepoints is not None:
        mask = adata.obs["timepoint"].isin(timepoints)
        n_before = adata.n_obs
        adata = adata[mask].copy()
        logger.info(
            f"  Filtered to {len(timepoints)} timepoints: "
            f"{n_before} → {adata.n_obs} cells"
        )

    # Optional coarse pre-filter on Clark 2019 cell type names
    available_types = sorted(adata.obs["cell_type"].unique())
    logger.info(f"  Available cell types ({len(available_types)}): {available_types}")

    if prefilter_types is not None:
        mask = adata.obs["cell_type"].isin(prefilter_types)
        n_before = adata.n_obs
        adata = adata[mask].copy()
        logger.info(
            f"  Pre-filtered to {prefilter_types}: {n_before} → {adata.n_obs} cells"
        )

    # NOTE: Final cell type labels (RPC1/2/3/MG) are NOT assigned here.
    # They come from ATAC data after integration via match_rna_atac_neighbors().

    return adata


# =========================================================================
#  PBMC 10X Multiome
# =========================================================================


def load_pbmc(
    data_dir: Path,
) -> tuple[ad.AnnData, ad.AnnData]:
    """Load PBMC 10X Multiome dataset, splitting into RNA and ATAC.

    Parameters
    ----------
    data_dir
        Path to ``data/pbmc_unsorted_10k/``.

    Returns
    -------
    Tuple of (rna_adata, atac_adata), both with the same cells.
    Cell type annotations are NOT included - use ``annotate_pbmc_cell_types``
    to add them.
    """
    data_dir = Path(data_dir)
    h5_path = data_dir / "pbmc_unsorted_10k_filtered_feature_bc_matrix.h5"

    logger.info(f"Loading PBMC from {h5_path.name}...")
    adata = sc.read_10x_h5(str(h5_path), gex_only=False)
    adata.var_names_make_unique()
    logger.info(f"  Full matrix: {adata.shape[0]} cells × {adata.shape[1]} features")

    # Split by feature type
    gex_mask = adata.var["feature_types"] == "Gene Expression"
    atac_mask = adata.var["feature_types"] == "Peaks"

    rna = adata[:, gex_mask].copy()
    atac = adata[:, atac_mask].copy()

    logger.info(f"  RNA:  {rna.shape}")
    logger.info(f"  ATAC: {atac.shape}")

    # Filter ATAC peaks to chr-prefixed only (remove scaffolds)
    chr_mask = atac.var_names.str.startswith("chr")
    n_before = atac.n_vars
    atac = atac[:, chr_mask].copy()
    logger.info(
        f"  ATAC filtered to chr-prefixed peaks: {n_before} → {atac.n_vars}"
    )

    return rna, atac


def annotate_pbmc_cell_types(
    rna: ad.AnnData,
    model_name: str = "Immune_All_Low.pkl",
) -> ad.AnnData:
    """Annotate PBMC cell types using CellTypist reference-based annotation.

    The original ScReNI pipeline uses Seurat reference-based label transfer
    (FindTransferAnchors + TransferData) against the Hao et al. 2021 PBMC
    reference. CellTypist is the Python equivalent: a pre-trained logistic
    regression model on the same immune cell atlas data.

    The model produces fine-grained immune subtypes which are stored in
    ``.obs['cell_type_celltypist']``. These are then mapped to the 8 ScReNI
    paper types via ``CELLTYPIST_TO_SCRENI`` and stored in ``.obs['cell_type']``.
    Cells that don't map to any of the 8 types get NaN.

    Parameters
    ----------
    rna
        Raw RNA AnnData from ``load_pbmc()``.
    model_name
        CellTypist model to use.

    Returns
    -------
    AnnData with ``.obs['cell_type']``, ``.obs['cell_type_celltypist']``.
    ``.X`` contains raw counts.
    """
    try:
        import celltypist
        from celltypist import models
    except ImportError:
        raise ImportError(
            "celltypist is required for PBMC annotation. "
            "Install with: pip install celltypist"
        )

    adata = rna.copy()

    # CellTypist needs log1p-normalized data (target_sum=1e4)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Load model - try local path first, then download
    logger.info(f"  Running CellTypist with model '{model_name}'...")
    model_path = Path(f"data/reference/{model_name}")
    if model_path.exists():
        logger.info(f"  Loading model from {model_path}")
        model = models.Model.load(model=str(model_path))
    else:
        logger.info("  Model not found locally, downloading...")
        models.download_models(model=model_name)
        model = models.Model.load(model=model_name)
    predictions = celltypist.annotate(adata, model=model, majority_voting=True)

    # Store CellTypist fine-grained labels
    adata.obs["cell_type_celltypist"] = predictions.predicted_labels["majority_voting"]
    logger.info(
        f"  CellTypist types: {adata.obs['cell_type_celltypist'].nunique()} unique\n"
        f"  Counts:\n{adata.obs['cell_type_celltypist'].value_counts().to_string()}"
    )

    # Map to ScReNI 8 types (unmapped → NaN)
    adata.obs["cell_type"] = adata.obs["cell_type_celltypist"].map(
        CELLTYPIST_TO_SCRENI
    )
    n_mapped = adata.obs["cell_type"].notna().sum()
    logger.info(f"  Mapped {n_mapped}/{adata.n_obs} cells to ScReNI types")

    # Restore raw counts
    adata.X = adata.layers["counts"]
    del adata.layers["counts"]

    return adata


def qc_filter(
    adata: ad.AnnData,
    min_genes: int = 200,
    max_genes: int = 4500,
    max_pct_mt: float = 15.0,
) -> ad.AnnData:
    """Standard QC filtering on gene counts and mitochondrial percentage.

    Parameters
    ----------
    adata
        AnnData with raw counts. Must have gene symbols as var_names.
    min_genes
        Minimum genes detected per cell.
    max_genes
        Maximum genes detected per cell.
    max_pct_mt
        Maximum percentage of mitochondrial counts.

    Returns
    -------
    Filtered AnnData (copy). QC metrics are stored in ``.obs``.
    """
    adata = adata.copy()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    n_before = adata.n_obs
    mask = (
        (adata.obs["n_genes_by_counts"] > min_genes)
        & (adata.obs["n_genes_by_counts"] < max_genes)
        & (adata.obs["pct_counts_mt"] < max_pct_mt)
    )
    adata = adata[mask].copy()
    logger.info(
        f"  QC filter: {n_before} → {adata.n_obs} cells "
        f"({n_before - adata.n_obs} removed)"
    )
    return adata


# =========================================================================
#  Diagnostic plots
# =========================================================================


def plot_pbmc_diagnostics(
    rna: ad.AnnData,
    atac: ad.AnnData,
    output_dir: Path,
) -> None:
    """Generate QC violin plots and UMAPs (RNA + ATAC) for PBMC data.

    Parameters
    ----------
    rna
        PBMC RNA AnnData with ``cell_type_celltypist`` and QC metrics in obs.
    atac
        PBMC ATAC AnnData (same cells as rna).
    output_dir
        Directory for output PNGs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure QC metrics exist
    if "n_genes_by_counts" not in rna.obs.columns:
        rna.var["mt"] = rna.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(rna, qc_vars=["mt"], inplace=True)

    # Use CellTypist labels for plots (more informative than mapped labels)
    plot_col = "cell_type_celltypist" if "cell_type_celltypist" in rna.obs.columns else "cell_type"

    # --- QC violin plots ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    sc.pl.violin(rna, keys="n_genes_by_counts", groupby=plot_col,
                 ax=axes[0], show=False, rotation=90)
    sc.pl.violin(rna, keys="total_counts", groupby=plot_col,
                 ax=axes[1], show=False, rotation=90)
    sc.pl.violin(rna, keys="pct_counts_mt", groupby=plot_col,
                 ax=axes[2], show=False, rotation=90)
    fig.tight_layout()
    fig.savefig(output_dir / "pbmc_qc_violin.png", dpi=150)
    logger.info(f"  Saved {output_dir / 'pbmc_qc_violin.png'}")
    plt.close(fig)

    # --- RNA UMAP ---
    # Use same params as CellTypist's internal graph (2500 HVGs, 50 PCs, k=10)
    rna_work = rna.copy()
    sc.pp.normalize_total(rna_work, target_sum=1e4)
    sc.pp.log1p(rna_work)
    sc.pp.highly_variable_genes(rna_work, n_top_genes=2500)
    rna_hvg = rna_work[:, rna_work.var["highly_variable"]].copy()
    sc.pp.scale(rna_hvg, max_value=10)
    sc.tl.pca(rna_hvg, n_comps=50)
    sc.pp.neighbors(rna_hvg, n_neighbors=10, n_pcs=50)
    sc.tl.umap(rna_hvg)
    rna.obsm["X_umap"] = rna_hvg.obsm["X_umap"]

    # --- ATAC UMAP (TF-IDF + LSI) ---
    atac_work = atac.copy()
    try:
        import muon as mu
        mu.atac.pp.tfidf(atac_work)
        mu.atac.tl.lsi(atac_work, n_comps=30)
        # Drop first LSI component (depth-correlated)
        atac_work.obsm["X_lsi"] = atac_work.obsm["X_lsi"][:, 1:]
        sc.pp.neighbors(atac_work, use_rep="X_lsi", n_neighbors=10)
        sc.tl.umap(atac_work)
        atac.obsm["X_umap"] = atac_work.obsm["X_umap"]
        has_atac_umap = True
    except Exception as e:
        logger.warning(f"  ATAC UMAP failed: {e}")
        has_atac_umap = False

    # Transfer cell type labels to ATAC for plotting
    if plot_col in rna.obs.columns:
        atac.obs[plot_col] = rna.obs[plot_col].values
    if "cell_type" in rna.obs.columns:
        atac.obs["cell_type"] = rna.obs["cell_type"].values

    # Three-panel UMAP: RNA ScReNI types, RNA CellTypist labels, ATAC ScReNI types
    n_panels = 3 if has_atac_umap else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(11 * n_panels, 8))

    rna_mapped = rna[rna.obs["cell_type"].notna()].copy()
    sc.pl.umap(rna_mapped, color="cell_type", ax=axes[0], show=False,
               title="RNA: ScReNI types")
    sc.pl.umap(rna, color=plot_col, ax=axes[1], show=False,
               title="RNA: CellTypist labels")
    if has_atac_umap:
        atac_mapped = atac[atac.obs["cell_type"].notna()].copy()
        sc.pl.umap(atac_mapped, color="cell_type", ax=axes[2], show=False,
                   title="ATAC: ScReNI types (LSI embedding)")

    fig.tight_layout()
    fig.savefig(output_dir / "pbmc_celltypes_umap.png", dpi=150)
    logger.info(f"  Saved {output_dir / 'pbmc_celltypes_umap.png'}")
    plt.close(fig)


# =========================================================================
#  Convenience: load all + save
# =========================================================================


def get_shared_timepoints(
    rna_timepoints: set[str],
    atac_timepoints: set[str],
) -> list[str]:
    """Find shared timepoints between RNA and ATAC datasets."""
    shared = sorted(rna_timepoints & atac_timepoints)
    logger.info(
        f"  RNA timepoints: {sorted(rna_timepoints)}\n"
        f"  ATAC timepoints: {sorted(atac_timepoints)}\n"
        f"  Shared: {shared}"
    )
    return shared


def load_and_save_all(
    data_root: Path,
    output_dir: Path,
    plot_dir: Path | None = None,
) -> dict[str, ad.AnnData]:
    """Load all datasets, apply QC + annotation, save as .h5ad with plots.

    Pipeline order:
    1. Load retinal RNA (Clark labels, no final cell type filter)
    2. Load retinal ATAC (filter to RPC1/2/3/MG)
    3. Filter both to shared timepoints, save
    4. Load PBMC, annotate with CellTypist, QC filter
    5. Map to ScReNI types, filter to mapped cells, save
    6. Generate diagnostic plots

    Final retinal RNA cell type labels (RPC1/2/3/MG) are assigned LATER
    during integration (Phase 1) by transferring ATAC labels via
    nearest-neighbor matching. See integration.match_rna_atac_neighbors().

    Parameters
    ----------
    data_root
        Root data directory (e.g., ``data/``).
    output_dir
        Where to save processed .h5ad files (e.g., ``data/processed/``).
    plot_dir
        Where to save diagnostic plots. Defaults to ``output/data_inspection/``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if plot_dir is None:
        plot_dir = Path("output/data_inspection")
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # --- Retinal ---
    ret_dir = Path(data_root) / "retinal_GSE181251"
    scrna_dir = ret_dir / "scRNAseq_clark2019"

    # Load RNA with coarse pre-filter (Clark labels, NOT final ScReNI labels)
    # Filter to the 9 timepoints used in the paper (Figure 2A).
    rna_full = None
    if scrna_dir.exists():
        rna_full = load_retinal_rna(
            scrna_dir,
            timepoints=RETINAL_TIMEPOINTS,
            prefilter_types=["Early RPCs", "Late RPCs", "Muller Glia",
                             "Neurogenic Cells"],
        )
    else:
        logger.warning(f"  Retinal RNA dir not found: {scrna_dir}")

    # Load ATAC (filter to RPC1/2/3/MG + 9 timepoints)
    atac_full = None
    if ret_dir.exists():
        atac_full = load_retinal_atac(ret_dir, timepoints=RETINAL_TIMEPOINTS)
        _check_counts(atac_full, EXPECTED_COUNTS["retinal_atac"], "retinal_atac")
    else:
        logger.warning(f"  Retinal ATAC dir not found: {ret_dir}")

    # Save retinal data
    if rna_full is not None:
        rna_full.write_h5ad(output_dir / "retinal_rna.h5ad")
        results["retinal_rna"] = rna_full
        logger.info(
            f"  Saved retinal_rna.h5ad: {rna_full.shape} "
            f"(Clark labels, pre-integration)"
        )

    if atac_full is not None:
        atac_full.write_h5ad(output_dir / "retinal_atac.h5ad")
        results["retinal_atac"] = atac_full
        logger.info(f"  Saved retinal_atac.h5ad: {atac_full.shape}")

    # --- PBMC ---
    pbmc_dir = Path(data_root) / "pbmc_unsorted_10k"
    if pbmc_dir.exists():
        pbmc_rna, pbmc_atac = load_pbmc(pbmc_dir)

        # Annotate with CellTypist
        pbmc_rna = annotate_pbmc_cell_types(pbmc_rna)

        # QC filter
        logger.info("  Applying QC filter to PBMC...")
        pbmc_rna = qc_filter(pbmc_rna)
        pbmc_atac = pbmc_atac[pbmc_rna.obs_names].copy()

        # Generate diagnostic plots (before dropping unmapped cells)
        logger.info("  Generating PBMC diagnostic plots...")
        plot_pbmc_diagnostics(pbmc_rna, pbmc_atac, plot_dir)

        # Save unfiltered version (all CellTypist types, unmapped = NaN)
        pbmc_atac.obs["cell_type"] = pbmc_rna.obs["cell_type"].values
        pbmc_atac.obs["cell_type_celltypist"] = pbmc_rna.obs["cell_type_celltypist"].values
        pbmc_rna.write_h5ad(output_dir / "pbmc_rna_all.h5ad")
        pbmc_atac.write_h5ad(output_dir / "pbmc_atac_all.h5ad")
        logger.info(f"  Saved pbmc_rna_all.h5ad: {pbmc_rna.shape} (all types, incl. unmapped)")

        # Filter to mapped ScReNI cell types only
        mask = pbmc_rna.obs["cell_type"].notna()
        pbmc_rna = pbmc_rna[mask].copy()
        pbmc_atac = pbmc_atac[mask].copy()
        logger.info(
            f"  PBMC after mapping to ScReNI types: {pbmc_rna.n_obs} cells "
            f"({(~mask).sum()} unmapped dropped)"
        )
        _check_counts(pbmc_rna, EXPECTED_COUNTS["pbmc"], "pbmc")

        pbmc_rna.write_h5ad(output_dir / "pbmc_rna.h5ad")
        pbmc_atac.write_h5ad(output_dir / "pbmc_atac.h5ad")
        results["pbmc_rna"] = pbmc_rna
        results["pbmc_atac"] = pbmc_atac
        logger.info(f"  Saved pbmc_rna.h5ad: {pbmc_rna.shape} (8 ScReNI types only)")
        logger.info(f"  Saved pbmc_atac.h5ad: {pbmc_atac.shape}")
    else:
        logger.warning(f"  PBMC dir not found: {pbmc_dir}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_and_save_all(data_root=Path("data"), output_dir=Path("data/processed"))
