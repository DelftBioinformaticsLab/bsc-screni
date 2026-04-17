"""Phase 0: Load PBMC dataset into AnnData objects.

Loads raw count matrices and cell annotations, applies QC filtering,
annotates cell types, and saves as .h5ad files with diagnostic plots.

Retinal data is loaded directly from the paper's Seurat exports
(see loading_paper.py).

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
import scipy.sparse as sp

logger = logging.getLogger(__name__)

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


def load_and_save_all(
    data_root: Path,
    output_dir: Path,
    plot_dir: Path | None = None,
) -> dict[str, ad.AnnData]:
    """Load PBMC dataset, apply QC + annotation, save as .h5ad with plots.

    Pipeline order:
    1. Load PBMC, annotate with CellTypist, QC filter
    2. Map to ScReNI types, filter to mapped cells, save
    3. Generate diagnostic plots

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
