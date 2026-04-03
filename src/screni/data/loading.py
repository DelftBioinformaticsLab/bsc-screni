"""Phase 0: Load retinal and PBMC datasets into AnnData objects.

Loads raw count matrices and cell annotations, filters to the cell types
used in the ScReNI paper, and saves as .h5ad files.

Retinal (unpaired):
    scATAC-seq: GSE181251 (Lyu et al. 2021) - 94,318 cells, 283,847 peaks
    scRNA-seq:  GSE118614 (Clark et al. 2019) - ~120k cells

PBMC 10X Multiome (paired):
    12,012 cells with both Gene Expression (36,601 genes) and ATAC Peaks (111,857)
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
            "This may be due to timepoint filtering differences."
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
    # Build peak names from the annotation
    peak_names = [peaks.columns[0]] + peaks.iloc[:, 0].tolist()
    # Standardize peak names: replace any chr1-start-end with chr1:start-end
    peak_names = [p.replace("-", ":", 1).replace("-", "-") for p in peak_names]
    # Actually the format is chr1-start-end, standardize to chr1:start-end
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
    # See REVIEW_NOTES.md for details.

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

    The model produces fine-grained immune subtypes which we then map to
    the 8 cell types used in the ScReNI paper.

    Parameters
    ----------
    rna
        Raw RNA AnnData from ``load_pbmc()``.
    model_name
        CellTypist model to use. ``"Immune_All_Low.pkl"`` gives fine-grained
        types comparable to Seurat's ``celltype.l2``.

    Returns
    -------
    AnnData with ``.obs['cell_type']`` and ``.obs['cell_type_fine']`` added.
    The ``.X`` contains raw counts.
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

    # CellTypist needs log-normalized data
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

    # Transfer labels
    adata.obs["cell_type_fine"] = predictions.predicted_labels["majority_voting"]
    logger.info(
        f"  CellTypist fine-grained types: "
        f"{adata.obs['cell_type_fine'].nunique()} unique"
    )
    logger.info(
        f"  Fine-grained counts:\n"
        f"{adata.obs['cell_type_fine'].value_counts().to_string()}"
    )

    # Map fine-grained CellTypist labels to the 8 ScReNI paper types.
    # CellTypist Immune_All_Low labels include types like:
    #   "Classical monocytes" → CD14 monocyte
    #   "Non-classical monocytes" → CD16 monocyte
    #   "CD4-positive, alpha-beta T cell" → CD4 naive cell
    #   etc.
    # This mapping covers known label variations; unmapped types → None.
    celltypist_to_screni = {
        # Monocytes
        "Classical monocytes": "CD14 monocyte",
        "CD14-positive monocyte": "CD14 monocyte",
        "CD14+ monocyte": "CD14 monocyte",
        "Non-classical monocytes": "CD16 monocyte",
        "CD16-positive monocyte": "CD16 monocyte",
        "CD16+ monocyte": "CD16 monocyte",
        # T cells
        "Naive CD4+ T cells": "CD4 naive cell",
        "CD4-positive, alpha-beta T cell": "CD4 naive cell",
        "Central memory CD4+ T cells": "CD4 naive cell",
        "Naive CD8+ T cells": "CD8 naive cell",
        "CD8-positive, alpha-beta T cell": "CD8 naive cell",
        "Central memory CD8+ T cells": "CD8 naive cell",
        # DCs
        "Conventional dendritic cells": "cDC",
        "Myeloid dendritic cells": "cDC",
        "cDC1": "cDC",
        "cDC2": "cDC",
        # B cells
        "Memory B cells": "Memory B cell",
        "Class-switched memory B cells": "Memory B cell",
        "Non-switched memory B cells": "Memory B cell",
        "Age-associated B cells": "Memory B cell",
        # NK
        "NK cells": "NK",
        "Natural killer cell": "NK",
        "CD16+ NK cells": "NK",
        "CD56bright NK cells": "NK",
        # Treg
        "Regulatory T cells": "Treg",
        "T regulatory cells": "Treg",
    }

    mapped = adata.obs["cell_type_fine"].map(celltypist_to_screni)
    n_mapped = mapped.notna().sum()
    n_total = len(mapped)
    logger.info(f"  Mapped {n_mapped}/{n_total} cells to ScReNI types")

    # Log unmapped types so we can extend the mapping
    unmapped_types = (
        adata.obs.loc[mapped.isna(), "cell_type_fine"]
        .value_counts()
    )
    if len(unmapped_types) > 0:
        logger.warning(
            f"  Unmapped CellTypist types ({mapped.isna().sum()} cells):\n"
            f"{unmapped_types.to_string()}"
        )

    adata.obs["cell_type"] = mapped

    # Report final counts
    ct_counts = adata.obs["cell_type"].value_counts()
    logger.info(f"  Final cell type counts:\n{ct_counts.to_string()}")

    # Restore raw counts
    adata.X = adata.layers["counts"]
    del adata.layers["counts"]

    return adata


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
) -> dict[str, ad.AnnData]:
    """Load all datasets, apply filters, and save as .h5ad.

    For retinal data, the correct pipeline order is:
    1. Load RNA with Clark 2019 labels (no final cell type filter)
    2. Load ATAC with Lyu 2021 labels (filter to RPC1/2/3/MG)
    3. Filter both to shared timepoints
    4. Save both as h5ad

    Final RNA cell type labels (RPC1/2/3/MG) are assigned LATER during
    integration (Phase 1) by transferring ATAC labels via nearest-neighbor
    matching. See integration.match_rna_atac_neighbors().

    Parameters
    ----------
    data_root
        Root data directory (e.g., ``data/``).
    output_dir
        Where to save processed .h5ad files (e.g., ``data/processed/``).

    Returns
    -------
    Dict with keys 'retinal_atac', 'retinal_rna', 'pbmc_rna', 'pbmc_atac'.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # --- Retinal ---
    ret_dir = Path(data_root) / "retinal_GSE181251"
    scrna_dir = ret_dir / "scRNAseq_clark2019"

    # Load RNA with coarse pre-filter (Clark labels, NOT final ScReNI labels)
    rna_full = None
    rna_timepoints: set[str] = set()
    if scrna_dir.exists():
        rna_full = load_retinal_rna(
            scrna_dir,
            prefilter_types=["Early RPCs", "Late RPCs", "Muller Glia",
                             "Neurogenic Cells"],
        )
        if "timepoint" in rna_full.obs.columns:
            rna_timepoints = set(rna_full.obs["timepoint"].dropna().unique())
    else:
        logger.warning(f"  Retinal RNA dir not found: {scrna_dir}")

    # Load ATAC (filter to RPC1/2/3/MG - these labels exist in ATAC data)
    atac_full = None
    atac_timepoints: set[str] = set()
    if ret_dir.exists():
        atac_full = load_retinal_atac(ret_dir)
        atac_timepoints = set(atac_full.obs["timepoint"].dropna().unique())
    else:
        logger.warning(f"  Retinal ATAC dir not found: {ret_dir}")

    # Filter both to shared timepoints
    if rna_timepoints and atac_timepoints:
        shared_tp = get_shared_timepoints(rna_timepoints, atac_timepoints)
        logger.info(f"Re-filtering to {len(shared_tp)} shared timepoints")

        if rna_full is not None:
            mask = rna_full.obs["timepoint"].isin(shared_tp)
            rna_full = rna_full[mask].copy()
            logger.info(f"  Retinal RNA after timepoint filter: {rna_full.n_obs}")

        if atac_full is not None:
            mask = atac_full.obs["timepoint"].isin(shared_tp)
            atac_full = atac_full[mask].copy()
            logger.info(f"  Retinal ATAC after timepoint filter: {atac_full.n_obs}")
            _check_counts(
                atac_full, EXPECTED_COUNTS["retinal_atac"], "retinal_atac"
            )

    # Save retinal data
    # NOTE: RNA has Clark labels, NOT final ScReNI labels. RNA cell type
    # count checks happen after integration (Phase 1).
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
        pbmc_rna = annotate_pbmc_cell_types(pbmc_rna)

        # Filter to known cell types
        mask = pbmc_rna.obs["cell_type"].isin(PBMC_CELL_TYPES)
        pbmc_rna = pbmc_rna[mask].copy()
        pbmc_atac = pbmc_atac[mask].copy()
        logger.info(f"  PBMC filtered to {len(PBMC_CELL_TYPES)} types: {pbmc_rna.n_obs} cells")
        _check_counts(pbmc_rna, EXPECTED_COUNTS["pbmc"], "pbmc")

        # Transfer cell type to ATAC (paired data, same cells)
        pbmc_atac.obs["cell_type"] = pbmc_rna.obs["cell_type"].values

        pbmc_rna.write_h5ad(output_dir / "pbmc_rna.h5ad")
        pbmc_atac.write_h5ad(output_dir / "pbmc_atac.h5ad")
        results["pbmc_rna"] = pbmc_rna
        results["pbmc_atac"] = pbmc_atac
        logger.info(f"  Saved pbmc_rna.h5ad: {pbmc_rna.shape}")
        logger.info(f"  Saved pbmc_atac.h5ad: {pbmc_atac.shape}")
    else:
        logger.warning(f"  PBMC dir not found: {pbmc_dir}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_and_save_all(data_root=Path("data"), output_dir=Path("data/processed"))
