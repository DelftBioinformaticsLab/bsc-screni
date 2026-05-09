"""Phase 0: Load and inspect the SEA-AD MTG dataset.

SEA-AD provides two h5ad files that each pool ALL donors:
  - RNA h5ad: singleome snRNA-seq (84 donors) + multiome RNA (28 donors)
  - ATAC h5ad: singleome snATAC-seq + multiome ATAC

A metadata column in each file distinguishes multiome from singleome nuclei.
This module:
  1. Inspects the schema (backed mode, no .X loaded)
  2. Audits whether multiome barcode pairing is recoverable
  3. Classifies donors by available modalities
  4. Loads, filters to chosen cell types, splits by modality, saves
  5. Propagates AD/control condition from ADNC + co-pathology covariates,
     filters donors by eligibility for differential analysis (SQ1)

The inspection step (1-2) must run BEFORE committing to any downstream
pipeline design. The pairing audit determines whether the paired WNN
branch is viable or whether all data must be processed as unpaired.
"""

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Cell types to keep for ScReNI analysis.
# AD-relevant subclasses with adequate multiome cell counts in SEA-AD.
# These are values in the Subclass column (exact name confirmed by inspection).
SEAAD_CELL_TYPES = ["Microglia-PVM", "Astrocyte", "Oligodendrocyte", "L2/3 IT"]

# Minimum cells per modality per donor for per-donor unpaired alignment
MIN_CELLS_PER_DONOR = 50

# AD/control binning for SQ1.  Forced by the donor counts in SEA-AD paired
# multiome (only 3 'Not AD' donors exist; only 2 have >=50 cells in both
# target cell types — a 2-vs-9 Wilcoxon is statistically toothless).
# We pool the lower two ADNC categories into 'control' and the upper two
# into 'ad'.  See progress_log.md (2026-05-09) for the full derivation.
ADNC_COL = "Overall AD neuropathological Change"
ADNC_TO_CONDITION = {
    "Not AD": "control",
    "Low": "control",
    "Intermediate": "ad",
    "High": "ad",
}

# Co-pathology obs columns + the value indicating absence.  Anything other
# than the absence value is collapsed to *_present = True.
LATE_COL = "LATE"
LATE_ABSENT = "Not Identified"
LBD_COL = "Highest Lewy Body Disease"
# LBD's "absent" label includes a parenthetical and varies by row.
LBD_ABSENT_PREFIX = "Not Identified"

# Cell-level inclusion criterion column (Allen Institute QC flag).
USED_IN_ANALYSIS_COL = "Used in analysis"


# =========================================================================
#  4a. Schema inspection
# =========================================================================


def inspect_seaad(
    rna_path: Path | str,
    atac_path: Path | str,
) -> dict:
    """Inspect SEA-AD h5ad files in backed mode (no .X loaded).

    Prints and returns schema information needed to configure the
    rest of the pipeline: column names, modality values, donor counts.

    Returns
    -------
    Dict with keys: rna_obs_cols, atac_obs_cols, rna_shape, atac_shape,
    and any identified column mappings.
    """
    rna_path = Path(rna_path)
    atac_path = Path(atac_path)

    info = {}

    for label, path in [("RNA", rna_path), ("ATAC", atac_path)]:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  {label}: {path.name}")
        logger.info(f"{'=' * 60}")

        adata = ad.read_h5ad(path, backed="r")
        info[f"{label.lower()}_shape"] = adata.shape
        info[f"{label.lower()}_obs_cols"] = adata.obs.columns.tolist()
        info[f"{label.lower()}_var_cols"] = adata.var.columns.tolist()

        logger.info(f"  Shape: {adata.shape}")

        # obs columns + dtypes
        logger.info(f"\n  obs columns ({len(adata.obs.columns)}):")
        for col in adata.obs.columns:
            dtype = adata.obs[col].dtype
            logger.info(f"    {col}: {dtype}")

        # var columns
        logger.info(f"\n  var columns ({len(adata.var.columns)}):")
        for col in adata.var.columns:
            logger.info(f"    {col}: {adata.var[col].dtype}")

        # First few var_names (peak format for ATAC)
        logger.info(f"\n  First 5 var_names: {adata.var_names[:5].tolist()}")

        # obsm / uns keys
        logger.info(f"  obsm keys: {list(adata.obsm.keys())}")
        logger.info(f"  uns keys: {list(adata.uns.keys())}")

        # Value counts for categorical obs columns
        logger.info(f"\n  Categorical value counts:")
        for col in adata.obs.columns:
            if hasattr(adata.obs[col], "cat") or adata.obs[col].dtype == "object":
                vc = adata.obs[col].value_counts()
                n_unique = len(vc)
                logger.info(f"\n    {col} ({n_unique} unique):")
                # Show all if <=30, otherwise top 15 + bottom 5
                if n_unique <= 30:
                    for val, count in vc.items():
                        logger.info(f"      {val}: {count:,}")
                else:
                    for val, count in vc.head(15).items():
                        logger.info(f"      {val}: {count:,}")
                    logger.info(f"      ... ({n_unique - 20} more)")
                    for val, count in vc.tail(5).items():
                        logger.info(f"      {val}: {count:,}")

        # Layers (where raw counts might be)
        if adata.layers:
            logger.info(f"\n  layers keys: {list(adata.layers.keys())}")
        else:
            logger.info(f"\n  layers: none")

        # X_normalization (if present)
        if "X_normalization" in adata.uns:
            logger.info(f"  X_normalization: {adata.uns['X_normalization']}")

        # Donor x method cross-tab (critical for understanding overlap)
        if "method" in adata.obs.columns and "Donor ID" in adata.obs.columns:
            crosstab = pd.crosstab(adata.obs["Donor ID"], adata.obs["method"])
            logger.info(f"\n  Donor x method cross-tab:\n{crosstab.to_string()}")

        adata.file.close()

    return info


# =========================================================================
#  4b. Pairing audit
# =========================================================================


def audit_multiome_pairing(
    rna_path: Path | str,
    atac_path: Path | str,
    modality_col: str,
    multiome_value: str,
) -> dict:
    """Test whether multiome RNA and ATAC cells can be paired by barcode.

    Tries several plausible key combinations to find matching cells
    across the two h5ad files. Reports overlap statistics.

    Parameters
    ----------
    rna_path, atac_path
        Paths to the SEA-AD h5ad files.
    modality_col
        Column name in obs that distinguishes multiome from singleome.
    multiome_value
        Value in modality_col that identifies multiome cells.

    Returns
    -------
    Dict with: n_rna_multi, n_atac_multi, best_key, best_overlap,
    best_overlap_pct, all_results (list of dicts per key tried).
    """
    logger.info("\n=== Multiome Pairing Audit ===")

    rna = ad.read_h5ad(rna_path, backed="r")
    atac = ad.read_h5ad(atac_path, backed="r")

    rna_multi = rna.obs[rna.obs[modality_col] == multiome_value]
    atac_multi = atac.obs[atac.obs[modality_col] == multiome_value]

    n_rna = len(rna_multi)
    n_atac = len(atac_multi)
    logger.info(f"  Multiome RNA cells: {n_rna:,}")
    logger.info(f"  Multiome ATAC cells: {n_atac:,}")

    results = []

    # Strategy 1: obs_names (barcodes) directly
    rna_names = set(rna_multi.index)
    atac_names = set(atac_multi.index)
    shared = rna_names & atac_names
    pct = len(shared) / max(min(n_rna, n_atac), 1) * 100
    logger.info(f"\n  Key: obs_names only")
    logger.info(f"    Shared: {len(shared):,} / {min(n_rna, n_atac):,} ({pct:.1f}%)")
    results.append({"key": "obs_names", "shared": len(shared), "pct": pct})

    # Strategy 2-N: compound keys using obs columns + obs_names
    rna_cols = set(rna_multi.columns)
    atac_cols = set(atac_multi.columns)

    candidate_key_cols = [
        ["sample_id"],
        ["bc"],
        ["sample_name"],
        ["library_prep"],
        ["Donor ID"],
        ["Donor ID", "bc"],
        ["sample_name", "bc"],
        ["library_prep", "bc"],
    ]

    for key_cols in candidate_key_cols:
        if not all(c in rna_cols and c in atac_cols for c in key_cols):
            continue

        # Build compound key: col1_val:col2_val:...:barcode
        def _make_keys(obs_df):
            parts = [obs_df[c].astype(str) for c in key_cols]
            prefix = parts[0]
            for p in parts[1:]:
                prefix = prefix + ":" + p
            return set(prefix + ":" + obs_df.index.astype(str))

        rna_keys = _make_keys(rna_multi)
        atac_keys = _make_keys(atac_multi)
        shared = rna_keys & atac_keys
        pct = len(shared) / max(min(n_rna, n_atac), 1) * 100
        key_name = "+".join(key_cols) + "+barcode"
        logger.info(f"\n  Key: {key_name}")
        logger.info(f"    Shared: {len(shared):,} / {min(n_rna, n_atac):,} ({pct:.1f}%)")
        results.append({"key": key_name, "shared": len(shared), "pct": pct})

    rna.file.close()
    atac.file.close()

    # Find best
    best = max(results, key=lambda r: r["shared"])
    logger.info(f"\n  Best key: {best['key']} ({best['shared']:,} shared, {best['pct']:.1f}%)")

    if best["pct"] >= 90:
        logger.info("  -> Pairing VIABLE: proceed with paired WNN branch")
    elif best["pct"] >= 50:
        logger.info("  -> Pairing PARTIAL: investigate further before committing")
    else:
        logger.info("  -> Pairing NOT RECOVERABLE: use unpaired (per-donor Harmony) for all")

    return {
        "n_rna_multi": n_rna,
        "n_atac_multi": n_atac,
        "best_key": best["key"],
        "best_overlap": best["shared"],
        "best_overlap_pct": best["pct"],
        "all_results": results,
    }


# =========================================================================
#  4c. Donor classification
# =========================================================================


def classify_donors(
    rna: ad.AnnData,
    atac: ad.AnnData,
    donor_col: str = "Donor ID",
    min_cells: int = MIN_CELLS_PER_DONOR,
) -> pd.DataFrame:
    """Classify donors by available modalities and cell counts.

    Parameters
    ----------
    rna, atac
        AnnData objects with donor_col in obs.
    donor_col
        Column containing donor identifiers.
    min_cells
        Minimum cells per modality to consider a donor usable.

    Returns
    -------
    DataFrame indexed by donor with columns: n_rna, n_atac,
    has_both, skip_reason.
    """
    logger.info("Classifying donors by modality availability...")

    rna_counts = rna.obs[donor_col].value_counts().to_dict()
    atac_counts = atac.obs[donor_col].value_counts().to_dict()

    all_donors = sorted(set(rna_counts) | set(atac_counts))

    rows = []
    for donor in all_donors:
        n_rna = rna_counts.get(donor, 0)
        n_atac = atac_counts.get(donor, 0)

        if n_rna == 0:
            skip = "no RNA cells"
        elif n_atac == 0:
            skip = "no ATAC cells"
        elif n_rna < min_cells:
            skip = f"too few RNA cells ({n_rna} < {min_cells})"
        elif n_atac < min_cells:
            skip = f"too few ATAC cells ({n_atac} < {min_cells})"
        else:
            skip = ""

        rows.append({
            "donor_id": donor,
            "n_rna": n_rna,
            "n_atac": n_atac,
            "has_both": skip == "",
            "skip_reason": skip,
        })

    df = pd.DataFrame(rows).set_index("donor_id")

    n_both = df["has_both"].sum()
    n_rna_only = ((df["n_rna"] > 0) & (df["n_atac"] == 0)).sum()
    n_atac_only = ((df["n_rna"] == 0) & (df["n_atac"] > 0)).sum()
    n_skip = (~df["has_both"] & (df["n_rna"] > 0) & (df["n_atac"] > 0)).sum()

    logger.info(
        f"  {len(df)} total donors: {n_both} with both modalities, "
        f"{n_rna_only} RNA-only, {n_atac_only} ATAC-only, "
        f"{n_skip} skipped (below threshold)"
    )

    return df


# =========================================================================
#  4d. Load, filter, split
# =========================================================================


def _load_subset(
    h5ad_path: Path | str,
    label: str,
    cell_type_col: str,
    cell_types: list[str],
) -> ad.AnnData:
    """Load an h5ad in backed mode, filter to cell types, then to_memory().

    This avoids loading the full .X matrix when we only keep a subset of cells.
    """
    logger.info(f"Loading SEA-AD {label} from {Path(h5ad_path).name} (backed)...")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    logger.info(f"  Full {label}: {adata.shape}")

    if cell_type_col not in adata.obs.columns:
        adata.file.close()
        raise KeyError(
            f"Column '{cell_type_col}' not found in {label} obs. "
            f"Available: {adata.obs.columns.tolist()}"
        )

    mask = adata.obs[cell_type_col].isin(cell_types)
    n_keep = mask.sum()
    logger.info(f"  Keeping {n_keep:,} / {adata.n_obs:,} cells -> loading into memory...")

    subset = adata[mask].to_memory()
    adata.file.close()

    logger.info(f"  {label} subset: {subset.shape}")
    return subset


def load_seaad(
    rna_path: Path | str,
    atac_path: Path | str,
    cell_types: list[str] = SEAAD_CELL_TYPES,
    cell_type_col: str = "Subclass",
) -> tuple[ad.AnnData, ad.AnnData]:
    """Load SEA-AD h5ad files and filter to selected cell types.

    Parameters
    ----------
    rna_path, atac_path
        Paths to the full SEA-AD h5ad files.
    cell_types
        Values in cell_type_col to keep.
    cell_type_col
        Column in obs containing cell type labels.

    Returns
    -------
    (rna, atac) AnnDatas filtered to selected cell types,
    with standardized obs column 'cell_type'.
    """
    # Load in backed mode first, filter on obs, then load only the
    # subset into memory.  This avoids materializing the full .X for
    # 1.4M x 36K (RNA) or 516K x 219K (ATAC) matrices.
    rna = _load_subset(rna_path, "RNA", cell_type_col, cell_types)
    atac = _load_subset(atac_path, "ATAC", cell_type_col, cell_types)

    # Standardize cell type column
    rna.obs["cell_type"] = rna.obs[cell_type_col].values
    atac.obs["cell_type"] = atac.obs[cell_type_col].values

    logger.info(
        f"  Filtered RNA: {rna.shape} "
        f"({rna.obs['cell_type'].value_counts().to_dict()})"
    )
    logger.info(
        f"  Filtered ATAC: {atac.shape} "
        f"({atac.obs['cell_type'].value_counts().to_dict()})"
    )

    # Verify raw counts
    for label, adata in [("RNA", rna), ("ATAC", atac)]:
        sample = adata.X[:100] if not hasattr(adata.X, "toarray") else adata.X[:100].toarray()
        if np.any(sample != sample.astype(int)):
            logger.warning(f"  {label} .X may not contain integer counts!")
        else:
            logger.info(f"  {label} .X contains integer counts")

    # Verify peak names are chr-prefixed (ATAC only)
    n_chr = atac.var_names.str.startswith("chr").sum()
    logger.info(f"  ATAC peaks with chr prefix: {n_chr} / {atac.n_vars}")
    if n_chr < atac.n_vars * 0.9:
        logger.warning("  Less than 90% of peaks have chr prefix!")

    return rna, atac


def split_by_modality(
    rna: ad.AnnData,
    atac: ad.AnnData,
    modality_col: str,
    multiome_value: str,
    pairing_key_col: str | None = None,
) -> dict[str, ad.AnnData]:
    """Split RNA and ATAC into multiome (paired) and singleome (unpaired).

    Parameters
    ----------
    rna, atac
        Full SEA-AD AnnData objects with modality_col in obs.
    modality_col
        Column distinguishing multiome from singleome.
    multiome_value
        Value identifying multiome cells.
    pairing_key_col
        If set, use this obs column + barcode to build the pairing key
        for multiome cells. If None, attempt pairing with barcodes only.

    Returns
    -------
    Dict with keys: 'paired_rna', 'paired_atac' (if pairing works),
    'unpaired_rna', 'unpaired_atac'.
    """
    logger.info(f"Splitting by modality (column: {modality_col})...")

    rna_multi_mask = rna.obs[modality_col] == multiome_value
    atac_multi_mask = atac.obs[modality_col] == multiome_value

    rna_multi = rna[rna_multi_mask].copy()
    atac_multi = atac[atac_multi_mask].copy()
    rna_single = rna[~rna_multi_mask].copy()
    atac_single = atac[~atac_multi_mask].copy()

    logger.info(
        f"  Multiome: {rna_multi.n_obs} RNA, {atac_multi.n_obs} ATAC\n"
        f"  Singleome: {rna_single.n_obs} RNA, {atac_single.n_obs} ATAC"
    )

    result = {
        "unpaired_rna": rna_single,
        "unpaired_atac": atac_single,
    }

    # Attempt barcode matching for paired branch
    if rna_multi.n_obs > 0 and atac_multi.n_obs > 0:
        if pairing_key_col is not None:
            rna_keys = (
                rna_multi.obs[pairing_key_col].astype(str)
                + ":" + rna_multi.obs_names.astype(str)
            )
            atac_keys = (
                atac_multi.obs[pairing_key_col].astype(str)
                + ":" + atac_multi.obs_names.astype(str)
            )
        else:
            rna_keys = pd.Series(rna_multi.obs_names, index=rna_multi.obs_names)
            atac_keys = pd.Series(atac_multi.obs_names, index=atac_multi.obs_names)

        shared = set(rna_keys.values) & set(atac_keys.values)
        overlap_pct = len(shared) / max(min(len(rna_keys), len(atac_keys)), 1) * 100

        logger.info(
            f"  Barcode overlap: {len(shared):,} / "
            f"{min(len(rna_keys), len(atac_keys)):,} ({overlap_pct:.1f}%)"
        )

        if overlap_pct >= 90:
            # Reindex both to shared barcodes
            rna_key_to_idx = dict(zip(rna_keys.values, rna_multi.obs_names))
            atac_key_to_idx = dict(zip(atac_keys.values, atac_multi.obs_names))

            shared_sorted = sorted(shared)
            rna_shared_idx = [rna_key_to_idx[k] for k in shared_sorted]
            atac_shared_idx = [atac_key_to_idx[k] for k in shared_sorted]

            result["paired_rna"] = rna_multi[rna_shared_idx].copy()
            result["paired_atac"] = atac_multi[atac_shared_idx].copy()

            logger.info(f"  Paired branch: {len(shared_sorted)} matched cells")

            # Verify cell type agreement
            rna_ct = result["paired_rna"].obs["cell_type"].values
            atac_ct = result["paired_atac"].obs["cell_type"].values
            agree = (rna_ct == atac_ct).mean() * 100
            logger.info(f"  Cell type agreement: {agree:.1f}%")
            if agree < 95:
                logger.warning(
                    f"  Cell type agreement below 95%! Check annotation consistency."
                )
        else:
            logger.warning(
                f"  Barcode overlap too low ({overlap_pct:.1f}%). "
                f"Treating multiome cells as unpaired."
            )
            # Add multiome cells to unpaired pools
            result["unpaired_rna"] = ad.concat(
                [rna_single, rna_multi], merge="same"
            )
            result["unpaired_atac"] = ad.concat(
                [atac_single, atac_multi], merge="same"
            )

    # Log overlapping donors
    if "paired_rna" in result:
        paired_donors = set(result["paired_rna"].obs["Donor ID"].unique()
                           if "donor_id" in result["paired_rna"].obs.columns else [])
        unpaired_rna_donors = set(result["unpaired_rna"].obs["Donor ID"].unique()
                                  if "donor_id" in result["unpaired_rna"].obs.columns else [])
        unpaired_atac_donors = set(result["unpaired_atac"].obs["Donor ID"].unique()
                                   if "donor_id" in result["unpaired_atac"].obs.columns else [])
        overlap = paired_donors & (unpaired_rna_donors | unpaired_atac_donors)
        if overlap:
            logger.info(
                f"  Overlapping donors (in both branches): {len(overlap)} "
                f"— kept in both for comparison"
            )

    return result


# =========================================================================
#  4e. QC summary
# =========================================================================


def qc_summary(
    adata_dict: dict[str, ad.AnnData],
    donor_col: str = "Donor ID",
) -> None:
    """Print QC summary for all splits."""
    logger.info("\n=== QC Summary ===")

    for name, adata in adata_dict.items():
        logger.info(f"\n  {name}: {adata.shape}")

        # Cell type counts
        if "cell_type" in adata.obs.columns:
            ct_counts = adata.obs["cell_type"].value_counts()
            logger.info(f"  Cell types: {ct_counts.to_dict()}")

        # Donor counts
        if donor_col in adata.obs.columns:
            n_donors = adata.obs[donor_col].nunique()
            logger.info(f"  Donors: {n_donors}")

        # ADNC distribution (if available)
        for adnc_col in ["Overall AD neuropathological Change",
                        "Continuous Pseudo-progression Score", "ADNC"]:
            if adnc_col in adata.obs.columns:
                vals = adata.obs[adnc_col].dropna()
                if len(vals) > 0:
                    logger.info(
                        f"  {adnc_col}: {vals.value_counts().to_dict()}"
                        if vals.dtype == "object" or hasattr(vals, "cat")
                        else f"  {adnc_col}: mean={vals.mean():.2f}, "
                             f"range=[{vals.min():.2f}, {vals.max():.2f}]"
                    )
                break


# =========================================================================
#  4f. Differential analysis prep (SQ1)
# =========================================================================


def add_condition_column(
    adata: ad.AnnData,
    adnc_col: str = ADNC_COL,
    inplace: bool = True,
) -> ad.AnnData:
    """Add a 'condition' obs column with values 'control' / 'ad' / NaN.

    The mapping pools 'Not AD' + 'Low' as control and 'Intermediate' + 'High'
    as ad — see progress_log.md (2026-05-09) for the donor-count rationale.
    Cells whose ADNC value is missing or unrecognised get NaN, which lets
    downstream filters drop them cleanly via ``.dropna()``.

    Parameters
    ----------
    adata
        AnnData with ``adnc_col`` in ``.obs``.
    adnc_col
        Name of the ADNC column.  Default matches SEA-AD's exact column name.
    inplace
        Modify ``adata`` in place (also returned).  Set False to copy first.

    Returns
    -------
    The (possibly copied) AnnData with ``obs['condition']`` set as a
    pandas Categorical with categories ``['control', 'ad']``.
    """
    if not inplace:
        adata = adata.copy()
    if adnc_col not in adata.obs.columns:
        raise KeyError(
            f"ADNC column {adnc_col!r} not found in obs. "
            f"Available: {list(adata.obs.columns)[:20]}..."
        )
    raw = adata.obs[adnc_col].astype(str)
    mapped = raw.map(ADNC_TO_CONDITION)
    adata.obs["condition"] = pd.Categorical(
        mapped, categories=["control", "ad"], ordered=False
    )
    n_ctrl = int((adata.obs["condition"] == "control").sum())
    n_ad = int((adata.obs["condition"] == "ad").sum())
    n_drop = int(mapped.isna().sum())
    logger.info(
        f"  condition: control={n_ctrl} ad={n_ad} unmapped={n_drop} "
        f"(unmapped cells will be dropped by downstream filters)"
    )
    return adata


def add_copathology_columns(
    adata: ad.AnnData,
    late_col: str = LATE_COL,
    lbd_col: str = LBD_COL,
    inplace: bool = True,
) -> ad.AnnData:
    """Add boolean 'LATE_present' and 'LBD_present' obs columns.

    These are covariates for the per-edge differential regression
    (``differential.py``).  Each is True iff the corresponding raw column
    indicates *any* level of co-pathology beyond the 'Not Identified' label.

    Missing values are treated as False (absent) — SEA-AD's missingness
    pattern is rare and the conservative reading keeps donors in the cohort.
    """
    if not inplace:
        adata = adata.copy()

    if late_col in adata.obs.columns:
        late_raw = adata.obs[late_col].astype(str)
        adata.obs["LATE_present"] = ~late_raw.str.startswith(LATE_ABSENT, na=False)
    else:
        logger.warning(f"  {late_col!r} not in obs; setting LATE_present = False")
        adata.obs["LATE_present"] = False

    if lbd_col in adata.obs.columns:
        lbd_raw = adata.obs[lbd_col].astype(str)
        adata.obs["LBD_present"] = ~lbd_raw.str.startswith(
            LBD_ABSENT_PREFIX, na=False
        )
    else:
        logger.warning(f"  {lbd_col!r} not in obs; setting LBD_present = False")
        adata.obs["LBD_present"] = False

    n_late = int(adata.obs["LATE_present"].sum())
    n_lbd = int(adata.obs["LBD_present"].sum())
    logger.info(
        f"  co-pathology: LATE_present={n_late} cells, LBD_present={n_lbd} cells"
    )
    return adata


def select_eligible_donors(
    adata: ad.AnnData,
    cell_type: str,
    min_cells_per_donor: int = MIN_CELLS_PER_DONOR,
    cell_type_col: str = "cell_type",
    donor_col: str = "Donor ID",
    require_condition: bool = True,
) -> pd.DataFrame:
    """Identify donors with adequate cells of one cell type for SQ1.

    For each donor, count cells matching ``cell_type`` and report the donor
    if the count is >= ``min_cells_per_donor``.  This is the per-cell-type
    eligibility filter that protects the per-edge regression from donors
    whose pseudobulk would be estimated from a handful of cells.

    Parameters
    ----------
    adata
        AnnData with ``donor_col``, ``cell_type_col``, ``condition`` in obs.
        Run :func:`add_condition_column` first.
    cell_type
        Single value of ``cell_type_col`` to count over (e.g. "Microglia-PVM").
    min_cells_per_donor
        Inclusive lower bound on cells per donor.
    require_condition
        If True, only donors mapped to control/ad are returned.  Set False
        to include all donors regardless of ADNC binning.

    Returns
    -------
    DataFrame with one row per eligible donor and columns:
    ``donor_id``, ``condition``, ``n_cells``, ``age``, ``sex``,
    ``LATE_present``, ``LBD_present``.  Sorted by donor_id.
    """
    if cell_type_col not in adata.obs.columns:
        raise KeyError(f"{cell_type_col!r} not in obs")
    if donor_col not in adata.obs.columns:
        raise KeyError(f"{donor_col!r} not in obs")

    obs = adata.obs
    mask = obs[cell_type_col] == cell_type
    if require_condition:
        if "condition" not in obs.columns:
            raise KeyError(
                "obs has no 'condition' column. "
                "Call add_condition_column(adata) first."
            )
        mask = mask & obs["condition"].notna()

    sub = obs.loc[mask]
    if len(sub) == 0:
        logger.warning(
            f"No cells matched cell_type={cell_type!r} (require_condition={require_condition})"
        )
        return pd.DataFrame(
            columns=["donor_id", "condition", "n_cells", "age", "sex",
                     "LATE_present", "LBD_present"]
        )

    counts = sub.groupby(donor_col, observed=True).size()
    eligible = counts[counts >= min_cells_per_donor].index.tolist()

    rows = []
    for d in eligible:
        d_cells = sub.loc[sub[donor_col] == d]
        first = d_cells.iloc[0]
        rows.append({
            "donor_id": d,
            "condition": str(first.get("condition", "")),
            "n_cells": int(len(d_cells)),
            "age": float(first["Age at Death"])
                if "Age at Death" in d_cells.columns
                and pd.notna(first["Age at Death"]) else np.nan,
            "sex": str(first["Sex"]) if "Sex" in d_cells.columns else "",
            "LATE_present": bool(first.get("LATE_present", False)),
            "LBD_present": bool(first.get("LBD_present", False)),
        })

    df = pd.DataFrame(rows).sort_values("donor_id").reset_index(drop=True)
    n_ctrl = int((df["condition"] == "control").sum())
    n_ad = int((df["condition"] == "ad").sum())
    logger.info(
        f"  {cell_type}: {len(df)} eligible donors "
        f"(>= {min_cells_per_donor} cells) — control={n_ctrl} ad={n_ad}"
    )
    return df


def subsample_cells_per_donor(
    adata: ad.AnnData,
    n_per_donor: int,
    donor_col: str = "Donor ID",
    cell_type: str | None = None,
    cell_type_col: str = "cell_type",
    seed: int = 0,
) -> ad.AnnData:
    """Subsample up to ``n_per_donor`` cells per donor.

    If a donor has fewer than ``n_per_donor`` cells, all of them are kept.
    Used to cap the wScReNI input size per (donor x cell type) slice so
    inference stays within the cluster job's wall-clock budget.

    Parameters
    ----------
    adata
        AnnData filtered to the donors of interest.
    n_per_donor
        Target cells per donor.  E.g. 50 for the standard SeaAD pilot.
    donor_col
        Donor identifier column.
    cell_type
        Optional pre-filter to one cell type before sampling.
    cell_type_col
        Cell type column name.
    seed
        Reproducible seed for the per-donor random draw.

    Returns
    -------
    AnnData containing the subsampled cells, in donor-grouped order.
    """
    rng = np.random.default_rng(seed)

    if cell_type is not None:
        if cell_type_col not in adata.obs.columns:
            raise KeyError(f"{cell_type_col!r} not in obs")
        adata = adata[adata.obs[cell_type_col] == cell_type].copy()

    if donor_col not in adata.obs.columns:
        raise KeyError(f"{donor_col!r} not in obs")

    chosen_idx: list[int] = []
    for donor, grp in adata.obs.groupby(donor_col, observed=True):
        positions = np.flatnonzero(adata.obs[donor_col].values == donor)
        if len(positions) <= n_per_donor:
            chosen_idx.extend(positions.tolist())
        else:
            picked = rng.choice(positions, size=n_per_donor, replace=False)
            chosen_idx.extend(sorted(picked.tolist()))

    chosen_idx = sorted(set(chosen_idx))
    sub = adata[chosen_idx].copy()
    logger.info(
        f"  subsampled to {sub.n_obs} cells across "
        f"{sub.obs[donor_col].nunique()} donors "
        f"(target {n_per_donor}/donor, cell_type={cell_type})"
    )
    return sub


# =========================================================================
#  Main
# =========================================================================


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data_dir = Path("data/seaad")
    out_dir = Path("data/processed/seaad")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find h5ad files
    rna_files = sorted(data_dir.glob("*RNA*h5ad")) + sorted(data_dir.glob("*rna*h5ad"))
    atac_files = sorted(data_dir.glob("*ATAC*h5ad")) + sorted(data_dir.glob("*atac*h5ad"))

    if not rna_files or not atac_files:
        logger.error(
            f"Could not find SEA-AD h5ad files in {data_dir}/\n"
            f"  RNA files found: {rna_files}\n"
            f"  ATAC files found: {atac_files}\n"
            f"Run slurm/download_seaad.sh first."
        )
        sys.exit(1)

    rna_path = rna_files[0]
    atac_path = atac_files[0]
    logger.info(f"RNA file: {rna_path}")
    logger.info(f"ATAC file: {atac_path}")

    # ----------------------------------------------------------------
    # Phase 0a: Inspect schema
    # ----------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Phase 0a: Schema Inspection")
    logger.info("=" * 60)

    info = inspect_seaad(rna_path, atac_path)

    # ----------------------------------------------------------------
    # Phase 0b: Pairing audit
    # ----------------------------------------------------------------
    # Column names confirmed from inspection output.
    MODALITY_COL = "method"
    MULTIOME_VALUE = "10xMulti"
    DONOR_COL = "Donor ID"
    CELL_TYPE_COL = "Subclass"

    logger.info("\n" + "=" * 60)
    logger.info("Phase 0b: Multiome Pairing Audit")
    logger.info("=" * 60)

    pairing = audit_multiome_pairing(
        rna_path, atac_path,
        modality_col=MODALITY_COL,
        multiome_value=MULTIOME_VALUE,
    )

    # ----------------------------------------------------------------
    # STOP: Review output above before proceeding
    # ----------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("INSPECTION COMPLETE")
    logger.info("=" * 60)
    logger.info(
        "\nReview the output above and confirm:\n"
        f"  1. Modality column: {MODALITY_COL} (correct?)\n"
        f"  2. Multiome value: {MULTIOME_VALUE} (correct?)\n"
        f"  3. Donor column: {DONOR_COL} (correct?)\n"
        f"  4. Cell type column: {CELL_TYPE_COL} (correct?)\n"
        f"  5. Pairing viable: {pairing['best_overlap_pct']:.1f}% overlap\n"
        "\nTo proceed with loading + splitting, run with --process flag."
    )

    if "--process" not in sys.argv:
        sys.exit(0)

    # ----------------------------------------------------------------
    # Phase 0c: Load RNA, split by modality, save, free memory
    # ----------------------------------------------------------------
    # Process one file at a time so we never hold both full matrices.
    import gc

    logger.info("\n" + "=" * 60)
    logger.info("Phase 0c: Load + split RNA")
    logger.info("=" * 60)

    logger.info("Loading full RNA h5ad...")
    rna = ad.read_h5ad(rna_path)
    rna.obs["cell_type"] = rna.obs[CELL_TYPE_COL]
    logger.info(f"  RNA: {rna.shape}")

    rna_multi_mask = rna.obs[MODALITY_COL] == MULTIOME_VALUE
    rna_obs = rna.obs.copy()  # keep metadata for barcode matching later

    # Save unpaired RNA
    logger.info("  Writing unpaired RNA...")
    rna_unpaired = rna[~rna_multi_mask].copy()
    rna_unpaired.write_h5ad(out_dir / "seaad_unpaired_rna.h5ad")
    logger.info(f"  Saved seaad_unpaired_rna.h5ad: {rna_unpaired.shape}")
    unpaired_rna_obs = rna_unpaired.obs.copy()
    del rna_unpaired

    # Save multiome RNA (before barcode matching — will rewrite after ATAC)
    logger.info("  Writing multiome RNA...")
    rna_multi = rna[rna_multi_mask].copy()
    rna_multi_names = set(rna_multi.obs_names)
    rna_multi.write_h5ad(out_dir / "seaad_paired_rna_all.h5ad")
    logger.info(f"  Saved seaad_paired_rna_all.h5ad: {rna_multi.shape}")
    del rna_multi, rna
    gc.collect()

    # ----------------------------------------------------------------
    # Phase 0d: Load ATAC, split by modality, save, free memory
    # ----------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Phase 0d: Load + split ATAC")
    logger.info("=" * 60)

    logger.info("Loading full ATAC h5ad...")
    atac = ad.read_h5ad(atac_path)
    atac.obs["cell_type"] = atac.obs[CELL_TYPE_COL]
    logger.info(f"  ATAC: {atac.shape}")

    atac_multi_mask = atac.obs[MODALITY_COL] == MULTIOME_VALUE

    # Save unpaired ATAC
    logger.info("  Writing unpaired ATAC...")
    atac_unpaired = atac[~atac_multi_mask].copy()
    atac_unpaired.write_h5ad(out_dir / "seaad_unpaired_atac.h5ad")
    logger.info(f"  Saved seaad_unpaired_atac.h5ad: {atac_unpaired.shape}")
    unpaired_atac_obs = atac_unpaired.obs.copy()
    del atac_unpaired

    # Barcode matching for paired cells
    atac_multi = atac[atac_multi_mask].copy()
    del atac
    gc.collect()

    shared_barcodes = sorted(rna_multi_names & set(atac_multi.obs_names))
    logger.info(
        f"  Barcode matching: {len(shared_barcodes):,} shared "
        f"(RNA multi: {len(rna_multi_names):,}, ATAC multi: {atac_multi.n_obs:,})"
    )

    # Save matched paired ATAC
    logger.info("  Writing paired ATAC...")
    atac_paired = atac_multi[shared_barcodes].copy()
    atac_paired.write_h5ad(out_dir / "seaad_paired_atac.h5ad")
    logger.info(f"  Saved seaad_paired_atac.h5ad: {atac_paired.shape}")

    # Verify cell type agreement
    rna_ct = rna_obs.loc[shared_barcodes, CELL_TYPE_COL].values
    atac_ct = atac_paired.obs[CELL_TYPE_COL].values
    agree = (rna_ct == atac_ct).mean() * 100
    logger.info(f"  Paired cell type agreement: {agree:.1f}%")
    del atac_multi, atac_paired
    gc.collect()

    # Rewrite paired RNA to matched barcodes only
    logger.info("  Rewriting paired RNA to matched barcodes...")
    rna_paired_all = ad.read_h5ad(out_dir / "seaad_paired_rna_all.h5ad")
    rna_paired = rna_paired_all[shared_barcodes].copy()
    rna_paired.write_h5ad(out_dir / "seaad_paired_rna.h5ad")
    logger.info(f"  Saved seaad_paired_rna.h5ad: {rna_paired.shape}")
    del rna_paired_all, rna_paired
    (out_dir / "seaad_paired_rna_all.h5ad").unlink()

    # ----------------------------------------------------------------
    # Phase 0e: Donor classification + QC
    # ----------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Phase 0e: Donor classification")
    logger.info("=" * 60)

    rna_stub = ad.AnnData(obs=unpaired_rna_obs)
    atac_stub = ad.AnnData(obs=unpaired_atac_obs)

    donor_info = classify_donors(rna_stub, atac_stub, donor_col=DONOR_COL)
    donor_info.to_csv(out_dir / "seaad_donor_classification.csv")
    logger.info(f"  Saved donor classification to {out_dir}")

    # QC summary
    logger.info(f"\n  Paired cell types:")
    logger.info(f"    {rna_obs.loc[shared_barcodes, CELL_TYPE_COL].value_counts().to_dict()}")
    logger.info(f"  Unpaired RNA cell types:")
    logger.info(f"    {unpaired_rna_obs['cell_type'].value_counts().to_dict()}")
    logger.info(f"  Unpaired ATAC cell types:")
    logger.info(f"    {unpaired_atac_obs['cell_type'].value_counts().to_dict()}")

    logger.info("\nDone.")
