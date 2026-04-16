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
    logger.info(f"Loading SEA-AD RNA from {Path(rna_path).name}...")
    rna = ad.read_h5ad(rna_path)
    logger.info(f"  Full RNA: {rna.shape}")

    logger.info(f"Loading SEA-AD ATAC from {Path(atac_path).name}...")
    atac = ad.read_h5ad(atac_path)
    logger.info(f"  Full ATAC: {atac.shape}")

    # Filter to chosen cell types
    for label, adata in [("RNA", rna), ("ATAC", atac)]:
        if cell_type_col not in adata.obs.columns:
            raise KeyError(
                f"Column '{cell_type_col}' not found in {label} obs. "
                f"Available: {adata.obs.columns.tolist()}"
            )

    rna_mask = rna.obs[cell_type_col].isin(cell_types)
    atac_mask = atac.obs[cell_type_col].isin(cell_types)

    rna = rna[rna_mask].copy()
    atac = atac[atac_mask].copy()

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
    # Phase 0c-e: Load, filter, split, classify, QC
    # ----------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Phase 0c: Loading and filtering")
    logger.info("=" * 60)

    rna, atac = load_seaad(
        rna_path, atac_path,
        cell_types=SEAAD_CELL_TYPES,
        cell_type_col=CELL_TYPE_COL,
    )

    logger.info("\n" + "=" * 60)
    logger.info("Phase 0d: Splitting by modality")
    logger.info("=" * 60)

    # Use the best pairing key from the audit
    pairing_key = None
    if pairing["best_overlap_pct"] >= 90 and pairing["best_key"] != "obs_names":
        # Extract the obs column name from the compound key name
        # e.g., "sample_id+barcode" -> "sample_id"
        key_parts = pairing["best_key"].replace("+barcode", "").split("+")
        if len(key_parts) == 1 and key_parts[0] in rna.obs.columns:
            pairing_key = key_parts[0]

    splits = split_by_modality(
        rna, atac,
        modality_col=MODALITY_COL,
        multiome_value=MULTIOME_VALUE,
        pairing_key_col=pairing_key,
    )

    logger.info("\n" + "=" * 60)
    logger.info("Phase 0e: Donor classification")
    logger.info("=" * 60)

    donor_info = classify_donors(
        splits["unpaired_rna"],
        splits["unpaired_atac"],
        donor_col=DONOR_COL,
    )

    qc_summary(splits, donor_col=DONOR_COL)

    # ----------------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Saving outputs")
    logger.info("=" * 60)

    if "paired_rna" in splits:
        splits["paired_rna"].write_h5ad(out_dir / "seaad_paired_rna.h5ad")
        splits["paired_atac"].write_h5ad(out_dir / "seaad_paired_atac.h5ad")
        logger.info(f"  Saved paired RNA/ATAC to {out_dir}")

    splits["unpaired_rna"].write_h5ad(out_dir / "seaad_unpaired_rna.h5ad")
    splits["unpaired_atac"].write_h5ad(out_dir / "seaad_unpaired_atac.h5ad")
    logger.info(f"  Saved unpaired RNA/ATAC to {out_dir}")

    donor_info.to_csv(out_dir / "seaad_donor_classification.csv")
    logger.info(f"  Saved donor classification to {out_dir}")

    logger.info("\nDone.")
