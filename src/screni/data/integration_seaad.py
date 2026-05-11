"""Phase 1: Integration of SEA-AD MTG multi-modal data.

Two branches:

Paired (multiome, 28 donors):
    Uses WNN via ``integrate_paired()`` from ``integration.py`` with
    ``batch_key="donor_id"`` for donor-aware HVG selection.

Unpaired (singleome, per-donor alignment):
    For each donor with both singleome RNA and singleome ATAC:
      1. Compute gene activity from ATAC peaks
      2. Align RNA and gene-activity via Harmony(modality)
      3. Pair RNA <-> ATAC cells (nearest cross-modal neighbor)
    No cross-donor batch correction — preserves AD-related variation.
    The resulting kNN graph is block-diagonal (no cross-donor neighbors).

Design decisions:
    - Per-donor alignment avoids conflating batch correction with the
      biological signal (ADNC, genotype) we want to preserve.
    - Overlapping donors (in both multiome and singleome) are kept in
      both branches to enable comparison.
    - Global HVGs are computed once across all singleome RNA donors,
      then used in every per-donor alignment for consistency.
"""

import logging
from pathlib import Path

import anndata as ad
import harmonypy as hm
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.spatial.distance import cdist

from screni.data.integration import integrate_paired
from screni.data.utils import compute_gene_activity
from screni.data.loading_seaad import MIN_CELLS_PER_DONOR
from screni.data.utils import load_gene_annotations

logger = logging.getLogger(__name__)

# Layer names to try for raw counts (SEA-AD ships normalized .X)
_RAW_LAYER_CANDIDATES = ["UMIs", "UMI", "raw", "counts", "raw_counts"]

# Obs columns to keep on integrated outputs. SEA-AD ships ~100 obs columns
# with mixed dtypes that break h5ad writes; we don't need them downstream.
_OBS_KEEP = ["modality", "cell_type", "Donor ID"]


def _swap_to_raw_counts(adata: ad.AnnData, label: str = "") -> bool:
    """Replace .X with raw counts from a layer, if available.

    SEA-AD RNA stores normalized data in .X and raw UMI counts in a `UMIs`
    layer. SEA-AD ATAC has no layers and stores raw integer Tn5-insertion
    counts directly in .X (each fragment contributes 2 because of the two
    Tn5 cuts per fragment) — this is what the downstream rank-based steps
    expect, so no warning is needed in that case.

    Returns
    -------
    True if a raw-counts layer was found and swapped in, False otherwise.
    """
    for candidate in _RAW_LAYER_CANDIDATES:
        if candidate in adata.layers:
            logger.info(f"  {label}: using .layers['{candidate}'] as raw counts")
            adata.X = adata.layers[candidate].copy()
            return True

    # Heuristic: if .X is already integer-valued, accept it silently.
    sample = adata.X[: min(500, adata.n_obs)]
    if hasattr(sample, "toarray"):
        sample = sample.toarray()
    if np.allclose(sample, np.round(sample)):
        logger.info(f"  {label}: .X is integer-valued, using as raw counts")
        return False
    logger.warning(
        f"  {label}: no raw count layer found "
        f"(layers: {list(adata.layers.keys()) if adata.layers else 'none'}) "
        f"and .X is not integer-valued. Using .X as-is."
    )
    return False


def _sanitize_obs_for_h5ad(
    adata: ad.AnnData, keep: list[str] | None = None
) -> None:
    """Make obs h5ad-writable: optionally restrict columns, coerce object to str.

    SEA-AD's full obs has mixed-dtype object columns (NaN+str+bool) that
    h5py rejects with `Can't implicitly convert non-string objects to strings`.
    """
    if keep is not None:
        cols = [c for c in keep if c in adata.obs.columns]
        adata.obs = adata.obs[cols].copy()
    for col in adata.obs.columns:
        if adata.obs[col].dtype == "object":
            adata.obs[col] = adata.obs[col].astype(str)


# =========================================================================
#  Paired branch
# =========================================================================


def integrate_seaad_paired(
    paired_rna: ad.AnnData,
    paired_atac: ad.AnnData,
    n_hvgs: int = 2000,
    n_pcs: int = 20,
    n_neighbors: int = 20,
) -> mu.MuData:
    """Integrate multiome donors via WNN.

    Thin wrapper around ``integrate_paired()`` that passes
    ``batch_key="donor_id"`` for donor-aware HVG selection.
    No additional batch correction beyond HVG selection.

    Parameters
    ----------
    paired_rna
        Multiome RNA AnnData with raw counts, donor_id and cell_type in obs.
    paired_atac
        Multiome ATAC AnnData, same cells as paired_rna.

    Returns
    -------
    MuData with WNN graph, UMAP, clusters, neighbor indices.
    """
    logger.info("=== SEA-AD Paired Integration (WNN) ===")
    logger.info(
        f"  {paired_rna.n_obs} cells, "
        f"{paired_rna.obs['Donor ID'].nunique()} donors"
    )

    # SEA-AD ships normalized data in .X; raw counts are in a layer.
    # integrate_paired() expects raw counts in .X.
    _swap_to_raw_counts(paired_rna, "RNA")
    _swap_to_raw_counts(paired_atac, "ATAC")

    mdata = integrate_paired(
        paired_rna, paired_atac,
        n_hvgs=n_hvgs,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        batch_key="Donor ID",
    )

    return mdata


# =========================================================================
#  Unpaired branch: per-donor alignment
# =========================================================================


def align_donor_unpaired(
    rna_donor: ad.AnnData,
    atac_donor: ad.AnnData,
    gene_annotations: pd.DataFrame,
    hvg_names: list[str] | None,
    donor_id: str,
    n_hvgs: int = 2000,
    n_harmony_dims: int = 20,
    harmony_lambda: float = 1.0,
    n_neighbors: int = 20,
) -> tuple[ad.AnnData, pd.DataFrame]:
    """Align RNA and ATAC for a single singleome donor.

    Steps:
      1. Gene activity from ATAC peaks
      2. HVGs (per-donor if hvg_names is None)
      3. Shared genes = intersection of HVGs and gene activity
      4. Log-normalize both
      5. Concatenate with modality flag
      6. PCA
      7. Harmony(modality)
      8. kNN + UMAP
      9. Pair RNA <-> ATAC (nearest cross-modal neighbor)

    Parameters
    ----------
    rna_donor
        RNA cells from one donor, raw counts.
    atac_donor
        ATAC cells from one donor, raw peak counts.
    gene_annotations
        Gene body coordinates from hg38 GTF.
    hvg_names
        Pre-computed HVGs. If None, computed on this donor's RNA.
    donor_id
        Donor identifier (for logging and output).
    n_hvgs
        Number of HVGs if computing per-donor.
    n_harmony_dims
        PCA dimensions for Harmony.
    harmony_lambda
        Harmony diversity penalty.
    n_neighbors
        k for kNN graph.

    Returns
    -------
    (merged, pairs) where merged is an AnnData with both modalities (obs
    restricted to modality / cell_type / Donor ID) and pairs is a DataFrame
    with columns rna_cell, atac_cell, cell_type, atac_cell_type, distance,
    donor_id, anchor. Pair count equals min(n_rna, n_atac).

    Raises
    ------
    ValueError
        If too few cells or shared genes.
    """
    n_rna = rna_donor.n_obs
    n_atac = atac_donor.n_obs

    if n_rna < MIN_CELLS_PER_DONOR:
        raise ValueError(f"Too few RNA cells ({n_rna} < {MIN_CELLS_PER_DONOR})")
    if n_atac < MIN_CELLS_PER_DONOR:
        raise ValueError(f"Too few ATAC cells ({n_atac} < {MIN_CELLS_PER_DONOR})")

    # Log RNA:ATAC ratio warning
    ratio = n_rna / max(n_atac, 1)
    if ratio > 10 or ratio < 0.1:
        logger.warning(
            f"  Donor {donor_id}: RNA:ATAC ratio = {ratio:.1f} "
            f"({n_rna}:{n_atac}), alignment may be biased"
        )

    # Step 1: Gene activity
    gene_activity = compute_gene_activity(atac_donor, gene_annotations)

    # Step 2: HVGs (per-donor if not provided)
    if hvg_names is None:
        work = rna_donor.copy()
        work.layers["counts"] = work.X.copy()
        sc.pp.normalize_total(work, target_sum=1e4)
        sc.pp.log1p(work)
        sc.pp.filter_genes(work, min_cells=3)
        sc.pp.highly_variable_genes(
            work, n_top_genes=min(n_hvgs, work.n_vars),
            flavor="seurat_v3", layer="counts", span=0.3,
        )
        hvg_names = work.var_names[work.var["highly_variable"]].tolist()
        del work
        logger.info(f"    Per-donor HVGs: {len(hvg_names)}")

    # Step 3: Shared genes
    shared_genes = sorted(
        set(hvg_names) & set(gene_activity.var_names) & set(rna_donor.var_names)
    )
    if len(shared_genes) < 200:
        raise ValueError(
            f"Too few shared genes ({len(shared_genes)} < 200) "
            f"between HVGs and gene activity"
        )

    # Step 3: Log-normalize
    rna_sub = rna_donor[:, shared_genes].copy()
    ga_sub = gene_activity[:, shared_genes].copy()

    sc.pp.normalize_total(rna_sub, target_sum=1e4)
    sc.pp.log1p(rna_sub)
    sc.pp.normalize_total(ga_sub, target_sum=1e4)
    sc.pp.log1p(ga_sub)

    # Step 4: Concatenate.
    # Slim obs to a whitelist before concat: SEA-AD ships ~100 obs columns
    # with mixed dtypes that (a) bloat memory and (b) break h5ad writes
    # downstream. Keep only what's needed for downstream phases.
    rna_sub.obs = rna_sub.obs[[c for c in _OBS_KEEP if c in rna_sub.obs.columns]].copy()
    ga_sub.obs = ga_sub.obs[[c for c in _OBS_KEEP if c in ga_sub.obs.columns]].copy()
    rna_sub.obs["modality"] = "RNA"
    ga_sub.obs["modality"] = "ATAC"

    # Make obs_names unique before concat
    rna_sub.obs_names = [f"rna_{i}" for i in range(rna_sub.n_obs)]
    ga_sub.obs_names = [f"atac_{i}" for i in range(ga_sub.n_obs)]

    # Preserve original obs_names for pairing output
    rna_original_names = rna_donor.obs_names.tolist()
    atac_original_names = atac_donor.obs_names.tolist()
    rna_celltypes = rna_donor.obs["cell_type"].astype(str).values
    atac_celltypes = atac_donor.obs["cell_type"].astype(str).values

    combined = ad.concat([rna_sub, ga_sub], merge="same")

    # Step 5: PCA
    sc.pp.scale(combined, max_value=10)
    sc.tl.pca(combined, n_comps=min(50, len(shared_genes) - 1, combined.n_obs - 1))

    # Step 6: Harmony
    pca_dims = min(n_harmony_dims, combined.obsm["X_pca"].shape[1])
    ho = hm.run_harmony(
        combined.obsm["X_pca"][:, :pca_dims],
        combined.obs,
        "modality",
        max_iter_harmony=20,
        lamb=harmony_lambda,
    )
    harmony_result = np.asarray(ho.Z_corr)
    if harmony_result.shape[0] == combined.n_obs:
        combined.obsm["X_harmony"] = harmony_result
    else:
        combined.obsm["X_harmony"] = harmony_result.T

    # Step 7: kNN + UMAP
    sc.pp.neighbors(combined, use_rep="X_harmony", n_neighbors=n_neighbors)
    sc.tl.umap(combined)

    # Step 8: Pair RNA <-> ATAC in Harmony space.
    # Anchor on the rarer modality: every anchor cell gets one cross-modal
    # neighbor, no dedup, which guarantees pair_count = min(n_rna, n_atac).
    # The old "RNA -> nearest ATAC then dedup-by-ATAC" scheme lost 90%+ of
    # cells per donor in this dataset because RNA cells crowded onto a small
    # subset of ATAC anchors.
    rna_mask = combined.obs["modality"] == "RNA"
    atac_mask = combined.obs["modality"] == "ATAC"
    rna_harmony = combined.obsm["X_harmony"][rna_mask.values]
    atac_harmony = combined.obsm["X_harmony"][atac_mask.values]

    if n_rna <= n_atac:
        # RNA is the rarer modality (or equal): RNA -> nearest ATAC
        anchor = "rna"
        dists = cdist(rna_harmony, atac_harmony, metric="euclidean")
        nearest = dists.argmin(axis=1)
        pair_distances = dists[np.arange(n_rna), nearest]
        pairs = pd.DataFrame({
            "rna_cell": rna_original_names,
            "atac_cell": [atac_original_names[i] for i in nearest],
            "cell_type": rna_celltypes,
            "atac_cell_type": atac_celltypes[nearest],
            "distance": pair_distances,
            "donor_id": donor_id,
            "anchor": anchor,
        })
    else:
        # ATAC is rarer: ATAC -> nearest RNA
        anchor = "atac"
        dists = cdist(atac_harmony, rna_harmony, metric="euclidean")
        nearest = dists.argmin(axis=1)
        pair_distances = dists[np.arange(n_atac), nearest]
        pairs = pd.DataFrame({
            "rna_cell": [rna_original_names[i] for i in nearest],
            "atac_cell": atac_original_names,
            "cell_type": rna_celltypes[nearest],
            "atac_cell_type": atac_celltypes,
            "distance": pair_distances,
            "donor_id": donor_id,
            "anchor": anchor,
        })

    logger.info(
        f"    Anchor: {anchor} (n_rna={n_rna}, n_atac={n_atac}) "
        f"-> {len(pairs)} pairs, mean dist={pairs['distance'].mean():.3f}"
    )

    return combined, pairs


def integrate_seaad_unpaired(
    unpaired_rna: ad.AnnData,
    unpaired_atac: ad.AnnData,
    gene_annotations: pd.DataFrame,
    donor_info: pd.DataFrame,
    donor_col: str = "Donor ID",
    n_hvgs: int = 2000,
    n_neighbors: int = 20,
) -> tuple[ad.AnnData | None, pd.DataFrame, pd.DataFrame]:
    """Integrate all singleome donors via per-donor unpaired alignment.

    Parameters
    ----------
    unpaired_rna, unpaired_atac
        Singleome RNA and ATAC AnnDatas.
    gene_annotations
        Gene body coordinates from hg38 GTF.
    donor_info
        From classify_donors(): has 'has_both' column.
    donor_col
        Column containing donor identifiers.
    n_hvgs
        Number of global HVGs.
    n_neighbors
        k for kNN graph.

    Returns
    -------
    (concatenated_merged, all_pairs, donor_summary) where
    concatenated_merged is an AnnData with block-diagonal kNN graph (or
    None if no donors aligned), all_pairs is a DataFrame of RNA-ATAC pairs
    across all donors, and donor_summary is a per-donor diagnostics
    DataFrame (n_rna, n_atac, n_pairs, anchor, mean_pair_distance,
    cell_type_agreement, status, skip_reason).
    """
    logger.info("=== SEA-AD Unpaired Integration (per-donor Harmony) ===")

    # Step 1: Loop over usable donors
    # HVGs are computed per-donor inside align_donor_unpaired to avoid
    # loading the full 1.2M-cell RNA matrix for a global HVG call.
    usable_donors = donor_info[donor_info["has_both"]].index.tolist()
    logger.info(f"  Usable donors: {len(usable_donors)}")

    all_merged = []
    all_pairs = []
    summary_rows: list[dict] = []

    for i, donor_id in enumerate(usable_donors):
        logger.info(f"\n  [{i + 1}/{len(usable_donors)}] Donor: {donor_id}")

        rna_d = unpaired_rna[unpaired_rna.obs[donor_col] == donor_id].copy()
        atac_d = unpaired_atac[unpaired_atac.obs[donor_col] == donor_id].copy()
        n_rna = rna_d.n_obs
        n_atac = atac_d.n_obs

        logger.info(f"    RNA: {n_rna}, ATAC: {n_atac}")

        row = {
            "donor_id": donor_id,
            "n_rna": n_rna,
            "n_atac": n_atac,
            "min_modality": min(n_rna, n_atac),
            "n_pairs": 0,
            "pair_fraction": float("nan"),
            "anchor": "",
            "mean_pair_distance": float("nan"),
            "cell_type_agreement": float("nan"),
            "status": "aligned",
            "skip_reason": "",
        }

        try:
            merged_d, pairs_d = align_donor_unpaired(
                rna_d, atac_d,
                gene_annotations=gene_annotations,
                hvg_names=None,  # compute per-donor
                donor_id=donor_id,
                n_hvgs=n_hvgs,
                n_neighbors=n_neighbors,
            )
            all_merged.append(merged_d)
            all_pairs.append(pairs_d)

            row["n_pairs"] = len(pairs_d)
            row["pair_fraction"] = (
                len(pairs_d) / row["min_modality"] if row["min_modality"] else float("nan")
            )
            row["anchor"] = pairs_d["anchor"].iloc[0] if len(pairs_d) else ""
            row["mean_pair_distance"] = float(pairs_d["distance"].mean())
            row["cell_type_agreement"] = float(
                (pairs_d["cell_type"].astype(str)
                 == pairs_d["atac_cell_type"].astype(str)).mean()
            )
        except ValueError as e:
            row["status"] = "skipped"
            row["skip_reason"] = str(e)
            logger.warning(f"    Skipped: {e}")

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index("donor_id")

    logger.info(
        f"\n  Aligned {(summary_df['status'] == 'aligned').sum()} / "
        f"{len(usable_donors)} donors"
    )
    skipped = summary_df[summary_df["status"] == "skipped"]
    if len(skipped):
        logger.info(f"  Skipped {len(skipped)} donors:")
        for d, r in skipped.iterrows():
            logger.info(f"    {d}: {r['skip_reason']}")

    if not all_merged:
        logger.warning("  No donors aligned! Returning empty results.")
        return None, pd.DataFrame(), summary_df

    concatenated = ad.concat(all_merged, merge="same")
    concatenated.obs_names_make_unique()

    all_pairs_df = pd.concat(all_pairs, ignore_index=True)

    # Summary
    logger.info(f"\n  Total cells: {concatenated.n_obs}")
    logger.info(f"  Total pairs: {len(all_pairs_df)}")
    aligned = summary_df[summary_df["status"] == "aligned"]
    if len(aligned):
        logger.info(
            f"  Pair fraction (of min_modality): "
            f"mean={aligned['pair_fraction'].mean():.3f}, "
            f"min={aligned['pair_fraction'].min():.3f}"
        )
        logger.info(
            f"  Cell type agreement: "
            f"mean={aligned['cell_type_agreement'].mean():.3f}, "
            f"min={aligned['cell_type_agreement'].min():.3f}"
        )
    if "cell_type" in all_pairs_df.columns:
        logger.info(
            f"  Pairs per cell type: "
            f"{all_pairs_df['cell_type'].value_counts().to_dict()}"
        )

    return concatenated, all_pairs_df, summary_df


# =========================================================================
#  Main
# =========================================================================


if __name__ == "__main__":
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data_dir = Path("data/processed/seaad")
    ref_dir = Path("data/reference")

    # Load gene annotations
    gene_ann = load_gene_annotations(ref_dir / "hg38.ensembl98.gtf.gz")

    # Load donor classification
    donor_info = pd.read_csv(data_dir / "seaad_donor_classification.csv", index_col=0)

    # ---- Paired branch ----
    # Skip if the integrated h5mu already exists. Re-running paired WNN
    # rewrites a 91 GB artifact that downstream students depend on; gate
    # behind a missing-output check so the job is idempotent.
    paired_rna_path = data_dir / "seaad_paired_rna.h5ad"
    paired_atac_path = data_dir / "seaad_paired_atac.h5ad"
    paired_out_path = data_dir / "seaad_paired_integrated.h5mu"

    if paired_out_path.exists():
        logger.info(
            f"\n{paired_out_path.name} already exists — skipping paired branch. "
            f"Delete the file to force a re-run."
        )
    elif paired_rna_path.exists() and paired_atac_path.exists():
        logger.info("\n" + "=" * 60)
        logger.info("SEA-AD Paired Integration (WNN)")
        logger.info("=" * 60)

        paired_rna = ad.read_h5ad(paired_rna_path)
        paired_atac = ad.read_h5ad(paired_atac_path)
        logger.info(f"Loaded paired RNA: {paired_rna.shape}, ATAC: {paired_atac.shape}")

        mdata = integrate_seaad_paired(paired_rna, paired_atac)
        # Sanitize obs on each modality before write: SEA-AD's mixed-dtype
        # obs columns will trip the h5mu writer the same way they trip h5ad.
        for mod_name in mdata.mod:
            _sanitize_obs_for_h5ad(mdata.mod[mod_name])
        _sanitize_obs_for_h5ad(mdata)
        try:
            mdata.write(str(paired_out_path))
            logger.info(f"Saved seaad_paired_integrated.h5mu")
        except Exception as e:
            logger.exception(f"Failed to write paired h5mu: {e}")

        del mdata, paired_rna, paired_atac
        import gc; gc.collect()
    else:
        logger.info("No paired data found — skipping WNN branch")

    # ---- Unpaired branch (per-donor alignment) ----
    logger.info("\n" + "=" * 60)
    logger.info("SEA-AD Unpaired Integration (per-donor Harmony)")
    logger.info("=" * 60)

    unpaired_rna = ad.read_h5ad(data_dir / "seaad_unpaired_rna.h5ad")
    unpaired_atac = ad.read_h5ad(data_dir / "seaad_unpaired_atac.h5ad")
    logger.info(
        f"Loaded unpaired RNA: {unpaired_rna.shape}, ATAC: {unpaired_atac.shape}"
    )
    _swap_to_raw_counts(unpaired_rna, "Unpaired RNA")
    _swap_to_raw_counts(unpaired_atac, "Unpaired ATAC")

    merged, pairs, donor_summary = integrate_seaad_unpaired(
        unpaired_rna, unpaired_atac,
        gene_annotations=gene_ann,
        donor_info=donor_info,
    )

    # Save the small / irreplaceable artifacts FIRST so a write failure on
    # the big h5ad doesn't take everything down with it.
    donor_summary.to_csv(data_dir / "seaad_unpaired_donor_summary.csv")
    logger.info(
        f"Saved seaad_unpaired_donor_summary.csv ({len(donor_summary)} donors)"
    )

    pairs.to_csv(data_dir / "seaad_unpaired_nn_pairs.csv", index=False)
    logger.info(f"Saved seaad_unpaired_nn_pairs.csv ({len(pairs)} pairs)")

    if merged is not None:
        _sanitize_obs_for_h5ad(merged, keep=_OBS_KEEP)
        try:
            merged.write_h5ad(data_dir / "seaad_unpaired_integrated.h5ad")
            logger.info(f"Saved seaad_unpaired_integrated.h5ad")
        except Exception as e:
            logger.exception(f"Failed to write merged h5ad: {e}")

    logger.info("\nDone.")
