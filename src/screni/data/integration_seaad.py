"""Phase 1: Integration of SEA-AD MTG multi-modal data.

Two branches:

Paired (multiome, 28 donors):
    Uses WNN via ``integrate_paired()`` from ``integration.py`` with
    ``batch_key="Donor ID"`` for donor-aware HVG selection.

Unpaired (singleome, global embedding + per-donor pairing):
    The earlier per-donor-everything design starved Harmony of data:
    per-donor PCA on ~10k cells × two modality batches left modalities
    in disjoint regions. The current design instead computes one global
    embedding for all 1.6M cells, then pairs per-donor in that embedding.

      1. Filter to donors with both modalities ≥ MIN_CELLS_PER_DONOR
      2. Gene activity from ATAC peaks (one global sparse matmul)
      3. Global HVGs from RNA (Seurat v3 VST on a donor-stratified sample)
      4. Per-modality log1p(normalize_total), subset to shared HVGs
      5. Concatenate RNA + gene-activity → ~1.6M cells
      6. Global PCA (sparse, no scaling)
      7. Harmony with batch_key="modality" only — donor effects deliberately
         preserved as biological variation
      8. Per-donor cross-modal NN pairing in X_harmony — anchor on the
         rarer modality, every pair shares donor_id on both sides.

Design decisions:
    - Global PCA gives Harmony enough cells per cluster to actually
      correct the modality batch (the per-donor variant could not).
    - Donor is excluded from the batch key so AD-related donor variation
      is preserved for downstream sub-questions.
    - Pairing stays per-donor so pair-level covariates (Donor ID, ADNC,
      genotype) are unambiguous.
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


def _pair_cross_modal(
    rna_coords: np.ndarray,
    atac_coords: np.ndarray,
    rna_names: np.ndarray,
    atac_names: np.ndarray,
    rna_celltypes: np.ndarray,
    atac_celltypes: np.ndarray,
    donor_id: str,
) -> pd.DataFrame:
    """Anchor cross-modal pairing on the rarer modality, no dedup.

    Returns a DataFrame with columns rna_cell, atac_cell, cell_type,
    atac_cell_type, distance, donor_id, anchor. Length = min(n_rna, n_atac).
    """
    n_rna = len(rna_coords)
    n_atac = len(atac_coords)

    if n_rna <= n_atac:
        anchor = "rna"
        dists = cdist(rna_coords, atac_coords, metric="euclidean")
        nearest = dists.argmin(axis=1)
        pair_dist = dists[np.arange(n_rna), nearest]
        return pd.DataFrame({
            "rna_cell": rna_names,
            "atac_cell": atac_names[nearest],
            "cell_type": rna_celltypes,
            "atac_cell_type": atac_celltypes[nearest],
            "distance": pair_dist,
            "donor_id": donor_id,
            "anchor": anchor,
        })
    anchor = "atac"
    dists = cdist(atac_coords, rna_coords, metric="euclidean")
    nearest = dists.argmin(axis=1)
    pair_dist = dists[np.arange(n_atac), nearest]
    return pd.DataFrame({
        "rna_cell": rna_names[nearest],
        "atac_cell": atac_names,
        "cell_type": rna_celltypes[nearest],
        "atac_cell_type": atac_celltypes,
        "distance": pair_dist,
        "donor_id": donor_id,
        "anchor": anchor,
    })


def _global_hvgs(
    rna: ad.AnnData,
    n_hvgs: int,
    donor_col: str,
    sample_cells: int = 200_000,
    seed: int = 0,
) -> list[str]:
    """Select HVGs from RNA on a donor-stratified subsample.

    Seurat v3 VST on the full 1.2M-cell matrix is feasible but slow and
    memory-heavy. A 200k-cell stratified subsample gives stable HVGs (Seurat v3
    is variance-based, not population-size-sensitive) at a fraction of cost.
    """
    if rna.n_obs <= sample_cells:
        sample_idx = np.arange(rna.n_obs)
    else:
        rng = np.random.RandomState(seed)
        per_donor = max(1, sample_cells // max(rna.obs[donor_col].nunique(), 1))
        sample_idx = []
        for _, group in rna.obs.groupby(donor_col, observed=True).groups.items():
            arr = np.asarray(group)
            k = min(per_donor, len(arr))
            sample_idx.append(rng.choice(arr, size=k, replace=False))
        sample_idx = np.concatenate(sample_idx)

    logger.info(f"  HVG: sampling {len(sample_idx)} of {rna.n_obs} cells for VST...")
    work = rna[sample_idx].to_memory() if rna.isbacked else rna[sample_idx].copy()
    work.layers["counts"] = work.X.copy()
    sc.pp.normalize_total(work, target_sum=1e4)
    sc.pp.log1p(work)
    sc.pp.highly_variable_genes(
        work, n_top_genes=n_hvgs,
        flavor="seurat_v3", layer="counts", span=0.3,
    )
    hvgs = work.var_names[work.var["highly_variable"]].tolist()
    logger.info(f"  HVG: selected {len(hvgs)} genes")
    return hvgs


def integrate_seaad_unpaired_global(
    unpaired_rna: ad.AnnData,
    unpaired_atac: ad.AnnData,
    gene_annotations: pd.DataFrame,
    donor_info: pd.DataFrame,
    donor_col: str = "Donor ID",
    n_hvgs: int = 2000,
    n_pcs: int = 50,
    n_harmony_dims: int = 20,
    harmony_lambda: float = 1.0,
) -> tuple[ad.AnnData, pd.DataFrame, pd.DataFrame]:
    """Integrate SEA-AD singleome via global embedding + per-donor pairing.

    Architecture (differs from the earlier per-donor design):

      1. Filter to usable donors (have both modalities, each ≥ MIN_CELLS_PER_DONOR)
      2. Gene activity from ATAC peaks (one global sparse matmul)
      3. Global HVGs from RNA (Seurat v3 VST on a stratified subsample)
      4. Subset both modalities to HVGs ∩ gene-activity vars
      5. Per-modality log1p(normalize_total) — sparse, in place
      6. Concatenate RNA + gene-activity → one combined AnnData (~1.6M cells)
      7. Global PCA on the sparse combined matrix (truncated SVD, no scaling)
      8. Harmony with batch_key="modality" only — donor effects are deliberately
         preserved in the embedding as biological variation
      9. Per-donor cross-modal NN pairing in X_harmony space — anchor on rare,
         every pair has the same donor_id on both sides

    The global PCA + global Harmony gives the modality correction enough data
    to actually mix the two clouds; per-donor pairing keeps the pair-level
    donor structure students need for AD-vs-control analyses.

    Returns
    -------
    (combined, all_pairs, donor_summary)
    """
    logger.info("=== SEA-AD Unpaired Integration (GLOBAL embedding) ===")

    # ----------------------------------------------------------------
    # 1) Filter to usable donors
    # ----------------------------------------------------------------
    usable_donors = donor_info[donor_info["has_both"]].index.tolist()
    logger.info(f"  Usable donors: {len(usable_donors)}")
    rna_mask = unpaired_rna.obs[donor_col].isin(usable_donors)
    atac_mask = unpaired_atac.obs[donor_col].isin(usable_donors)
    rna = unpaired_rna[rna_mask.values].copy()
    atac = unpaired_atac[atac_mask.values].copy()
    logger.info(f"  After donor filter — RNA: {rna.shape}, ATAC: {atac.shape}")

    # ----------------------------------------------------------------
    # 2) Gene activity (global)
    # ----------------------------------------------------------------
    gene_activity = compute_gene_activity(atac, gene_annotations)

    # ----------------------------------------------------------------
    # 3) Global HVGs (on stratified subsample)
    # ----------------------------------------------------------------
    hvgs = _global_hvgs(rna, n_hvgs=n_hvgs, donor_col=donor_col)

    # ----------------------------------------------------------------
    # 4) Shared gene set
    # ----------------------------------------------------------------
    shared_genes = sorted(
        set(hvgs) & set(gene_activity.var_names) & set(rna.var_names)
    )
    logger.info(f"  Shared genes: {len(shared_genes)} (HVG ∩ gene-activity)")
    if len(shared_genes) < 200:
        raise ValueError(f"Too few shared genes ({len(shared_genes)} < 200)")

    rna_sub = rna[:, shared_genes].copy()
    ga_sub = gene_activity[:, shared_genes].copy()
    del rna, gene_activity, atac
    import gc; gc.collect()

    # ----------------------------------------------------------------
    # 5) Log-normalize per modality
    # ----------------------------------------------------------------
    logger.info("  Normalizing RNA and gene-activity ...")
    sc.pp.normalize_total(rna_sub, target_sum=1e4)
    sc.pp.log1p(rna_sub)
    sc.pp.normalize_total(ga_sub, target_sum=1e4)
    sc.pp.log1p(ga_sub)

    # ----------------------------------------------------------------
    # 6) Slim obs, tag modality, preserve original obs_names, concat
    # ----------------------------------------------------------------
    rna_sub.obs = rna_sub.obs[
        [c for c in _OBS_KEEP if c in rna_sub.obs.columns]
    ].copy()
    ga_sub.obs = ga_sub.obs[
        [c for c in _OBS_KEEP if c in ga_sub.obs.columns]
    ].copy()
    rna_sub.obs["modality"] = "RNA"
    ga_sub.obs["modality"] = "ATAC"
    rna_sub.obs["_original_obs"] = rna_sub.obs_names.values
    ga_sub.obs["_original_obs"] = ga_sub.obs_names.values

    # Give rows distinct obs_names so concat doesn't trip on duplicates.
    rna_sub.obs_names = [f"rna_{i}" for i in range(rna_sub.n_obs)]
    ga_sub.obs_names = [f"atac_{i}" for i in range(ga_sub.n_obs)]

    logger.info(
        f"  Concatenating RNA ({rna_sub.n_obs}) + gene-activity ({ga_sub.n_obs}) ..."
    )
    combined = ad.concat([rna_sub, ga_sub], merge="same")
    del rna_sub, ga_sub
    gc.collect()
    logger.info(f"  Combined: {combined.shape}")

    # ----------------------------------------------------------------
    # 7) Global PCA — sparse, no scaling (densifying 1.6M × 2k is ~12 GB)
    # ----------------------------------------------------------------
    logger.info(f"  Running PCA ({n_pcs} components, sparse) ...")
    n_pcs = min(n_pcs, len(shared_genes) - 1, combined.n_obs - 1)
    sc.pp.pca(combined, n_comps=n_pcs, zero_center=False)

    # ----------------------------------------------------------------
    # 8) Harmony with modality as the only batch
    # ----------------------------------------------------------------
    logger.info(
        f"  Running Harmony (batch=modality, dims={n_harmony_dims}, "
        f"lamb={harmony_lambda}) ..."
    )
    pca_dims = min(n_harmony_dims, combined.obsm["X_pca"].shape[1])
    ho = hm.run_harmony(
        combined.obsm["X_pca"][:, :pca_dims],
        combined.obs,
        "modality",
        max_iter_harmony=20,
        lamb=harmony_lambda,
    )
    harmony_result = np.asarray(ho.Z_corr)
    if harmony_result.shape[0] != combined.n_obs:
        harmony_result = harmony_result.T
    combined.obsm["X_harmony"] = harmony_result
    logger.info(f"  Harmony embedding: {harmony_result.shape}")

    # ----------------------------------------------------------------
    # 9) Per-donor pairing in X_harmony space
    # ----------------------------------------------------------------
    rna_mask = (combined.obs["modality"] == "RNA").values
    atac_mask = (combined.obs["modality"] == "ATAC").values
    donor_arr = combined.obs[donor_col].astype(str).values
    ct_arr = combined.obs["cell_type"].astype(str).values
    original_arr = combined.obs["_original_obs"].astype(str).values
    embedding = combined.obsm["X_harmony"]

    all_pairs: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for i, donor_id in enumerate(usable_donors):
        donor_mask = donor_arr == donor_id
        d_rna = donor_mask & rna_mask
        d_atac = donor_mask & atac_mask
        n_rna_d = int(d_rna.sum())
        n_atac_d = int(d_atac.sum())

        row = {
            "donor_id": donor_id,
            "n_rna": n_rna_d,
            "n_atac": n_atac_d,
            "min_modality": min(n_rna_d, n_atac_d),
            "n_pairs": 0,
            "pair_fraction": float("nan"),
            "anchor": "",
            "mean_pair_distance": float("nan"),
            "cell_type_agreement": float("nan"),
            "status": "aligned",
            "skip_reason": "",
        }

        if n_rna_d < MIN_CELLS_PER_DONOR or n_atac_d < MIN_CELLS_PER_DONOR:
            row["status"] = "skipped"
            row["skip_reason"] = (
                f"too few cells (RNA={n_rna_d}, ATAC={n_atac_d}, "
                f"min={MIN_CELLS_PER_DONOR})"
            )
            logger.warning(f"  [{i + 1}/{len(usable_donors)}] {donor_id}: skipped — {row['skip_reason']}")
            summary_rows.append(row)
            continue

        pairs_d = _pair_cross_modal(
            embedding[d_rna], embedding[d_atac],
            original_arr[d_rna], original_arr[d_atac],
            ct_arr[d_rna], ct_arr[d_atac],
            donor_id,
        )
        all_pairs.append(pairs_d)

        row["n_pairs"] = len(pairs_d)
        row["pair_fraction"] = len(pairs_d) / row["min_modality"]
        row["anchor"] = pairs_d["anchor"].iloc[0]
        row["mean_pair_distance"] = float(pairs_d["distance"].mean())
        row["cell_type_agreement"] = float(
            (pairs_d["cell_type"] == pairs_d["atac_cell_type"]).mean()
        )
        summary_rows.append(row)
        logger.info(
            f"  [{i + 1}/{len(usable_donors)}] {donor_id}: "
            f"n_rna={n_rna_d}, n_atac={n_atac_d}, anchor={row['anchor']}, "
            f"pairs={row['n_pairs']}, agree={row['cell_type_agreement']:.3f}"
        )

    summary_df = pd.DataFrame(summary_rows).set_index("donor_id")
    all_pairs_df = (
        pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    )

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    aligned = summary_df[summary_df["status"] == "aligned"]
    logger.info(f"\n  Aligned {len(aligned)} / {len(summary_df)} donors")
    if len(aligned):
        logger.info(
            f"  Pair count: total={aligned['n_pairs'].sum():,}, "
            f"median per donor={int(aligned['n_pairs'].median())}"
        )
        logger.info(
            f"  Cell type agreement: "
            f"mean={aligned['cell_type_agreement'].mean():.3f}, "
            f"min={aligned['cell_type_agreement'].min():.3f}, "
            f"max={aligned['cell_type_agreement'].max():.3f}"
        )
        logger.info(
            f"  Mean pair distance: "
            f"mean={aligned['mean_pair_distance'].mean():.3f}"
        )
    if len(all_pairs_df) and "cell_type" in all_pairs_df.columns:
        logger.info(
            f"  Pairs per cell type (RNA-side): "
            f"{all_pairs_df['cell_type'].value_counts().to_dict()}"
        )

    # Restore the meaningful obs_names so the merged AnnData is round-trippable.
    combined.obs_names = pd.Index(combined.obs["_original_obs"].values)
    combined.obs_names_make_unique()
    combined.obs.drop(columns=["_original_obs"], inplace=True)

    return combined, all_pairs_df, summary_df


# Public name kept stable for __main__ and any imports.
integrate_seaad_unpaired = integrate_seaad_unpaired_global


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
