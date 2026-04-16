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
from screni.data.integration_retinal import compute_gene_activity
from screni.data.loading_seaad import MIN_CELLS_PER_DONOR
from screni.data.utils import load_gene_annotations

logger = logging.getLogger(__name__)


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
        f"{paired_rna.obs['donor_id'].nunique()} donors"
    )

    mdata = integrate_paired(
        paired_rna, paired_atac,
        n_hvgs=n_hvgs,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        batch_key="donor_id",
    )

    return mdata


# =========================================================================
#  Unpaired branch: per-donor alignment
# =========================================================================


def compute_global_hvgs(
    rna: ad.AnnData,
    n_hvgs: int = 2000,
) -> list[str]:
    """Compute HVGs across ALL singleome RNA donors.

    These HVGs are used in every per-donor alignment for a consistent
    shared feature space.

    Parameters
    ----------
    rna
        All singleome RNA cells (all donors), raw counts in .X.
    n_hvgs
        Number of HVGs to select.

    Returns
    -------
    List of HVG names.
    """
    logger.info(f"Computing global HVGs across {rna.n_obs} singleome RNA cells...")

    work = rna.copy()
    work.layers["counts"] = work.X.copy()
    sc.pp.normalize_total(work, target_sum=1e4)
    sc.pp.log1p(work)
    sc.pp.highly_variable_genes(
        work, n_top_genes=n_hvgs, flavor="seurat_v3", layer="counts",
    )
    hvgs = work.var_names[work.var["highly_variable"]].tolist()
    logger.info(f"  Selected {len(hvgs)} global HVGs")
    return hvgs


def align_donor_unpaired(
    rna_donor: ad.AnnData,
    atac_donor: ad.AnnData,
    gene_annotations: pd.DataFrame,
    hvg_names: list[str],
    donor_id: str,
    n_harmony_dims: int = 20,
    harmony_lambda: float = 1.0,
    n_neighbors: int = 20,
) -> tuple[ad.AnnData, pd.DataFrame]:
    """Align RNA and ATAC for a single singleome donor.

    Steps:
      1. Gene activity from ATAC peaks
      2. Shared genes = intersection of hvg_names and gene activity
      3. Log-normalize both
      4. Concatenate with modality flag
      5. PCA (50 comps)
      6. Harmony(modality)
      7. kNN + UMAP
      8. Pair RNA <-> ATAC (nearest cross-modal neighbor)

    Parameters
    ----------
    rna_donor
        RNA cells from one donor, raw counts.
    atac_donor
        ATAC cells from one donor, raw peak counts.
    gene_annotations
        Gene body coordinates from hg38 GTF.
    hvg_names
        Pre-computed global HVGs.
    donor_id
        Donor identifier (for logging and output).
    n_harmony_dims
        PCA dimensions for Harmony.
    harmony_lambda
        Harmony diversity penalty.
    n_neighbors
        k for kNN graph.

    Returns
    -------
    (merged, pairs) where merged is an AnnData with both modalities
    and pairs is a DataFrame with rna_cell, atac_cell, cell_type, donor_id.

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

    # Step 2: Shared genes
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

    # Step 4: Concatenate
    rna_sub.obs["modality"] = "RNA"
    ga_sub.obs["modality"] = "ATAC"

    # Make obs_names unique before concat
    rna_sub.obs_names = [f"rna_{i}" for i in range(rna_sub.n_obs)]
    ga_sub.obs_names = [f"atac_{i}" for i in range(ga_sub.n_obs)]

    # Preserve original obs_names for pairing output
    rna_original_names = rna_donor.obs_names.tolist()
    atac_original_names = atac_donor.obs_names.tolist()

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

    # Step 8: Pair RNA <-> ATAC (nearest cross-modal neighbor in Harmony space)
    rna_mask = combined.obs["modality"] == "RNA"
    atac_mask = combined.obs["modality"] == "ATAC"

    rna_harmony = combined.obsm["X_harmony"][rna_mask.values]
    atac_harmony = combined.obsm["X_harmony"][atac_mask.values]

    # For each RNA cell, find nearest ATAC cell
    dists = cdist(rna_harmony, atac_harmony, metric="euclidean")
    nearest_atac_idx = dists.argmin(axis=1)
    nearest_dists = dists[np.arange(len(rna_original_names)), nearest_atac_idx]

    pairs = pd.DataFrame({
        "rna_cell": rna_original_names,
        "atac_cell": [atac_original_names[i] for i in nearest_atac_idx],
        "distance": nearest_dists,
        "donor_id": donor_id,
    })

    # Add cell type from RNA (more reliable annotation)
    rna_celltypes = rna_donor.obs["cell_type"].values
    pairs["cell_type"] = rna_celltypes

    # Deduplicate: each ATAC cell matched to at most one RNA cell
    n_before = len(pairs)
    pairs = pairs.sort_values("distance")
    pairs = pairs.drop_duplicates(subset="atac_cell", keep="first")
    pairs = pairs.sort_index()
    if n_before > len(pairs):
        logger.info(
            f"    Deduplicated: {n_before} -> {len(pairs)} pairs "
            f"({n_before - len(pairs)} duplicates removed)"
        )

    return combined, pairs


def integrate_seaad_unpaired(
    unpaired_rna: ad.AnnData,
    unpaired_atac: ad.AnnData,
    gene_annotations: pd.DataFrame,
    donor_info: pd.DataFrame,
    donor_col: str = "donor_id",
    n_hvgs: int = 2000,
    n_neighbors: int = 20,
) -> tuple[ad.AnnData | None, pd.DataFrame]:
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
    (concatenated_merged, all_pairs) where concatenated_merged is an
    AnnData with block-diagonal kNN graph (or None if no donors aligned)
    and all_pairs is a DataFrame of RNA-ATAC pairs across all donors.
    """
    logger.info("=== SEA-AD Unpaired Integration (per-donor Harmony) ===")

    # Step 1: Global HVGs
    hvg_names = compute_global_hvgs(unpaired_rna, n_hvgs=n_hvgs)

    # Step 2: Loop over usable donors
    usable_donors = donor_info[donor_info["has_both"]].index.tolist()
    logger.info(f"  Usable donors: {len(usable_donors)}")

    all_merged = []
    all_pairs = []
    skipped = []

    for i, donor_id in enumerate(usable_donors):
        logger.info(f"\n  [{i + 1}/{len(usable_donors)}] Donor: {donor_id}")

        rna_d = unpaired_rna[unpaired_rna.obs[donor_col] == donor_id].copy()
        atac_d = unpaired_atac[unpaired_atac.obs[donor_col] == donor_id].copy()

        logger.info(f"    RNA: {rna_d.n_obs}, ATAC: {atac_d.n_obs}")

        try:
            merged_d, pairs_d = align_donor_unpaired(
                rna_d, atac_d,
                gene_annotations=gene_annotations,
                hvg_names=hvg_names,
                donor_id=donor_id,
                n_neighbors=n_neighbors,
            )
            all_merged.append(merged_d)
            all_pairs.append(pairs_d)
            logger.info(f"    Aligned: {len(pairs_d)} pairs")
        except ValueError as e:
            skipped.append((donor_id, str(e)))
            logger.warning(f"    Skipped: {e}")

    # Step 3: Concatenate
    logger.info(f"\n  Aligned {len(all_merged)} / {len(usable_donors)} donors")
    if skipped:
        logger.info(f"  Skipped {len(skipped)} donors:")
        for d, reason in skipped:
            logger.info(f"    {d}: {reason}")

    if not all_merged:
        logger.warning("  No donors aligned! Returning empty results.")
        return None, pd.DataFrame()

    concatenated = ad.concat(all_merged, merge="same")
    concatenated.obs_names_make_unique()

    all_pairs_df = pd.concat(all_pairs, ignore_index=True)

    # Summary
    logger.info(f"\n  Total cells: {concatenated.n_obs}")
    logger.info(f"  Total pairs: {len(all_pairs_df)}")
    if "cell_type" in all_pairs_df.columns:
        logger.info(
            f"  Pairs per cell type: "
            f"{all_pairs_df['cell_type'].value_counts().to_dict()}"
        )

    return concatenated, all_pairs_df


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

    # ---- Paired branch (if pairing worked) ----
    paired_rna_path = data_dir / "seaad_paired_rna.h5ad"
    paired_atac_path = data_dir / "seaad_paired_atac.h5ad"

    if paired_rna_path.exists() and paired_atac_path.exists():
        logger.info("\n" + "=" * 60)
        logger.info("SEA-AD Paired Integration (WNN)")
        logger.info("=" * 60)

        paired_rna = ad.read_h5ad(paired_rna_path)
        paired_atac = ad.read_h5ad(paired_atac_path)
        logger.info(f"Loaded paired RNA: {paired_rna.shape}, ATAC: {paired_atac.shape}")

        mdata = integrate_seaad_paired(paired_rna, paired_atac)
        mdata.write(str(data_dir / "seaad_paired_integrated.h5mu"))
        logger.info(f"Saved seaad_paired_integrated.h5mu")

        del mdata, paired_rna, paired_atac
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

    merged, pairs = integrate_seaad_unpaired(
        unpaired_rna, unpaired_atac,
        gene_annotations=gene_ann,
        donor_info=donor_info,
    )

    if merged is not None:
        merged.write_h5ad(data_dir / "seaad_unpaired_integrated.h5ad")
        logger.info(f"Saved seaad_unpaired_integrated.h5ad")

    pairs.to_csv(data_dir / "seaad_unpaired_nn_pairs.csv", index=False)
    logger.info(f"Saved seaad_unpaired_nn_pairs.csv ({len(pairs)} pairs)")

    logger.info("\nDone.")
