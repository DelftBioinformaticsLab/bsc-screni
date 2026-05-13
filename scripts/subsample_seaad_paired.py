"""Subsample SEA-AD paired HVG/HVP files into Phase 3 input files.

Reads the per-cell-type files produced by `scripts/run_seaad_hvg_selection.py`
(`seaad_paired_rna_hvg.h5ad`, `seaad_paired_atac_hvp.h5ad`) and writes a
matched RNA/ATAC subsample at:

    data/processed/seaad/seaad_paired_rna_sub{seed}.h5ad
    data/processed/seaad/seaad_paired_atac_sub{seed}.h5ad

The seed appears in the filename so multiple sub-questions can coexist
without overwriting each other; Phase 3 picks them up via glob and emits
output files prefixed `seaad_paired_sub{seed}_*` accordingly.

The RNA and ATAC cells stay row-aligned (same barcodes in the same order)
— required for Phase 3's correlation step.

Usage:
    pixi run python scripts/subsample_seaad_paired.py --seed 42
    pixi run python scripts/subsample_seaad_paired.py --seed 7 --n-per-type 100
    pixi run python scripts/subsample_seaad_paired.py --seed 99 \\
        --cell-types Microglia-PVM Astrocyte Oligodendrocyte "L2/3 IT"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA = Path("data/processed/seaad")
RNA_HVG = DATA / "seaad_paired_rna_hvg.h5ad"
ATAC_HVP = DATA / "seaad_paired_atac_hvp.h5ad"
DEFAULT_K = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--seed", type=int, required=True,
        help="Random seed. Also embedded in the output filename as "
             "'_sub{seed}.h5ad' so multiple subsamples don't collide.",
    )
    p.add_argument(
        "--n-per-type", type=int, default=50,
        help="Cells to sample from each cell type. Types with fewer cells "
             "are kept in full (with a warning). Default: 50.",
    )
    p.add_argument(
        "--cell-types", nargs="*", default=None,
        help="Restrict to these cell types (matches obs['cell_type']). "
             "If omitted, all 24 SEA-AD subclasses are included. Subclasses "
             "with spaces/slashes must be quoted, e.g. \"L2/3 IT\".",
    )
    p.add_argument(
        "--k", type=int, default=DEFAULT_K,
        help=f"Neighbors for the KNN graph stored in uns['knn_indices'] "
             f"(default: {DEFAULT_K}, matches the paper). The KNN is "
             f"computed on obsm['X_pca'] of the subsample.",
    )
    p.add_argument(
        "--rna",  type=Path, default=RNA_HVG,
        help=f"Input RNA HVG file (default: {RNA_HVG}).",
    )
    p.add_argument(
        "--atac", type=Path, default=ATAC_HVP,
        help=f"Input ATAC HVP file (default: {ATAC_HVP}).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    logger.info(f"Reading {args.rna} ...")
    rna = ad.read_h5ad(args.rna)
    logger.info(f"Reading {args.atac} ...")
    atac = ad.read_h5ad(args.atac)

    if not (rna.obs_names == atac.obs_names).all():
        raise RuntimeError(
            "RNA and ATAC obs_names are not row-aligned. Phase 3 correlation "
            "would be wrong on this input — refusing to subsample."
        )

    if "cell_type" not in rna.obs.columns:
        raise KeyError("Missing 'cell_type' column in RNA obs.")

    # Optional cell-type filter
    if args.cell_types:
        missing = set(args.cell_types) - set(rna.obs["cell_type"].unique())
        if missing:
            raise ValueError(
                f"--cell-types not in obs: {sorted(missing)}. "
                f"Available: {sorted(rna.obs['cell_type'].unique())}"
            )
        mask = rna.obs["cell_type"].isin(args.cell_types).values
        rna = rna[mask].copy()
        atac = atac[mask].copy()
        logger.info(f"Filtered to {rna.n_obs} cells across {len(args.cell_types)} types")

    # Stratified subsample by cell_type
    idx_list = []
    for ct, group in rna.obs.groupby("cell_type", observed=True).groups.items():
        arr = np.asarray(group)
        k = min(args.n_per_type, len(arr))
        if k < args.n_per_type:
            logger.warning(
                f"  {ct}: only {len(arr)} cells available (requested {args.n_per_type})"
            )
        picks = rng.choice(arr, size=k, replace=False)
        idx_list.extend(picks.tolist())
        logger.info(f"  {ct}: {len(arr):>6} available, picked {k}")

    name_to_pos = {n: i for i, n in enumerate(rna.obs_names)}
    pos_idx = sorted(name_to_pos[n] for n in idx_list)
    logger.info(f"Total cells picked: {len(pos_idx)}")

    rna_sub = rna[pos_idx].copy()
    atac_sub = atac[pos_idx].copy()
    assert (rna_sub.obs_names == atac_sub.obs_names).all()

    # Replace the full-set WNN KNN (which would still reference 138k positions
    # — useless on this subsample) with a freshly computed k=20 KNN on the
    # subsample's joint WNN-input embedding (obsm['X_pca']). This is what
    # wScReNI consumes as the cell-neighborhood structure.
    if "X_pca" not in rna_sub.obsm:
        raise KeyError(
            "Missing obsm['X_pca'] in HVG input — can't recompute subsample KNN. "
            "Re-run scripts/run_seaad_hvg_selection.py to attach it."
        )
    rna_sub.uns.pop("wnn_neighbor_indices", None)
    atac_sub.uns.pop("wnn_neighbor_indices", None)

    logger.info(f"Computing k={args.k} KNN on subsample obsm['X_pca'] ...")
    nn = NearestNeighbors(n_neighbors=args.k).fit(rna_sub.obsm["X_pca"])
    _, knn_indices = nn.kneighbors(rna_sub.obsm["X_pca"])
    knn_indices = knn_indices.astype(np.int64)
    rna_sub.uns["knn_indices"] = knn_indices
    atac_sub.uns["knn_indices"] = knn_indices
    logger.info(f"  knn_indices: shape={knn_indices.shape}")

    rna_out = DATA / f"seaad_paired_rna_sub{args.seed}.h5ad"
    atac_out = DATA / f"seaad_paired_atac_sub{args.seed}.h5ad"
    rna_sub.write_h5ad(rna_out)
    atac_sub.write_h5ad(atac_out)
    logger.info(f"\nWrote {rna_out}: {rna_sub.shape}")
    logger.info(f"Wrote {atac_out}: {atac_sub.shape}")
    logger.info(
        f"\nNext: `pixi run gene-peak` (will pick up sub{args.seed} files via glob)"
    )


if __name__ == "__main__":
    main()
