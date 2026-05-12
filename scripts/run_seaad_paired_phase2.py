"""Phase 2 (HVG/HVP selection + cell subsampling) for SEA-AD PAIRED only.

Standalone counterpart to ``screni.data.feature_selection.__main__`` that
processes the SEA-AD paired branch and nothing else. Useful while the
unpaired integration job is still in queue or running: students need fresh
paired files to develop against.

Selection order vs the original pipeline:

The original `prepare_subsample()` SUBSAMPLES first, then picks HVGs from
the ~200-cell subsample. That follows the paper, but VST on a small sample
is noisy. Here we instead pick HVGs/HVPs on the FULL cell set (~138k
cells), giving a more stable feature set, and produce TWO outputs per
modality so students can choose their own cell selection:

  seaad_paired_rna_hvg.h5ad   — full cells × 500 HVGs    (raw counts)
  seaad_paired_atac_hvp.h5ad  — full cells × 10000 HVPs  (raw counts)
  seaad_paired_rna_sub.h5ad   — 50/type cells × 500 HVGs  (default subsample)
  seaad_paired_atac_sub.h5ad  — 50/type cells × 10000 HVPs (default subsample)

The HVGs/HVPs in the `_sub` files are identical to those in the `_hvg/_hvp`
files — only the cell set differs. Students who want a different cell
selection (different n_per_type, custom cell-type subset, different
sampling strategy) can start from `_hvg.h5ad` / `_hvp.h5ad` and subsample
in their own code.

Inputs (must exist):
  data/processed/seaad/seaad_paired_rna.h5ad
  data/processed/seaad/seaad_paired_atac.h5ad

Why these files instead of the 91 GB h5mu:
  - The h5ad files are the post-barcode-match, pre-WNN paired inputs.
    The cell set is identical to mdata.mod['rna'/'atac'] in the h5mu.
  - Loading is ~28 GB instead of ~100 GB.
  - We only need raw counts and cell_type for Phase 2; the WNN embedding
    matters only for the optional KNN-indices output (wScReNI input).

Run via slurm/run_seaad_paired_phase2.sh or:
  pixi run python scripts/run_seaad_paired_phase2.py
"""

import gc
import logging
from pathlib import Path

import anndata as ad

from screni.data.feature_selection import (
    filter_chr_peaks,
    select_variable_features,
    subsample_cells,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SEAAD_DIR = Path("data/processed/seaad")
RNA_PATH = SEAAD_DIR / "seaad_paired_rna.h5ad"
ATAC_PATH = SEAAD_DIR / "seaad_paired_atac.h5ad"

N_HVGS = 500
N_HVPS = 10_000
N_PER_TYPE = 50
SEED = 42


def _pull_cell_type(adata: ad.AnnData) -> ad.AnnData:
    """Return obs with a 'cell_type' column (copies from Subclass if needed)."""
    if "cell_type" in adata.obs.columns:
        return adata.obs[["cell_type"]].copy()
    if "Subclass" in adata.obs.columns:
        return (
            adata.obs[["Subclass"]]
            .rename(columns={"Subclass": "cell_type"})
            .copy()
        )
    raise KeyError(
        f"Neither 'cell_type' nor 'Subclass' in obs (have: "
        f"{list(adata.obs.columns)[:10]}...)"
    )


def _raw_counts_X(adata: ad.AnnData, label: str):
    """Return raw counts as the X matrix.

    SEA-AD RNA puts raw UMIs in layers['UMIs'] and normalized values in .X.
    SEA-AD ATAC puts integer Tn5 cut counts directly in .X (no layers).
    """
    for layer_name in ("UMIs", "UMI", "counts", "raw", "raw_counts"):
        if layer_name in adata.layers:
            logger.info(f"  {label}: using layers['{layer_name}'] as raw counts")
            return adata.layers[layer_name]
    logger.info(f"  {label}: no raw-counts layer; using .X as-is (assumed raw)")
    return adata.X


def _process_rna() -> None:
    if not RNA_PATH.exists():
        raise FileNotFoundError(f"Missing input: {RNA_PATH}")

    logger.info(f"\n{'=' * 60}\n  RNA\n{'=' * 60}")
    logger.info(f"Reading {RNA_PATH} ...")
    full = ad.read_h5ad(RNA_PATH)
    logger.info(f"  shape: {full.shape}")

    rna = ad.AnnData(
        X=_raw_counts_X(full, "RNA").copy(),
        obs=_pull_cell_type(full),
        var=full.var.copy(),
    )
    del full
    gc.collect()

    logger.info(f"  cell types: {rna.obs['cell_type'].value_counts().to_dict()}")

    logger.info(f"Selecting {N_HVGS} HVGs from full RNA (Seurat v3 VST) ...")
    rna_hvg = select_variable_features(rna, n_features=N_HVGS)
    logger.info(f"  HVG output: {rna_hvg.shape}")
    del rna
    gc.collect()

    hvg_path = SEAAD_DIR / "seaad_paired_rna_hvg.h5ad"
    rna_hvg.write_h5ad(hvg_path)
    logger.info(f"Wrote {hvg_path}: {rna_hvg.shape}")

    logger.info(f"Subsampling {N_PER_TYPE} cells per type ...")
    rna_sub = subsample_cells(rna_hvg, n_per_type=N_PER_TYPE, seed=SEED)
    sub_path = SEAAD_DIR / "seaad_paired_rna_sub.h5ad"
    rna_sub.write_h5ad(sub_path)
    logger.info(f"Wrote {sub_path}: {rna_sub.shape}")


def _process_atac() -> None:
    if not ATAC_PATH.exists():
        raise FileNotFoundError(f"Missing input: {ATAC_PATH}")

    logger.info(f"\n{'=' * 60}\n  ATAC\n{'=' * 60}")
    logger.info(f"Reading {ATAC_PATH} ...")
    full = ad.read_h5ad(ATAC_PATH)
    logger.info(f"  shape: {full.shape}")

    atac = ad.AnnData(
        X=_raw_counts_X(full, "ATAC").copy(),
        obs=_pull_cell_type(full),
        var=full.var.copy(),
    )
    del full
    gc.collect()

    logger.info("Filtering chr-prefixed peaks ...")
    atac = filter_chr_peaks(atac)

    logger.info(f"Selecting {N_HVPS} HVPs from full ATAC (Seurat v3 VST, span=0.5) ...")
    # Wider LOESS span for ATAC's near-degenerate mean-variance relation,
    # matching what prepare_subsample() uses internally.
    atac_hvp = select_variable_features(atac, n_features=N_HVPS, span=0.5)
    logger.info(f"  HVP output: {atac_hvp.shape}")
    del atac
    gc.collect()

    hvp_path = SEAAD_DIR / "seaad_paired_atac_hvp.h5ad"
    atac_hvp.write_h5ad(hvp_path)
    logger.info(f"Wrote {hvp_path}: {atac_hvp.shape}")

    logger.info(f"Subsampling {N_PER_TYPE} cells per type ...")
    atac_sub = subsample_cells(atac_hvp, n_per_type=N_PER_TYPE, seed=SEED)
    sub_path = SEAAD_DIR / "seaad_paired_atac_sub.h5ad"
    atac_sub.write_h5ad(sub_path)
    logger.info(f"Wrote {sub_path}: {atac_sub.shape}")


def main() -> None:
    _process_rna()
    gc.collect()
    _process_atac()
    logger.info("\nDone.")


if __name__ == "__main__":
    main()
