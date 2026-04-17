"""Load the retinal benchmark data from the ScReNI paper's Seurat exports.

The paper provides pre-processed Seurat objects for the mouse retinal
dataset (mmRetina_RPCMG).  These were exported to MatrixMarket + CSV
via R (see docs for the extraction script).  This module loads those
exports into AnnData objects ready for Phases 2-3.

Files expected in ``data/paper/datasets/``:
    retinal_rna_counts.mtx     RNA count matrix  (genes x cells)
    retinal_rna_genes.txt      Gene names
    retinal_rna_cells.txt      Cell barcodes
    retinal_rna_metadata.csv   Cell metadata (celltypes, samples, ...)

    retinal_atac_counts.mtx    ATAC count matrix (peaks x cells)
    retinal_atac_peaks.txt     Peak names (chr-start-end format)
    retinal_atac_cells.txt     Cell barcodes
    retinal_atac_metadata.csv  Cell metadata

    mmRetina_RPCMG_Cell100_annotation.csv
        The exact 400 subsampled cells (100 per type) used in the paper,
        with columns: undup.rna.points.id, undup.atac.points.id,
        undup.cell.types.
"""

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

from screni.data.utils import standardize_peak_name

logger = logging.getLogger(__name__)

# Paper uses RPCs_S1/S2/S3 naming; we normalize to RPC1/2/3 for consistency
PAPER_CELLTYPE_MAP = {
    "RPCs_S1": "RPC1",
    "RPCs_S2": "RPC2",
    "RPCs_S3": "RPC3",
    "MG": "MG",
}


def load_paper_retinal(
    paper_dir: Path | str,
) -> tuple[ad.AnnData, ad.AnnData, pd.DataFrame]:
    """Load the paper's retinal RNA, ATAC, and cell annotations.

    Parameters
    ----------
    paper_dir
        Path to ``data/paper/datasets/``.

    Returns
    -------
    Tuple of (rna, atac, annotation):
        - rna: AnnData (cells x genes) with raw counts and cell_type in .obs
        - atac: AnnData (cells x peaks) with raw counts and cell_type in .obs
        - annotation: DataFrame with the paper's 400 subsampled cell pairs
    """
    paper_dir = Path(paper_dir)

    # --- RNA ---
    logger.info("Loading paper's retinal RNA...")
    rna_mtx = sio.mmread(str(paper_dir / "retinal_rna_counts.mtx"))
    rna_mtx = sp.csc_matrix(rna_mtx).T.tocsr()  # genes x cells -> cells x genes
    rna_genes = (paper_dir / "retinal_rna_genes.txt").read_text().strip().split("\n")
    rna_cells = (paper_dir / "retinal_rna_cells.txt").read_text().strip().split("\n")
    rna_meta = pd.read_csv(paper_dir / "retinal_rna_metadata.csv", index_col=0)

    rna = ad.AnnData(
        X=rna_mtx,
        obs=pd.DataFrame(index=rna_cells[:rna_mtx.shape[0]]),
        var=pd.DataFrame(index=rna_genes[:rna_mtx.shape[1]]),
    )
    rna.var_names_make_unique()

    # Add cell type from metadata
    rna.obs["cell_type"] = rna_meta.loc[rna.obs_names, "celltypes"].map(
        PAPER_CELLTYPE_MAP
    ).values

    logger.info(
        f"  RNA: {rna.shape}, max={rna.X.max():.0f}\n"
        f"  Cell types: {rna.obs['cell_type'].value_counts().to_dict()}"
    )

    # --- ATAC ---
    logger.info("Loading paper's retinal ATAC...")
    atac_mtx = sio.mmread(str(paper_dir / "retinal_atac_counts.mtx"))
    atac_mtx = sp.csc_matrix(atac_mtx).T.tocsr()  # peaks x cells -> cells x peaks
    atac_peaks = (paper_dir / "retinal_atac_peaks.txt").read_text().strip().split("\n")
    atac_cells = (paper_dir / "retinal_atac_cells.txt").read_text().strip().split("\n")
    atac_meta = pd.read_csv(paper_dir / "retinal_atac_metadata.csv", index_col=0)

    # Standardize peak names: chr1-start-end -> chr1:start-end
    atac_peaks_std = [standardize_peak_name(p) for p in atac_peaks[:atac_mtx.shape[1]]]

    atac = ad.AnnData(
        X=atac_mtx,
        obs=pd.DataFrame(index=atac_cells[:atac_mtx.shape[0]]),
        var=pd.DataFrame(index=atac_peaks_std),
    )
    atac.var_names_make_unique()

    atac.obs["cell_type"] = atac_meta.loc[atac.obs_names, "celltypes"].map(
        PAPER_CELLTYPE_MAP
    ).values

    atac_max = atac.X.max() if sp.issparse(atac.X) else np.max(atac.X)
    logger.info(
        f"  ATAC: {atac.shape}, max={atac_max:.0f}\n"
        f"  Cell types: {atac.obs['cell_type'].value_counts().to_dict()}"
    )

    # --- Cell100 annotation (the paper's subsampled 400 cells) ---
    ann_path = paper_dir / "mmRetina_RPCMG_Cell100_annotation.csv"
    annotation = pd.read_csv(ann_path)
    # Normalize to match subsample_pairs() expected columns
    annotation = annotation.rename(columns={
        "undup.rna.points.id": "rna_cell",
        "undup.atac.points.id": "atac_cell",
        "undup.cell.types": "cell_type",
    })
    # Keep only the columns we need
    annotation = annotation[["rna_cell", "atac_cell", "cell_type"]].copy()
    # Map cell types
    annotation["cell_type"] = annotation["cell_type"].map(PAPER_CELLTYPE_MAP)

    logger.info(
        f"  Annotation: {len(annotation)} cell pairs\n"
        f"  Types: {annotation['cell_type'].value_counts().to_dict()}"
    )

    return rna, atac, annotation


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    paper_dir = Path("data/paper/datasets")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    rna, atac, annotation = load_paper_retinal(paper_dir)

    rna.write_h5ad(out_dir / "retinal_rna.h5ad")
    atac.write_h5ad(out_dir / "retinal_atac.h5ad")
    annotation.to_csv(out_dir / "retinal_nn_pairs.csv", index=False)

    logger.info(f"Saved to {out_dir}")
