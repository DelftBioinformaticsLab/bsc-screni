"""Inspect downloaded datasets and print summary statistics.

Usage (on cluster):
  apptainer exec --writable-tmpfs --pwd /opt/app --containall \
    --bind src/:/opt/app/src/ --bind data/:/opt/app/data/ --bind output/:/opt/app/output/ \
    --env PYTHONPATH=/opt/app/src \
    container.sif pixi run --manifest-path /opt/app/pixi.toml python src/inspect_data.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata as ad

DATA_DIR = Path("data")
OUT_DIR = Path("output/data_inspection")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# =============================================================================
#  PBMC 10X Multiome (paired scRNA-seq + scATAC-seq)
# =============================================================================


def inspect_pbmc():
    section("PBMC 10X Multiome (paired)")
    pbmc_dir = DATA_DIR / "pbmc_unsorted_10k"

    if not pbmc_dir.exists():
        print(f"  [SKIP] {pbmc_dir} not found")
        return

    # --- Filtered feature barcode matrix (HDF5) ---
    h5_path = pbmc_dir / "pbmc_unsorted_10k_filtered_feature_bc_matrix.h5"
    if h5_path.exists():
        print(f"Reading {h5_path.name} ...")
        adata = sc.read_10x_h5(str(h5_path), gex_only=False)
        print(f"  Full matrix: {adata.shape[0]} cells x {adata.shape[1]} features")
        print(f"  Feature types: {adata.var['feature_types'].value_counts().to_dict()}")

        # Split into GEX and ATAC
        gex_mask = adata.var["feature_types"] == "Gene Expression"
        atac_mask = adata.var["feature_types"] == "Peaks"
        n_gex = gex_mask.sum()
        n_atac = atac_mask.sum()
        print(f"  Gene Expression features: {n_gex}")
        print(f"  ATAC Peaks features: {n_atac}")

        # GEX stats
        gex = adata[:, gex_mask].copy()
        print(f"\n  --- Gene Expression ---")
        print(f"  Shape: {gex.shape}")
        if sp.issparse(gex.X):
            nnz = gex.X.nnz
            total = gex.shape[0] * gex.shape[1]
            print(f"  Sparsity: {1 - nnz / total:.2%} zeros")
            print(f"  Total counts: {gex.X.sum():.0f}")
            counts_per_cell = np.array(gex.X.sum(axis=1)).flatten()
            genes_per_cell = np.array((gex.X > 0).sum(axis=1)).flatten()
        else:
            counts_per_cell = gex.X.sum(axis=1)
            genes_per_cell = (gex.X > 0).sum(axis=1)
        print(f"  Counts/cell: median={np.median(counts_per_cell):.0f}, "
              f"mean={np.mean(counts_per_cell):.0f}, "
              f"min={np.min(counts_per_cell):.0f}, max={np.max(counts_per_cell):.0f}")
        print(f"  Genes/cell:  median={np.median(genes_per_cell):.0f}, "
              f"mean={np.mean(genes_per_cell):.0f}")

        # Plot distributions
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(counts_per_cell, bins=50, edgecolor="black")
        axes[0].set_title("PBMC: Total counts per cell (GEX)")
        axes[0].set_xlabel("Total counts")
        axes[0].set_ylabel("Number of cells")

        axes[1].hist(genes_per_cell, bins=50, edgecolor="black")
        axes[1].set_title("PBMC: Genes detected per cell")
        axes[1].set_xlabel("Number of genes")
        axes[1].set_ylabel("Number of cells")

        # ATAC stats
        atac = adata[:, atac_mask].copy()
        if sp.issparse(atac.X):
            fragments_per_cell = np.array(atac.X.sum(axis=1)).flatten()
            peaks_per_cell = np.array((atac.X > 0).sum(axis=1)).flatten()
        else:
            fragments_per_cell = atac.X.sum(axis=1)
            peaks_per_cell = (atac.X > 0).sum(axis=1)

        print(f"\n  --- ATAC Peaks ---")
        print(f"  Shape: {atac.shape}")
        print(f"  Fragments/cell: median={np.median(fragments_per_cell):.0f}, "
              f"mean={np.mean(fragments_per_cell):.0f}")
        print(f"  Peaks/cell:     median={np.median(peaks_per_cell):.0f}, "
              f"mean={np.mean(peaks_per_cell):.0f}")

        axes[2].hist(fragments_per_cell, bins=50, edgecolor="black")
        axes[2].set_title("PBMC: ATAC fragments per cell")
        axes[2].set_xlabel("Total fragments")
        axes[2].set_ylabel("Number of cells")

        plt.tight_layout()
        fig.savefig(OUT_DIR / "pbmc_distributions.png", dpi=150)
        print(f"\n  Saved: {OUT_DIR / 'pbmc_distributions.png'}")
        plt.close(fig)

        # Top expressed genes
        fig, ax = plt.subplots(figsize=(10, 5))
        sc.pl.highest_expr_genes(gex, n_top=20, ax=ax, show=False)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "pbmc_top_genes.png", dpi=150)
        print(f"  Saved: {OUT_DIR / 'pbmc_top_genes.png'}")
        plt.close(fig)

    else:
        print(f"  [SKIP] {h5_path.name} not found")

    # --- Peak annotations ---
    annot_path = pbmc_dir / "pbmc_unsorted_10k_atac_peak_annotation.tsv"
    if annot_path.exists():
        peak_annot = pd.read_csv(annot_path, sep="\t")
        print(f"\n  --- Peak Annotations ---")
        print(f"  Shape: {peak_annot.shape}")
        print(f"  Columns: {list(peak_annot.columns)}")
        if "peak_type" in peak_annot.columns:
            print(f"  Peak types:\n{peak_annot['peak_type'].value_counts().to_string()}")

    # --- Peaks BED ---
    bed_path = pbmc_dir / "pbmc_unsorted_10k_atac_peaks.bed"
    if bed_path.exists():
        peaks_bed = pd.read_csv(bed_path, sep="\t", header=None, names=["chr", "start", "end"])
        print(f"\n  --- Peaks BED ---")
        print(f"  Total peaks: {len(peaks_bed)}")
        print(f"  Chromosomes: {sorted(peaks_bed['chr'].unique())}")
        peak_widths = peaks_bed["end"] - peaks_bed["start"]
        print(f"  Peak width: median={peak_widths.median():.0f}bp, "
              f"mean={peak_widths.mean():.0f}bp")

    # --- Summary CSV ---
    summary_path = pbmc_dir / "pbmc_unsorted_10k_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        print(f"\n  --- Run Summary ---")
        for _, row in summary.iterrows():
            for col in summary.columns:
                print(f"  {col}: {row[col]}")


# =============================================================================
#  Retinal Development (unpaired scRNA-seq + scATAC-seq)
# =============================================================================


def load_mtx_with_annotations(prefix: str, data_dir: Path):
    """Load a GEO-style sparse matrix + cell/gene/peak annotations."""
    mtx_path = data_dir / f"{prefix}_raw_matrix.mtx.gz"
    cell_path = data_dir / f"{prefix}_cell_annotation.txt.gz"

    # Try both gene and peak annotations
    gene_path = data_dir / f"{prefix}_gene_annotation.txt.gz"
    peak_path = data_dir / f"{prefix}_peak_annotation.txt.gz"

    if not mtx_path.exists():
        return None, None, None

    print(f"  Reading {mtx_path.name} ...")
    mat = sio.mmread(str(mtx_path))
    if sp.issparse(mat):
        mat = mat.tocsr()
    print(f"  Raw matrix shape: {mat.shape}")

    cells = None
    if cell_path.exists():
        cells = pd.read_csv(cell_path, sep="\t", compression="gzip")
        print(f"  Cell annotations: {cells.shape[0]} cells, columns={list(cells.columns)}")

    features = None
    if gene_path.exists():
        features = pd.read_csv(gene_path, sep="\t", compression="gzip")
        print(f"  Gene annotations: {features.shape[0]} genes, columns={list(features.columns)}")
    elif peak_path.exists():
        features = pd.read_csv(peak_path, sep="\t", compression="gzip")
        print(f"  Peak annotations: {features.shape[0]} peaks, columns={list(features.columns)}")

    return mat, cells, features


def inspect_retinal():
    section("Retinal Development - Mouse (unpaired)")
    ret_dir = DATA_DIR / "retinal_GSE181251"

    if not ret_dir.exists():
        print(f"  [SKIP] {ret_dir} not found")
        return

    # List all files
    files = sorted(ret_dir.glob("*"))
    print(f"  Files in {ret_dir}:")
    for f in files:
        size_mb = f.stat().st_size / 1e6
        print(f"    {f.name:60s} {size_mb:8.1f} MB")

    # --- Main developmental scATAC-seq ---
    print(f"\n  --- Single Cell ATACseq (developmental timepoints) ---")
    mat, cells, peaks = load_mtx_with_annotations(
        "GSE181251_Single_Cell_ATACseq", ret_dir
    )

    if mat is not None:
        print(f"  Matrix: {mat.shape[0]} x {mat.shape[1]}")
        if sp.issparse(mat):
            nnz = mat.nnz
            total = mat.shape[0] * mat.shape[1]
            print(f"  Sparsity: {1 - nnz / total:.2%} zeros")
            print(f"  Non-zero entries: {nnz:,}")

        if cells is not None:
            print(f"\n  Cell annotation columns: {list(cells.columns)}")
            # Look for cell type column
            for col in cells.columns:
                n_unique = cells[col].nunique()
                if n_unique < 50:
                    print(f"  '{col}' ({n_unique} unique values):")
                    print(f"    {cells[col].value_counts().to_dict()}")

            # Plot cell type distribution
            type_col = None
            for candidate in ["cell_type", "CellType", "celltype", "type", "cluster"]:
                if candidate in cells.columns:
                    type_col = candidate
                    break
            # If no standard name, pick first column with <30 unique values
            if type_col is None:
                for col in cells.columns:
                    if cells[col].nunique() < 30 and cells[col].dtype == object:
                        type_col = col
                        break

            if type_col is not None:
                fig, ax = plt.subplots(figsize=(10, 5))
                cells[type_col].value_counts().plot.barh(ax=ax)
                ax.set_title(f"Retinal scATAC-seq: {type_col} distribution")
                ax.set_xlabel("Number of cells")
                plt.tight_layout()
                fig.savefig(OUT_DIR / "retinal_atacseq_celltypes.png", dpi=150)
                print(f"\n  Saved: {OUT_DIR / 'retinal_atacseq_celltypes.png'}")
                plt.close(fig)

    # --- Perturbation datasets (brief summary) ---
    perturbation_prefixes = [
        ("GSE181251_E14E16_NFI_overexpression_scRNAseq", "E14-E16 NFI OE scRNA-seq"),
        ("GSE181251_E14E16_overexpression_scATACseq", "E14-E16 OE scATAC-seq"),
        ("GSE181251_E14P0_NFI_overexpression_scRNAseq", "E14-P0 NFI OE scRNA-seq"),
        ("GSE181251_P0P5_KO", "P0-P5 KO scRNA-seq"),
        ("GSE181251_P0P5_Overexpression", "P0-P5 OE scRNA-seq"),
        ("GSE181251_P2_Nfi_KO_scRNAseq", "P2 Nfi KO scRNA-seq"),
        ("GSE181251_P2_KO_scATACseq", "P2 KO scATAC-seq"),
        ("GSE181251_P14_KO_scATACseq", "P14 KO scATAC-seq"),
    ]

    print(f"\n  --- Perturbation Datasets (summary) ---")
    summary_rows = []
    for prefix, label in perturbation_prefixes:
        mtx_path = ret_dir / f"{prefix}_raw_matrix.mtx.gz"
        if mtx_path.exists():
            mat = sio.mmread(str(mtx_path))
            summary_rows.append({
                "dataset": label,
                "rows": mat.shape[0],
                "cols": mat.shape[1],
                "nnz": mat.nnz if sp.issparse(mat) else "dense",
            })
            print(f"  {label:40s}  {mat.shape[0]:>8,} x {mat.shape[1]:>8,}")

    if summary_rows:
        fig, ax = plt.subplots(figsize=(10, 5))
        df = pd.DataFrame(summary_rows)
        ax.barh(df["dataset"], df["rows"])
        ax.set_xlabel("Number of cells/rows")
        ax.set_title("Retinal GSE181251: dataset sizes")
        plt.tight_layout()
        fig.savefig(OUT_DIR / "retinal_dataset_sizes.png", dpi=150)
        print(f"\n  Saved: {OUT_DIR / 'retinal_dataset_sizes.png'}")
        plt.close(fig)


# =============================================================================
#  Main
# =============================================================================

if __name__ == "__main__":
    print(f"Python: {sys.version}")
    print(f"scanpy: {sc.__version__}")
    print(f"anndata: {ad.__version__}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Data directory: {DATA_DIR.resolve()}")
    print(f"Output directory: {OUT_DIR.resolve()}")

    inspect_pbmc()
    inspect_retinal()

    section("Done")
    print(f"  Figures saved to: {OUT_DIR.resolve()}")
    print(f"  Files:")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"    {f.name}")
