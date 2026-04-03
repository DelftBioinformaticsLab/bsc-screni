"""Diagnostic UMAPs for Phase 1 integration results."""
import sys
from pathlib import Path

sys.path.insert(0, "src")

import anndata as ad
import muon as mu
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("output/data_inspection")
OUT.mkdir(parents=True, exist_ok=True)

# ---- Retinal (unpaired) ----
ret_path = Path("data/processed/retinal_integrated.h5ad")
if ret_path.exists():
    print("=== Retinal Integration ===")
    merged = ad.read_h5ad(ret_path)
    print(f"Shape: {merged.shape}")
    print(f"Datatypes: {merged.obs['datatype'].value_counts().to_dict()}")
    print(f"Cell types: {merged.obs['cell_type'].value_counts(dropna=False).to_dict()}")

    fig, axes = plt.subplots(1, 3, figsize=(30, 8))

    sc.pl.umap(merged, color="datatype", ax=axes[0], show=False,
               title="Retinal: data type (RNA vs ATAC)")
    sc.pl.umap(merged, color="cell_type", ax=axes[1], show=False,
               title="Retinal: cell type")

    # Color by timepoint if available
    if "timepoint" in merged.obs.columns:
        sc.pl.umap(merged, color="timepoint", ax=axes[2], show=False,
                   title="Retinal: timepoint")
    else:
        axes[2].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "retinal_integration_umap.png", dpi=150)
    print(f"Saved: {OUT / 'retinal_integration_umap.png'}")
    plt.close(fig)
else:
    print(f"Retinal integration not found: {ret_path}")

# ---- PBMC (paired) ----
pbmc_path = Path("data/processed/pbmc_integrated.h5mu")
if pbmc_path.exists():
    print("\n=== PBMC Integration ===")
    mdata = mu.read(pbmc_path)
    print(f"Shape: {mdata.shape}")
    print(f"Clusters: {mdata.obs['leiden'].nunique()}")

    # Get cell type from rna modality
    ct_col = "rna:cell_type" if "rna:cell_type" in mdata.obs.columns else "cell_type"

    fig, axes = plt.subplots(1, 3, figsize=(30, 8))

    sc.pl.umap(mdata, color=ct_col, ax=axes[0], show=False,
               title="PBMC: cell type (WNN UMAP)")
    sc.pl.umap(mdata, color="leiden", ax=axes[1], show=False,
               title="PBMC: Leiden clusters (WNN)")

    # Modality weights
    wt_col = "rna:mod_weight" if "rna:mod_weight" in mdata.obs.columns else None
    if wt_col:
        sc.pl.umap(mdata, color=wt_col, ax=axes[2], show=False,
                   title="PBMC: RNA modality weight", cmap="RdYlBu_r")
    else:
        axes[2].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "pbmc_integration_umap.png", dpi=150)
    print(f"Saved: {OUT / 'pbmc_integration_umap.png'}")
    plt.close(fig)
else:
    print(f"PBMC integration not found: {pbmc_path}")
