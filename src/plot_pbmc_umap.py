"""Quick UMAP plot of PBMC cell types.

Uses the SAME preprocessing as CellTypist's majority voting
(2500 HVGs, 50 PCs, k=10) so that the UMAP is consistent
with the annotation graph.
"""
from pathlib import Path

import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("output/data_inspection")
OUT.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad("data/processed/pbmc_rna.h5ad")
print(f"Shape: {adata.shape}")
print(f"Cell types:\n{adata.obs['cell_type'].value_counts(dropna=False).to_string()}")

# Drop unmapped cells
adata = adata[adata.obs["cell_type"].notna()].copy()
print(f"After dropping unmapped: {adata.n_obs}")

# Normalize (same as what CellTypist expects)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Match CellTypist's internal graph construction exactly:
# - 2500 HVGs, scale max_value=10, 50 PCs, k=10
sc.pp.highly_variable_genes(adata, n_top_genes=2500)
adata_hvg = adata[:, adata.var["highly_variable"]].copy()
sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, n_comps=50)
sc.pp.neighbors(adata_hvg, n_neighbors=10, n_pcs=50)
sc.tl.umap(adata_hvg)

# Transfer UMAP back
adata.obsm["X_umap"] = adata_hvg.obsm["X_umap"]

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.umap(adata, color="cell_type", ax=ax, show=False,
           title="PBMC cell types (CellTypist, Healthy_COVID19_PBMC)")
fig.tight_layout()
fig.savefig(OUT / "pbmc_celltypes_umap.png", dpi=150)
print(f"Saved: {OUT / 'pbmc_celltypes_umap.png'}")

# Also plot fine-grained if available
if "cell_type_fine" in adata.obs.columns:
    fig, ax = plt.subplots(figsize=(12, 8))
    sc.pl.umap(adata, color="cell_type_fine", ax=ax, show=False,
               title="CellTypist fine-grained labels")
    fig.tight_layout()
    fig.savefig(OUT / "pbmc_celltypes_fine_umap.png", dpi=150)
    print(f"Saved: {OUT / 'pbmc_celltypes_fine_umap.png'}")
