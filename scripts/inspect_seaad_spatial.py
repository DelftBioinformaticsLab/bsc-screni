"""Quick schema inspection of the SEA-AD MTG MERFISH h5ad.

Mirrors loading_seaad.inspect_seaad() but tailored for the spatial file:
- expects 2D spatial coords in obsm
- expects a small targeted gene panel (~140 genes), not transcriptome-wide
"""

from pathlib import Path

import anndata as ad
import numpy as np

PATH = Path("data/seaad_spatial/SEAAD_MTG_MERFISH.2024-12-11.h5ad")


def main() -> None:
    print(f"Reading {PATH} (backed)...")
    adata = ad.read_h5ad(PATH, backed="r")

    print(f"\nShape (cells x genes): {adata.shape}")
    print(f"X dtype: {adata.X.dtype if hasattr(adata.X, 'dtype') else type(adata.X)}")

    print(f"\nobsm keys: {list(adata.obsm.keys())}")
    for k, v in adata.obsm.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    print(f"\nuns keys: {list(adata.uns.keys())}")
    print(f"layers keys: {list(adata.layers.keys())}")

    print(f"\nvar columns ({len(adata.var.columns)}):")
    for col in adata.var.columns:
        print(f"  {col}: {adata.var[col].dtype}")
    print(f"First 10 genes: {adata.var_names[:10].tolist()}")
    print(f"Last 5 genes: {adata.var_names[-5:].tolist()}")

    print(f"\nobs columns ({len(adata.obs.columns)}):")
    for col in adata.obs.columns:
        print(f"  {col}: {adata.obs[col].dtype}")

    print(f"\nCategorical value counts:")
    for col in adata.obs.columns:
        s = adata.obs[col]
        if hasattr(s, "cat") or s.dtype == "object":
            vc = s.value_counts(dropna=False)
            n = len(vc)
            print(f"\n  {col} ({n} unique):")
            head = vc.head(15)
            for v, c in head.items():
                print(f"    {v}: {c:,}")
            if n > 15:
                print(f"    ... ({n - 15} more)")

    # Spatial coordinate ranges
    for k in ("spatial", "X_spatial", "X_umap"):
        if k in adata.obsm:
            arr = np.asarray(adata.obsm[k])
            print(f"\n{k} ranges:")
            for i in range(arr.shape[1]):
                print(f"  dim{i}: min={arr[:, i].min():.2f}, "
                      f"max={arr[:, i].max():.2f}, "
                      f"mean={arr[:, i].mean():.2f}")

    # Section / donor breakdown if columns exist
    for donor_col in ("donor_id", "Donor ID", "donor"):
        if donor_col in adata.obs.columns:
            print(f"\n{donor_col}: {adata.obs[donor_col].nunique()} unique")
            break
    for section_col in ("section_id", "section", "library_id", "sample_id"):
        if section_col in adata.obs.columns:
            print(f"{section_col}: {adata.obs[section_col].nunique()} unique")
            break

    # Sanity-check raw counts (load a small block of X)
    sample = adata.X[:1000]
    if hasattr(sample, "toarray"):
        sample = sample.toarray()
    print(f"\nX[:1000] sample stats:")
    print(f"  dtype: {sample.dtype}")
    print(f"  min={sample.min()}, max={sample.max()}, mean={sample.mean():.3f}")
    print(f"  integer-valued: {np.allclose(sample, sample.astype(int))}")
    print(f"  non-zero fraction: {(sample != 0).mean():.3f}")

    adata.file.close()


if __name__ == "__main__":
    main()
