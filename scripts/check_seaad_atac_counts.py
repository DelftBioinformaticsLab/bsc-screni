"""Check whether SEA-AD ATAC .X holds raw counts or normalized values.

The integration pipeline warns "no raw count layer found" for SEA-AD ATAC
because there is no named `UMIs`/`counts` layer. That warning is silent on
the actual content of .X: it may still be integer fragment counts, just
without a separate normalized version. This script reads a sample of .X
from the processed SEA-AD ATAC files and reports the values directly.

Outputs to stdout:
  - dtype, shape, layer list
  - sample .X stats: min, max, mean, sparsity
  - integer-valued check (within float tolerance)
  - histogram of value frequencies
  - first few var columns (peak metadata, sometimes mentions provenance)

Run via slurm/run_check_seaad_atac_counts.sh or interactively.
"""

from pathlib import Path

import anndata as ad
import numpy as np


CANDIDATES = [
    Path("data/processed/seaad/seaad_unpaired_atac.h5ad"),
    Path("data/processed/seaad/seaad_paired_atac.h5ad"),
]


def report(path: Path) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {path}")
    print(f"{'=' * 70}")

    if not path.exists():
        print(f"  MISSING")
        return

    a = ad.read_h5ad(path, backed="r")
    print(f"  shape: {a.shape}")
    print(f"  layers: {list(a.layers.keys()) if a.layers else 'none'}")
    print(f"  uns keys: {list(a.uns.keys())}")
    if "X_normalization" in a.uns:
        print(f"  uns['X_normalization']: {a.uns['X_normalization']}")

    # Sample first N cells
    n_sample = min(2000, a.n_obs)
    x = a.X[:n_sample]
    if hasattr(x, "toarray"):
        x = x.toarray()
    print(f"\n  .X[:{n_sample}] sample")
    print(f"    dtype: {x.dtype}")
    print(f"    min / max / mean: {x.min()} / {x.max()} / {x.mean():.4f}")
    print(f"    non-zero fraction: {(x != 0).mean():.4f}")
    is_int = np.allclose(x, np.round(x))
    print(f"    integer-valued (within tol): {is_int}")

    # Value histogram on non-zero entries
    nz = x[x != 0]
    if nz.size > 0:
        print(f"\n  Non-zero value distribution:")
        for q in (0.5, 0.9, 0.99, 0.999, 1.0):
            print(f"    quantile {q:.3f}: {np.quantile(nz, q):.4f}")
        # Count of integer 1s, 2s, 3s ... (signature of raw fragment counts)
        if is_int:
            xi = np.round(nz).astype(int)
            unique, counts = np.unique(xi, return_counts=True)
            top = sorted(zip(unique, counts), key=lambda t: -t[1])[:10]
            total = counts.sum()
            print(f"\n  Top integer values:")
            for v, c in top:
                print(f"    {v}: {c:,} ({c / total * 100:.2f}%)")

    # First var columns can mention "fragments" / "tile" / "binarized"
    print(f"\n  var columns: {list(a.var.columns)[:10]}")
    print(f"  first var rows:")
    print(a.var.head(3).to_string())

    a.file.close()


def verdict(path: Path) -> str:
    """Return a short string verdict on whether .X looks like raw counts."""
    if not path.exists():
        return "missing"
    a = ad.read_h5ad(path, backed="r")
    x = a.X[: min(2000, a.n_obs)]
    if hasattr(x, "toarray"):
        x = x.toarray()
    is_int = np.allclose(x, np.round(x))
    max_v = float(x.max())
    a.file.close()
    if is_int and max_v <= 200:
        return f"LIKELY RAW COUNTS (integer, max={max_v:.0f})"
    if is_int and max_v <= 1:
        return f"LIKELY BINARIZED (0/1 only)"
    if not is_int and 0 <= x.min() and max_v <= 20:
        return f"LIKELY NORMALIZED (float, max={max_v:.4f})"
    return f"AMBIGUOUS (int={is_int}, min={x.min()}, max={max_v})"


def main() -> None:
    for p in CANDIDATES:
        report(p)
    print(f"\n{'=' * 70}")
    print("  Verdict")
    print(f"{'=' * 70}")
    for p in CANDIDATES:
        print(f"  {p.name}: {verdict(p)}")


if __name__ == "__main__":
    main()
