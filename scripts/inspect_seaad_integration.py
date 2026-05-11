"""Phase 1 SEA-AD integration QC: UMAP figures + summary stats.

Reads the artifacts produced by `pixi run integrate-seaad`:
  - data/processed/seaad/seaad_paired_integrated.h5mu
  - data/processed/seaad/seaad_unpaired_integrated.h5ad
  - data/processed/seaad/seaad_unpaired_donor_summary.csv
  - data/processed/seaad/seaad_unpaired_nn_pairs.csv

Produces two PNGs under output/data_inspection/:
  - seaad_paired_integration_umap.png   (Subclass, Donor on WNN UMAP)
  - seaad_unpaired_integration_umap.png (top-4 donors x [modality, Subclass])

And prints donor-summary statistics to stdout (pair fraction, cell-type
agreement, anchor distribution).

Subsampling is aggressive: the paired UMAP plots ~10k cells stratified by
Subclass, and the unpaired plot uses up to 2000 cells per donor. The
unpaired UMAPs live in per-donor coordinate systems (per-donor Harmony),
so they are plotted one donor per panel rather than as one global UMAP.
"""

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path("data/processed/seaad")
OUT_DIR = Path("output/data_inspection")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAIRED_H5MU = DATA_DIR / "seaad_paired_integrated.h5mu"
UNPAIRED_H5AD = DATA_DIR / "seaad_unpaired_integrated.h5ad"
DONOR_SUMMARY_CSV = DATA_DIR / "seaad_unpaired_donor_summary.csv"
PAIRS_CSV = DATA_DIR / "seaad_unpaired_nn_pairs.csv"


def _stratified_sample(
    df: pd.DataFrame, col: str, n_total: int, seed: int = 0
) -> np.ndarray:
    """Return positional indices for a roughly stratified subsample."""
    rng = np.random.RandomState(seed)
    groups = df.groupby(col, observed=True).indices
    per_group = max(1, n_total // max(len(groups), 1))
    picks = []
    for vals in groups.values():
        k = min(per_group, len(vals))
        picks.extend(rng.choice(vals, size=k, replace=False).tolist())
    return np.array(sorted(picks))


def _scatter_by_category(ax, coords, labels, title, top_n=12, point_size=2):
    """Color a scatter by category, gray for everything outside top_n labels."""
    s = pd.Series(labels)
    top = s.value_counts().head(top_n).index.tolist()
    colors = plt.cm.tab20(np.linspace(0, 1, len(top)))
    for lab, c in zip(top, colors):
        m = s == lab
        ax.scatter(coords[m, 0], coords[m, 1], s=point_size, c=[c],
                   label=str(lab), alpha=0.7, linewidths=0)
    other = ~s.isin(top)
    if other.any():
        ax.scatter(coords[other, 0], coords[other, 1], s=point_size * 0.5,
                   c="lightgray", alpha=0.3, linewidths=0, label="other")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    ax.legend(markerscale=4, loc="best", fontsize=6, frameon=False)


def plot_paired() -> None:
    if not PAIRED_H5MU.exists():
        print(f"SKIP paired: {PAIRED_H5MU} missing")
        return
    print(f"\nReading {PAIRED_H5MU} ...")
    import muon as mu  # noqa: WPS433 — lazy import (heavy)
    mdata = mu.read(str(PAIRED_H5MU))
    print(f"  shape: {mdata.shape}")
    print(f"  modalities: {list(mdata.mod.keys())}")

    obs = mdata.obs
    if "X_wnn_umap" in mdata.obsm:
        umap_key = "X_wnn_umap"
    elif "X_umap" in mdata.obsm:
        umap_key = "X_umap"
    else:
        print(f"  no UMAP in mdata.obsm: {list(mdata.obsm.keys())} — skipping plot")
        return
    print(f"  using {umap_key}")

    # Stratified subsample by Subclass for ~10k cells.
    if "Subclass" in obs.columns:
        strat_col = "Subclass"
    elif "cell_type" in obs.columns:
        strat_col = "cell_type"
    else:
        strat_col = obs.columns[0]
    idx = _stratified_sample(obs, strat_col, n_total=10_000)
    coords = np.asarray(mdata.obsm[umap_key])[idx]
    sub_obs = obs.iloc[idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _scatter_by_category(
        axes[0], coords, sub_obs[strat_col].astype(str).values,
        title=f"Paired WNN UMAP — {strat_col} (n={len(idx):,})",
    )
    donor_col = "Donor ID" if "Donor ID" in sub_obs.columns else (
        "donor_id" if "donor_id" in sub_obs.columns else None
    )
    if donor_col is not None:
        _scatter_by_category(
            axes[1], coords, sub_obs[donor_col].astype(str).values,
            title=f"Paired WNN UMAP — {donor_col} (n={len(idx):,})",
        )
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "seaad_paired_integration_umap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_unpaired() -> None:
    if not UNPAIRED_H5AD.exists():
        print(f"SKIP unpaired UMAP: {UNPAIRED_H5AD} missing")
        return
    print(f"\nReading {UNPAIRED_H5AD} (backed) ...")
    a = ad.read_h5ad(UNPAIRED_H5AD, backed="r")
    print(f"  shape: {a.shape}")
    print(f"  obsm keys: {list(a.obsm.keys())}")

    donor_col = "Donor ID" if "Donor ID" in a.obs.columns else "donor_id"
    if donor_col not in a.obs.columns:
        print(f"  no donor column — skipping plot")
        a.file.close()
        return

    donor_counts = a.obs[donor_col].value_counts()
    top_donors = donor_counts.head(4).index.tolist()
    print(f"  top donors: {top_donors}")

    fig, axes = plt.subplots(len(top_donors), 2, figsize=(12, 4 * len(top_donors)))
    if len(top_donors) == 1:
        axes = axes[None, :]

    rng = np.random.RandomState(0)
    for i, donor in enumerate(top_donors):
        donor_mask = (a.obs[donor_col] == donor).to_numpy()
        donor_idx = np.where(donor_mask)[0]
        if len(donor_idx) > 2000:
            donor_idx = rng.choice(donor_idx, size=2000, replace=False)
            donor_idx.sort()

        sub_obs = a.obs.iloc[donor_idx]
        coords = np.asarray(a.obsm["X_umap"])[donor_idx]

        _scatter_by_category(
            axes[i, 0], coords, sub_obs["modality"].astype(str).values,
            title=f"{donor}: modality (n={len(donor_idx):,})",
            top_n=2,
        )
        ct_col = "cell_type" if "cell_type" in sub_obs.columns else "Subclass"
        _scatter_by_category(
            axes[i, 1], coords, sub_obs[ct_col].astype(str).values,
            title=f"{donor}: {ct_col} (n={len(donor_idx):,})",
        )

    plt.tight_layout()
    out = OUT_DIR / "seaad_unpaired_integration_umap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    a.file.close()


def print_summary() -> None:
    if not DONOR_SUMMARY_CSV.exists():
        print(f"\nSKIP summary: {DONOR_SUMMARY_CSV} missing")
        return
    print(f"\nReading {DONOR_SUMMARY_CSV} ...")
    df = pd.read_csv(DONOR_SUMMARY_CSV, index_col=0)
    aligned = df[df["status"] == "aligned"]
    skipped = df[df["status"] == "skipped"]
    print(f"  donors: {len(df)} total, {len(aligned)} aligned, {len(skipped)} skipped")
    print(f"\n  n_pairs:")
    print(f"    total: {aligned['n_pairs'].sum():,}")
    print(f"    median per donor: {aligned['n_pairs'].median():.0f}")
    print(f"\n  pair_fraction (n_pairs / min(n_rna, n_atac)):")
    print(f"    mean: {aligned['pair_fraction'].mean():.3f}")
    print(f"    min:  {aligned['pair_fraction'].min():.3f}")
    print(f"    max:  {aligned['pair_fraction'].max():.3f}")
    print(f"\n  cell_type_agreement:")
    print(f"    mean: {aligned['cell_type_agreement'].mean():.3f}")
    print(f"    min:  {aligned['cell_type_agreement'].min():.3f}")
    print(f"    max:  {aligned['cell_type_agreement'].max():.3f}")
    print(f"\n  anchor distribution:")
    print(aligned["anchor"].value_counts().to_string())
    if len(skipped):
        print(f"\n  skipped donors:")
        for d, reason in skipped["skip_reason"].items():
            print(f"    {d}: {reason}")

    if PAIRS_CSV.exists():
        print(f"\nReading {PAIRS_CSV} ...")
        pairs = pd.read_csv(PAIRS_CSV)
        print(f"  pairs: {len(pairs):,}")
        if "cell_type" in pairs.columns:
            print(f"  per cell type (RNA-side):")
            for v, c in pairs["cell_type"].value_counts().head(20).items():
                print(f"    {v}: {c:,}")


def main() -> None:
    print_summary()
    plot_paired()
    plot_unpaired()
    print("\nDone.")


if __name__ == "__main__":
    main()
