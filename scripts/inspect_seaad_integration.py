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


def _scatter_by_category(
    ax, coords, labels, title,
    top_n=12, point_size=2, palette=None, category_order=None,
):
    """Color a scatter by category, gray for everything outside top_n labels.

    palette : optional list of colors; cycled across categories. If None,
        uses tab20 (good for many categories) or a colorblind-friendly
        two-color palette when there are exactly 2 categories.
    category_order : optional list specifying which categories to plot
        first / in legend order; defaults to most-frequent first.
    """
    s = pd.Series(labels)
    if category_order is not None:
        cats = [c for c in category_order if c in s.unique()]
    else:
        cats = s.value_counts().head(top_n).index.tolist()
    if palette is None:
        if len(cats) == 2:
            # Okabe-Ito blue + vermillion, colorblind-safe and high contrast.
            palette = ["#0072B2", "#D55E00"]
        else:
            palette = [plt.cm.tab20(x) for x in np.linspace(0, 1, len(cats))]
    for lab, c in zip(cats, palette):
        m = s == lab
        ax.scatter(coords[m, 0], coords[m, 1], s=point_size, c=[c],
                   label=str(lab), alpha=0.7, linewidths=0)
    other = ~s.isin(cats)
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

    if "X_wnn_umap" in mdata.obsm:
        umap_key = "X_wnn_umap"
    elif "X_umap" in mdata.obsm:
        umap_key = "X_umap"
    else:
        print(f"  no UMAP in mdata.obsm: {list(mdata.obsm.keys())} — skipping plot")
        return
    print(f"  using {umap_key}")

    # Combine candidate columns from top-level obs AND per-modality obs.
    # The paired h5mu was written before our slim-obs edit, so columns can
    # live in either place depending on muon's pull-on-update behavior.
    def _lookup(col_candidates):
        for src in [mdata.obs] + [mdata.mod[m].obs for m in mdata.mod]:
            for c in col_candidates:
                if c in src.columns and src.index.equals(mdata.obs.index):
                    return src[c]
        return None

    cell_type_series = _lookup(["Subclass", "cell_type"])
    donor_series = _lookup(["Donor ID", "donor_id"])

    if cell_type_series is None:
        print(
            f"  no Subclass/cell_type column found in mdata.obs or any "
            f"modality obs — top-level cols: {list(mdata.obs.columns)}, "
            f"rna mod cols: {list(mdata.mod['rna'].obs.columns)[:20]}"
        )
        return

    # Stratified subsample by cell type for ~10k cells.
    obs_for_strat = pd.DataFrame({"cell_type": cell_type_series.values})
    idx = _stratified_sample(obs_for_strat, "cell_type", n_total=10_000)
    coords = np.asarray(mdata.obsm[umap_key])[idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _scatter_by_category(
        axes[0], coords, cell_type_series.astype(str).values[idx],
        title=f"Paired WNN UMAP — cell type (n={len(idx):,})",
    )
    if donor_series is not None:
        _scatter_by_category(
            axes[1], coords, donor_series.astype(str).values[idx],
            title=f"Paired WNN UMAP — donor (n={len(idx):,})",
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

    # Plot all cells per donor (no subsampling — per-donor sizes are
    # typically 1k–30k, which matplotlib handles fine with small markers).
    # Modality colors are forced to a high-contrast palette; cell type uses
    # the default tab20 across the top subclasses.
    for i, donor in enumerate(top_donors):
        donor_idx = np.where((a.obs[donor_col] == donor).to_numpy())[0]
        sub_obs = a.obs.iloc[donor_idx]
        coords = np.asarray(a.obsm["X_umap"])[donor_idx]
        # Scale marker size with cell count so dense donors stay readable.
        pt = max(0.3, 4.0 / np.sqrt(max(len(donor_idx), 1)))

        _scatter_by_category(
            axes[i, 0], coords, sub_obs["modality"].astype(str).values,
            title=f"{donor}: modality (n={len(donor_idx):,})",
            top_n=2, point_size=pt,
            palette=["#0072B2", "#D55E00"],
            category_order=["RNA", "ATAC"],
        )
        ct_col = "cell_type" if "cell_type" in sub_obs.columns else "Subclass"
        _scatter_by_category(
            axes[i, 1], coords, sub_obs[ct_col].astype(str).values,
            title=f"{donor}: {ct_col} (n={len(donor_idx):,})",
            point_size=pt,
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
