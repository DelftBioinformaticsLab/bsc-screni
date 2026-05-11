"""Quick exploration of SEA-AD MERFISH: normalization, gene panel, one section plot."""

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

PATH = Path("data/seaad_spatial/SEAAD_MTG_MERFISH.2024-12-11.h5ad")
OUT = Path("output/data_inspection")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    adata = ad.read_h5ad(PATH, backed="r")

    print("=== uns metadata ===")
    for key in ("title", "X_normalization", "default_embedding", "batch_condition",
                "original_taxonomy_path"):
        if key in adata.uns:
            print(f"  {key}: {adata.uns[key]}")

    # Gene panel breakdown: real vs Blank controls
    var_names = adata.var_names.to_numpy()
    is_blank = np.char.startswith(var_names.astype(str), "Blank-")
    print(f"\n=== Gene panel ===")
    print(f"  Total features: {len(var_names)}")
    print(f"  Real genes: {(~is_blank).sum()}")
    print(f"  Blank controls: {is_blank.sum()}")

    # Full Subclass breakdown
    print(f"\n=== Subclass (full) ===")
    for v, c in adata.obs["Subclass"].value_counts().items():
        print(f"  {v}: {c:,}")

    # AD-relevant cell types we already use elsewhere in the pipeline
    SCRENI_TYPES = ["Microglia-PVM", "Astrocyte", "Oligodendrocyte", "L2/3 IT"]
    n_screni = adata.obs["Subclass"].isin(SCRENI_TYPES).sum()
    print(f"\n  Cells in ScReNI subclasses {SCRENI_TYPES}: "
          f"{n_screni:,} ({n_screni / adata.n_obs * 100:.1f}%)")

    # Pick the largest section and load it into memory for plotting
    section_counts = adata.obs["Section"].value_counts()
    top_section = section_counts.index[0]
    print(f"\n=== Plotting largest section ===")
    print(f"  Section: {top_section} ({section_counts.iloc[0]:,} cells)")

    mask = (adata.obs["Section"] == top_section).to_numpy()
    obs_sub = adata.obs.loc[mask]
    coords = np.asarray(adata.obsm["spatial"])[mask]
    layer = obs_sub["Layer annotation"].astype(str).to_numpy()
    subclass = obs_sub["Subclass"].astype(str).to_numpy()

    # Two-panel plot: by cortical layer, by subclass (top 6 + Other)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Panel 1: cortical layer
    layers = ["L1", "L2/3", "L4", "L5", "L6"]
    colors_layer = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    for lay, col in zip(layers, colors_layer):
        m = layer == lay
        axes[0].scatter(coords[m, 0], coords[m, 1], s=0.3, c=[col], label=lay, alpha=0.6)
    other = ~np.isin(layer, layers)
    axes[0].scatter(coords[other, 0], coords[other, 1], s=0.2, c="lightgray",
                    label="other/nan", alpha=0.3)
    axes[0].set_aspect("equal")
    axes[0].set_title(f"Cortical layer\n{top_section}")
    axes[0].legend(markerscale=8, loc="best", fontsize=8)
    axes[0].set_xlabel("spatial dim 0 (um)")
    axes[0].set_ylabel("spatial dim 1 (um)")

    # Panel 2: top subclasses
    top_sub = obs_sub["Subclass"].value_counts().head(6).index.tolist()
    colors_sub = plt.cm.tab10(np.linspace(0, 1, len(top_sub)))
    for sc, col in zip(top_sub, colors_sub):
        m = subclass == sc
        axes[1].scatter(coords[m, 0], coords[m, 1], s=0.3, c=[col], label=sc, alpha=0.6)
    other_sub = ~np.isin(subclass, top_sub)
    axes[1].scatter(coords[other_sub, 0], coords[other_sub, 1], s=0.2, c="lightgray",
                    label="other", alpha=0.3)
    axes[1].set_aspect("equal")
    axes[1].set_title(f"Top 6 cell subclasses\n{top_section}")
    axes[1].legend(markerscale=8, loc="best", fontsize=8)
    axes[1].set_xlabel("spatial dim 0 (um)")

    plt.tight_layout()
    out_path = OUT / "seaad_spatial_section_overview.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")

    adata.file.close()


if __name__ == "__main__":
    main()
