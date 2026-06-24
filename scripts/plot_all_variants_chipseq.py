"""Plot ChIP-seq precision by retinal cell type for all formula variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = ["w_ij", "k_ij", "w_refined", "w_hybrid", "z_ij", "dw"]
LABELS = {
    "w_ij": r"$w_{i,j}$",
    "k_ij": r"$k_{i,j}$",
    "w_refined": r"$s_{i,j}$",
    "w_hybrid": r"$v_{i,j}$",
    "z_ij": r"$z_{i,j}$",
    "dw": r"$d_{i,j}^{w}$",
}
COLORS = {
    "w_ij": "#4C78A8",
    "k_ij": "#9E9E9E",
    "w_refined": "#1B9E77",
    "w_hybrid": "#E6862A",
    "z_ij": "#8E6BBE",
    "dw": "#D65F5F",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="output/weight_decomposition_retina/precision_recall_celltype",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = data_dir / "all_variants_per_celltype.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    data = pd.read_csv(csv_path)
    required = {"variant", "cell_type", "precision_mean", "precision_se", "n_cells"}
    if not required.issubset(data.columns):
        raise ValueError(f"{csv_path} is missing columns: {sorted(required - set(data.columns))}")
    missing_variants = set(ORDER) - set(data["variant"])
    if missing_variants:
        raise ValueError(f"Results are missing variants: {sorted(missing_variants)}")

    metadata_path = data_dir / "all_variants_metadata.json"
    top_k = 500
    if metadata_path.exists():
        top_k = json.loads(metadata_path.read_text(encoding="utf-8")).get("top_k", top_k)
    _ = top_k

    preferred_cell_types = ["MG", "RPC1", "RPC2", "RPC3"]
    available_cell_types = set(data["cell_type"].astype(str))
    cell_types = [ct for ct in preferred_cell_types if ct in available_cell_types]
    cell_types.extend(sorted(available_cell_types - set(cell_types)))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })

    fig, ax = plt.subplots(figsize=(6.4, 3.9))

    x = np.arange(len(cell_types))
    group_width = 0.84
    bar_width = group_width / len(ORDER)
    offsets = (np.arange(len(ORDER)) - (len(ORDER) - 1) / 2) * bar_width

    for vi, variant in enumerate(ORDER):
        subset = data[data["variant"] == variant].set_index("cell_type")
        means = [float(subset.loc[ct, "precision_mean"]) for ct in cell_types]
        ses = [
            0.0 if pd.isna(subset.loc[ct, "precision_se"])
            else float(subset.loc[ct, "precision_se"])
            for ct in cell_types
        ]
        ax.bar(
            x + offsets[vi],
            means,
            width=bar_width * 0.86,
            color=COLORS[variant],
            label=LABELS[variant],
            zorder=3,
        )
        ax.errorbar(
            x + offsets[vi],
            means,
            yerr=ses,
            fmt="none",
            ecolor="#222222",
            elinewidth=1.0,
            capsize=2.5,
            capthick=1.0,
            zorder=4,
        )

    y_max_data = float((data["precision_mean"] + data["precision_se"].fillna(0)).max())
    y_max = max(0.01, np.ceil(y_max_data * 120) / 100)
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.linspace(0, y_max, 6))
    ax.set_yticklabels([f"{v:.2f}" for v in np.linspace(0, y_max, 6)])
    ax.set_ylabel("ChIP-seq precision", labelpad=8)

    ax.set_xticks(x)
    ax.set_xticklabels(cell_types)
    ax.set_xlim(-0.48, len(cell_types) - 0.52)
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.8, zorder=0)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
        columnspacing=2.4,
        handlelength=1.4,
    )
    fig.subplots_adjust(left=0.115, right=0.995, top=0.985, bottom=0.29)

    data_dir.mkdir(parents=True, exist_ok=True)
    png_path = data_dir / "precision_by_celltype_all_variants.png"
    pdf_path = data_dir / "precision_by_celltype_all_variants.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved chart to {png_path}")
    print(f"Saved chart to {pdf_path}")


if __name__ == "__main__":
    main()
