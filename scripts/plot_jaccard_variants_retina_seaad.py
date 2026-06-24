"""Plot retina and SEA-AD Jaccard similarity to kScReNI for each formula."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VARIANTS = ["w_ij", "w_refined", "w_hybrid", "z_ij", "dw"]
LABELS = {
    "w_ij": r"$w_{i,j}$",
    "w_refined": r"$s_{i,j}$",
    "w_hybrid": r"$v_{i,j}$",
    "z_ij": r"$z_{i,j}$",
    "dw": r"$d_{i,j}^{w}$",
}
COLORS = {
    "w_ij": "#4C78A8",
    "w_refined": "#1B9E77",
    "w_hybrid": "#E6862A",
    "z_ij": "#8E6BBE",
    "dw": "#D65F5F",
}
DATASET_LABELS = {
    "retina": "Mouse retina",
    "seaad": "SEA-AD MTG",
}


def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = [variant for variant in VARIANTS if variant not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing variant columns: {missing}")
    df = df.copy()
    df["dataset"] = label
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-retina",
        default="output/weight_decomposition_retina/jaccard_variants/jaccard_per_cell.csv",
    )
    parser.add_argument(
        "--csv-seaad",
        default="output/seaad_grn/weight_decomposition/jaccard_variants/jaccard_per_cell.csv",
    )
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    csv_retina = Path(args.csv_retina)
    csv_seaad = Path(args.csv_seaad)
    out_dir = Path(args.out_dir) if args.out_dir else csv_retina.parent

    combined = pd.concat(
        [load_csv(csv_retina, "retina"), load_csv(csv_seaad, "seaad")],
        ignore_index=True,
    )
    datasets = ["retina", "seaad"]
    stats = {
        dataset: {
            variant: (
                float(group[variant].mean()),
                0.0 if pd.isna(group[variant].std()) else float(group[variant].std()),
            )
            for variant in VARIANTS
        }
        for dataset, group in combined.groupby("dataset")
    }
    missing_datasets = [dataset for dataset in datasets if dataset not in stats]
    if missing_datasets:
        raise ValueError(f"Missing datasets in inputs: {missing_datasets}")

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

    fig, ax = plt.subplots(figsize=(5.7, 3.75))

    x = np.arange(len(datasets))
    group_width = 0.76
    bar_width = group_width / len(VARIANTS)
    offsets = (np.arange(len(VARIANTS)) - (len(VARIANTS) - 1) / 2) * bar_width

    for vi, variant in enumerate(VARIANTS):
        means = [stats[dataset][variant][0] for dataset in datasets]
        sds = [stats[dataset][variant][1] for dataset in datasets]
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
            yerr=sds,
            fmt="none",
            ecolor="#222222",
            elinewidth=1.0,
            capsize=2.5,
            capthick=1.0,
            zorder=4,
        )

    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_ylabel(r"Jaccard similarity to $k_{i,j}$", labelpad=8)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[dataset] for dataset in datasets])
    ax.set_xlim(-0.42, len(datasets) - 0.58)
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.8, zorder=0)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=5,
        frameon=False,
        columnspacing=1.6,
        handlelength=1.4,
    )
    fig.subplots_adjust(left=0.13, right=0.995, top=0.985, bottom=0.28)

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "jaccard_variants.png"
    pdf_path = out_dir / "jaccard_variants.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved chart to {png_path}")
    print(f"Saved chart to {pdf_path}")


if __name__ == "__main__":
    main()
