"""Plot wScReNI weight composition by cell type.

Reads ``weight_composition_per_celltype.csv`` from
``analyze_weight_composition.py`` and writes a horizontal three-panel bar plot.
The three panels show the mean per-cell percent contribution of d_i,j^w,
r_i,j, and t_i,j.  Cell-type labels are shown only once on the
shared y-axis.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = (
    ("direct_j_w", r"$d_{i,j}^{w}$", "mean_pct_direct_j_w", "std_pct_direct_j_w", "#4E79A7"),
    ("peak_j_coef", r"$r_{i,j}$", "mean_pct_peak_j_coef", "std_pct_peak_j_coef", "#59A14F"),
    ("atac_boost", r"$t_{i,j}$", "mean_pct_atac_boost", "std_pct_atac_boost", "#F28E2B"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        default="output/weight_decomposition_retina/weight_composition",
        help="Directory containing weight_composition_per_celltype.csv.",
    )
    parser.add_argument("--csv", default=None, help="CSV path. Overrides --data-dir.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to the CSV parent.",
    )
    parser.add_argument(
        "--sort-by",
        default="peak_j_coef",
        choices=["peak_j_coef", "direct_j_w", "atac_boost", "cell_type"],
        help="Cell-type order on the y-axis.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Accepted for compatibility with existing run scripts; not drawn.",
    )
    return parser.parse_args()


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{csv_path} is empty")

    required = {"cell_type", *(mean for _, _, mean, _, _ in COMPONENTS)}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    return rows


def value(row: dict[str, str], col: str) -> float:
    return float(row[col])


def sort_cell_types(rows: list[dict[str, str]], sort_by: str) -> list[dict[str, str]]:
    sort_col = {
        "direct_j_w": "mean_pct_direct_j_w",
        "peak_j_coef": "mean_pct_peak_j_coef",
        "atac_boost": "mean_pct_atac_boost",
        "cell_type": "cell_type",
    }[sort_by]
    if sort_by == "cell_type":
        return sorted(rows, key=lambda row: row["cell_type"].lower())
    return sorted(rows, key=lambda row: value(row, sort_col), reverse=True)


def axis_bounds(values: np.ndarray, errors: np.ndarray) -> tuple[float, float, np.ndarray]:
    lower_data = float(np.nanmin(values - errors)) if len(values) else 0.0
    upper_data = float(np.nanmax(values + errors)) if len(values) else 0.0
    if not math.isfinite(upper_data) or upper_data <= 0:
        return 0.0, 1.0, np.array([0.0, 1.0])

    span = max(upper_data - lower_data, 1.0)
    if upper_data <= 8:
        step = 1
    elif upper_data <= 25:
        step = 5
    else:
        step = 10

    if lower_data >= 10:
        padding = max(0.08 * span, 1.0)
        lower = max(0.0, math.floor((lower_data - padding) / 10.0) * 10.0)
    else:
        padding = max(0.12 * upper_data, 0.5)
        lower = 0.0

    upper = min(100.0, math.ceil((upper_data + padding) / step) * step)
    if upper <= lower:
        upper = min(100.0, lower + step)
    if upper - lower < 2 * step and upper < 100:
        upper = min(100.0, lower + 2 * step)

    ticks = np.arange(lower, upper + 0.5 * step, step)
    return lower, upper, ticks


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv) if args.csv else Path(args.data_dir) / "weight_composition_per_celltype.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run analyze_weight_composition.py first.")

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [row for row in read_rows(csv_path) if row["cell_type"] != "Overall"]
    rows = sort_cell_types(rows, args.sort_by)
    if not rows:
        raise ValueError("No cell-type rows found")

    labels = [row["cell_type"] for row in rows]
    y = np.arange(len(rows))
    height = max(3.0, min(7.4, 0.30 * len(rows) + 1.0))
    bottom = 0.18 if len(rows) <= 6 else 0.11
    left = 0.085 if max(len(label) for label in labels) <= 8 else 0.185

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.5,
            "axes.titlesize": 10.8,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(6.8, height),
        sharey=True,
        gridspec_kw={"wspace": 0.14},
    )

    for ax, (_, label, mean_col, std_col, color) in zip(axes, COMPONENTS):
        values = np.array([value(row, mean_col) for row in rows], dtype=float)
        errors = np.array([value(row, std_col) if std_col in row and row[std_col] else 0.0 for row in rows], dtype=float)
        ax.barh(
            y,
            values,
            xerr=errors,
            height=0.68,
            color=color,
            edgecolor="none",
            error_kw={"ecolor": "#222222", "elinewidth": 1.1, "capsize": 2.5, "capthick": 1.1},
        )
        ax.set_title(label, pad=5)
        xmin, xmax, ticks = axis_bounds(values, errors)
        ax.set_xlim(xmin, xmax)
        ax.set_xticks(ticks)
        ax.grid(axis="x", color="#D2D2D2", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.tick_params(axis="x", length=3, pad=2)
        ax.tick_params(axis="y", length=0, pad=4)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
        ax.spines["left"].set_visible(False)
    fig.supxlabel("Mean contribution (%)", y=0.04, fontsize=10.2)

    fig.subplots_adjust(left=left, right=0.955, top=0.90, bottom=bottom)
    png_path = out_dir / "weight_composition_plot.png"
    pdf_path = out_dir / "weight_composition_plot.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved -> {png_path}")
    print(f"Saved -> {pdf_path}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
