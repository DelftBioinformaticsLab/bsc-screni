"""Create a compact, publication-ready clustering ARI line plot.

Reads clustering_ari.csv produced by analyze_clustering_ari_variants.py.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--data-dir", default="output/weight_decomposition_retina",
               help="Directory containing clustering_ari/clustering_ari.csv")
p.add_argument("--out-dir", default=None,
               help="Output directory (default: data-dir/clustering_ari)")
args = p.parse_args()

data_dir = Path(args.data_dir)
csv_path = data_dir / "clustering_ari" / "clustering_ari.csv"
out_dir = Path(args.out_dir) if args.out_dir else data_dir / "clustering_ari"
out_dir.mkdir(parents=True, exist_ok=True)

if not csv_path.exists():
    sys.exit(f"ERROR: {csv_path} not found.\nRun analyze_clustering_ari_variants.py first.")

df = pd.read_csv(csv_path)
k_cols = [c for c in df.columns if c.startswith("ari_top")]
if not k_cols:
    sys.exit("ERROR: no ari_top* columns found in CSV.")
PLOT_MIN_K = 400
PLOT_MAX_K = 1400
k_vals = sorted(
    k for k in (int(c.removeprefix("ari_top")) for c in k_cols)
    if PLOT_MIN_K <= k <= PLOT_MAX_K
)
if not k_vals:
    sys.exit(f"ERROR: no ari_top* columns found between {PLOT_MIN_K} and {PLOT_MAX_K}.")
k_cols = [f"ari_top{k}" for k in k_vals]

STYLE: dict[str, dict] = {
    "w_ij":      {"color": "#4C78A8", "ls": "-",
                  "label": r"$w_{i,j}$"},
    "k_ij":      {"color": "#9E9E9E", "ls": (0, (5, 2)),
                  "label": r"$k_{i,j}$"},
    "w_refined": {"color": "#1B9E77", "ls": (0, (3, 1.5)),
                  "label": r"$s_{i,j}$"},
    "w_hybrid":  {"color": "#E6862A", "ls": (0, (6, 1.5, 1.5, 1.5)),
                  "label": r"$v_{i,j}$"},
    "z_ij":      {"color": "#8E6BBE", "ls": (0, (1.5, 1.5)),
                  "label": r"$z_{i,j}$"},
    "dw":        {"color": "#D65F5F", "ls": (0, (7, 1.5, 1.5, 1.5, 1.5, 1.5)),
                  "label": r"$d_{i,j}^{w}$"},
}

all_y = df[k_cols].to_numpy(dtype=float).ravel()
finite_y = [v for v in all_y if math.isfinite(v)]
if not finite_y:
    sys.exit("ERROR: no finite ARI values found in CSV.")

# Round outward to tenths. Ensure enough vertical room for the line strokes.
data_min = min(finite_y)
data_max = max(finite_y)
y_min = max(-1.0, math.floor(data_min * 10) / 10)
y_max = min(1.0, math.ceil(data_max * 10) / 10)
if data_min - y_min < 0.03:
    y_min = max(-1.0, round(y_min - 0.1, 10))
if y_max - data_max < 0.03:
    y_max = min(1.0, round(y_max + 0.1, 10))
if y_max - y_min < 0.2:
    y_max = min(1.0, round(y_min + 0.2, 10))
    if y_max - y_min < 0.2:
        y_min = max(-1.0, round(y_max - 0.2, 10))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

MM = 1 / 25.4
FONT_SIZE = 7.5
LINE_WIDTH = 1.9
fig, ax = plt.subplots(figsize=(88 * MM, 68 * MM))

for order, (_, row) in enumerate(df.iterrows()):
    name = row["variant"]
    if name not in STYLE:
        continue
    st = STYLE[name]
    line, = ax.plot(
        k_vals, [row[f"ari_top{k}"] for k in k_vals],
        color=st["color"], linestyle=st["ls"], linewidth=LINE_WIDTH,
        alpha=0.92, solid_capstyle="round", dash_capstyle="round",
        label=st["label"],
        zorder=3 + order,
    )
    line.set_path_effects([pe.Stroke(linewidth=LINE_WIDTH + 0.8,
                                     foreground="white"), pe.Normal()])

x_ticks = [k for k in range(PLOT_MIN_K, PLOT_MAX_K + 1, 200)
           if k_vals[0] <= k <= k_vals[-1]]
if not x_ticks:
    x_ticks = k_vals
ax.set_xticks(x_ticks)
ax.set_xlabel("Top-$k$ edges per cell", fontsize=FONT_SIZE)
ax.set_ylabel("ARI", fontsize=FONT_SIZE)
ax.set_ylim(y_min, y_max)
x_pad = max(40, (k_vals[-1] - k_vals[0]) * 0.04)
ax.set_xlim(k_vals[0] - x_pad, k_vals[-1] + x_pad)
ax.tick_params(labelsize=FONT_SIZE)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, zorder=0)

# Indicate that the displayed ARI axis is truncated above zero.
if y_min > 0:
    break_style = dict(transform=ax.transAxes, color="black", clip_on=False,
                       linewidth=0.8)
    ax.plot((-0.018, 0.018), (-0.025, 0.025), **break_style)
    ax.plot((-0.018, 0.018), (0.015, 0.065), **break_style)

ax.legend(fontsize=FONT_SIZE, frameon=False, ncol=3, loc="upper center",
          bbox_to_anchor=(0.5, -0.26), columnspacing=1.15,
          handlelength=2.2, handletextpad=0.45, labelspacing=0.35)

fig.tight_layout(pad=0.4)
fig.subplots_adjust(bottom=0.29)

for ext in ("pdf", "png"):
    out = out_dir / f"clustering_ari_plot.{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=300 if ext == "png" else None)
    print(f"Saved -> {out}")

plt.close(fig)
