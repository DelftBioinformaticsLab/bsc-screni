"""Compute wScReNI weight composition by cell type.

This script reads a saved ``decomposed_sample.npz`` and computes how much of
the total edge weight comes from each requested component:

  direct_j^w   -> direct_j_wscreni
  peak_j_coef  -> peak_j_coef
  ATAC boost   -> atac_boost

The main reported values are per-cell percentages averaged within each cell
type, plus an ``Overall`` row across all sampled cells.  Pooled percentages are
also included for traceability; these are computed from summed component
weights within a group.

Outputs
-------
<out-dir>/weight_composition_per_cell.csv
<out-dir>/weight_composition_per_celltype.csv
<out-dir>/weight_composition_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


COMPONENTS = (
    ("direct_j_w", "direct_j_wscreni"),
    ("peak_j_coef", "peak_j_coef"),
    ("atac_boost", "atac_boost"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        default="output/weight_decomposition_retina",
        help="Directory containing decomposed_sample.npz and cell_meta.csv.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <data-dir>/weight_composition.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset label written to outputs. Defaults to an inferred label.",
    )
    return parser.parse_args()


def infer_dataset_name(data_dir: Path) -> str:
    path_text = str(data_dir).replace("\\", "/").lower()
    if "seaad" in path_text or "sea-ad" in path_text:
        return "SEA-AD"
    if "retina" in path_text:
        return "Mouse retina"
    return data_dir.name


def detect_cell_type_column(meta: pd.DataFrame) -> str:
    for col in ("cell_type", "subclass", "Subclass", "celltype", "CellType"):
        if col in meta.columns:
            return col
    raise ValueError(
        "Could not find a cell-type column in cell_meta.csv. "
        "Expected one of: cell_type, subclass, Subclass, celltype, CellType."
    )


def load_components(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as npz:
        missing = [npz_key for _, npz_key in COMPONENTS if npz_key not in npz.files]
        if missing:
            raise KeyError(f"{npz_path} is missing required arrays: {missing}")
        arrays = {name: npz[npz_key].astype(np.float64) for name, npz_key in COMPONENTS}

    base_shape = arrays["direct_j_w"].shape
    if len(base_shape) != 3:
        raise ValueError(f"direct_j_wscreni must be 3D (cells, genes, genes), got {base_shape}")
    for name, arr in arrays.items():
        if arr.shape != base_shape:
            raise ValueError(f"{name} has shape {arr.shape}, expected {base_shape}")
    return arrays


def percent_columns(component_sums: pd.DataFrame) -> pd.DataFrame:
    total = component_sums[["sum_direct_j_w", "sum_peak_j_coef", "sum_atac_boost"]].sum(axis=1)
    safe_total = total.replace(0, np.nan)
    out = component_sums.copy()
    out["total_weight"] = total
    out["pct_direct_j_w"] = 100.0 * out["sum_direct_j_w"] / safe_total
    out["pct_peak_j_coef"] = 100.0 * out["sum_peak_j_coef"] / safe_total
    out["pct_atac_boost"] = 100.0 * out["sum_atac_boost"] / safe_total
    return out


def summarize_group(
    dataset: str,
    cell_type: str,
    group: pd.DataFrame,
) -> dict[str, object]:
    pooled_total = group[["sum_direct_j_w", "sum_peak_j_coef", "sum_atac_boost"]].sum().sum()
    if pooled_total <= 0:
        pooled = {
            "pooled_pct_direct_j_w": np.nan,
            "pooled_pct_peak_j_coef": np.nan,
            "pooled_pct_atac_boost": np.nan,
        }
    else:
        pooled = {
            "pooled_pct_direct_j_w": 100.0 * group["sum_direct_j_w"].sum() / pooled_total,
            "pooled_pct_peak_j_coef": 100.0 * group["sum_peak_j_coef"].sum() / pooled_total,
            "pooled_pct_atac_boost": 100.0 * group["sum_atac_boost"].sum() / pooled_total,
        }

    return {
        "dataset": dataset,
        "cell_type": cell_type,
        "n_cells": int(len(group)),
        "mean_pct_direct_j_w": float(group["pct_direct_j_w"].mean()),
        "std_pct_direct_j_w": float(group["pct_direct_j_w"].std(ddof=0)),
        "mean_pct_peak_j_coef": float(group["pct_peak_j_coef"].mean()),
        "std_pct_peak_j_coef": float(group["pct_peak_j_coef"].std(ddof=0)),
        "mean_pct_atac_boost": float(group["pct_atac_boost"].mean()),
        "std_pct_atac_boost": float(group["pct_atac_boost"].std(ddof=0)),
        **pooled,
    }


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "weight_composition"
    dataset = args.dataset_name or infer_dataset_name(data_dir)

    npz_path = data_dir / "decomposed_sample.npz"
    meta_path = data_dir / "cell_meta.csv"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing decomposed sample: {npz_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing cell metadata: {meta_path}")

    print("Weight composition analysis")
    print(f"  dataset : {dataset}")
    print(f"  data dir: {data_dir}")
    print(f"  out dir : {out_dir}")

    arrays = load_components(npz_path)
    meta = pd.read_csv(meta_path)
    ct_col = detect_cell_type_column(meta)

    n_cells, n_genes, _ = arrays["direct_j_w"].shape
    if len(meta) != n_cells:
        raise ValueError(
            f"cell_meta.csv has {len(meta)} rows but decomposed_sample.npz has {n_cells} cells"
        )

    cell_ids = meta["cell"] if "cell" in meta.columns else pd.Series([f"cell_{i}" for i in range(n_cells)])
    cell_types = meta[ct_col].astype(str)

    per_cell = pd.DataFrame(
        {
            "dataset": dataset,
            "cell": cell_ids.astype(str).to_numpy(),
            "cell_type": cell_types.to_numpy(),
            "sum_direct_j_w": arrays["direct_j_w"].sum(axis=(1, 2)),
            "sum_peak_j_coef": arrays["peak_j_coef"].sum(axis=(1, 2)),
            "sum_atac_boost": arrays["atac_boost"].sum(axis=(1, 2)),
        }
    )
    per_cell = percent_columns(per_cell)

    if per_cell[["pct_direct_j_w", "pct_peak_j_coef", "pct_atac_boost"]].isna().any().any():
        raise ValueError("At least one cell has zero total component weight; percentages are undefined.")

    rows = []
    for cell_type, group in per_cell.groupby("cell_type", sort=True):
        rows.append(summarize_group(dataset, str(cell_type), group))
    rows.append(summarize_group(dataset, "Overall", per_cell))
    per_celltype = pd.DataFrame(rows)

    pct_cols = [
        "pct_direct_j_w",
        "pct_peak_j_coef",
        "pct_atac_boost",
        "mean_pct_direct_j_w",
        "std_pct_direct_j_w",
        "mean_pct_peak_j_coef",
        "std_pct_peak_j_coef",
        "mean_pct_atac_boost",
        "std_pct_atac_boost",
        "pooled_pct_direct_j_w",
        "pooled_pct_peak_j_coef",
        "pooled_pct_atac_boost",
    ]
    for df in (per_cell, per_celltype):
        for col in pct_cols:
            if col in df.columns:
                df[col] = df[col].round(6)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_cell.to_csv(out_dir / "weight_composition_per_cell.csv", index=False)
    per_celltype.to_csv(out_dir / "weight_composition_per_celltype.csv", index=False)

    overall = per_celltype.loc[per_celltype["cell_type"] == "Overall"].iloc[0].to_dict()
    summary = {
        "dataset": dataset,
        "data_dir": str(data_dir),
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
        "cell_type_column": ct_col,
        "n_cell_types": int(per_cell["cell_type"].nunique()),
        "overall": overall,
        "outputs": {
            "per_cell": str(out_dir / "weight_composition_per_cell.csv"),
            "per_celltype": str(out_dir / "weight_composition_per_celltype.csv"),
        },
    }
    with open(out_dir / "weight_composition_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print()
    print("Overall mean composition:")
    print(f"  direct_j^w  : {overall['mean_pct_direct_j_w']:.2f}%")
    print(f"  peak_j_coef : {overall['mean_pct_peak_j_coef']:.2f}%")
    print(f"  ATAC boost  : {overall['mean_pct_atac_boost']:.2f}%")
    print()
    print(f"Saved -> {out_dir / 'weight_composition_per_cell.csv'}")
    print(f"Saved -> {out_dir / 'weight_composition_per_celltype.csv'}")
    print(f"Saved -> {out_dir / 'weight_composition_summary.json'}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
