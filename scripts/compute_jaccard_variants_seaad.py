"""Compute Jaccard similarity of formula variants against kScReNI.

Run once for each decomposition output directory:

  python compute_jaccard_variants_seaad.py \
      --data-dir output/weight_decomposition_retina --dataset-name retina
  python compute_jaccard_variants_seaad.py \
      --data-dir output/seaad_grn/weight_decomposition --dataset-name seaad

Variants, all compared with k_ij = direct_j_kscreni:
  w_ij      normal wScReNI full network
  w_refined wScReNI without peak_j_coef
  w_hybrid  kScReNI with I_ij boost
  z_ij      wScReNI with peak_j_coef and no I_ij boost
  dw        wScReNI RF expression importances only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VARIANT_ORDER = ["w_ij", "w_refined", "w_hybrid", "z_ij", "dw"]
DESCRIPTIONS = {
    "w_ij": "normal wScReNI full network",
    "w_refined": "direct_j_wscreni + atac_boost",
    "w_hybrid": "direct_j_kscreni + atac_boost",
    "z_ij": "direct_j_wscreni + peak_j_coef",
    "dw": "direct_j_wscreni",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default="output/weight_decomposition_retina",
        help="Directory containing decomposed_sample.npz and cell_meta.csv",
    )
    parser.add_argument(
        "--dataset-name",
        default="retina",
        help="Dataset label stored in outputs, for example retina or seaad",
    )
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <data-dir>/jaccard_variants)",
    )
    return parser.parse_args()


def load_components(npz_path: Path) -> dict[str, np.ndarray]:
    required = ["direct_j_wscreni", "direct_j_kscreni", "peak_j_coef", "atac_boost"]
    with np.load(npz_path) as npz:
        missing = [key for key in required if key not in npz.files]
        if missing:
            raise KeyError(f"{npz_path} is missing required arrays: {missing}")
        arrays = {key: npz[key].astype(np.float32) for key in required}
        # Prefer stored full matrices because they are the exact networks generated
        # by the decomposition script. Fall back to component sums for older NPZs.
        arrays["w_ij"] = (
            npz["w_matrices"].astype(np.float32)
            if "w_matrices" in npz.files
            else arrays["direct_j_wscreni"] + arrays["peak_j_coef"] + arrays["atac_boost"]
        )
        arrays["z_ij"] = (
            npz["z_matrices"].astype(np.float32)
            if "z_matrices" in npz.files
            else arrays["direct_j_wscreni"] + arrays["peak_j_coef"]
        )
    shape = arrays["direct_j_kscreni"].shape
    if len(shape) != 3 or shape[1] != shape[2]:
        raise ValueError(f"Expected (cells, genes, genes) arrays, got {shape}")
    for key, array in arrays.items():
        if array.shape != shape:
            raise ValueError(f"{key} has shape {array.shape}, expected {shape}")
    return arrays


def detect_cell_type_col(meta: pd.DataFrame) -> str:
    for column in ["cell_type", "subclass", "Subclass"]:
        if column in meta.columns:
            return column
    raise ValueError(
        "cell_meta.csv must contain one of: cell_type, subclass, Subclass"
    )


def jaccard_top_k(a: np.ndarray, b: np.ndarray, k: int) -> float:
    if k <= 0:
        raise ValueError("--top-k must be positive")
    flat_a = a.ravel(order="C")
    flat_b = b.ravel(order="C")
    effective_k = min(k, flat_a.size, flat_b.size)
    set_a = set(np.argpartition(flat_a, -effective_k)[-effective_k:].tolist())
    set_b = set(np.argpartition(flat_b, -effective_k)[-effective_k:].tolist())
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union else 1.0


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "jaccard_variants"
    npz_path = data_dir / "decomposed_sample.npz"
    meta_path = data_dir / "cell_meta.csv"
    for path in [npz_path, meta_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    arrays = load_components(npz_path)
    dw = arrays["direct_j_wscreni"]
    dk = arrays["direct_j_kscreni"]
    pk = arrays["peak_j_coef"]
    bo = arrays["atac_boost"]
    variants = {
        "w_ij": arrays["w_ij"],
        "w_refined": dw + bo,
        "w_hybrid": dk + bo,
        "z_ij": arrays["z_ij"],
        "dw": dw,
    }

    n_cells, n_genes, _ = dk.shape
    meta = pd.read_csv(meta_path)
    cell_type_col = detect_cell_type_col(meta)
    if "cell" not in meta.columns:
        raise ValueError("cell_meta.csv must contain a 'cell' column")
    if len(meta) != n_cells:
        raise ValueError(f"NPZ has {n_cells} cells but metadata has {len(meta)} rows")

    print(f"Dataset: {args.dataset_name}")
    print(f"Cells: {n_cells}; genes: {n_genes}; top-k: {args.top_k}")
    print(f"Cell-type column: {cell_type_col}")

    rows: list[dict] = []
    for i in range(n_cells):
        row = {
            "dataset": args.dataset_name,
            "cell": meta.iloc[i]["cell"],
            "cell_type": str(meta.iloc[i][cell_type_col]),
            "top_k": args.top_k,
        }
        for variant in VARIANT_ORDER:
            row[variant] = round(jaccard_top_k(variants[variant][i], dk[i], args.top_k), 6)
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_cell = pd.DataFrame(rows)
    per_cell.to_csv(out_dir / "jaccard_per_cell.csv", index=False)

    overall = (
        per_cell[VARIANT_ORDER]
        .agg(["mean", "std"])
        .T
        .reset_index()
        .rename(columns={"index": "variant", "mean": "mean_jaccard", "std": "sd_jaccard"})
    )
    overall["dataset"] = args.dataset_name
    overall["top_k"] = args.top_k
    overall["formula"] = overall["variant"].map(DESCRIPTIONS)
    overall = overall[["dataset", "variant", "formula", "top_k", "mean_jaccard", "sd_jaccard"]]
    overall.to_csv(out_dir / "jaccard_summary.csv", index=False)

    per_celltype = (
        per_cell.groupby("cell_type")[VARIANT_ORDER]
        .mean()
        .reset_index()
        .melt(id_vars="cell_type", var_name="variant", value_name="mean_jaccard")
    )
    per_celltype["dataset"] = args.dataset_name
    per_celltype.to_csv(out_dir / "jaccard_per_celltype.csv", index=False)

    summary = {
        "dataset": args.dataset_name,
        "top_k": args.top_k,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "reference": "k_ij = direct_j_kscreni",
        "variants": {
            row["variant"]: {
                "formula": row["formula"],
                "mean_jaccard": round(float(row["mean_jaccard"]), 6),
                "sd_jaccard": round(float(row["sd_jaccard"]), 6),
            }
            for _, row in overall.iterrows()
        },
        "per_celltype": {
            cell_type: {
                variant: round(float(value), 6)
                for variant, value in group.set_index("variant")["mean_jaccard"].items()
            }
            for cell_type, group in per_celltype.groupby("cell_type")
        },
    }
    (out_dir / "jaccard_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nMean Jaccard vs k_ij:")
    print(overall[["variant", "mean_jaccard", "sd_jaccard"]].to_string(index=False))
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
