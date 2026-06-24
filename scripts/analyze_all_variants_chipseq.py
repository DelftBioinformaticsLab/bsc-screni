"""Evaluate ChIP-seq precision/recall by retinal cell type for six formulas.

The calculation follows the original ScReNI R evaluation: rank the complete
gene-by-gene matrix (self-edges included), take the top-k entries, and compare
their TF-target labels with the ChIP-Atlas ground truth.

Variants:
  w_ij      = dw + pk + bo_paper   published wScReNI shared-boost formula
  k_ij      = dk                   published kScReNI
  w_refined = dw + bo              wScReNI without peak_j_coef
  w_hybrid  = dk + bo              kScReNI with corrected I_ij boost
  z_ij      = dw + pk              wScReNI without the I_ij boost
  dw        = dw                   wScReNI RF expression importances only
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VARIANT_ORDER = ["w_ij", "k_ij", "w_refined", "w_hybrid", "z_ij", "dw"]
FORMULAS = {
    "w_ij": "dw + pk + bo_paper",
    "k_ij": "dk",
    "w_refined": "dw + bo",
    "w_hybrid": "dk + bo",
    "z_ij": "dw + pk",
    "dw": "dw",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="output/weight_decomposition_retina")
    parser.add_argument("--npz-path", default=None)
    parser.add_argument("--meta-path", default=None)
    parser.add_argument(
        "--gene-labels", default="data/processed/retinal_gene_labels.csv",
        help="CSV whose gene column matches the NPZ matrix order",
    )
    parser.add_argument(
        "--chip-path", default="../refer/mmp9.TSV.5kb_TF_target.df.txt",
    )
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--allow-corrected-as-wij", action="store_true",
        help="Use corrected atac_boost for w_ij if atac_boost_paper is absent. "
             "This is only for legacy smoke tests and is not the published formula.",
    )
    return parser.parse_args()


def load_chip_atlas(path: Path) -> set[tuple[str, str]]:
    """Load the documented ChIP-Atlas format: row index, TF, target."""
    try:
        frame = pd.read_csv(path, sep="\t", header=0, index_col=0, dtype=str)
        if frame.shape[1] < 2:
            raise ValueError("fewer than two data columns")
        clean = frame.iloc[:, :2].copy()
        clean.columns = ["tf", "target"]
        clean["tf"] = clean["tf"].str.strip()
        clean["target"] = clean["target"].str.strip()
        clean = clean.dropna(subset=["tf", "target"])
    except Exception as exc:
        raise ValueError(
            f"Could not read ChIP-Atlas file {path} as row-index/TF/target TSV"
        ) from exc
    return set(zip(clean["tf"], clean["target"]))


def load_components(path: Path, allow_corrected_as_wij: bool) -> dict[str, np.ndarray]:
    required = ["direct_j_wscreni", "direct_j_kscreni", "peak_j_coef", "atac_boost"]
    with np.load(path) as npz:
        missing = [key for key in required if key not in npz.files]
        if missing:
            raise KeyError(f"{path} is missing required arrays: {missing}")
        arrays = {key: npz[key].astype(np.float32) for key in required}
        if "atac_boost_paper" in npz.files:
            arrays["atac_boost_paper"] = npz["atac_boost_paper"].astype(np.float32)
        elif allow_corrected_as_wij:
            warnings.warn(
                "atac_boost_paper is absent; using corrected atac_boost for w_ij. "
                "This does not represent normal published wScReNI."
            )
            arrays["atac_boost_paper"] = arrays["atac_boost"]
        else:
            raise KeyError(
                f"{path} has no atac_boost_paper array. Normal w_ij requires the "
                "published shared-boost component. Re-run "
                "analyze_weight_decomposition_retina.py, or use "
                "--allow-corrected-as-wij only for legacy smoke tests."
            )
    shape = arrays[required[0]].shape
    if len(shape) != 3 or shape[1] != shape[2]:
        raise ValueError(f"Expected component shape (cells, genes, genes), got {shape}")
    for key, array in arrays.items():
        if array.shape != shape:
            raise ValueError(f"{key} has shape {array.shape}, expected {shape}")
    return arrays


def top_k_pr(
    matrix: np.ndarray,
    chip_flat_indices: set[int],
    top_k: int,
) -> tuple[float, float, int]:
    """Return R-compatible top-k precision and recall for one network matrix."""
    flat = matrix.ravel(order="C")
    valid = np.flatnonzero(~np.isnan(flat))
    effective_k = min(top_k, len(valid))
    if effective_k == 0:
        return float("nan"), float("nan"), 0

    values = flat[valid]
    if effective_k < len(valid):
        selected = np.argpartition(-values, effective_k - 1)[:effective_k]
        selected = selected[np.argsort(-values[selected], kind="stable")]
    else:
        selected = np.argsort(-values, kind="stable")
    top_indices = valid[selected[:effective_k]]
    true_positives = sum(int(index) in chip_flat_indices for index in top_indices)
    precision = true_positives / effective_k
    recall = true_positives / len(chip_flat_indices) if chip_flat_indices else float("nan")
    return precision, recall, true_positives


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    data_dir = Path(args.data_dir)
    npz_path = Path(args.npz_path) if args.npz_path else data_dir / "decomposed_sample.npz"
    meta_path = Path(args.meta_path) if args.meta_path else data_dir / "cell_meta.csv"
    gene_labels_path = Path(args.gene_labels)
    chip_path = Path(args.chip_path)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "precision_recall_celltype"

    for path in [npz_path, meta_path, gene_labels_path, chip_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    components = load_components(npz_path, args.allow_corrected_as_wij)
    dw = components["direct_j_wscreni"]
    dk = components["direct_j_kscreni"]
    pk = components["peak_j_coef"]
    bo = components["atac_boost"]
    bo_paper = components["atac_boost_paper"]
    n_cells, n_genes, _ = dw.shape

    variants = {
        "w_ij": dw + pk + bo_paper,
        "k_ij": dk,
        "w_refined": dw + bo,
        "w_hybrid": dk + bo,
        "z_ij": dw + pk,
        "dw": dw,
    }

    meta = pd.read_csv(meta_path)
    cell_type_col = next(
        (col for col in ["cell_type", "subclass", "Subclass"] if col in meta.columns),
        None,
    )
    if "cell" not in meta.columns or cell_type_col is None:
        raise ValueError("cell_meta.csv must contain cell and cell-type columns")
    if len(meta) != n_cells:
        raise ValueError(f"NPZ has {n_cells} cells but metadata has {len(meta)} rows")

    gene_labels = pd.read_csv(gene_labels_path)
    if "gene" not in gene_labels.columns or len(gene_labels) != n_genes:
        raise ValueError(
            f"{gene_labels_path} must contain exactly {n_genes} rows in a gene column"
        )
    genes = gene_labels["gene"].astype(str).tolist()
    gene_to_index = {gene: index for index, gene in enumerate(genes)}

    chip_pairs = load_chip_atlas(chip_path)
    chip_flat_indices = {
        gene_to_index[tf] * n_genes + gene_to_index[target]
        for tf, target in chip_pairs
        if tf in gene_to_index and target in gene_to_index
    }
    if not chip_flat_indices:
        raise ValueError("No ChIP-Atlas pairs overlap the network gene set")

    print(f"Cells: {n_cells}; genes: {n_genes}; top-k: {args.top_k}")
    print(f"ChIP-Atlas pairs in network gene set: {len(chip_flat_indices):,}")
    rows: list[dict] = []
    for variant in VARIANT_ORDER:
        for cell_index in range(n_cells):
            precision, recall, true_positives = top_k_pr(
                variants[variant][cell_index], chip_flat_indices, args.top_k
            )
            rows.append({
                "variant": variant,
                "formula": FORMULAS[variant],
                "cell": meta.iloc[cell_index]["cell"],
                "cell_type": str(meta.iloc[cell_index][cell_type_col]),
                "top_k": args.top_k,
                "true_positives": true_positives,
                "precision": precision,
                "recall": recall,
            })

    per_cell = pd.DataFrame(rows)
    summary = (
        per_cell.groupby(["variant", "formula", "cell_type"], sort=False)
        .agg(
            n_cells=("cell", "count"),
            precision_mean=("precision", "mean"),
            precision_sd=("precision", "std"),
            precision_se=("precision", "sem"),
            recall_mean=("recall", "mean"),
            recall_sd=("recall", "std"),
            recall_se=("recall", "sem"),
        )
        .reset_index()
    )
    summary["variant"] = pd.Categorical(
        summary["variant"], categories=VARIANT_ORDER, ordered=True
    )
    summary = summary.sort_values(["cell_type", "variant"]).reset_index(drop=True)
    summary["variant"] = summary["variant"].astype(str)

    overall = (
        per_cell.groupby(["variant", "formula"], sort=False)
        .agg(
            n_cells=("cell", "count"),
            precision_mean=("precision", "mean"),
            precision_sd=("precision", "std"),
            precision_se=("precision", "sem"),
            recall_mean=("recall", "mean"),
            recall_sd=("recall", "std"),
            recall_se=("recall", "sem"),
        )
        .reset_index()
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    per_cell.to_csv(out_dir / "all_variants_chipseq_pr.csv", index=False)
    summary.to_csv(out_dir / "all_variants_per_celltype.csv", index=False)
    overall.to_csv(out_dir / "all_variants_overall.csv", index=False)
    metadata = {
        "top_k": args.top_k,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "chip_pairs_in_network_gene_set": len(chip_flat_indices),
        "variants": FORMULAS,
        "self_edges_included": True,
        "published_wij_uses_atac_boost_paper": True,
    }
    (out_dir / "all_variants_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nOverall mean precision:")
    print(overall[["variant", "precision_mean", "recall_mean"]].to_string(index=False))
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
