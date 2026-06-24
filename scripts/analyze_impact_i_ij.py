"""Compute weight composition on edges where I_ij=1 and boost > 0.

For every cell, this analysis keeps only edges that:

1. are theoretically eligible for the ATAC boost according to the triplet
   table (I_ij=1), and
2. received a strictly positive saved ``atac_boost`` in that cell.

It then computes the direct_j^w, peak_j_coef, and ATAC-boost percentages over
that filtered edge set. Results are reported per cell, per cell type, and for
all sampled cells combined.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="output/weight_decomposition_retina")
    parser.add_argument(
        "--gene-labels",
        default=None,
        help=(
            "Optional fallback gene-label CSV. If decomposed_sample.npz contains "
            "gene_names, those names are used as the authoritative matrix axis."
        ),
    )
    parser.add_argument(
        "--gene-h5ad",
        default=None,
        help=(
            "Optional fallback AnnData file whose var_names match the component "
            "matrix gene axis. Used only when NPZ gene_names are absent and the "
            "gene-label CSV does not match the matrix size."
        ),
    )
    parser.add_argument("--triplets", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dataset-name", default=None)
    return parser.parse_args()


def infer_dataset_name(data_dir: Path) -> str:
    text = str(data_dir).replace("\\", "/").lower()
    if "seaad" in text or "sea-ad" in text:
        return "SEA-AD"
    if "retina" in text:
        return "Mouse retina"
    return data_dir.name


def detect_cell_type_column(meta: pd.DataFrame) -> str:
    for col in ("cell_type", "subclass", "Subclass", "celltype", "CellType"):
        if col in meta.columns:
            return col
    raise ValueError("Could not detect a cell-type column in cell_meta.csv")


def detect_triplet_columns(triplets: pd.DataFrame) -> tuple[str, str]:
    if "TF" not in triplets.columns:
        raise ValueError("Triplets CSV must contain a 'TF' column")
    for col in ("target_gene", "target", "gene", "gene.name"):
        if col in triplets.columns:
            return "TF", col
    raise ValueError("Could not detect target-gene column in triplets CSV")


def build_ij_mask(triplets: pd.DataFrame, genes: list[str]) -> tuple[np.ndarray, int]:
    tf_col, target_col = detect_triplet_columns(triplets)
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    mask = np.zeros((len(genes), len(genes)), dtype=bool)
    skipped = 0
    for tf, target in zip(triplets[tf_col].astype(str), triplets[target_col].astype(str)):
        tf_idx = gene_to_idx.get(tf)
        target_idx = gene_to_idx.get(target)
        if tf_idx is None or target_idx is None:
            skipped += 1
            continue
        mask[tf_idx, target_idx] = True
    return mask, skipped


def load_components(npz_path: Path) -> tuple[dict[str, np.ndarray], list[str] | None]:
    keys = {
        "direct_j_w": "direct_j_wscreni",
        "peak_j_coef": "peak_j_coef",
        "atac_boost": "atac_boost",
    }
    with np.load(npz_path) as npz:
        missing = [key for key in keys.values() if key not in npz.files]
        if missing:
            raise KeyError(f"{npz_path} is missing required arrays: {missing}")
        arrays = {name: np.maximum(npz[key].astype(np.float64), 0) for name, key in keys.items()}
        gene_names = (
            [str(g) for g in npz["gene_names"].tolist()]
            if "gene_names" in npz.files
            else None
        )
    shape = arrays["direct_j_w"].shape
    if len(shape) != 3 or shape[1] != shape[2]:
        raise ValueError(f"Expected square 3D component arrays, got {shape}")
    for name, arr in arrays.items():
        if arr.shape != shape:
            raise ValueError(f"{name} has shape {arr.shape}, expected {shape}")
    return arrays, gene_names


def resolve_gene_axis(
    n_genes: int,
    npz_gene_names: list[str] | None,
    gene_labels_path: Path | None,
    gene_h5ad_path: Path | None,
) -> list[str]:
    if npz_gene_names is not None:
        if len(npz_gene_names) != n_genes:
            raise ValueError(
                f"NPZ gene_names has {len(npz_gene_names)} entries but matrices have {n_genes} genes"
            )
        return npz_gene_names

    label_error = None
    if gene_labels_path is not None:
        if not gene_labels_path.exists():
            raise FileNotFoundError(gene_labels_path)

        gene_labels = pd.read_csv(gene_labels_path)
        if "gene" not in gene_labels.columns:
            raise ValueError(f"{gene_labels_path} must contain a 'gene' column")
        genes = gene_labels["gene"].astype(str).tolist()
        if len(genes) == n_genes:
            return genes
        label_error = f"Gene labels has {len(genes)} rows but matrices have {n_genes} genes"

    if gene_h5ad_path is not None and gene_h5ad_path.exists():
        try:
            import anndata as ad

            adata = ad.read_h5ad(gene_h5ad_path, backed="r")
            genes = [str(g) for g in adata.var_names.tolist()]
            adata.file.close()
        except Exception as exc:  # pragma: no cover - depends on cluster I/O stack
            raise ValueError(f"Failed to read gene axis from {gene_h5ad_path}: {exc}") from exc
        if len(genes) == n_genes:
            return genes
        h5ad_error = f"{gene_h5ad_path} has {len(genes)} var_names but matrices have {n_genes} genes"
    else:
        h5ad_error = f"{gene_h5ad_path} not found" if gene_h5ad_path is not None else "no --gene-h5ad provided"

    raise ValueError(
        "Could not resolve the component matrix gene axis. "
        "Use a decomposition NPZ that stores gene_names, or provide a matching "
        "gene-label CSV/H5AD. "
        f"CSV check: {label_error or 'not provided'}; H5AD check: {h5ad_error}."
    )


def add_percentages(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    total = out[["sum_direct_j_w", "sum_peak_j_coef", "sum_atac_boost"]].sum(axis=1)
    safe = total.replace(0, np.nan)
    out["total_weight"] = total
    out["pct_direct_j_w"] = 100.0 * out["sum_direct_j_w"] / safe
    out["pct_peak_j_coef"] = 100.0 * out["sum_peak_j_coef"] / safe
    out["pct_atac_boost"] = 100.0 * out["sum_atac_boost"] / safe
    return out


def summarize_group(dataset: str, cell_type: str, group: pd.DataFrame) -> dict[str, object]:
    sum_cols = ["sum_direct_j_w", "sum_peak_j_coef", "sum_atac_boost"]
    pooled_total = float(group[sum_cols].sum().sum())
    pooled = {
        f"pooled_pct_{name}": 100.0 * float(group[f"sum_{name}"].sum()) / pooled_total
        if pooled_total > 0 else np.nan
        for name in ("direct_j_w", "peak_j_coef", "atac_boost")
    }
    return {
        "dataset": dataset,
        "cell_type": cell_type,
        "n_cells": int(len(group)),
        "mean_selected_edges": float(group["selected_edges"].mean()),
        "std_selected_edges": float(group["selected_edges"].std(ddof=0)),
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
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "impact_i_ij"
    dataset = args.dataset_name or infer_dataset_name(data_dir)
    npz_path = data_dir / "decomposed_sample.npz"
    meta_path = data_dir / "cell_meta.csv"
    gene_labels_path = Path(args.gene_labels) if args.gene_labels else None
    gene_h5ad_path = Path(args.gene_h5ad) if args.gene_h5ad else None
    triplets_path = Path(args.triplets)
    for path in (npz_path, meta_path, triplets_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if gene_labels_path is not None and not gene_labels_path.exists():
        raise FileNotFoundError(gene_labels_path)
    if gene_h5ad_path is not None and not gene_h5ad_path.exists():
        print(f"  WARNING: --gene-h5ad not found, will ignore: {gene_h5ad_path}")
        gene_h5ad_path = None

    arrays, npz_gene_names = load_components(npz_path)
    n_cells, n_genes, _ = arrays["direct_j_w"].shape
    meta = pd.read_csv(meta_path)
    if len(meta) != n_cells:
        raise ValueError(f"Metadata has {len(meta)} rows but NPZ has {n_cells} cells")
    ct_col = detect_cell_type_column(meta)

    genes = resolve_gene_axis(n_genes, npz_gene_names, gene_labels_path, gene_h5ad_path)
    if npz_gene_names is not None:
        gene_source = "decomposed_sample.npz:gene_names"
    elif gene_labels_path is not None and len(pd.read_csv(gene_labels_path, usecols=["gene"])) == n_genes:
        gene_source = str(gene_labels_path)
    else:
        gene_source = str(gene_h5ad_path)

    ij_mask, skipped_triplet_rows = build_ij_mask(pd.read_csv(triplets_path), genes)
    if not ij_mask.any():
        raise ValueError("No I_ij=1 pairs mapped to the decomposition gene set")

    cell_ids = (
        meta["cell"].astype(str).to_numpy()
        if "cell" in meta.columns
        else np.asarray([f"cell_{idx}" for idx in range(n_cells)])
    )
    cell_types = meta[ct_col].astype(str).to_numpy()
    rows = []
    for idx in range(n_cells):
        practical_mask = ij_mask & (arrays["atac_boost"][idx] > 0)
        rows.append(
            {
                "dataset": dataset,
                "cell": cell_ids[idx],
                "cell_type": cell_types[idx],
                "theoretical_iij_edges": int(ij_mask.sum()),
                "selected_edges": int(practical_mask.sum()),
                "selected_pct_of_iij": 100.0 * practical_mask.sum() / ij_mask.sum(),
                "sum_direct_j_w": float(arrays["direct_j_w"][idx][practical_mask].sum()),
                "sum_peak_j_coef": float(arrays["peak_j_coef"][idx][practical_mask].sum()),
                "sum_atac_boost": float(arrays["atac_boost"][idx][practical_mask].sum()),
            }
        )

    per_cell = add_percentages(pd.DataFrame(rows))
    if per_cell[["pct_direct_j_w", "pct_peak_j_coef", "pct_atac_boost"]].isna().any().any():
        raise ValueError("At least one cell has no positive-boost I_ij edges or zero selected weight")

    summaries = [
        summarize_group(dataset, str(cell_type), group)
        for cell_type, group in per_cell.groupby("cell_type", sort=True)
    ]
    summaries.append(summarize_group(dataset, "Overall", per_cell))
    per_celltype = pd.DataFrame(summaries)

    for df in (per_cell, per_celltype):
        numeric = df.select_dtypes(include=[np.number]).columns
        df[numeric] = df[numeric].round(6)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_cell_path = out_dir / "impact_i_ij_per_cell.csv"
    per_type_path = out_dir / "impact_i_ij_per_celltype.csv"
    summary_path = out_dir / "impact_i_ij_summary.json"
    per_cell.to_csv(per_cell_path, index=False)
    per_celltype.to_csv(per_type_path, index=False)

    overall = per_celltype.loc[per_celltype["cell_type"] == "Overall"].iloc[0].to_dict()
    summary = {
        "dataset": dataset,
        "filter": "theoretical I_ij=1 AND per-cell atac_boost > 0",
        "n_cells": n_cells,
        "n_genes": n_genes,
        "gene_axis_source": gene_source,
        "theoretical_iij_edges": int(ij_mask.sum()),
        "skipped_triplet_rows": skipped_triplet_rows,
        "overall": overall,
        "outputs": {"per_cell": str(per_cell_path), "per_celltype": str(per_type_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{dataset} impact of practically boosted I_ij edges")
    print(f"  gene axis source        : {gene_source}")
    print(f"  theoretical I_ij edges : {int(ij_mask.sum()):,}")
    print(f"  mean selected per cell : {overall['mean_selected_edges']:.1f}")
    print(f"  direct_j^w             : {overall['mean_pct_direct_j_w']:.2f}%")
    print(f"  peak_j_coef            : {overall['mean_pct_peak_j_coef']:.2f}%")
    print(f"  ATAC boost             : {overall['mean_pct_atac_boost']:.2f}%")
    print(f"Saved -> {out_dir}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
