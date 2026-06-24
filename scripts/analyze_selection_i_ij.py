"""Count the I_ij selection funnel for mouse retina and SEA-AD.

For each dataset, report:

1. Total directed gene-gene pairs, excluding self-pairs: G * (G - 1).
2. Candidate pairs where regulator gene j is a TF: n_TF * (G - 1).
3. Selected pairs where I_ij = 1: a TF has a motif in a correlated peak
   linked to the target gene.

The output is a small CSV and JSON summary suitable for paper tables or
figure annotations.

Run from the bsc-screni project root:

  python scripts/analyze_selection_i_ij.py
  python scripts/analyze_selection_i_ij.py --no-seaad
  python scripts/analyze_selection_i_ij.py --seaad-prefix sub42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--no-retina", action="store_true", help="Skip mouse retina")
    parser.add_argument("--no-seaad", action="store_true", help="Skip SEA-AD")
    parser.add_argument(
        "--seaad-prefix",
        default="sub42",
        help="SEA-AD processed file prefix, default: sub42",
    )
    parser.add_argument(
        "--retina-gene-labels",
        default="data/processed/retinal_gene_labels.csv",
    )
    parser.add_argument(
        "--retina-triplets",
        default="data/processed/retinal_triplets.csv",
    )
    parser.add_argument(
        "--seaad-dir",
        default="data/processed/seaad",
        help="Directory containing SEA-AD gene-label and triplet CSVs",
    )
    parser.add_argument(
        "--out-dir",
        default="output/iij_selection",
        help="Output directory, default: output/iij_selection",
    )
    args = parser.parse_args()
    if args.no_retina and args.no_seaad:
        parser.error("--no-retina and --no-seaad cannot both be set")
    return args


def detect_target_col(triplets: pd.DataFrame) -> str:
    for col in ("target_gene", "target", "gene", "gene.name"):
        if col in triplets.columns and col != "TF":
            return col
    raise ValueError(
        "Triplet CSV must contain a target column named one of: "
        "target_gene, target, gene, gene.name"
    )


def load_gene_labels(path: Path) -> tuple[list[str], set[str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    labels = pd.read_csv(path)
    if "gene" not in labels.columns:
        raise ValueError(f"{path} must contain a 'gene' column")
    type_col = next((c for c in ("type", "label", "gene_type") if c in labels.columns), None)
    if type_col is None:
        raise ValueError(f"{path} must contain a TF/type column")

    genes = labels["gene"].dropna().astype(str).tolist()
    if len(genes) != len(set(genes)):
        raise ValueError(f"{path} contains duplicate gene names")

    type_values = labels[type_col].fillna("").astype(str).str.upper()
    tf_genes = set(labels.loc[type_values == "TF", "gene"].astype(str))
    return genes, tf_genes


def analyze_dataset(
    dataset: str,
    gene_labels_path: Path,
    triplets_path: Path,
) -> dict[str, object]:
    genes, tf_genes = load_gene_labels(gene_labels_path)
    if not triplets_path.exists():
        raise FileNotFoundError(triplets_path)
    triplets = pd.read_csv(triplets_path)
    if "TF" not in triplets.columns:
        raise ValueError(f"{triplets_path} must contain a 'TF' column")
    target_col = detect_target_col(triplets)

    gene_set = set(genes)
    n_genes = len(genes)
    n_tfs = len(tf_genes)
    total_gene_gene_pairs = n_genes * max(n_genes - 1, 0)
    tf_regulator_pairs = n_tfs * max(n_genes - 1, 0)

    raw_pairs = set()
    mapped_iij_pairs = set()
    missing_tf_names = set()
    missing_target_names = set()
    non_hvg_tf_names = set()
    self_pairs = set()

    for tf, target in zip(triplets["TF"].astype(str), triplets[target_col].astype(str)):
        raw_pairs.add((tf, target))
        if tf not in gene_set:
            missing_tf_names.add(tf)
            continue
        if target not in gene_set:
            missing_target_names.add(target)
            continue
        if tf not in tf_genes:
            non_hvg_tf_names.add(tf)
            continue
        if tf == target:
            self_pairs.add((tf, target))
            continue
        mapped_iij_pairs.add((tf, target))

    n_iij = len(mapped_iij_pairs)
    pct_tf_of_all = 100.0 * tf_regulator_pairs / total_gene_gene_pairs if total_gene_gene_pairs else 0.0
    pct_iij_of_all = 100.0 * n_iij / total_gene_gene_pairs if total_gene_gene_pairs else 0.0
    pct_iij_of_tf = 100.0 * n_iij / tf_regulator_pairs if tf_regulator_pairs else 0.0

    return {
        "dataset": dataset,
        "gene_labels_path": str(gene_labels_path),
        "triplets_path": str(triplets_path),
        "n_genes": n_genes,
        "n_tfs": n_tfs,
        "total_gene_gene_pairs": total_gene_gene_pairs,
        "tf_regulator_pairs": tf_regulator_pairs,
        "iij_pairs": n_iij,
        "pct_tf_regulator_pairs_of_all": pct_tf_of_all,
        "pct_iij_pairs_of_all": pct_iij_of_all,
        "pct_iij_pairs_of_tf_regulator_pairs": pct_iij_of_tf,
        "triplet_rows": int(len(triplets)),
        "unique_raw_triplet_tf_target_pairs": len(raw_pairs),
        "unique_missing_tf_names": len(missing_tf_names),
        "unique_missing_target_names": len(missing_target_names),
        "unique_non_hvg_tf_names": len(non_hvg_tf_names),
        "self_pairs_excluded": len(self_pairs),
    }


def print_summary(rows: list[dict[str, object]]) -> None:
    print("\nI_ij selection funnel")
    print("=" * 78)
    print(
        f"{'Dataset':<12} {'Genes':>7} {'TFs':>6} {'All pairs':>14} "
        f"{'TF pairs':>14} {'I_ij=1':>12}"
    )
    print("-" * 78)
    for row in rows:
        print(
            f"{row['dataset']:<12} "
            f"{row['n_genes']:>7,} "
            f"{row['n_tfs']:>6,} "
            f"{row['total_gene_gene_pairs']:>14,} "
            f"{row['tf_regulator_pairs']:>14,} "
            f"{row['iij_pairs']:>12,}"
        )
        print(
            f"{'':<12} {'':>7} {'':>6} {'100.00%':>14} "
            f"{row['pct_tf_regulator_pairs_of_all']:>13.2f}% "
            f"{row['pct_iij_pairs_of_all']:>11.2f}%"
        )
        print(
            f"{'':<12} I_ij=1 as % of TF-regulator candidate pairs: "
            f"{row['pct_iij_pairs_of_tf_regulator_pairs']:.2f}%"
        )
    print("=" * 78)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    if not args.no_retina:
        rows.append(
            analyze_dataset(
                "mouse_retina",
                Path(args.retina_gene_labels),
                Path(args.retina_triplets),
            )
        )
    if not args.no_seaad:
        seaad_dir = Path(args.seaad_dir)
        rows.append(
            analyze_dataset(
                "sea_ad",
                seaad_dir / f"seaad_paired_{args.seaad_prefix}_gene_labels.csv",
                seaad_dir / f"seaad_paired_{args.seaad_prefix}_triplets.csv",
            )
        )

    print_summary(rows)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "selection_i_ij_summary.csv"
    json_path = out_dir / "selection_i_ij_summary.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\nSaved CSV  -> {csv_path}")
    print(f"Saved JSON -> {json_path}")


if __name__ == "__main__":
    main()
