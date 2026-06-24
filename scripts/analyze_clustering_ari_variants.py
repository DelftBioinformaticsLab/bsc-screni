"""Paper-compatible hierarchical clustering ARI for retina network variants.

Reproduces R ``calculate_scNetwork_degree()`` from the ScReNI paper (Xu et al.
2025, Fig 3B) and evaluates all six formula variants of interest:

  w_ij       dw + pk + bo_paper    original wScReNI (R shared-boost formula)
  k_ij       dk                    original kScReNI (RNA-only)
  w_refined  dw + bo_corrected     proposed formula: removes peak_j_coef
  w_hybrid   dk + bo_corrected     kScReNI base + per-TF boost
  z_ij       dw + pk               wScReNI with peak_j_coef, no boost
  dw         dw                    wScReNI RF expression importances only

The paper reference for n=400 (top-500, complete linkage):
  kScReNI=0.388   wScReNI=0.481

R evaluation path (reproduced here):
  1. Rank weight matrix entries in column-major (Fortran) order.
  2. Take top-k entries, including self-edges (as in R).
  3. Compute colSums of the binary top-k indicator matrix → out-degree per gene.
  4. Cluster on dist(cor(log(degree + 1))), complete linkage.
  5. Cut to n_cell_types clusters, compute ARI vs known labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

SEED = 42
np.random.seed(SEED)

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--data-dir", default="output/weight_decomposition_retina",
                    help="Directory with decomposed_sample.npz and cell_meta.csv")
parser.add_argument("--top-ks", nargs="+", type=int,
                    default=list(range(400, 1401, 100)),
                    help="Top-k thresholds (default: 400 to 1400 in steps of 100)")
args = parser.parse_args()

data_dir = Path(args.data_dir)
npz_path  = data_dir / "decomposed_sample.npz"
meta_path = data_dir / "cell_meta.csv"
out_dir   = data_dir / "clustering_ari"
out_dir.mkdir(parents=True, exist_ok=True)
top_ks = sorted(set(args.top_ks))

if any(k <= 0 for k in top_ks):
    parser.error("--top-ks values must be positive integers")

for fp in [npz_path, meta_path]:
    if not fp.exists():
        sys.exit(f"ERROR: {fp} not found.\n  Run run_decomp_retina_n400.sh first.")

# ── Load NPZ ─────────────────────────────────────────────────────────────────
print(f"Loading {npz_path} …")
npz = np.load(str(npz_path))

dw = npz["direct_j_wscreni"].astype(np.float32)   # (C, G, G)
pk = npz["peak_j_coef"].astype(np.float32)
bo = npz["atac_boost"].astype(np.float32)           # corrected per-TF boost
dk = npz["direct_j_kscreni"].astype(np.float32)

# Paper's original shared boost — use if present, else fall back to corrected
bo_paper = (npz["atac_boost_paper"].astype(np.float32)
            if "atac_boost_paper" in npz.files else bo)

n_cells, n_genes, _ = dw.shape
print(f"  {n_cells} cells × {n_genes} genes")

# ── Load metadata ─────────────────────────────────────────────────────────────
meta = pd.read_csv(meta_path)
if len(meta) != n_cells:
    sys.exit(f"ERROR: metadata has {len(meta)} rows; NPZ has {n_cells} cells.")
cell_types  = meta["cell_type"].astype(str).to_numpy()
unique_types = sorted(set(cell_types))
n_types      = len(unique_types)
type_to_int  = {t: i for i, t in enumerate(unique_types)}
true_labels  = np.array([type_to_int[t] for t in cell_types])
print(f"  Cell types ({n_types}): {unique_types}")

# ── Six formula variants ──────────────────────────────────────────────────────
# Names and order defined by the user's specification.
VARIANTS: list[tuple[str, np.ndarray]] = [
    ("w_ij",      dw + pk + bo_paper),   # original wScReNI (R shared-boost formula)
    ("k_ij",      dk),                   # original kScReNI
    ("w_refined", dw + bo),              # proposed: removes peak_j_coef
    ("w_hybrid",  dk + bo),              # kScReNI expression + corrected boost
    ("z_ij",      dw + pk),              # wScReNI with peak_j_coef, no boost
    ("dw",        dw),                   # wScReNI RF expression importances only
]

# ── R-compatible helpers ──────────────────────────────────────────────────────

def outdegree_matrices(W: np.ndarray, thresholds: list[int]) -> dict[int, np.ndarray]:
    """
    Compute gene out-degree at every threshold using one ranking per cell.

    R vectorises weight matrices column-major (Fortran order).  The outdegree
    assigned to gene j in the R paper is the number of targets for which j
    appears in the top-k edge set — equivalent to colSums of the binary
    top-k indicator matrix (each column is a target gene, each row a regulator).
    """
    n_c, n_g, _ = W.shape
    effective = {k: min(k, n_g * n_g) for k in thresholds}
    matrices = {k: np.zeros((n_c, n_g), dtype=np.float32) for k in thresholds}
    for c in range(n_c):
        flat_f = W[c].ravel(order="F")
        max_k = max(effective.values())
        target_cols = np.argsort(-flat_f, kind="stable")[:max_k] // n_g
        counts = np.zeros(n_g, dtype=np.float32)
        previous = 0
        for k in thresholds:
            current = effective[k]
            np.add.at(counts, target_cols[previous:current], 1)
            matrices[k][c] = counts
            previous = current
    return matrices


def hclust_ari(deg: np.ndarray) -> float:
    """
    Cluster cells on dist(cor(log(degree + 1))), complete linkage → ARI.
    Matches the R evaluation exactly.
    """
    log_deg = np.log1p(deg.astype(np.float64))
    corr    = np.corrcoef(log_deg)
    if not np.isfinite(corr).all():
        return float("nan")
    dists   = pdist(corr, metric="euclidean")
    Z       = linkage(dists, method="complete")
    pred    = fcluster(Z, n_types, criterion="maxclust")
    return float(adjusted_rand_score(true_labels, pred))


# ── Main evaluation loop ──────────────────────────────────────────────────────
print()
print("=" * 60)
print("CLUSTERING ARI  (R-compatible evaluation)")
print(f"  n_cells={n_cells}  n_genes={n_genes}  n_types={n_types}")
print(f"  seed={SEED}  (deterministic for fixed inputs)")
print(f"  top-k tested: {top_ks}")
print("=" * 60)
header = f"  {'Variant':<14}" + "".join(f"  ARI@{k:>4}" for k in top_ks)
print(header)
print("  " + "─" * (len(header) - 2))

records: list[dict] = []
for name, W in VARIANTS:
    row: dict = {"variant": name}
    vals = []
    degree_by_k = outdegree_matrices(W, top_ks)
    for k in top_ks:
        ari = hclust_ari(degree_by_k[k])
        row[f"ari_top{k}"] = round(ari, 4)
        vals.append(ari)
    records.append(row)
    val_str = "".join(f"  {v:>8.4f}" for v in vals)
    print(f"  {name:<14}{val_str}")

print()
print("  Paper reference (Xu et al. 2025, Fig 3B, n=400, top-500):")
print("    kScReNI=0.388   wScReNI=0.481")

# ── Save ──────────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
df.to_csv(out_dir / "clustering_ari.csv", index=False)

summary = {
    "n_cells":   n_cells,
    "n_genes":   n_genes,
    "n_types":   n_types,
    "seed":      SEED,
    "cell_types": unique_types,
    "top_ks":    top_ks,
    "evaluation": {
        "edge_ranking":  "column-major (Fortran order), self-edges retained",
        "degree_def":    "colSums of binary top-k indicator (R paper definition)",
        "distance":      "dist(cor(log(degree + 1)))",
        "linkage":       "complete (R hclust default)",
    },
    "variants": {
        "w_ij":       "dw + pk + bo_paper  (original wScReNI, R shared-boost formula)",
        "k_ij":       "dk                  (original kScReNI, RNA-only)",
        "w_refined":  "dw + bo             (proposed: removes peak_j_coef)",
        "w_hybrid":   "dk + bo             (kScReNI expression + corrected boost)",
        "z_ij":       "dw + pk             (wScReNI with peak_j_coef, no boost)",
        "dw":         "dw                  (wScReNI RF expression importances only)",
    },
    "results": records,
    "paper_reference": {"kScReNI": 0.388, "wScReNI": 0.481,
                        "source": "Xu et al. 2025, Fig 3B"},
}
with open(out_dir / "clustering_ari_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Saved → {out_dir}/clustering_ari.csv")
print(f"  Saved → {out_dir}/clustering_ari_summary.json")
print("Done.")
