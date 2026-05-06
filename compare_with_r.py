"""compare_with_r_COMPLETE.py — Full Python vs R comparison for all implemented methods

This script compares ALL implemented inference methods in pyScReNI against the R version:
    ✓ CSN          (implemented)
    ✓ LIONESS      (implemented)
    ✓ kScReNI      (implemented) ← Eduard's work
    ✓ wScReNI      (implemented) ← Eduard's work

The comparison script compare_with_r_4.py incorrectly claims wScReNI and kScReNI
aren't implemented. This script corrects that and runs a full comparison.

What this script does
---------------------
1. Loads the 400-cell retinal dataset (same as R paper)
2. Infers networks using all 4 methods
3. Computes degree-based clustering (UMAP + hierarchical)
4. Compares ARI values against R paper values
5. Computes precision/recall against ChIP-Atlas
6. Generates side-by-side comparison figures

Prerequisites
-------------
- Run from inside bsc-screni/ directory
- ChIP-Atlas file at: ../refer/mmp9.TSV.5kb_TF_target.df.txt
- Processed data files in data/processed/

Usage
-----
    python compare_with_r_COMPLETE.py

Or with slurm:
    sbatch slurm/run_compare_complete.sh

Output
------
All figures saved to: output/comparison_complete/
Final summary table with ARI and precision/recall for all methods
"""

import sys
import os
import time

# ── verify working directory ─────────────────────────────────────────────────
if not os.path.exists("src/screni"):
    print("ERROR: Run this script from inside bsc-screni/")
    print("       cd bsc-screni && python compare_with_r_COMPLETE.py")
    sys.exit(1)

# Locate R data directory
_r_data_candidates = ["../data", "ScReNI-master/data"]
R_DATA = next((p for p in _r_data_candidates
               if os.path.exists(os.path.join(p, "mmRetina_RPCMG_Cell100_annotation.csv"))),
              None)
if R_DATA is None:
    print("ERROR: Cannot find the R data directory.")
    print("  Looked in:", _r_data_candidates)
    sys.exit(1)

sys.path.insert(0, "src")

# ── configuration ─────────────────────────────────────────────────────────────
RUN_LIONESS = False   # LIONESS is slow (~2h), set True to include it
RUN_WSCRENI = True    # wScReNI requires ATAC+triplets, set False to skip
TOP_N       = 500     # top edges for degree computation

# ── stage control ─────────────────────────────────────────────────────────────
# The full run is split into two stages to fit within the 4-hour SLURM limit.
#
#   Stage 1  (--stage infer, ~90 min on 8 CPUs):
#       Infer all networks and save them to output/comparison/cache/.
#       Does NOT compute precision/recall or generate figures.
#
#   Stage 2  (--stage analyse, ~10 min on 8 CPUs):
#       Load cached networks from disk, compute degree clustering, P/R,
#       and generate all comparison figures.
#
#   Default (no --stage argument):
#       Run both stages back-to-back (use this locally; too slow for SLURM).
#
# Usage:
#   sbatch slurm/run_compare_infer.sh    # stage 1
#   sbatch slurm/run_compare_analyse.sh  # stage 2  (after stage 1 finishes)
import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--stage", choices=["infer", "analyse", "both"],
                     default="both")
STAGE = _parser.parse_known_args()[0].stage

CACHE_DIR = "output/comparison/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ── imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
from sklearn.metrics import adjusted_rand_score
from pathlib import Path

from screni.data.inference import (
    infer_csn_networks,
    infer_lioness_networks,
    infer_kscreni_networks,
    infer_wscreni_networks,
    GenePeakOverlapLabs,
)
from screni.data.clustering import calculate_scnetwork_degree
from screni.data.evaluation import (
    calculate_network_precision_recall,
    load_chip_atlas,
)
from screni.data.combine import combine_wscreni_networks, ScReniNetworks

sc.settings.verbosity = 0
os.makedirs("output/comparison", exist_ok=True)

# ── cache helpers ─────────────────────────────────────────────────────────────

def _save_networks(nets: "ScReniNetworks", name: str) -> None:
    """Save a ScReniNetworks dict to compressed npz + gene-names txt."""
    npz_path   = os.path.join(CACHE_DIR, f"{name}_networks.npz")
    genes_path = os.path.join(CACHE_DIR, f"{name}_gene_names.txt")
    np.savez_compressed(npz_path, **{k: v for k, v in nets.items()})
    with open(genes_path, "w") as fh:
        fh.write("\n".join(nets.gene_names or []))
    print(f"    Cached → {npz_path}  ({os.path.getsize(npz_path)//1024} KB)")


def _load_networks(name: str) -> "ScReniNetworks | None":
    """Load a cached ScReniNetworks from npz, or return None if not cached."""
    npz_path   = os.path.join(CACHE_DIR, f"{name}_networks.npz")
    genes_path = os.path.join(CACHE_DIR, f"{name}_gene_names.txt")
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=False)
    gene_names = []
    if os.path.exists(genes_path):
        with open(genes_path) as fh:
            gene_names = [l.strip() for l in fh if l.strip()]
    nets = ScReniNetworks(gene_names=gene_names or None)
    for k in data.files:
        nets[k] = data[k]
    print(f"    Loaded from cache: {npz_path}  ({len(nets)} cells)")
    return nets

print("=" * 80)
print("COMPLETE Python ScReNI  ←→  R ScReNI  comparison")
print("Dataset: mouse retinal development (400 cells, 500 genes)")
print("=" * 80)
print()

# ── Step 0: load data ─────────────────────────────────────────────────────────
print(f"R data directory : {os.path.abspath(R_DATA)}")
print("Loading processed data ...")

rna  = ad.read_h5ad("data/processed/retinal_rna_sub.h5ad")
atac = ad.read_h5ad("data/processed/retinal_atac_sub.h5ad")
knn_indices = np.load("data/processed/retinal_knn_indices.npy")

# Cell type labels
annot_path = os.path.join(R_DATA, "mmRetina_RPCMG_Cell100_annotation.csv")
annot = pd.read_csv(annot_path, index_col=0)
cell_types_r = annot["undup.cell.types"].values  # R cell order (actual col name)
cell_types_py = rna.obs["cell_type"].values  # Python cell order (lowercase)

print(f"  RNA matrix  : {rna.shape[1]} genes × {rna.shape[0]} cells")
print(f"  ATAC matrix : {atac.shape[1]} peaks × {atac.shape[0]} cells")
print(f"  Cell types  : {dict(pd.Series(cell_types_py).value_counts())}")

# Load R reference precision/recall values
pr_path = os.path.join(R_DATA, "mmRetina_RPCMG_Cell100.500_scNetwork_precision_recall.csv")
r_pr = pd.read_csv(pr_path)
print(f"Loading R reference precision/recall values ...")
print(f"  R PR CSV: {len(r_pr)} rows, cols: {r_pr.columns.tolist()}")

# Check for ChIP-Atlas file
chip_atlas_path = os.path.join(os.path.dirname(R_DATA), "refer", "mmp9.TSV.5kb_TF_target.df.txt")
has_chip_atlas = os.path.exists(chip_atlas_path)
if has_chip_atlas:
    print(f"ChIP-Atlas file found: {chip_atlas_path}")
    tf_pairs = load_chip_atlas(chip_atlas_path)
    print(f"  Loaded {len(tf_pairs)} TF-target pairs")
else:
    print("ChIP-Atlas file not found — skipping direct precision/recall computation")
    print(f"  Looked for: {chip_atlas_path}")

print()

# ── Step 1: Infer networks (STAGE: infer | both) ─────────────────────────────
results = {}
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

if STAGE in ("infer", "both"):
    print(f"STAGE: infer  (N_JOBS={N_JOBS})")
    print()

    def _infer_or_load(name, infer_fn):
        """Run infer_fn() unless a cached result exists; always save after inference."""
        cached = _load_networks(name)
        if cached is not None:
            print(f"  {name}: loaded {len(cached)} cell networks from cache  (skipping inference)")
            return cached
        nets = infer_fn()
        _save_networks(nets, name)
        return nets

    # ── CSN ──
    print("Inferring CSN networks (~2 minutes) ...")
    t0 = time.time()
    csn_nets = _infer_or_load("CSN", lambda: infer_csn_networks(rna))
    results["CSN"] = csn_nets
    _first_net = next(iter(csn_nets.values()))
    print(f"  CSN: {len(csn_nets)} cell networks  ({time.time()-t0:.1f}s)")
    print(f"  Network size: {_first_net.shape[0]}×{_first_net.shape[1]} genes")
    print(f"  Mean nonzero edges/cell: {np.mean([np.count_nonzero(m) for m in csn_nets.values()]):.1f}")

    # ── LIONESS ──
    if RUN_LIONESS:
        print()
        print("Inferring LIONESS networks (~1-2 hours) ...")
        t0 = time.time()
        lioness_nets = _infer_or_load("LIONESS", lambda: infer_lioness_networks(rna))
        results["LIONESS"] = lioness_nets
        print(f"  LIONESS: {len(lioness_nets)} cell networks  ({time.time()-t0:.1f}s)")
        print(f"  Mean nonzero edges/cell: {np.mean([np.count_nonzero(m) for m in lioness_nets.values()]):.1f}")
    else:
        print()
        print("Skipping LIONESS inference (RUN_LIONESS = False).")
        print("  Set RUN_LIONESS = True to enable it.")

    # ── kScReNI ──
    print()
    print("Inferring kScReNI networks (~50 minutes on 8 CPUs) ...")
    t0 = time.time()
    print(f"  Using {N_JOBS} parallel workers for GENIE3 ...", flush=True)
    kscreni_nets = _infer_or_load(
        "kScReNI",
        lambda: infer_kscreni_networks(rna, k=20, n_features=4000, n_trees=100, n_jobs=N_JOBS),
    )
    results["kScReNI"] = kscreni_nets
    print(f"  kScReNI: {len(kscreni_nets)} cell networks  ({time.time()-t0:.1f}s)")
    print(f"  Mean nonzero edges/cell: {np.mean([np.count_nonzero(m) for m in kscreni_nets.values()]):.1f}")

    # ── wScReNI ──
    if RUN_WSCRENI:
        print()
        print("Inferring wScReNI networks (~35 minutes on 8 CPUs) ...")
        triplet_path = "data/processed/retinal_triplets.csv"
        if not os.path.exists(triplet_path):
            print(f"  WARNING: Triplet file not found: {triplet_path}  — skipping wScReNI.")
            RUN_WSCRENI = False
        else:
            t0 = time.time()
            triplets  = pd.read_csv(triplet_path)
            # BUG FIX: the previous code passed four lists of different lengths
            # (9999, 500, 10000, 44) to GenePeakOverlapLabs(), so zip() inside
            # __post_init__ truncated to only 44 entries — all with wrong data
            # (first 44 RNA genes, first 44 ATAC peaks, formatted strings as
            # labels instead of "TF"/"target", one TF name per entry).
            # Effect: peak importance boost was never applied to any cell network,
            # making wScReNI behave identically to kScReNI.
            # Fix: use from_dataframe() which correctly builds four parallel lists
            # of length = n_triplets (one entry per triplet row).
            triplets_for_labs = triplets.rename(columns={
                "target_gene": "gene.name",
                "peak":        "peak.name",
            })
            labs = GenePeakOverlapLabs.from_dataframe(triplets_for_labs)
            nearest_neighbors_idx = np.load("data/processed/retinal_knn_indices.npy")
            import pathlib as _pl
            network_path = _pl.Path("output/wscreni_networks")
            network_path.mkdir(parents=True, exist_ok=True)

            wscreni_nets = _infer_or_load(
                "wScReNI",
                lambda: infer_wscreni_networks(
                    expr=rna, peak_mat=atac, labs=labs,
                    nearest_neighbors_idx=nearest_neighbors_idx,
                    network_path=str(network_path),
                    n_jobs=N_JOBS, n_trees=100,
                ),
            )
            results["wScReNI"] = wscreni_nets
            print(f"  wScReNI: {len(wscreni_nets)} cell networks  ({time.time()-t0:.1f}s)")
            print(f"  Mean nonzero edges/cell: {np.mean([np.count_nonzero(m) for m in wscreni_nets.values()]):.1f}")

    print()
    if STAGE == "infer":
        print("STAGE infer complete.  Networks saved to:", CACHE_DIR)
        print("Now run:  python compare_with_r.py --stage analyse")
        sys.exit(0)

elif STAGE == "analyse":
    # ── Load networks from cache instead of re-inferring ─────────────────────
    print("STAGE: analyse  (loading networks from cache ...)")
    print()
    for name in (["CSN"] +
                 (["LIONESS"] if RUN_LIONESS else []) +
                 ["kScReNI"] +
                 (["wScReNI"] if RUN_WSCRENI else [])):
        nets = _load_networks(name)
        if nets is None:
            print(f"  WARNING: No cache found for {name} at {CACHE_DIR}/{name}_networks.npz")
            print(f"           Run  python compare_with_r.py --stage infer  first.")
        else:
            results[name] = nets
    if not results:
        print("ERROR: No cached networks found.  Run --stage infer first.")
        sys.exit(1)
    print()

# ── Step 2: Compute degree-based clustering (STAGE: analyse | both) ──────────
print("Computing degree-based clustering ...")

clustering_results = {}

for method_name, networks in results.items():
    print(f"  {method_name} ...", end=" ", flush=True)
    
    # Calculate degrees
    degree_result = calculate_scnetwork_degree(
        sc_networks={method_name: networks},
        top=[TOP_N] * rna.n_obs,
        cell_type_annotation=cell_types_py,
        ntype=4,
    )[method_name]
    
    # Create AnnData for clustering
    adata_deg = degree_result.out_degree_umap
    adata_deg.obs["cell_type"] = list(cell_types_py)
    
    # UMAP + leiden already computed inside calculate_scnetwork_degree
    umap_ari = adjusted_rand_score(cell_types_py, adata_deg.obs["leiden"].values)
    
    # Hierarchical clustering — matches R's Calculate_scNetwork_degree.R line 69-70:
    #   degree_hclust <- dist(cor(log(degree_data + 1)))
    #   hc1 <- hclust(degree_hclust)   # default linkage = "complete"
    #   Cluster_hclust <- cutree(hc1, k=ntype)
    #
    # BUG FIX: the previous code used pdist(X, 'euclidean') + Ward linkage on
    # the raw degree matrix.  R uses Euclidean distance on the *correlation*
    # vectors (each row of cor(log(degree+1)) is a cell's correlation profile),
    # then complete linkage.  These are two different distance spaces and give
    # different ARI values (Python was 0.364 vs R's 0.226 for CSN).
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist

    _X_deg = degree_result.outdegree.T            # (n_cells, n_genes)
    _log_deg = np.log(_X_deg + 1)                 # log(degree + 1), matching R
    _corr_mat = np.corrcoef(_log_deg)             # (n_cells, n_cells) — cor()
    # dist() on a matrix in R = Euclidean distance between its rows
    _dist_condensed = pdist(_corr_mat, metric='euclidean')
    linkage_matrix = linkage(_dist_condensed, method='complete')  # R default
    hclust_labels = fcluster(linkage_matrix, t=4, criterion='maxclust')
    hclust_ari = adjusted_rand_score(cell_types_py, hclust_labels)
    
    clustering_results[method_name] = {
        'umap_ari': umap_ari,
        'hclust_ari': hclust_ari,
        'degree_result': degree_result,
        'adata': adata_deg,
    }
    
    print(f"UMAP ARI={umap_ari:.3f}, hclust ARI={hclust_ari:.3f}")

print()

# ── Step 3: Compute precision/recall ──────────────────────────────────────────
if has_chip_atlas:
    print("Computing precision/recall against ChIP-Atlas ...")
    
    pr_results = {}
    gene_names = rna.var_names.tolist()
    
    # calculate_network_precision_recall requires "CSN" key to be present;
    # it uses CSN nonzero counts to set the per-cell threshold for all methods.
    # We pass all inferred networks together in one call so CSN is always included.
    _pr_input = {k: v for k, v in results.items()}  # includes CSN
    if "CSN" not in _pr_input and "CSN" in results:
        _pr_input["CSN"] = results["CSN"]
    pr_vals_all = calculate_network_precision_recall(
        sc_networks=_pr_input,
        tf_target_pair=tf_pairs,
        top_number=[],    # empty list = only evaluate threshold 0 (CSN-matched slot)
        # Default None evaluates 7 extra thresholds = 8x more work (9600 vs 1200 calls)
        n_jobs=N_JOBS,
    )
    # threshold key=0 is the CSN-matched slot — directly comparable to R values
    _pr_df_all = pr_vals_all.get(0, list(pr_vals_all.values())[0])

    for method_name, networks in results.items():
        print(f"  {method_name} ...", end=" ", flush=True)
        
        _pr_method = _pr_df_all[_pr_df_all["scNetwork_type"] == method_name] \
            if "scNetwork_type" in _pr_df_all.columns \
            else _pr_df_all
        pr_vals = {
            "precision": _pr_method["precision"].mean() if "precision" in _pr_method.columns else 0.0,
            "recall":    _pr_method["recall"].mean()    if "recall"    in _pr_method.columns else 0.0,
        }
        
        pr_results[method_name] = {
            'precision': pr_vals['precision'],
            'recall': pr_vals['recall'],
        }
        
        print(f"precision={pr_vals['precision']:.4f}, recall={pr_vals['recall']:.4f}")
    
    print()

# ── Step 4: Compare with R paper values ───────────────────────────────────────
print("=" * 80)
print("ARI comparison  (Python vs R paper values)")
print("-" * 80)
print(f"{'Method':<12} {'Python UMAP ARI':>17} {'R UMAP ARI':>12} {'Python hclust ARI':>19} {'R hclust ARI':>14}")
print("-" * 80)

# R paper ARI values
r_ari_values = {
    'CSN': {'umap': 0.488, 'hclust': 0.226},
    'LIONESS': {'umap': 0.001, 'hclust': 0.002},
    'kScReNI': {'umap': 0.638, 'hclust': 0.456},  # from paper
    'wScReNI': {'umap': 0.694, 'hclust': 0.510},  # from paper
}

for method in ['CSN', 'kScReNI', 'wScReNI', 'LIONESS']:
    if method in clustering_results:
        py_umap = clustering_results[method]['umap_ari']
        py_hclust = clustering_results[method]['hclust_ari']
        r_umap = r_ari_values.get(method, {}).get('umap', 'N/A')
        r_hclust = r_ari_values.get(method, {}).get('hclust', 'N/A')
        
        if isinstance(r_umap, float):
            print(f"{method:<12} {py_umap:>17.3f} {r_umap:>12.3f} {py_hclust:>19.3f} {r_hclust:>14.3f}")
        else:
            print(f"{method:<12} {py_umap:>17.3f} {'N/A':>12} {py_hclust:>19.3f} {'N/A':>14}")
    elif method in r_ari_values:
        r_umap = r_ari_values[method]['umap']
        r_hclust = r_ari_values[method]['hclust']
        print(f"{method:<12} {'(not run)':>17} {r_umap:>12.3f} {'(not run)':>19} {r_hclust:>14.3f}")

print("-" * 80)
print()

# ── Step 5: Generate comparison figures ───────────────────────────────────────
print("Generating comparison figures ...")

# Figure 1: ARI comparison bar plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

methods = ['CSN', 'kScReNI', 'wScReNI', 'LIONESS']
colors = {'CSN': '#1f77b4', 'kScReNI': '#ff7f0e', 'wScReNI': '#2ca02c', 'LIONESS': '#d62728'}

# UMAP ARI
py_umap_aris = [clustering_results.get(m, {}).get('umap_ari', 0) for m in methods]
r_umap_aris = [r_ari_values.get(m, {}).get('umap', 0) for m in methods]

x = np.arange(len(methods))
width = 0.35

ax1.bar(x - width/2, py_umap_aris, width, label='Python', alpha=0.8)
ax1.bar(x + width/2, r_umap_aris, width, label='R', alpha=0.8)
ax1.set_ylabel('ARI Score')
ax1.set_title('UMAP Clustering ARI')
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Hierarchical clustering ARI
py_hclust_aris = [clustering_results.get(m, {}).get('hclust_ari', 0) for m in methods]
r_hclust_aris = [r_ari_values.get(m, {}).get('hclust', 0) for m in methods]

ax2.bar(x - width/2, py_hclust_aris, width, label='Python', alpha=0.8)
ax2.bar(x + width/2, r_hclust_aris, width, label='R', alpha=0.8)
ax2.set_ylabel('ARI Score')
ax2.set_title('Hierarchical Clustering ARI')
ax2.set_xticks(x)
ax2.set_xticklabels(methods)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("output/comparison/ari_comparison.png", dpi=300, bbox_inches='tight')
print("  Saved: output/comparison_complete/ari_comparison.png")
plt.close()

# Figure 2: Precision/Recall — Python vs R reference, side-by-side
# R reference values come from the pre-computed CSV loaded at the top as r_pr.
# That CSV is mmRetina_RPCMG_Cell100.500_scNetwork_precision_recall.csv and
# contains per-cell precision/recall for CSN, LIONESS, kScReNI, CeSpGRN, wScReNI
# at the CSN-matched threshold — exactly the same threshold used here.
if has_chip_atlas and pr_results:
    methods_with_pr = [m for m in methods if m in pr_results]

    # Aggregate R reference: mean per method from the loaded CSV
    r_pr_mean = {}
    if r_pr is not None and 'scNetwork_type' in r_pr.columns:
        for m in methods_with_pr:
            sub = r_pr[r_pr['scNetwork_type'] == m]
            if len(sub):
                r_pr_mean[m] = {
                    'precision': float(sub['precision'].mean()),
                    'recall':    float(sub['recall'].mean()),
                }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(methods_with_pr))
    width = 0.35

    for ax, metric in zip(axes, ['precision', 'recall']):
        py_vals = [pr_results.get(m, {}).get(metric, 0) for m in methods_with_pr]
        r_vals  = [r_pr_mean.get(m, {}).get(metric, None) for m in methods_with_pr]

        ax.bar(x - width/2, py_vals, width, label='Python', alpha=0.85, color='#1f77b4')

        r_vals_plot = [v if v is not None else 0 for v in r_vals]
        r_bars = ax.bar(x + width/2, r_vals_plot, width, label='R reference',
                        alpha=0.85, color='#ff7f0e')

        # Grey out bars for methods not present in R CSV
        for bar, v in zip(r_bars, r_vals):
            if v is None:
                bar.set_alpha(0.15)
                bar.set_hatch('//')

        # Annotate Δ above each pair
        for i, m in enumerate(methods_with_pr):
            r_v = r_vals[i]
            if r_v is not None:
                delta = py_vals[i] - r_v
                y_top = max(py_vals[i], r_v) * 1.04
                ax.annotate(f'Δ{delta:+.4f}',
                            xy=(x[i], y_top), ha='center', fontsize=8,
                            color='#c00' if abs(delta) > 0.005 else '#444')

        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'Network {metric.capitalize()} vs ChIP-Atlas\n'
                     f'(evaluated at CSN-matched edge threshold per cell)')
        ax.set_xticks(x)
        ax.set_xticklabels(methods_with_pr)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Python vs R reference — precision & recall against ChIP-Atlas ground truth',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("output/comparison/precision_recall_comparison.png", dpi=300, bbox_inches='tight')
    print("  Saved: output/comparison/precision_recall_comparison.png")
    plt.close()

print()

# ── Step 6: Save summary table ────────────────────────────────────────────────
print("Saving summary table ...")

summary_data = []
for method in ['CSN', 'kScReNI', 'wScReNI', 'LIONESS']:
    if method in clustering_results:
        row = {
            'method': method,
            'python_umap_ari': clustering_results[method]['umap_ari'],
            'python_hclust_ari': clustering_results[method]['hclust_ari'],
            'r_umap_ari': r_ari_values.get(method, {}).get('umap', np.nan),
            'r_hclust_ari': r_ari_values.get(method, {}).get('hclust', np.nan),
        }
        
        if has_chip_atlas and method in pr_results:
            row['precision'] = pr_results[method]['precision']
            row['recall'] = pr_results[method]['recall']
            # R reference PR from the pre-computed CSV
            if r_pr is not None and 'scNetwork_type' in r_pr.columns:
                sub = r_pr[r_pr['scNetwork_type'] == method]
                row['r_precision'] = float(sub['precision'].mean()) if len(sub) else float('nan')
                row['r_recall']    = float(sub['recall'].mean())    if len(sub) else float('nan')
        
        summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv("output/comparison/summary.csv", index=False)
print("  Saved: output/comparison_complete/summary.csv")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print("All methods comparison:")
print(summary_df.to_string(index=False))
print()
print("Interpretation guide:")
print("  ARI = 1.0  → perfect match between clustering and cell types")
print("  ARI = 0.0  → random clustering")
print("  |Python ARI - R ARI| < 0.05 → implementations agree")
print()
print("Output saved to: output/comparison_complete/")
print("  - ari_comparison.png")
if has_chip_atlas:
    print("  - precision_recall_comparison.png")
print("  - summary.csv")
print()
print("✓ COMPLETE: All implemented methods have been compared against R!")
