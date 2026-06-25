"""compare_with_r.py — Full Python vs Reference comparison

What this script does
---------------------
1. Loads the 400-cell retinal dataset.
2. Calculates the intersection of Global HVGs and Cell-Specific HVGs.
3. Filters the ChIP-Atlas ground truth strictly to this intersection.
4. Infers networks using current methods (CSN, kScReNI, wScReNI subspace with padding).
5. Loads a pre-computed Reference wScReNI (Global, no padding).
6. Computes degree-based clustering (UMAP + hierarchical).
7. Computes precision/recall dynamically against the filtered ground truth.
8. Generates side-by-side comparison figures against the Python Reference.

Prerequisites
-------------
- Processed data files in data/processed/
- Reference wScReNI cache in output/comparison/cache/wScReNI_networks.npz
"""

import sys
import os
import time
import argparse as _ap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
from sklearn.metrics import adjusted_rand_score
from pathlib import Path

# ── verify working directory ─────────────────────────────────────────────────
if not os.path.exists("src/screni"):
    print("ERROR: Run this script from inside bsc-screni/")
    print("       cd bsc-screni && python compare_with_r.py")
    sys.exit(1)

# Locate R data directory (still used for annotations and ChIP-Atlas)
_r_data_candidates = ["data", "ScReNI-master/data"]
R_DATA = next((p for p in _r_data_candidates
               if os.path.exists(os.path.join(p, "mmRetina_RPCMG_Cell100_annotation.csv"))),
              None)
if R_DATA is None:
    print("ERROR: Cannot find the data directory.")
    sys.exit(1)

sys.path.insert(0, "src")

# ── configuration ─────────────────────────────────────────────────────────────
RUN_LIONESS = False   
RUN_WSCRENI = True    
TOP_N       = 500     

_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--stage", choices=["infer", "analyse", "both"], default="both")
STAGE = _parser.parse_known_args()[0].stage

CACHE_DIR = "output/comparison/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("output/comparison", exist_ok=True)

# ── imports ───────────────────────────────────────────────────────────────────
from screni.data.inference_type import (
    infer_csn_networks,
    infer_lioness_networks,
    infer_kscreni_networks,
    infer_wscreni_networks,
    GenePeakOverlapLabs,
)
from screni.data.clustering import calculate_scnetwork_degree
from screni.data.evaluation import load_chip_atlas
from screni.data.combine import combine_wscreni_networks, ScReniNetworks
sc.settings.verbosity = 0


# ── cache helpers ─────────────────────────────────────────────────────────────
def _save_networks(nets: "ScReniNetworks", name: str) -> None:
    npz_path   = os.path.join(CACHE_DIR, f"{name}_networks_type.npz")
    genes_path = os.path.join(CACHE_DIR, f"{name}_gene_names_type.txt")
    np.savez_compressed(npz_path, **{k: v for k, v in nets.items()})
    with open(genes_path, "w") as fh:
        fh.write("\n".join(nets.gene_names or []))
    print(f"    Cached → {npz_path}  ({os.path.getsize(npz_path)//1024} KB)")

def _load_networks(name: str) -> "ScReniNetworks | None":
    npz_path   = os.path.join(CACHE_DIR, f"{name}_networks_type.npz")
    genes_path = os.path.join(CACHE_DIR, f"{name}_gene_names_type.txt")
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

def plot_pr_bar_per_celltype(cell_type_metrics, output_dir="output/comparison", filename="pr_bar_by_celltype.png"):
    cell_types = list(cell_type_metrics.keys())
    precisions = [cell_type_metrics[ct]['precision'] for ct in cell_types]
    recalls = [cell_type_metrics[ct]['recall'] for ct in cell_types]
    x = np.arange(len(cell_types))
    width = 0.35  
    fig, ax = plt.subplots(figsize=(10, 6))
    rects_p = ax.bar(x - width/2, precisions, width, label='Precision', color='#2ca02c', edgecolor='black')
    rects_r = ax.bar(x + width/2, recalls, width, label='Recall', color='#1f77b4', edgecolor='black')
    ax.set_ylabel('Score')
    ax.set_title('Precision and Recall by Cell Type (Type-Specific HVGs)', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(cell_types, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.bar_label(rects_p, padding=3, fmt='%.3f', fontsize=9)
    ax.bar_label(rects_r, padding=3, fmt='%.3f', fontsize=9)
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pr_curves_per_celltype(cell_type_curves, output_dir="output/comparison", filename="pr_curves_by_celltype.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap('tab10')
    for i, (ct, metrics) in enumerate(cell_type_curves.items()):
        ax.plot(metrics['recall'], metrics['precision'], marker='o', linewidth=2, 
                label=ct, color=cmap(i % 10))
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves by Cell Type (Type-Specific HVGs)')
    ax.legend(title="Cell Types", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_pr_curves_per_celltype(cell_specific_metrics, global_metrics, output_dir="output/comparison", filename="combined_pr_curves_by_celltype.png"):
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap('tab10')

    cell_types = list(cell_specific_metrics.keys())
    
    for i, ct in enumerate(cell_types):
        color = cmap(i % 10)
        
        # 1. Plot Cell-specific (Solid line, circle marker)
        cs_p = cell_specific_metrics[ct]['precision']
        cs_r = cell_specific_metrics[ct]['recall']
        ax.plot(cs_r, cs_p, marker='o', linestyle='-', linewidth=2, color=color, 
                label=f"{ct} (type-specific)")

        # 2. Plot Global Reference (Dashed line, square marker)
        if ct in global_metrics:
            g_p = global_metrics[ct]['precision']
            g_r = global_metrics[ct]['recall']
            ax.plot(g_r, g_p, marker='s', linestyle='--', linewidth=2, color=color, alpha=0.7,
                    label=f"{ct} (gwScReNI)")

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves by Cell Type\n(Type-Specific vs gwScReNI)', fontsize=14)
    
    # Put the legend outside the plot so it doesn't cover the lines
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined cell-type PR curves to {save_path}")

print("=" * 80)
print("ScReNI Evaluation: cell-specific vs global")
print("Dataset: mouse retinal development (400 cells)")
print("=" * 80)
print()


# ── Step 0: load data ─────────────────────────────────────────────────────────
print("Loading processed data ...")
rna  = ad.read_h5ad("data/processed/retinal_rna_sub_type.h5ad")
atac = ad.read_h5ad("data/processed/retinal_atac_sub_type.h5ad")
knn_indices = np.load("data/processed/retinal_knn_indices_type.npy")

# Annotations
annot_path = os.path.join(R_DATA, "mmRetina_RPCMG_Cell100_annotation.csv")
annot = pd.read_csv(annot_path, index_col=0)
cell_types_py = rna.obs["cell_type"].values

# ==== Determine HVG Subspace Intersection ====
global_rna_path = "data/processed/retinal_rna_sub.h5ad"
if os.path.exists(global_rna_path):
    rna_global = ad.read_h5ad(global_rna_path)
    global_hvgs = set(rna_global.var_names)
else:
    print(f"WARNING: Global RNA file not found at {global_rna_path}. Falling back to current var_names.")
    global_hvgs = set(rna.var_names)

union_cell_hvgs = set(rna.var_names)
valid_genes = global_hvgs.intersection(union_cell_hvgs)
print(f"  HVG Subspace Intersection: {len(valid_genes)} genes shared between Global and Cell-Specific Union")

print(f"  RNA matrix (Cell-Specific) : {rna.shape[1]} genes × {rna.shape[0]} cells")
print(f"  ATAC matrix                : {atac.shape[1]} peaks × {atac.shape[0]} cells")

# Check for ChIP-Atlas file
chip_atlas_path = os.path.join(os.path.dirname(R_DATA), "refer", "mmp9.TSV.5kb_TF_target.df.txt")
has_chip_atlas = os.path.exists(chip_atlas_path)
if has_chip_atlas:
    print(f"ChIP-Atlas file found: {chip_atlas_path}")
    tf_pairs = load_chip_atlas(chip_atlas_path)
    print(f"  Loaded {len(tf_pairs)} raw TF-target pairs")
    
    # ==== Filter ground truth strictly to the intersection ====
    filtered_tf_pairs = set()
    for pair in tf_pairs:
        tf, target = pair.split('_')
        if tf in valid_genes and target in valid_genes:
            filtered_tf_pairs.add(pair)
            
    tf_pairs = filtered_tf_pairs
    print(f"  Filtered to {len(tf_pairs)} valid pairs strictly within the intersecting HVG subspace.")
else:
    print("ChIP-Atlas file not found — skipping direct precision/recall computation")

print()


# ── Step 1: Infer networks (STAGE: infer | both) ─────────────────────────────
results = {}
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

if STAGE in ("infer", "both"):
    print(f"STAGE: infer  (N_JOBS={N_JOBS})\n")

    def _infer_or_load(name, infer_fn):
        cached = _load_networks(name)
        if cached is not None:
            print(f"  {name}: loaded {len(cached)} cell networks from cache")
            return cached
        nets = infer_fn()
        _save_networks(nets, name)
        return nets

    # ── CSN ──
    print("Inferring CSN networks (~2 minutes) ...")
    t0 = time.time()
    csn_nets = _infer_or_load("CSN", lambda: infer_csn_networks(rna))
    results["CSN"] = csn_nets
    print(f"  CSN: {len(csn_nets)} cell networks  ({time.time()-t0:.1f}s)")

    # ── LIONESS ──
    if RUN_LIONESS:
        print("\nInferring LIONESS networks (~1-2 hours) ...")
        t0 = time.time()
        lioness_nets = _infer_or_load("LIONESS", lambda: infer_lioness_networks(rna))
        results["LIONESS"] = lioness_nets
        print(f"  LIONESS: {len(lioness_nets)} cell networks  ({time.time()-t0:.1f}s)")

    # ── kScReNI ──
    print("\nInferring kScReNI networks (~50 minutes on 8 CPUs) ...")
    t0 = time.time()
    kscreni_nets = _infer_or_load(
        "kScReNI",
        lambda: infer_kscreni_networks(rna, k=20, n_features=4000, n_trees=100, n_jobs=N_JOBS),
    )
    results["kScReNI"] = kscreni_nets
    print(f"  kScReNI: {len(kscreni_nets)} cell networks  ({time.time()-t0:.1f}s)")

    # ── wScReNI (Current Subspace) ──
    if RUN_WSCRENI:
        print("\nInferring wScReNI networks (Subspace Run) ...")
        
        labs_dict = {}
        all_valid_peaks = set()
        
        for ct in rna.obs["cell_type"].unique():
            triplet_path = f"data/processed/retinal_{ct}_triplets_type.csv"
            if os.path.exists(triplet_path):
                triplets = pd.read_csv(triplet_path)
                triplets_for_labs = triplets.rename(columns={"target_gene": "gene.name", "peak": "peak.name"})
                labs_dict[ct] = GenePeakOverlapLabs.from_dataframe(triplets_for_labs)
                all_valid_peaks.update(triplets["peak"].unique())
        
        if labs_dict:
            t0 = time.time()
            valid_peaks_list = [p for p in atac.var_names if p in all_valid_peaks]
            import scipy.sparse as _sp
            peak_overlap_arr = atac[:, valid_peaks_list].X
            peak_overlap_arr = peak_overlap_arr.toarray() if _sp.issparse(peak_overlap_arr) else np.asarray(peak_overlap_arr)
            peak_overlap_arr = peak_overlap_arr.astype(np.float64)

            peak_overlap_arr += np.random.default_rng(seed=100).normal(0, 1e-5, peak_overlap_arr.shape)
            network_path = Path("output/wscreni_networks")
            network_path.mkdir(parents=True, exist_ok=True)

            wscreni_nets = _infer_or_load(
                "wScReNI",
                lambda: infer_wscreni_networks(
                    expr=rna, peak_mat=peak_overlap_arr, peak_names=valid_peaks_list,
                    labs=labs_dict, nearest_neighbors_idx=knn_indices,
                    network_path=str(network_path), n_jobs=N_JOBS, n_trees=100,
                ),
            )
            
            # --- PADDING LAYER for Subspace Matrix Alignment ---
            padded_nets = ScReniNetworks(gene_names=rna.var_names.tolist())
            for cell_i, cell_name in enumerate(rna.obs_names):
                net = wscreni_nets[cell_name]
                if net.shape == (rna.n_vars, rna.n_vars):
                    padded_nets[cell_name] = net
                else:
                    padded_net = np.full((rna.n_vars, rna.n_vars), np.nan, dtype=np.float64)
                    ct = rna.obs['cell_type'].iloc[cell_i]
                    ct_hvgs = rna.uns['cell_type_hvgs'][ct]
                    global_indices = [i for i, g in enumerate(rna.var_names) if g in ct_hvgs]
                    mesh_i, mesh_j = np.meshgrid(global_indices, global_indices, indexing='ij')
                    padded_net[mesh_i, mesh_j] = net
                    padded_nets[cell_name] = padded_net

            results["wScReNI"] = padded_nets
            print(f"  wScReNI: {len(padded_nets)} cell networks  ({time.time()-t0:.1f}s)")

    # ── Step 1.5: Load Reference Python Networks (Global wScReNI) ───────────
    print("\nLoading Reference wScReNI networks for baseline comparison ...")
    ref_npz_path = os.path.join(CACHE_DIR, "wScReNI_networks.npz")  # Notice: No _type
    ref_genes_path = os.path.join(CACHE_DIR, "wScReNI_gene_names.txt")

    if os.path.exists(ref_npz_path):
        ref_data = np.load(ref_npz_path, allow_pickle=False)
        ref_gene_names = []
        if os.path.exists(ref_genes_path):
            with open(ref_genes_path) as fh:
                ref_gene_names = [l.strip() for l in fh if l.strip()]
        
        # Loaded identically as created, NO np.nan padding logic applied here
        ref_nets = ScReniNetworks(gene_names=ref_gene_names or None)
        for k in ref_data.files:
            ref_nets[k] = ref_data[k]
            
        print(f"  Loaded Reference wScReNI: {len(ref_nets)} cells")
        results["Reference_wScReNI"] = ref_nets
    else:
        print(f"  WARNING: Reference networks not found at {ref_npz_path}. Analysis will skip reference.")
    print()



# ── Step 3: Compute precision/recall ──────────────────────────────────────────
if has_chip_atlas:
    print("Computing precision/recall against Filtered Intersecting ChIP-Atlas ...")
    pr_results = {}
    gene_names = rna.var_names.tolist()
    
    from screni.data.precision_recall import calculate_precision_recall
    
    # 1. Evaluate Global Methods (CSN, kScReNI, LIONESS) at 1% threshold dynamically
    eval_pct_global = 0.01
    for method_name in ['CSN', 'kScReNI', 'LIONESS']:
        if method_name in results:
            print(f"  {method_name} (Threshold: Top {eval_pct_global*100}% of subspace) ...", end=" ", flush=True)
            method_precision, method_recall = [], []
            nets = results[method_name]
            current_genes = nets.gene_names if nets.gene_names else gene_names
            max_possible_edges = len(current_genes) * len(current_genes)
            dynamic_top_n = max(1, int(max_possible_edges * eval_pct_global))
            
            for cell_i, cell_name in enumerate(rna.obs_names):
                p, r = calculate_precision_recall(
                    scnetwork_weights=nets[cell_name],
                    tf_target_pair=tf_pairs,
                    top_number=dynamic_top_n,
                    gene_names=current_genes
                )
                if not np.isnan(p): method_precision.append(p)
                if not np.isnan(r): method_recall.append(r)
            
            pr_results[method_name] = {
                'precision': np.mean(method_precision) if method_precision else 0.0,
                'recall': np.mean(method_recall) if method_recall else 0.0,
            }
            print(f"precision={pr_results[method_name]['precision']:.4f}, recall={pr_results[method_name]['recall']:.4f}")

    # 2. Evaluate Subspace & Reference wScReNI Methods dynamically
    w_variants = [m for m in results.keys() if "wScReNI" in m]
    if w_variants:
        eval_pct = 0.01  # Evaluate at 1% of valid edges
        
        for w_method in w_variants:
            print(f"  {w_method} (Threshold: Top {eval_pct*100}% of valid capacity) ...", end=" ", flush=True)
            w_precision, w_recall = [], []
            w_nets = results[w_method]
            
            # Determine appropriate gene labels (Reference loads its own global labels)
            current_genes = w_nets.gene_names if w_nets.gene_names else gene_names
            
            for cell_i, cell_name in enumerate(rna.obs_names):
                ct = rna.obs['cell_type'].iloc[cell_i]
                
                # Dynamic Capacity calculation
                if w_method == "wScReNI" and 'cell_type_hvgs' in rna.uns:
                    # Current Run is a subspace method, cap at ct_hvgs size
                    ct_hvgs = rna.uns['cell_type_hvgs'][ct]
                    max_possible_edges = len(ct_hvgs) * len(ct_hvgs)
                else:
                    # Reference is global, cap at total global genes in its network
                    max_possible_edges = len(current_genes) * len(current_genes)
                    
                dynamic_top_n = max(1, int(max_possible_edges * eval_pct))
                
                p, r = calculate_precision_recall(
                    scnetwork_weights=w_nets[cell_name],
                    tf_target_pair=tf_pairs,  # This is already correctly filtered to intersection!
                    top_number=dynamic_top_n,
                    gene_names=current_genes
                )
                
                if not np.isnan(p): w_precision.append(p)
                if not np.isnan(r): w_recall.append(r)
                
            pr_results[w_method] = {
                'precision': np.mean(w_precision) if w_precision else 0.0,
                'recall': np.mean(w_recall) if w_recall else 0.0,
            }
            print(f"precision={pr_results[w_method]['precision']:.4f}, recall={pr_results[w_method]['recall']:.4f}")
    
    print()


# ── Step 4: Compare with Reference ───────────────────────────────────────────

methods = ['CSN', 'kScReNI', 'wScReNI', 'LIONESS']


# ── Step 5: Generate comparison figures ───────────────────────────────────────
print("Generating comparison figures ...")


# Figure 2: Precision/Recall 
if has_chip_atlas and pr_results:
    methods_with_pr = [m for m in methods if m in pr_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(methods_with_pr))
    width = 0.35

    for ax, metric in zip(axes, ['precision', 'recall']):
        py_vals = [pr_results[m].get(metric, 0) for m in methods_with_pr]
        ref_vals = [pr_results.get("Reference_wScReNI", {}).get(metric, 0) for _ in methods_with_pr]

        ax.bar(x - width/2, py_vals, width, label='type-specific wScReNI', alpha=0.85, color='#1f77b4')
        ax.bar(x + width/2, ref_vals, width, label='gwScReNI', alpha=0.85, color='#ff7f0e')

        for i, m in enumerate(methods_with_pr):
            r_v = ref_vals[i]
            if r_v:
                delta = py_vals[i] - r_v
                y_top = max(py_vals[i], r_v) * 1.04
                ax.annotate(f'Δ{delta:+.4f}',
                            xy=(x[i], y_top), ha='center', fontsize=8,
                            color='#c00' if abs(delta) > 0.005 else '#444')

        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'Network {metric.capitalize()} vs Filtered ChIP-Atlas')
        ax.set_xticks(x)
        ax.set_xticklabels(methods_with_pr)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Type-specific wScReNI vs gwScReNI — Precision & Recall on Shared Subspace', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("output/comparison/precision_recall_comparison_type.png", dpi=300, bbox_inches='tight')
    plt.close()


# ── Step 6: Save summary table ────────────────────────────────────────────────
print("Saving summary table ...")

summary_data = []
for method in ['CSN', 'kScReNI', 'wScReNI', 'LIONESS']:
    row = {
        'method': method
    }
    
    if has_chip_atlas and method in pr_results:
        row['current_precision'] = pr_results[method]['precision']
        row['current_recall'] = pr_results[method]['recall']
        row['ref_precision'] = pr_results.get("Reference_wScReNI", {}).get('precision', np.nan)
        row['ref_recall']    = pr_results.get("Reference_wScReNI", {}).get('recall', np.nan)
        
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv("output/comparison/summary_type.csv", index=False)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(summary_df.to_string(index=False))
print("\n✓ COMPLETE: Evaluated against Unpadded Global Reference natively.")


# =============================================================================
# ── ADVANCED SUBSPACE VISUALIZATIONS ─────────────────────────────────────────
# =============================================================================


#  Edge Weight Distribution
if 'wScReNI' in results and 'Reference_wScReNI' in results:
    print("  Generating wScReNI edge weight distribution plot with 1% thresholds ...")
    
    curr_weights = []
    curr_thresh_list = []
    eval_pct = 0.01
    
    # Collect non-NaN, non-zero weights & thresholds for Current Subspace Run
    for cell_i, cell in enumerate(rna.obs_names):
        if cell in results['wScReNI']:
            net = results['wScReNI'][cell]
            valid_edges = net[~np.isnan(net)]
            valid_edges = valid_edges[np.abs(valid_edges) > 1e-6] 
            curr_weights.append(valid_edges)
            
            # Determine capacity to find the top 1% threshold for this cell
            if 'cell_type_hvgs' in rna.uns:
                ct = rna.obs['cell_type'].iloc[cell_i]
                ct_hvgs = rna.uns['cell_type_hvgs'][ct]
                max_possible_edges = len(ct_hvgs) * len(ct_hvgs)
            else:
                current_genes = results['wScReNI'].gene_names if results['wScReNI'].gene_names else rna.var_names.tolist()
                max_possible_edges = len(current_genes) * len(current_genes)
                
            top_n = max(1, int(max_possible_edges * eval_pct))
            if len(valid_edges) >= top_n:
                sorted_edges = np.sort(valid_edges)[::-1]
                curr_thresh_list.append(sorted_edges[top_n - 1])
            elif len(valid_edges) > 0:
                curr_thresh_list.append(np.min(valid_edges))
        
    ref_weights = []
    ref_thresh_list = []
    ref_global_genes = results['Reference_wScReNI'].gene_names if results['Reference_wScReNI'].gene_names else rna.var_names.tolist()
    max_possible_edges_ref = len(ref_global_genes) * len(ref_global_genes)
    top_n_ref = max(1, int(max_possible_edges_ref * eval_pct))
    
    # Collect non-NaN, non-zero weights & thresholds for Global Reference Run
    for cell in rna.obs_names:
        if cell in results['Reference_wScReNI']:
            net = results['Reference_wScReNI'][cell]
            valid_edges = net[~np.isnan(net)]
            valid_edges = valid_edges[np.abs(valid_edges) > 1e-6]
            ref_weights.append(valid_edges)
            
            if len(valid_edges) >= top_n_ref:
                sorted_edges = np.sort(valid_edges)[::-1]
                ref_thresh_list.append(sorted_edges[top_n_ref - 1])
            elif len(valid_edges) > 0:
                ref_thresh_list.append(np.min(valid_edges))
        
    if curr_weights and ref_weights:
        # Concatenate cell matrices into flat 1D arrays for histogram computation
        curr_weights = np.concatenate(curr_weights)
        ref_weights = np.concatenate(ref_weights)
        
        curr_thresh_mean = np.mean(curr_thresh_list) if curr_thresh_list else 0
        ref_thresh_mean = np.mean(ref_thresh_list) if ref_thresh_list else 0
        
        fig, ax = plt.subplots(figsize=(9, 6))
        
        # 'stepfilled' is cleaner for overlapping histograms
        # density=True normalizes both curves despite different capacities
        ax.hist(curr_weights, bins=100, density=True, histtype='stepfilled', 
                alpha=0.6, label='Type-specific wScReNI', color='#1f77b4', edgecolor='#1f77b4')
        ax.hist(ref_weights, bins=100, density=True, histtype='stepfilled', 
                alpha=0.6, label='gwScReNI', color='#ff7f0e', edgecolor='#ff7f0e')
        
        # Add Vertical Dotted Threshold Lines
        if curr_thresh_mean > 0:
            ax.axvline(x=curr_thresh_mean, color='#003366', linestyle='--', linewidth=2.5, 
                       label=f'1% Threshold (Type-specific: {curr_thresh_mean:.4f})')
        if ref_thresh_mean > 0:
            ax.axvline(x=ref_thresh_mean, color='#990000', linestyle='--', linewidth=2.5, 
                       label=f'1% Threshold (gwScReNI: {ref_thresh_mean:.4f})')
        
        ax.set_yscale('log')
        ax.set_xlabel('Edge Weight')
        ax.set_ylabel('Density (Log Scale)')
        ax.set_title('Distribution of Active Edge Weights\n(with Mean 1% Top Edge Thresholds)')
        
        # Adjust legend to fit everything nicely without obscuring data
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        plt.savefig("output/comparison/weight_distribution_type.png", dpi=300, bbox_inches='tight')
        plt.close()
# ==============================================================================

if has_chip_atlas and 'wScReNI' in results and 'cell_type_hvgs' in rna.uns:
    print("\n" + "=" * 80)
    print("Generating Advanced Subspace Diagnostic Plots ...")
    print("=" * 80)

    import seaborn as sns
    output_dir_adv = Path("output/comparison")
    
    # # 1. Ground-Truth Box Density Plot (Safely constrained to valid intersection)
    gt_counts = {}
    cell_types_list = [ct for ct in rna.uns['cell_type_hvgs'].keys() if ct != 'union']
    for ct in cell_types_list:
        hvg_set = set(rna.uns['cell_type_hvgs'][ct])
        valid_edges = sum(
            1 for pair in tf_pairs 
            if pair.split('_')[0] in hvg_set and pair.split('_')[1] in hvg_set
        )
        gt_counts[ct] = valid_edges
        
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=list(gt_counts.keys()), y=list(gt_counts.values()), palette="viridis")
    plt.ylabel("Filtered Ground-Truth Edges")
    plt.title("Available Ground-Truth Edges within Intersection Subspace")
    for i, v in enumerate(gt_counts.values()):
        ax.text(i, v + max(gt_counts.values())*0.02, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir_adv / "ground_truth_density_type.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Subspace-Normalized Precision-Recall Curve
    from screni.data.precision_recall import calculate_precision_recall
    relative_thresholds = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9, 1]
    curve_precision = {pct: [] for pct in relative_thresholds}
    curve_recall = {pct: [] for pct in relative_thresholds}
    
    cell_type_curves = {ct: {'precision': {pct: [] for pct in relative_thresholds}, 
                             'recall': {pct: [] for pct in relative_thresholds}} 
                        for ct in cell_types_list}
    
    w_nets = results['wScReNI']
    global_genes = rna.var_names.tolist()
    
    for cell_i, cell_name in enumerate(rna.obs_names):
        ct = rna.obs['cell_type'].iloc[cell_i]
        ct_hvgs = rna.uns['cell_type_hvgs'][ct]
        max_possible_edges = len(ct_hvgs) * len(ct_hvgs)
        
        for pct in relative_thresholds:
            dynamic_top_n = max(1, int(max_possible_edges * pct))
            p, r = calculate_precision_recall(
                scnetwork_weights=w_nets[cell_name],
                tf_target_pair=tf_pairs,
                top_number=dynamic_top_n,
                gene_names=global_genes
            )
            if not np.isnan(p): curve_precision[pct].append(p)
            if not np.isnan(r): curve_recall[pct].append(r)
            
            if ct in cell_type_curves:
                if not np.isnan(p): cell_type_curves[ct]['precision'][pct].append(p)
                if not np.isnan(r): cell_type_curves[ct]['recall'][pct].append(r)
            
    mean_p = [np.mean(curve_precision[pct]) for pct in relative_thresholds]
    mean_r = [np.mean(curve_recall[pct]) for pct in relative_thresholds]
    
    curve_metrics = {}
    for ct in cell_types_list:
        mean_p_list = [np.mean(cell_type_curves[ct]['precision'][pct]) for pct in relative_thresholds]
        mean_r_list = [np.mean(cell_type_curves[ct]['recall'][pct]) for pct in relative_thresholds]
        curve_metrics[ct] = {'precision': mean_p_list, 'recall': mean_r_list}
    
    # Plot Bar Chart locked at a specific threshold (e.g. 1%)
    chosen_pct = 0.01
    bar_metrics = {}
    for ct in cell_types_list:
        ct_mean_p = np.mean(cell_type_curves[ct]['precision'][chosen_pct])
        ct_mean_r = np.mean(cell_type_curves[ct]['recall'][chosen_pct])
        bar_metrics[ct] = {'precision': ct_mean_p, 'recall': ct_mean_r}
        
    plot_pr_bar_per_celltype(bar_metrics, output_dir=str(output_dir_adv), filename=f"pr_bar_by_celltype_{int(chosen_pct*100)}pct.png")

    # =========================================================================
    # 3. Subspace-Normalized Precision-Recall Curve for Reference wScReNI
    # =========================================================================
    if 'Reference_wScReNI' in results:
        ref_nets = results['Reference_wScReNI']
        ref_global_genes = ref_nets.gene_names if ref_nets.gene_names else global_genes
        
        ref_curve_precision = {pct: [] for pct in relative_thresholds}
        ref_curve_recall = {pct: [] for pct in relative_thresholds}
        
        ref_cell_type_curves = {ct: {'precision': {pct: [] for pct in relative_thresholds}, 
                                     'recall': {pct: [] for pct in relative_thresholds}} 
                                for ct in cell_types_list}
        
        for cell_i, cell_name in enumerate(rna.obs_names):
            ct = rna.obs['cell_type'].iloc[cell_i]
            
            # Reference is global, cap at total global genes in its network
            max_possible_edges = len(ref_global_genes) * len(ref_global_genes)
            
            for pct in relative_thresholds:
                dynamic_top_n = max(1, int(max_possible_edges * pct))
                p, r = calculate_precision_recall(
                    scnetwork_weights=ref_nets[cell_name],
                    tf_target_pair=tf_pairs,
                    top_number=dynamic_top_n,
                    gene_names=ref_global_genes
                )
                if not np.isnan(p): ref_curve_precision[pct].append(p)
                if not np.isnan(r): ref_curve_recall[pct].append(r)
                
                if ct in ref_cell_type_curves:
                    if not np.isnan(p): ref_cell_type_curves[ct]['precision'][pct].append(p)
                    if not np.isnan(r): ref_cell_type_curves[ct]['recall'][pct].append(r)
            
        ref_mean_p = [np.mean(ref_curve_precision[pct]) for pct in relative_thresholds]
        ref_mean_r = [np.mean(ref_curve_recall[pct]) for pct in relative_thresholds]
        
        ref_curve_metrics = {}
        for ct in cell_types_list:
            ref_mean_p_list = [np.mean(ref_cell_type_curves[ct]['precision'][pct]) for pct in relative_thresholds]
            ref_mean_r_list = [np.mean(ref_cell_type_curves[ct]['recall'][pct]) for pct in relative_thresholds]
            ref_curve_metrics[ct] = {'precision': ref_mean_p_list, 'recall': ref_mean_r_list}
        
        # Plot Bar Chart locked at a specific threshold for Reference
        ref_bar_metrics = {}
        for ct in cell_types_list:
            ct_mean_p = np.mean(ref_cell_type_curves[ct]['precision'][chosen_pct])
            ct_mean_r = np.mean(ref_cell_type_curves[ct]['recall'][chosen_pct])
            ref_bar_metrics[ct] = {'precision': ct_mean_p, 'recall': ct_mean_r}
            
        plot_pr_bar_per_celltype(ref_bar_metrics, output_dir=str(output_dir_adv), filename=f"pr_bar_by_celltype_ref_{int(chosen_pct*100)}pct.png")

        # =========================================================================
        # 4. COMBINED OVERALL MEAN PRECISION-RECALL CURVE
        # =========================================================================
        plt.figure(figsize=(8, 6))
        
        # Plot Cell-Specific
        plt.plot(mean_r, mean_p, marker='o', linewidth=2, color='#2ca02c', label='type-specific wScReNI')
        # Plot Global
        plt.plot(ref_mean_r, ref_mean_p, marker='s', linestyle='--', linewidth=2, color='#ff7f0e', label='gwScReNI')
        
        # Annotate thresholds for both (offset slightly so they don't overlap)
        for i, pct in enumerate(relative_thresholds):
            plt.annotate(f"{pct*100:.1f}%", (mean_r[i], mean_p[i]), textcoords="offset points", xytext=(8,5), ha='left', fontsize=8, color='#2ca02c')
            plt.annotate(f"{pct*100:.1f}%", (ref_mean_r[i], ref_mean_p[i]), textcoords="offset points", xytext=(8,-12), ha='left', fontsize=8, color='#ff7f0e')
            
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Combined Precision-Recall Curve\n(Type-Specific vs gwScReNI)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir_adv / "combined_normalized_pr_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # =========================================================================
        # 5. COMBINED CELL-TYPE PRECISION-RECALL CURVES
        # =========================================================================
        plot_combined_pr_curves_per_celltype(
            cell_specific_metrics=curve_metrics, 
            global_metrics=ref_curve_metrics, 
            output_dir=str(output_dir_adv)
        )
