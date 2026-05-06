# This script generates all reproducible graphs from the ScReNI tutorial.
# Run it from inside the bsc-screni-main folder.
# See REPRODUCE_GRAPHS.md for full setup instructions.

import sys
import os

# Print IMMEDIATELY before any heavy imports so the user knows the script is alive.
print("ScReNI tutorial graph reproduction", flush=True)
print(f"Python {sys.version}", flush=True)
print(f"Working directory: {os.getcwd()}", flush=True)
print(flush=True)

# --- verify we're in the right place ---
if not os.path.exists("data/processed/retinal_rna_sub.h5ad"):
    print("ERROR: Run this script from inside your bsc-screni-main folder.", flush=True)
    print("       Expected: bsc-screni-main/data/processed/retinal_rna_sub.h5ad", flush=True)
    sys.exit(1)

# Heavy imports - each can take 5-30 s on Windows (numba JIT, umap-learn, etc.)
print("Loading numpy ...", flush=True)
import numpy as np
print("Loading pandas ...", flush=True)
import pandas as pd
print("Loading matplotlib ...", flush=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
print("Loading seaborn ...", flush=True)
import seaborn as sns
print("Loading anndata ...", flush=True)
import anndata as ad
print("Loading scanpy (slowest - may take 30-120 s on Windows first run) ...", flush=True)
import scanpy as sc
sc.settings.verbosity = 0
print("All libraries loaded.", flush=True)
print(flush=True)

os.makedirs("output/graphs", exist_ok=True)
print("Output folder: output/graphs/", flush=True)
print(flush=True)

# ---------------------------------------------------------------------------
# Auto-detect where the ScReNI-master data folder is.
#
# Two valid layouts:
#   A) Siblings  (what REPRODUCE_GRAPHS.md recommends):
#      parent/
#        ScReNI-master/data/
#        bsc-screni-main/          ← cwd
#
#   B) Nested  (what the zip actually creates):
#      ScReNI-master/
#        data/                     ← one level up from cwd
#        bsc-screni-main/          ← cwd
# ---------------------------------------------------------------------------
def _find_screni_data() -> str:
    """Return the path to ScReNI-master/data, whichever layout is in use."""
    candidates = [
        "ScReNI-master/data",   # layout A: sibling
        "../data",              # layout B: nested (zip default)
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return "ScReNI-master/data"   # fallback — will produce a clear SKIP message

SCRENI_DATA = _find_screni_data()
print(f"ScReNI-master data folder detected at: {SCRENI_DATA}", flush=True)
print(flush=True)

# ============================================================
# GRAPH 1 — Precision & Recall boxplots  (Step 3)
# Source: ScReNI-master/data/mmRetina_RPCMG_Cell100.500_scNetwork_precision_recall.csv
# R equivalent: Plot_scNetwork_precision_recall()
# ============================================================
print("Graph 1: Precision & Recall boxplots ...", flush=True)

PR_CSV = os.path.join(SCRENI_DATA, "mmRetina_RPCMG_Cell100.500_scNetwork_precision_recall.csv")
if not os.path.exists(PR_CSV):
    print(f"  SKIP — file not found: {PR_CSV}", flush=True)
    print("  Put ScReNI-master/ next to bsc-screni-main/ (see REPRODUCE_GRAPHS.md)", flush=True)
else:
    df_pr = pd.read_csv(PR_CSV, index_col=0)

    # Exact colour palette from tutorial R code (ggplot default cycle for 5 groups)
    palette = {
        "CSN":     "#F8766D",
        "CeSpGRN": "#A3A500",
        "LIONESS": "#00BF7D",
        "kScReNI": "#00B0F6",
        "wScReNI": "#E76BF3",
    }
    order = ["CSN", "CeSpGRN", "LIONESS", "kScReNI", "wScReNI"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric in zip(axes, ["precision", "recall"]):
        sns.boxplot(
            data=df_pr, x="scNetwork_type", y=metric,
            hue="scNetwork_type",
            order=order, palette=palette, ax=ax,
            linewidth=1.0, fliersize=2, legend=False,
        )
        ax.set_title(metric.capitalize(), fontsize=14, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(metric, fontsize=12)
        ax.tick_params(axis="x", labelsize=11)
        sns.despine(ax=ax)

    fig.suptitle(
        "Precision & Recall at CSN-matched threshold (400 cells)",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    plt.savefig("output/graphs/graph1_precision_recall_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: output/graphs/graph1_precision_recall_boxplots.png", flush=True)


# ============================================================
# GRAPH 2 — K-means heatmap of smoothed expression  (Step 4.3)
# Source: ScReNI-master/data/clustering_revised.txt
# R equivalent: IReNA::plot_kmeans_pheatmap()
# ============================================================
print("Graph 2: K-means heatmap ...", flush=True)

CLUSTER_TXT = os.path.join(SCRENI_DATA, "clustering_revised.txt")
if not os.path.exists(CLUSTER_TXT):
    print(f"  SKIP — file not found: {CLUSTER_TXT}", flush=True)
else:
    clustering = pd.read_csv(CLUSTER_TXT, sep=" ", index_col=0)
    smexp_cols = [c for c in clustering.columns if c.startswith("SmExp")]
    kmeans_groups = clustering["KmeansGroup"]
    expr = clustering[smexp_cols]

    # Sort genes by cluster (ascending) — matches R's order(KmeansGroup, decreasing=F)
    sorted_idx = kmeans_groups.sort_values(ascending=True).index
    expr_sorted   = expr.loc[sorted_idx]
    groups_sorted = kmeans_groups.loc[sorted_idx]

    # Clip to [-2, 2] for display — matches R heatmap default colour scale
    expr_clipped = expr_sorted.values.clip(-2, 2)

    # Exact RGB values from tutorial Block 39
    module_colors = {
        1: (174/255, 199/255, 232/255),
        2: (103/255, 193/255, 227/255),
        3: ( 91/255, 166/255, 218/255),
        4: (  0/255, 114/255, 189/255),
        5: (253/255, 209/255, 176/255),
        6: (239/255, 153/255,  81/255),
    }

    fig = plt.figure(figsize=(16, 9))
    gs  = fig.add_gridspec(1, 3, width_ratios=[0.03, 0.97, 0.05], wspace=0.02)
    ax_bar  = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])
    ax_cbar = fig.add_subplot(gs[2])

    # Side colour bar — cluster membership
    bar_colors = np.array([module_colors[g] for g in groups_sorted.values])
    ax_bar.imshow(bar_colors.reshape(-1, 1, 3), aspect="auto", interpolation="none")
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    ax_bar.set_ylabel("Genes (n=1995)", fontsize=11)

    # Main heatmap
    im = ax_heat.imshow(
        expr_clipped, aspect="auto", cmap="RdBu_r",
        vmin=-2, vmax=2, interpolation="none",
    )
    ax_heat.set_xticks(np.linspace(0, len(smexp_cols)-1, 6, dtype=int))
    ax_heat.set_xticklabels(
        [f"Bin {int(v)+1}" for v in np.linspace(0, len(smexp_cols)-1, 6)],
        fontsize=9,
    )
    ax_heat.set_yticks([])
    ax_heat.set_xlabel("Pseudotime bins (50 total)", fontsize=11)
    ax_heat.set_title("K-means clustering of smoothed expression across pseudotime", fontsize=12)

    plt.colorbar(im, cax=ax_cbar, label="Scaled expression (clipped ±2)")

    # Legend
    legend_handles = [
        Patch(facecolor=module_colors[k], label=f"Module {k}")
        for k in sorted(module_colors) if k in groups_sorted.unique()
    ]
    ax_heat.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.22, 1), loc="upper left",
        fontsize=9, title="Module", title_fontsize=9,
    )

    plt.savefig("output/graphs/graph2_kmeans_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: output/graphs/graph2_kmeans_heatmap.png", flush=True)


# ============================================================
# GRAPH 3 — Integration UMAP: RNA + ATAC coloured by cell type  (Step 1.2)
# Source: data/processed/retinal_rna_sub.h5ad  +  data/processed/retinal_atac_sub.h5ad
# R equivalent: DimPlot(coembed, group.by = c('datatypes','samples','celltypes'))
# NOTE: The R version uses a pre-saved integrated Seurat object.
#       Here we re-run the integration (Harmony, ~1 min) directly from the
#       processed h5ad files that are already in your repo.
# ============================================================
print("Graph 3: Integration UMAP ...", flush=True)

try:
    from harmonypy import run_harmony
    _harmony_available = True
except ImportError:
    _harmony_available = False
    print("  NOTE: harmonypy not installed — skipping integration UMAP", flush=True)
    print("  Install with:  pip install harmonypy", flush=True)

if _harmony_available:
    rna  = ad.read_h5ad("data/processed/retinal_rna_sub.h5ad")
    atac = ad.read_h5ad("data/processed/retinal_atac_sub.h5ad")

    # Load the pre-computed gene activity matrix (peaks → gene scores)
    # Shape: (400 cells, 217 genes) — already in the repo
    gene_act_mat = np.load("data/processed/retinal_peak_overlap_matrix.npz")["peak_matrix"]
    peak_info    = pd.read_csv("data/processed/retinal_peak_info.csv", index_col=0)
    gene_act_genes = peak_info["associated_genes"].values  # 217 gene names

    # Find shared genes — deduplicate (multiple peaks can map to same gene)
    import scipy.sparse as sp_mod
    rna_gene_list = list(rna.var_names)
    rna_gene_set  = set(rna_gene_list)
    seen = set()
    shared_genes, shared_idx_ga, shared_idx_rna = [], [], []
    for i, g in enumerate(gene_act_genes):
        if g in rna_gene_set and g not in seen:
            seen.add(g)
            shared_genes.append(g)
            shared_idx_ga.append(i)
            shared_idx_rna.append(rna_gene_list.index(g))

    # Build gene-activity AnnData restricted to shared genes
    X_rna = rna.X.toarray() if sp_mod.issparse(rna.X) else np.array(rna.X)

    rna_sub  = ad.AnnData(X=X_rna[:, shared_idx_rna].copy(), obs=rna.obs.copy())
    rna_sub.var_names  = shared_genes
    rna_sub.obs_names  = [f"RNA_{n}" for n in rna_sub.obs_names]   # ensure unique
    rna_sub.obs["modality"] = "RNA"
    rna_sub.obs["dataset"]  = rna.obs["cell_type"].astype(str) + "_RNA"

    atac_sub = ad.AnnData(X=gene_act_mat[:, shared_idx_ga].copy(), obs=atac.obs.copy())
    atac_sub.var_names  = shared_genes
    atac_sub.obs_names  = [f"ATAC_{n}" for n in atac_sub.obs_names]  # ensure unique
    atac_sub.obs["modality"] = "ATAC"
    atac_sub.obs["dataset"]  = atac.obs["cell_type"].astype(str) + "_ATAC"

    # Normalise both modalities identically; replace NaN from zero-variance genes
    for a in [rna_sub, atac_sub]:
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        sc.pp.scale(a, zero_center=True)
        X_fixed = a.X if not sp_mod.issparse(a.X) else a.X.toarray()
        a.X = np.nan_to_num(X_fixed, nan=0.0, posinf=0.0, neginf=0.0)
        sc.tl.pca(a, n_comps=20)

    # Concatenate — carry over per-modality PCA embeddings explicitly.
    # ad.concat drops obsm by default, so we stack and re-attach before Harmony.
    pca_rna, pca_atac = rna_sub.obsm["X_pca"].copy(), atac_sub.obsm["X_pca"].copy()
    combined = ad.concat([rna_sub, atac_sub], merge="same")
    combined.obsm["X_pca"] = np.vstack([pca_rna, pca_atac])  # (800, 20)

    # Run Harmony and manually extract Z_corr as a plain numpy array.
    #
    # Why not sc.external.pp.harmony_integrate or run_harmony().Z_corr.T directly?
    # harmonypy 0.0.9 uses a PyTorch backend. Z_corr is a torch tensor, and
    # anndata 0.11+ shape-validation fails when the tensor is assigned without
    # first converting to a contiguous numpy array — even via scanpy's own wrapper.
    #
    # We also check orientation explicitly: Z_corr is (n_pcs, n_cells) by
    # convention, so we need .T for obsm. But if a future version flips it
    # we handle that too.
    from harmonypy import run_harmony as _run_harmony
    _ho = _run_harmony(
        combined.obsm["X_pca"], combined.obs, "modality", random_state=42
    )
    _Z = _ho.Z_corr
    # Convert torch tensor → numpy if needed
    if hasattr(_Z, "detach"):
        _Z = _Z.detach().cpu().numpy()
    else:
        _Z = np.asarray(_Z)
    # Ensure shape is (n_cells, n_pcs) for obsm
    if _Z.shape[0] != combined.n_obs:
        _Z = _Z.T
    assert _Z.shape[0] == combined.n_obs, (
        f"Harmony Z_corr shape {_Z.shape} doesn't match n_obs={combined.n_obs}"
    )
    combined.obsm["X_harmony"] = _Z
    sc.pp.neighbors(combined, use_rep="X_harmony", n_pcs=20)
    sc.tl.umap(combined, random_state=42)

    # Three-panel plot matching R's DimPlot(coembed, group.by=c(...))
    cell_type_colors = {
        "MG": "#F8766D", "RPC1": "#00BA38", "RPC2": "#619CFF", "RPC3": "#F564E3",
        "MG_RNA": "#F8766D", "RPC1_RNA": "#00BA38", "RPC2_RNA": "#619CFF", "RPC3_RNA": "#F564E3",
        "MG_ATAC": "#D73027", "RPC1_ATAC": "#006837", "RPC2_ATAC": "#4575B4", "RPC3_ATAC": "#762A83",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col, title in zip(
        axes,
        ["modality",   "dataset",    "cell_type"],
        ["Data type",  "Sample",     "Cell type"],
    ):
        sc.pl.umap(combined, color=col, ax=ax, show=False,
                   title=title, frameon=True, legend_loc="right margin", size=12)

    fig.suptitle("Step 1.2 — Integrated scRNA-seq + scATAC-seq UMAP", fontsize=13)
    plt.tight_layout()
    plt.savefig("output/graphs/graph3_integration_umap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: output/graphs/graph3_integration_umap.png", flush=True)


# ============================================================
# GRAPH 4 — Degree correlation heatmap + UMAP  (Step 4.2)
# Source: data/processed/retinal_rna_sub.h5ad (expression matrix for CSN)
# R equivalent: Heatmap(cor(log(outdegree+1))) + DimPlot(out.degree, group.by="lab")
# Uses: infer_csn_networks + calculate_scnetwork_degree from screni.data
# NOTE: CSN on 400 cells × 500 genes takes ~3-4 minutes.
# ============================================================
print("Graph 4: Degree heatmap + UMAP (runs CSN inference, ~3 min) ...", flush=True)

try:
    sys.path.insert(0, "src")
    # BUG FIX: function is infer_csn_networks, not infer_csn_scnetworks
    from screni.data.inference  import infer_csn_networks
    from screni.data.clustering import calculate_scnetwork_degree

    rna = ad.read_h5ad("data/processed/retinal_rna_sub.h5ad")
    # Expression matrix: cells × genes (Python/pandas convention for screni).
    # NOTE: _resolve_input treats DataFrame index=cells, columns=genes.
    #       DO NOT transpose — pass X_raw directly.
    #       The original script used .T (rows=genes, cols=cells) which is the
    #       R convention and caused CSN to swap cells and genes.
    import scipy.sparse as _sp
    X_raw = rna.X.toarray() if _sp.issparse(rna.X) else np.array(rna.X)
    expr = pd.DataFrame(X_raw, index=rna.obs_names, columns=rna.var_names)

    cell_types = rna.obs["cell_type"].values
    cell_type_colors = {"MG": "red", "RPC1": "green", "RPC2": "blue", "RPC3": "purple"}

    # Infer CSN
    print("  Running CSN inference (this takes ~3 minutes) ...", flush=True)
    csn_nets = infer_csn_networks(expr, alpha=0.01, boxsize=0.1, weighted=False)

    # Degree analysis — top 500 edges per cell (matches R's top=rep(500, ncell))
    n_cells = rna.n_obs
    degree_results = calculate_scnetwork_degree(
        sc_networks={"CSN": csn_nets},
        top=[500] * n_cells,
        cell_type_annotation=cell_types,
        ntype=4,
    )

    result   = degree_results["CSN"]
    outdeg   = result.outdegree      # (n_genes, n_cells)
    out_adata = result.out_degree_umap

    # Attach cell type labels to adata (matches R's out.degree[['lab']] <- sub.celltype)
    out_adata.obs["cell_type"] = list(cell_types)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: correlation heatmap (matches R's Heatmap(cor(log(outdegree+1))))
    log_out  = np.log(outdeg + 1)
    corr_mat = np.corrcoef(log_out.T)   # (n_cells, n_cells) cell × cell

    # Sort cells by cell type for a clean block diagonal
    sort_order = np.argsort(cell_types)
    corr_sorted = corr_mat[np.ix_(sort_order, sort_order)]
    sorted_types = cell_types[sort_order]

    im = axes[0].imshow(corr_sorted, cmap="RdBu_r", vmin=-1, vmax=1,
                        aspect="auto", interpolation="none")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_title("CSN — cor(log(out-degree + 1))", fontsize=12)
    plt.colorbar(im, ax=axes[0], shrink=0.6, label="Pearson r")

    # Cell type colour bar along top (matches R's HeatmapAnnotation)
    bar_colors = np.array([mcolors.to_rgb(cell_type_colors[ct]) for ct in sorted_types])
    ax_top = axes[0].inset_axes([0, 1.01, 1, 0.025])
    ax_top.imshow(bar_colors.reshape(1, -1, 3), aspect="auto", interpolation="none")
    ax_top.set_xticks([]); ax_top.set_yticks([])

    # Add legend for colour bar
    handles = [Patch(color=c, label=t) for t, c in cell_type_colors.items()]
    axes[0].legend(handles=handles, loc="lower right", fontsize=8,
                   title="Cell type", framealpha=0.9)

    # Right: UMAP coloured by cell type (matches R's DimPlot(out.degree, group.by="lab"))
    sc.pl.umap(out_adata, color="cell_type", ax=axes[1], show=False,
               palette=cell_type_colors, title="CSN — UMAP (out-degree)",
               frameon=True, legend_loc="right margin")

    fig.suptitle("Step 4.2 — Cell clustering based on degree matrix (CSN)", fontsize=13)
    plt.tight_layout()
    plt.savefig("output/graphs/graph4_degree_heatmap_umap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: output/graphs/graph4_degree_heatmap_umap.png", flush=True)

except Exception as e:
    print(f"  SKIP — error during degree graph: {e}", flush=True)


# ============================================================
# GRAPH 5 — Pseudotime trajectory  (Step 4.3)
# Source: data/processed/retinal_rna_sub.h5ad
# R equivalent: plot_cells(cds, color_by='pseudotime') + FeaturePlot(pseudotime)
# NOTE: uses scanpy DPT instead of Monocle3
# ============================================================
print("Graph 5: Pseudotime plots ...", flush=True)

rna = ad.read_h5ad("data/processed/retinal_rna_sub.h5ad")
adata_pt = rna.copy()

sc.pp.normalize_total(adata_pt, target_sum=1e4)
sc.pp.log1p(adata_pt)
sc.pp.highly_variable_genes(adata_pt, n_top_genes=500)
sc.tl.pca(adata_pt, n_comps=20, mask_var="highly_variable")
sc.pp.neighbors(adata_pt, n_pcs=20, n_neighbors=15)
sc.tl.umap(adata_pt, random_state=42)
sc.tl.leiden(adata_pt, resolution=0.5, key_added="leiden",
             flavor="igraph", n_iterations=2, directed=False)

# Root cell: first RPC1 cell — the earliest progenitor (matching R's choice)
rpc1_mask = adata_pt.obs["cell_type"].isin(["RPCs_S1", "RPC1"])
rpc1_cells = adata_pt.obs_names[rpc1_mask]
if len(rpc1_cells) == 0:
    # Fallback: use whichever cell type appears first alphabetically
    rpc1_cells = adata_pt.obs_names[:1]

# BUG FIX: np.where returns a tuple of arrays; need [0][0] not [0]
iroot_arr = np.where(adata_pt.obs_names == rpc1_cells[0])[0]
adata_pt.uns["iroot"] = int(iroot_arr[0])
sc.tl.dpt(adata_pt)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sc.pl.umap(adata_pt, color="leiden",          ax=axes[0], show=False,
           title="p1: Clusters",    frameon=True)
sc.pl.umap(adata_pt, color="dpt_pseudotime",  ax=axes[1], show=False,
           title="p2: Pseudotime (DPT)", cmap="viridis", frameon=True)
sc.pl.umap(adata_pt, color="dpt_pseudotime",  ax=axes[2], show=False,
           title="p3: Pseudotime (feature)", cmap="plasma", frameon=True,
           colorbar_loc="right")

fig.suptitle("Step 4.3 — Pseudotime trajectory (scanpy DPT ≈ Monocle3)", fontsize=13)
plt.tight_layout()
plt.savefig("output/graphs/graph5_pseudotime.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: output/graphs/graph5_pseudotime.png", flush=True)


# ============================================================
# GRAPH 6 — GO enrichment bar chart  (Step 4.3)
# Source: ScReNI-master/data/clustering_revised.txt
# R equivalent: ggplot geom_bar on enrichment_GO (clusterProfiler)
# Uses: enrich_module from screni.data (calls g:Profiler API)
# NOTE: requires internet access. Takes ~30 seconds.
# ============================================================
print("Graph 6: GO enrichment bar chart ...", flush=True)

if not os.path.exists(CLUSTER_TXT):
    print(f"  SKIP — file not found: {CLUSTER_TXT}", flush=True)
else:
    try:
        sys.path.insert(0, "src")
        from screni.data.clustering import enrich_module

        # Reuse clustering loaded earlier
        clustering = pd.read_csv(CLUSTER_TXT, sep=" ", index_col=0)
        # Add a 'Symbol' column from the index (gene symbols are the row names)
        clustering["Symbol"] = clustering.index

        enrichment_GO = enrich_module(
            clustering, organism="mmusculus",
            enrich_db="GO", fun_num=5,
        )

        if enrichment_GO.empty:
            print("  SKIP — no enrichment results returned (check internet connection)", flush=True)
        else:
            # Match R's Mm_func construction:
            # Mm_func = enrichment_GO[order(module, decreasing=T), c(Description, module, p_value)]
            # Mm_func$score = -log10(qvalue)  (we use p_value_log10)
            Mm_func = enrichment_GO[["module", "term_name", "p_value_log10"]].copy()
            Mm_func = Mm_func.sort_values("module", ascending=False).reset_index(drop=True)
            Mm_func["rank"] = range(len(Mm_func))
            Mm_func["module"] = Mm_func["module"].astype("category")

            # Exact RGB from tutorial Block 39
            col = [
                (174/255, 199/255, 232/255),
                (103/255, 193/255, 227/255),
                ( 91/255, 166/255, 218/255),
                (  0/255, 114/255, 189/255),
                (253/255, 209/255, 176/255),
                (239/255, 153/255,  81/255),
            ]

            modules = sorted(Mm_func["module"].cat.categories, key=int)
            color_map = {m: col[i % len(col)] for i, m in enumerate(modules)}

            fig, ax = plt.subplots(figsize=(10, max(5, len(Mm_func) * 0.35)))
            for _, row in Mm_func.iterrows():
                color = color_map.get(row["module"], "grey")
                ax.barh(row["rank"], row["p_value_log10"],
                        color=color, edgecolor="none")

            ax.set_yticks(Mm_func["rank"])
            ax.set_yticklabels(Mm_func["term_name"], fontsize=8)
            ax.set_xlabel("-log10(p-value)", fontsize=11, fontweight="bold")
            ax.set_title("GO enrichment per k-means module (g:Profiler ≈ clusterProfiler)",
                         fontsize=11)
            ax.invert_yaxis()

            # Legend
            handles = [Patch(facecolor=color_map[m], label=f"Module {m}") for m in modules]
            ax.legend(handles=handles, loc="lower right", fontsize=9,
                      title="Module", framealpha=0.9)
            sns.despine(ax=ax)
            plt.tight_layout()
            plt.savefig("output/graphs/graph6_go_enrichment_bars.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
            print("  Saved: output/graphs/graph6_go_enrichment_bars.png", flush=True)

    except Exception as e:
        print(f"  SKIP — error during enrichment graph: {e}", flush=True)


print(flush=True)
print("Done. Check output/graphs/ for all saved PNGs.", flush=True)
