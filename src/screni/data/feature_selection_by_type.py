"""Phase 2: Cell subsampling, feature selection, and KNN computation.

Matches the original R ``select_features()`` and
``Select_partial_cells_for_scNewtorks()`` functions.

Key parameters from the paper:
    - 100 cells per cell type (retinal, 4 types = 400 total)
    - 50 cells per cell type (PBMC, 8 types = 400 total)
    - 500 HVGs for network inference (main benchmark)
    - 2000 HVGs for clustering benchmark
    - 10,000 HV peaks for ATAC
    - Feature selection uses Seurat v3 VST
    - Returns RAW COUNTS (not normalized) for selected features
    - KNN (k=20) computed on integrated embedding for wScReNI
"""

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def subsample_cells(
    adata: ad.AnnData,
    n_per_type: int,
    cell_type_col: str = "cell_type",
    seed: int = 42,
) -> ad.AnnData:
    """Randomly subsample a fixed number of cells per cell type."""
    rng = np.random.RandomState(seed)
    indices = []

    for ct in sorted(adata.obs[cell_type_col].unique()):
        ct_idx = np.where(adata.obs[cell_type_col] == ct)[0]
        n_available = len(ct_idx)
        if n_available < n_per_type:
            logger.warning(
                f"  Cell type '{ct}' has only {n_available} cells, "
                f"requested {n_per_type}. Using all available."
            )
            indices.extend(ct_idx.tolist())
        else:
            sampled = rng.choice(ct_idx, size=n_per_type, replace=False)
            indices.extend(sorted(sampled.tolist()))

    result = adata[indices].copy()
    logger.info(
        f"  Subsampled {adata.n_obs} -> {result.n_obs} cells "
        f"({n_per_type} per type x {adata.obs[cell_type_col].nunique()} types)"
    )
    logger.info(f"  Cell type counts: {result.obs[cell_type_col].value_counts().to_dict()}")
    return result


def select_variable_features(
    adata: ad.AnnData,
    n_features: int,
    flavor: str = "seurat_v3",
    span: float = 0.3,
) -> ad.AnnData:
    """Select highly variable features using VST."""
    if sp.issparse(adata.X):
        feature_sums = np.array(adata.X.sum(axis=0)).flatten()
    else:
        feature_sums = adata.X.sum(axis=0)
    nonzero_mask = feature_sums > 0
    n_removed = (~nonzero_mask).sum()
    if n_removed > 0:
        logger.info(f"  Removed {n_removed} zero-sum features")
        adata = adata[:, nonzero_mask].copy()

    n_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=3)
    n_filtered = n_before - adata.n_vars
    if n_filtered > 0:
        logger.info(f"  Filtered {n_filtered} features expressed in <3 cells")

    if flavor == "seurat_v3":
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_features,
            flavor="seurat_v3",
            span=span,
        )
    else:
        work = adata.copy()
        sc.pp.normalize_total(work, target_sum=1e4)
        sc.pp.log1p(work)
        sc.pp.highly_variable_genes(work, n_top_genes=n_features, flavor=flavor)
        adata.var["highly_variable"] = work.var["highly_variable"]

    hvg_mask = adata.var["highly_variable"]
    result = adata[:, hvg_mask].copy()

    if sp.issparse(result.X):
        max_val = result.X.max()
    else:
        max_val = result.X.max()

    logger.info(
        f"  Selected {result.n_vars} variable features "
        f"(max value: {max_val:.1f}, should be >> 1 if raw counts)"
    )

    return result


def filter_chr_peaks(adata: ad.AnnData) -> ad.AnnData:
    """Filter ATAC peaks to chr-prefixed only, removing scaffolds."""
    chr_mask = adata.var_names.str.startswith("chr")
    n_before = adata.n_vars
    result = adata[:, chr_mask].copy()
    n_removed = n_before - result.n_vars
    if n_removed > 0:
        logger.info(f"  Filtered peaks: {n_before} -> {result.n_vars} (removed {n_removed} non-chr)")
    return result


def subsample_pairs(
    rna: ad.AnnData,
    atac: ad.AnnData,
    pairs: pd.DataFrame,
    n_per_type: int,
    seed: int = 42,
) -> tuple[ad.AnnData, ad.AnnData]:
    """Subsample matched RNA-ATAC cell pairs for unpaired datasets."""
    rng = np.random.RandomState(seed)
    sampled_indices = []
    for ct in sorted(pairs["cell_type"].unique()):
        ct_idx = pairs.index[pairs["cell_type"] == ct].tolist()
        n = min(len(ct_idx), n_per_type)
        sampled = rng.choice(ct_idx, size=n, replace=False)
        sampled_indices.extend(sorted(sampled.tolist()))

    pairs_sub = pairs.loc[sampled_indices]

    rna_sub = rna[pairs_sub["rna_cell"].values].copy()
    atac_sub = atac[pairs_sub["atac_cell"].values].copy()

    rna_sub.obs["cell_type"] = pairs_sub["cell_type"].values
    atac_sub.obs["cell_type"] = pairs_sub["cell_type"].values
    rna_sub.obs["_original_rna_cell"] = pairs_sub["rna_cell"].values

    shared_names = [f"cell_{i}" for i in range(len(pairs_sub))]
    rna_sub.obs_names = shared_names
    atac_sub.obs_names = shared_names

    logger.info(
        f"  Subsampled {len(pairs)} pairs -> {len(pairs_sub)} "
        f"({n_per_type} per type x {pairs['cell_type'].nunique()} types)"
    )
    return rna_sub, atac_sub


def compute_knn(
    embedding: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """Compute k-nearest-neighbor indices from an embedding matrix."""
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(embedding)
    indices = nn.kneighbors(return_distance=False)
    logger.info(f"  KNN: {indices.shape} (k={k}) from {embedding.shape[1]}d embedding")
    return indices


def _select_by_name(adata: ad.AnnData, names: list[str], label: str) -> ad.AnnData:
    """Subset an AnnData to a pre-defined list of feature names."""
    available = [n for n in names if n in adata.var_names]
    if len(available) < len(names):
        logger.warning(
            f"  {label}: {len(names) - len(available)} features not found "
            f"in data ({len(available)}/{len(names)} available)"
        )
    result = adata[:, available].copy()
    return result


def select_variable_features_per_celltype(
    adata: ad.AnnData,
    genes_per_type: int = 500,
    cell_type_key: str = "cell_type",
    flavor: str = "seurat_v3",
    span: float = 0.3,
    min_cells_per_type: int = 30
) -> ad.AnnData:
    """Select highly variable features per cell type using VST with dynamic span expansion."""
    
    if cell_type_key not in adata.obs:
        raise ValueError(f"Key '{cell_type_key}' not found in adata.obs")

    # Step 1: Remove global zero-sum features
    if sp.issparse(adata.X):
        feature_sums = np.array(adata.X.sum(axis=0)).flatten()
    else:
        feature_sums = adata.X.sum(axis=0)
        
    nonzero_mask = feature_sums > 0
    n_removed = (~nonzero_mask).sum()
    if n_removed > 0:
        logger.info(f"  Removed {n_removed} global zero-sum features")
        adata = adata[:, nonzero_mask].copy()

    unique_cell_types = adata.obs[cell_type_key].dropna().unique()
    all_selected_features = {"union": set()}

    # Step 2: Iterate over cell types to compute HVGs locally (Dynamic Span)
    for ct in unique_cell_types:
        ct_mask = adata.obs[cell_type_key] == ct
        adata_ct = adata[ct_mask].copy()
        
        # Skip if subset is too small for robust LOESS regression
        if adata_ct.n_obs < min_cells_per_type or adata_ct.n_vars <= genes_per_type:
            logger.warning(f"  Skipping {ct}: Only {adata_ct.n_obs} cells (requires {min_cells_per_type}).")
            continue

        sc.pp.filter_genes(adata_ct, min_cells=3)
        
        success = False
        current_span = span

        # Iteratively widen the LOESS window to resolve statistical singularities
        while current_span <= 1.0 and not success:
            try:
                sc.pp.highly_variable_genes(
                    adata_ct,
                    n_top_genes=genes_per_type,
                    flavor="seurat_v3",
                    span=current_span,
                )
                success = True
                if current_span > span:
                    logger.info(f"  Successfully bypassed LOESS singularity for {ct} using span={current_span:.2f}")
                    
            except ValueError as e:
                if "singularities" in str(e).lower() or "loess" in str(e).lower():
                    current_span += 0.1
                else:
                    raise e

        # Ultimate fallback with stricter sparsity filtering
        if not success:
            logger.warning(f"  seurat_v3 failed for {ct} even at span=1.0. Applying stricter zero-filter.")
            stricter_min = max(3, int(adata_ct.n_obs * 0.10))
            sc.pp.filter_genes(adata_ct, min_cells=stricter_min)
            sc.pp.highly_variable_genes(
                adata_ct, 
                n_top_genes=genes_per_type, 
                flavor="seurat_v3", 
                span=0.5
            )

        # Extract and add to union set
        ct_hvgs = adata_ct.var_names[adata_ct.var["highly_variable"]]
        all_selected_features[ct] = ct_hvgs
        all_selected_features["union"].update(ct_hvgs)
        
        logger.info(f"  Selected {len(ct_hvgs)} features for {ct}")

    # Step 3 & 4: Subset original object with the union of all selected features
    final_features = list(all_selected_features["union"])
    
    if not final_features:
        raise RuntimeError("No highly variable features were selected across any cell types.")
        
    result = adata[:, final_features].copy()

    # Save the cell-type specific HVGs into the unstructured data (uns)
    result.uns['cell_type_hvgs'] = {k: list(v) for k, v in all_selected_features.items()}

    # Verify we have raw counts
    if sp.issparse(result.X):
        max_val = result.X.max()
    else:
        max_val = result.X.max()

    logger.info(
        f"  Total unique features selected across all cell types: {result.n_vars} "
        f"(max value: {max_val:.1f}, should be >> 1 if raw counts)"
    )

    return result


def prepare_subsample(
    rna: ad.AnnData,
    atac: ad.AnnData,
    n_per_type: int = 100,
    n_genes: int = 500,
    n_peaks: int = 10000,
    seed: int = 42,
    pairs: pd.DataFrame | None = None,
    hvg_list: list[str] | None = None,
    hvp_list: list[str] | None = None,
    embedding: np.ndarray | None = None,
    embedding_cell_names: list[str] | np.ndarray | None = None,
    knn_k: int = 20,
) -> dict[str, ad.AnnData | np.ndarray]:
    """Full Phase 2 pipeline: subsample cells, select features, compute KNN."""
    mode = "R-reference" if (hvg_list is not None or hvp_list is not None) else "Python"
    logger.info(f"=== Phase 2: Cell Subsampling & Feature Selection ({mode} mode) ===")

    if pairs is not None:
        logger.info("Subsampling matched RNA-ATAC pairs (unpaired mode)...")
        rna_sub, atac_sub = subsample_pairs(
            rna, atac, pairs, n_per_type=n_per_type, seed=seed,
        )
    else:
        logger.info("Subsampling RNA cells...")
        rna_sub = subsample_cells(rna, n_per_type=n_per_type, seed=seed)

        logger.info("Subsampling ATAC cells...")
        atac_sub = subsample_cells(atac, n_per_type=n_per_type, seed=seed)

    atac_sub = filter_chr_peaks(atac_sub)

    # Feature Selection Logic
    if hvg_list is not None:
        logger.info(f"Using R-reference HVGs ({len(hvg_list)} genes) as global baseline...")
        rna_sub = _select_by_name(rna_sub, hvg_list, "HVGs")
    else:
        # Utilize the PURE cell-specific algorithm
        logger.info("Selecting HVGs using strictly cell-specific algorithm...")
        
        rna_sub = select_variable_features_per_celltype(
            rna_sub, 
            genes_per_type=n_genes, 
            cell_type_key="cell_type"
        )
        
    if hvp_list is not None:
        logger.info(f"Using R-reference HVPs ({len(hvp_list)} peaks)...")
        atac_sub = _select_by_name(atac_sub, hvp_list, "HVPs")
    else:
        logger.info(f"Selecting {n_peaks} HV peaks from ATAC (Python VST)...")
        atac_sub = select_variable_features(atac_sub, n_features=n_peaks, span=0.5)

    result = {"rna": rna_sub, "atac": atac_sub}

    if embedding is not None and embedding_cell_names is not None:
        logger.info("Computing KNN from integrated embedding...")
        cell_lookup = {str(name): i for i, name in enumerate(embedding_cell_names)}

        if "_original_rna_cell" in rna_sub.obs.columns:
            lookup_names = rna_sub.obs["_original_rna_cell"].values
        else:
            lookup_names = rna_sub.obs_names

        sub_indices = [cell_lookup[str(n)] for n in lookup_names if str(n) in cell_lookup]
        sub_embedding = embedding[sub_indices]

        if len(sub_embedding) == rna_sub.n_obs:
            result["knn_indices"] = compute_knn(sub_embedding, k=knn_k)
        else:
            logger.warning(
                f"  KNN: matched {len(sub_embedding)}/{rna_sub.n_obs} "
                f"cells in embedding, skipping"
            )

    logger.info(
        f"Phase 2 complete:\n"
        f"  RNA:  {rna_sub.shape} (raw counts)\n"
        f"  ATAC: {atac_sub.shape} (raw counts)"
        + (f"\n  KNN:  {result['knn_indices'].shape}" if "knn_indices" in result else "")
    )

    return result


def _load_feature_list(path: Path) -> list[str] | None:
    if path.exists():
        return path.read_text().strip().split("\n")
    return None

def plot_hvg_overlap(rna_adata, genes_per_type=500, cell_type_col="cell_type", output_path="hvg_overlap.png"):
    """
    Plots the overlap between the union of cell-specific HVGs and global HVGs
    using a standard matplotlib bar chart.
    """
    print("1. Running cell-specific HVG selection...")
    rna_cell_specific = select_variable_features_per_celltype(
        rna_adata.copy(),
        genes_per_type=genes_per_type,
        cell_type_key=cell_type_col
    )
    
    cell_specific_hvgs = set(rna_cell_specific.var_names)
    n_union = len(cell_specific_hvgs)
    print(f"  -> Cell-specific selection resulted in a union of {n_union} unique HVGs.")

    print(f"\n2. Running global HVG selection (forcing n_features={n_union})...")
    rna_global = select_variable_features(
        rna_adata.copy(),
        n_features=n_union,
        flavor="seurat_v3"
    )
    
    global_hvgs = set(rna_global.var_names)
    print(f"  -> Global selection grabbed {len(global_hvgs)} HVGs.")

    # Calculate overlaps
    overlap = cell_specific_hvgs.intersection(global_hvgs)
    only_cell_specific = cell_specific_hvgs - global_hvgs
    only_global = global_hvgs - cell_specific_hvgs

    # Print results to terminal
    overlap_pct = (len(overlap) / n_union) * 100 if n_union > 0 else 0
    print("\n=== OVERLAP RESULTS ===")
    print(f"Shared (Intersection): {len(overlap)} genes ({overlap_pct:.1f}%)")
    print(f"Unique to Cell-Specific: {len(only_cell_specific)} genes")
    print(f"Unique to Global: {len(only_global)} genes")

    # Generate Bar Chart Plot
    plt.figure(figsize=(8, 5))
    
    labels = ['Shared\n(Intersection)', 'Unique to\nType-Specific', 'Unique to\nGlobal']
    sizes = [len(overlap), len(only_cell_specific), len(only_global)]
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']
    
    bars = plt.bar(labels, sizes, color=colors, alpha=0.85, edgecolor='black')
    
    plt.title(f"Union HVG Overlap Breakdown\n(Evaluated at exactly {n_union} genes per method)", 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel("Number of Genes", fontsize=12)
    
    # Add percentage text directly above the bars
    for bar, size in zip(bars, sizes):
        pct = (size / n_union) * 100 if n_union > 0 else 0
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + (n_union * 0.02),
                 f"{size}\n({pct:.1f}%)", ha='center', va='bottom', 
                 fontsize=11, fontweight='bold')
        
    plt.ylim(0, max(sizes) * 1.25)  # Scale y-axis to fit text annotations
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved overlap bar chart to: {output_path}")
    plt.close()


def plot_per_celltype_hvg_overlap(rna_adata, genes_per_type=500, cell_type_col="cell_type", output_path="hvg_per_celltype_overlap.png"):
    """
    Plots the overlap between cell-specific HVGs and global HVGs for EACH individual cell type
    as a stacked bar chart.
    """
    print("\n--- Running Per-Cell-Type Overlap Analysis ---")
    print("1. Retrieving per-cell-type HVGs...")
    rna_cell_specific = select_variable_features_per_celltype(
        rna_adata.copy(),
        genes_per_type=genes_per_type,
        cell_type_key=cell_type_col
    )
    
    cell_type_hvgs_dict = rna_cell_specific.uns.get('cell_type_hvgs', {})
    n_union = len(rna_cell_specific.var_names)
    
    print(f"2. Running global HVG selection to compare (n_features={n_union})...")
    rna_global = select_variable_features(
        rna_adata.copy(),
        n_features=n_union,
        flavor="seurat_v3"
    )
    global_hvgs = set(rna_global.var_names)
    
    print("\n=== PER CELL-TYPE OVERLAP RESULTS ===")
    cell_types = []
    shared_counts = []
    unique_counts = []
    
    for ct, hvgs in cell_type_hvgs_dict.items():
        if ct == "union":
            continue
        ct_hvgs_set = set(hvgs)
        shared = ct_hvgs_set.intersection(global_hvgs)
        unique = ct_hvgs_set - global_hvgs
        
        cell_types.append(str(ct))
        shared_counts.append(len(shared))
        unique_counts.append(len(unique))
        
        pct_shared = (len(shared) / len(ct_hvgs_set)) * 100 if len(ct_hvgs_set) > 0 else 0
        print(f"{ct}: {len(shared)} shared ({pct_shared:.1f}%), {len(unique)} unique")

    # Generate Stacked Bar Chart Plot
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(cell_types))
    width = 0.6
    
    p1 = plt.bar(x, shared_counts, width, label='Shared with Global', color='#2ca02c', edgecolor='black', alpha=0.85)
    p2 = plt.bar(x, unique_counts, width, bottom=shared_counts, label='Unique to Cell Type', color='#1f77b4', edgecolor='black', alpha=0.85)
    
    plt.title(f"HVG Overlap with Global Set per Cell Type\n(Target: {genes_per_type} genes/type vs {n_union} global)", 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Cell Type", fontsize=12)
    plt.ylabel("Number of Genes", fontsize=12)
    plt.xticks(x, cell_types, rotation=45, ha='right')
    plt.legend()
    
    # Add percentage and total labels
    for i in range(len(cell_types)):
        total = shared_counts[i] + unique_counts[i]
        pct = (shared_counts[i] / total) * 100 if total > 0 else 0
        
        # Label inside the shared portion
        if shared_counts[i] > 0:
            plt.text(x[i], shared_counts[i] / 2, f"{pct:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
            
        # Total label above the bar
        plt.text(x[i], total + (max(shared_counts + unique_counts) * 0.02), f"{total}", ha='center', va='bottom', fontweight='bold', fontsize=10)
        
    # Scale y-axis to fit annotations
    plt.ylim(0, max([s+u for s,u in zip(shared_counts, unique_counts)]) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved per-cell-type overlap stacked bar chart to: {output_path}")
    plt.close()


def plot_individual_celltype_overlaps(rna_adata, genes_per_type=500, cell_type_col="cell_type", output_dir="."):
    """
    Creates individual 3-bar charts for EACH cell type, comparing its specific HVGs
    against the baseline global HVG set. Saves one PNG file per cell type.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("\n--- Generating Individual Cell-Type Overlap Plots ---")
    
    # 1. Run cell-specific selection to retrieve the dictionary
    rna_cell_specific = select_variable_features_per_celltype(
        rna_adata.copy(),
        genes_per_type=genes_per_type,
        cell_type_key=cell_type_col
    )
    cell_type_hvgs_dict = rna_cell_specific.uns.get('cell_type_hvgs', {})
    n_union = len(rna_cell_specific.var_names)
    
    # 2. Run global selection to act as the comparative baseline
    rna_global = select_variable_features(
        rna_adata.copy(),
        n_features=n_union,
        flavor="seurat_v3"
    )
    global_hvgs = set(rna_global.var_names)
    
    # 3. Iterate through each valid cell type and generate its 3-bar plot
    for ct, hvgs in cell_type_hvgs_dict.items():
        if ct == "union":
            continue
            
        ct_set = set(hvgs)
        overlap = ct_set.intersection(global_hvgs)
        only_ct = ct_set - global_hvgs
        only_global = global_hvgs - ct_set
        
        plt.figure(figsize=(8, 5))
        labels = ['Shared\n(Intersection)', f'Unique to\n{ct}', 'Unique to\nGlobal']
        sizes = [len(overlap), len(only_ct), len(only_global)]
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e']
        
        bars = plt.bar(labels, sizes, color=colors, alpha=0.85, edgecolor='black')
        
        plt.title(f"HVG Overlap: {ct} vs Global\n(Type: {len(ct_set)} genes vs Global: {len(global_hvgs)} genes)", 
                  fontsize=14, fontweight='bold', pad=20)
        plt.ylabel("Number of Genes", fontsize=12)
        
        # Add labels dynamically
        for i, (bar, size) in enumerate(zip(bars, sizes)):
            height = bar.get_height()
            if i < 2:  # 'Shared' or 'Unique to CT' is out of the cell type's total
                pct = (size / len(ct_set)) * 100 if len(ct_set) > 0 else 0
                text = f"{size}\n({pct:.1f}% of type)"
            else:      # 'Unique to Global' is out of the global set's total
                pct = (size / len(global_hvgs)) * 100 if len(global_hvgs) > 0 else 0
                text = f"{size}\n({pct:.1f}% of global)"
                
            plt.text(bar.get_x() + bar.get_width() / 2, height + (max(sizes) * 0.02),
                     text, ha='center', va='bottom', fontsize=11, fontweight='bold')
            
        plt.ylim(0, max(sizes) * 1.25)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        safe_ct = str(ct).replace(" ", "_").replace("/", "_")
        file_path = out_path / f"hvg_overlap_{safe_ct}.png"
        plt.savefig(file_path, dpi=300)
        print(f"  -> Saved {file_path}")
        plt.close()


if __name__ == "__main__":
    import logging
    import sys
    from pathlib import Path

    import muon as mu

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data_dir = Path("data/processed")
    out_dir = Path("data/processed")
    paper_dir = Path("data/paper/datasets")

    use_r_ref = "--r-reference" in sys.argv

    r_hvgs = _load_feature_list(paper_dir / "r_hvg_500.txt") if use_r_ref else None
    r_hvps = _load_feature_list(paper_dir / "r_hvp_10000.txt") if use_r_ref else None
    
    if use_r_ref:
        if r_hvgs and r_hvps:
            logger.info(
                f"R-reference mode: {len(r_hvgs)} HVGs, {len(r_hvps)} HVPs "
                f"from {paper_dir}"
            )
        else:
            logger.warning(
                "R-reference mode requested but feature lists not found. "
                "Run scripts/run_paper_phase3.R first. Falling back to Python."
            )
            r_hvgs = r_hvps = None

    # --- Retinal (unpaired, supports R-reference mode) ---
    logger.info("\n" + "=" * 60)
    logger.info("Retinal (unpaired)")
    logger.info("=" * 60)

    pairs = pd.read_csv(data_dir / "retinal_nn_pairs.csv")
    rna_full = ad.read_h5ad(data_dir / "retinal_rna.h5ad")
    atac_full = ad.read_h5ad(data_dir / "retinal_atac.h5ad")

    # =====================================================================
    # Generating HVG Overlap Charts
    # =====================================================================
    logger.info("Generating HVG Overlap Charts...")
    # 1. Create a quick subsample of the raw data just for the plot
    rna_for_plot, _ = subsample_pairs(rna_full, atac_full, pairs, n_per_type=100, seed=42)
    
    # 2. Run the union overlap plot
    plot_hvg_overlap(
        rna_for_plot, 
        genes_per_type=500, 
        cell_type_col="cell_type", 
        output_path=str(out_dir / "hvg_overlap_bar.png")
    )
    
    # 3. Run the PER CELL TYPE stacked overlap plot
    plot_per_celltype_hvg_overlap(
        rna_for_plot,
        genes_per_type=500,
        cell_type_col="cell_type",
        output_path=str(out_dir / "hvg_per_celltype_overlap_bar.png")
    )

    # 4. Run the newly added INDIVIDUAL cell-type overlap plots
    plot_individual_celltype_overlaps(
        rna_for_plot,
        genes_per_type=500,
        cell_type_col="cell_type",
        output_dir=out_dir
    )
    # =====================================================================

    harmony_path = paper_dir / "seurat_obj_harmony.csv"
    ret_emb = None
    ret_emb_names = None
    if harmony_path.exists():
        harmony_df = pd.read_csv(harmony_path, index_col=0)
        ret_emb = harmony_df.values
        ret_emb_names = list(harmony_df.index)
        logger.info(f"Loaded Harmony embedding: {ret_emb.shape}")

    retinal = prepare_subsample(
        rna=rna_full, atac=atac_full,
        n_per_type=100, n_genes=500, n_peaks=10000, seed=42,
        pairs=pairs,
        hvg_list=r_hvgs,
        hvp_list=r_hvps,
        embedding=ret_emb,
        embedding_cell_names=ret_emb_names,
    )

    retinal["rna"].write_h5ad(out_dir / "retinal_rna_sub_type.h5ad")
    retinal["atac"].write_h5ad(out_dir / "retinal_atac_sub_type.h5ad")
    if "knn_indices" in retinal:
        np.save(out_dir / "retinal_knn_indices_type.npy", retinal["knn_indices"])
    logger.info(f"Saved retinal subsampled data to {out_dir}")
    
    print(f"Number of cells: {retinal['rna'].n_obs}")
    print(f"Number of genes (HVGs only): {retinal['rna'].n_vars}") 

    if 'sparse' in str(type(retinal["rna"].X)).lower():
        print(f"Max expression value: {retinal['rna'].X.max()}")
    else:
        print(f"Max expression value: {retinal['rna'].X.max()}")

    print(retinal["rna"].obs['cell_type'].value_counts())
    
    logger.info("\nGenerating global HVG equivalent to match union size...")
    
    # A. Recover the raw RNA data for the exact same subsampled cells
    original_cells = retinal["rna"].obs["_original_rna_cell"].values
    rna_global_raw = rna_full[original_cells].copy()
    
    # B. Ensure obs names and cell_type annotations match the other files exactly
    rna_global_raw.obs_names = retinal["rna"].obs_names
    rna_global_raw.obs["cell_type"] = retinal["rna"].obs["cell_type"].copy()
    
    # C. Select global variable features using the exact size of the cell-specific union
    n_union = retinal["rna"].n_vars
    logger.info(f"  Selecting {n_union} global HVGs...")
    rna_global_hvg = select_variable_features(rna_global_raw, n_features=n_union, flavor="seurat_v3")
    
    # D. Save the global versions (without "_type")
    rna_global_hvg.write_h5ad(out_dir / "retinal_rna_sub.h5ad")
    
    # ATAC and KNN don't rely on RNA HVGs, so we can just re-save the existing 
    # objects under the global filenames to ensure the pipeline doesn't break
    retinal["atac"].write_h5ad(out_dir / "retinal_atac_sub.h5ad")
    if "knn_indices" in retinal:
        np.save(out_dir / "retinal_knn_indices.npy", retinal["knn_indices"])
        
    logger.info(f"Saved global equivalent data to {out_dir}")
    del retinal, rna_full, atac_full

    logger.info("\nDone.")