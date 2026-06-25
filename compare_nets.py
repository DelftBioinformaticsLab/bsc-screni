import os
import numpy as np
import warnings
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress
import networkx as nx
import networkx.algorithms.community as nx_comm
import pandas as pd

def plot_grn(matrix, gene_names, title, output_path, top_pct=0.0005,num_edges_opt=0):
    """
    Creates a NetworkX directed graph from an adjacency matrix and plots it.
    Filters to the top X% of edges and the largest connected component for readability.
    Sizes nodes based on degree and labels top hub genes.
    """
    mat_clean = np.nan_to_num(matrix, nan=0.0)
    np.fill_diagonal(mat_clean, 0.0)
    
    # Calculate the number of edges to extract based on the top percentage
    
    if top_pct < 1.0:
        num_edges = int(top_pct * mat_clean.size)
    else:
        num_edges = int(top_pct)
    if num_edges_opt!=0:
        num_edges=num_edges_opt
    flat = mat_clean.flatten()
    if num_edges > 0 and num_edges < len(flat):
        threshold_val = np.partition(flat, -num_edges)[-num_edges]
    else:
        threshold_val = 0
        
    sources, targets = np.where(mat_clean >= threshold_val)
    
    G = nx.DiGraph()
    for s, t in zip(sources, targets):
        if mat_clean[s, t] > 0:
            G.add_edge(gene_names[s], gene_names[t], weight=mat_clean[s, t])
            
    plt.figure(figsize=(12, 12))
    if len(G.edges()) == 0:
        plt.text(0.5, 0.5, "No edges found", ha="center", va="center")
        plt.title(title)
        plt.axis("off")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
        
    # Get largest weakly connected component for cleaner visualization
    if len(G.nodes()) > 50:
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        
    # Calculate Louvain communities on the subgraph for node coloring
    G_undirected = G.to_undirected()
    try:
        communities = nx_comm.louvain_communities(G_undirected, seed=42)
    except Exception:
        communities = [set(G.nodes())]
        
    node_to_comm = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = i
            
    num_comms = len(communities)
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    
    if num_comms <= 10:
        color_palette = [mcolors.to_hex(cm.tab10(i)) for i in range(num_comms)]
    elif num_comms <= 20:
        color_palette = [mcolors.to_hex(cm.tab20(i)) for i in range(num_comms)]
    else:
        color_palette = [mcolors.to_hex(cm.turbo(i / num_comms)) for i in range(num_comms)]
        
    node_colors = [color_palette[node_to_comm.get(n, 0)] for n in G.nodes()]
        
    # Layout optimization
    # Force directed layout: small 'k' and higher 'iterations' packs connected communities tightly
    pos = nx.spring_layout(G, k=0.15, iterations=400, seed=42)
    
    out_degrees = dict(G.out_degree())
    in_degrees = dict(G.in_degree())
    
    # Node size proportional to total connections
    node_sizes = [100 + (out_degrees.get(n, 0) + in_degrees.get(n, 0)) * 50 for n in G.nodes()]
    
    # Draw nodes and edges (using module-based node colors)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, edgecolors='gray')
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color='gray', arrows=True, arrowsize=10)
    
    # Label nodes with highest degree (Master Regulators)
    degrees = [out_degrees.get(n, 0) + in_degrees.get(n, 0) for n in G.nodes()]
    if len(degrees) > 0:
        p80 = np.percentile(degrees, 80)
        high_degree_nodes = [n for n in G.nodes() if (out_degrees.get(n, 0) + in_degrees.get(n, 0)) >= p80]
    else:
        high_degree_nodes = list(G.nodes())
        
    if len(high_degree_nodes) < 5:
        high_degree_nodes = list(G.nodes())
        
    labels = {n: n for n in high_degree_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold')
    
    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_modularity_heatmap(matrix, gene_names, communities, title, output_path, edge_threshold_pct=0.01):
    """
    Plots a heatmap of the adjacency matrix ordered by Louvain communities,
    so that communities form blocks along the diagonal, colored by cluster.
    """
    if not communities:
        return
        
    # Sort communities by size, largest first
    sorted_comms = sorted(communities, key=len, reverse=True)
    
    # Create an ordered list of genes and record community boundaries
    ordered_genes = []
    comm_boundaries = [0]
    
    for comm in sorted_comms:
        # Sort genes within the community for consistency
        ordered_genes.extend(sorted(list(comm)))
        comm_boundaries.append(len(ordered_genes))
        
    # Find the indices of the ordered genes in the original matrix
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    
    # Exclude genes that might not map correctly (failsafe)
    ordered_idx = [gene_to_idx[g] for g in ordered_genes if g in gene_to_idx]
    
    if len(ordered_idx) == 0:
        return
        
    # Sub-matrix permuted
    ordered_matrix = matrix[np.ix_(ordered_idx, ordered_idx)]
    
    # Binarize using the same threshold logic as compute_topology
    mat_clean = np.nan_to_num(ordered_matrix, nan=0.0)
    np.fill_diagonal(mat_clean, 0.0)
    
    orig_clean = np.nan_to_num(matrix, nan=0.0)
    np.fill_diagonal(orig_clean, 0.0)
    
    if edge_threshold_pct < 1.0:
        num_edges = int(edge_threshold_pct * orig_clean.size)
    else:
        num_edges = int(edge_threshold_pct)
        
    flat = orig_clean.flatten()
    if num_edges > 0 and num_edges < len(flat):
        threshold_val = np.partition(flat, -num_edges)[-num_edges]
    else:
        threshold_val = 0
        
    binary_mat = (mat_clean >= threshold_val).astype(int)
    
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    from matplotlib.patches import Patch
    
    # Prepare display matrix with different integer values for different clusters
    num_comms = len(sorted_comms)
    display_mat = np.zeros_like(binary_mat, dtype=int)
    display_mat[binary_mat == 1] = 1 # Inter-cluster edges
    
    for k in range(num_comms):
        start = comm_boundaries[k]
        end = comm_boundaries[k+1]
        internal_mask = binary_mat[start:end, start:end] == 1
        sub_view = display_mat[start:end, start:end]
        sub_view[internal_mask] = k + 2 # Cluster-specific colors start at 2
        
    # Generate colors
    color_list = ['#f5f5f5', '#7f8c8d'] # background, inter-cluster edge
    if num_comms <= 10:
        cluster_colors = [mcolors.to_hex(cm.tab10(i)) for i in range(num_comms)]
    elif num_comms <= 20:
        cluster_colors = [mcolors.to_hex(cm.tab20(i)) for i in range(num_comms)]
    else:
        cluster_colors = [mcolors.to_hex(cm.turbo(i / num_comms)) for i in range(num_comms)]
        
    color_list.extend(cluster_colors)
    cmap = mcolors.ListedColormap(color_list)
    bounds = list(range(num_comms + 3))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    plt.figure(figsize=(12, 10)) # Wider to accommodate legend
    
    # Plot matrix
    plt.imshow(display_mat, cmap=cmap, norm=norm, interpolation='none', aspect='auto')
    
    # Draw colored squares for boundaries
    ax = plt.gca()
    for k in range(num_comms):
        start = comm_boundaries[k]
        end = comm_boundaries[k+1]
        size = end - start
        if size > 1:
            rect = plt.Rectangle((start-0.5, start-0.5), size, size, 
                                 fill=False, edgecolor=cluster_colors[k], linewidth=1.5, alpha=0.9)
            ax.add_patch(rect)
            
    plt.title(title, fontsize=14)
    plt.xlabel("Genes (Ordered by Modularity Cluster)")
    plt.ylabel("Genes (Ordered by Modularity Cluster)")
    
    # Custom legend
    legend_elements = [Patch(facecolor='#7f8c8d', edgecolor='none', label='Inter-cluster Edge')]
    max_legend = min(num_comms, 15)
    for k in range(max_legend):
        c_size = comm_boundaries[k+1] - comm_boundaries[k]
        legend_elements.append(Patch(facecolor=cluster_colors[k], edgecolor='none', label=f'Cluster {k+1} (Size {c_size})'))
    if num_comms > 15:
        legend_elements.append(Patch(facecolor='none', edgecolor='none', label=f'... and {num_comms-15} more'))
        
    # Place legend outside
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def compare_wscreni_networks(py_matrix, r_matrix, py_gene_names, r_gene_names, cell_name="Mean_All_Cells", top_pct=0.01, enrichment_top_pct=0.01, output_dir="output/comparison"):
    """
    Compares two wScReNI networks (Python vs R) using Weighted Jaccard Similarity 
    on the edges between genes in the intersection of their HVGs.
    Extracts the unique genes from the top X% of edges for downstream enrichment analysis.
    Safely handles differently ordered matrices by using explicit gene name lists.
    Evaluates Topological properties and exports Modularity clusters.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # =======================================================================
    # Background/Universe Gene List Generation
    # =======================================================================
    common_genes = set(py_gene_names).intersection(set(r_gene_names))
    background_file_path = os.path.join(output_dir, f"background_hvg_intersection_{cell_name}.txt")
    
    with open(background_file_path, "w") as f:
        for gene in sorted(list(common_genes)):
            f.write(f"{gene}\n")
            
    print(f"[{cell_name}] Saved Background Gene Universe ({len(common_genes)} genes) to: {background_file_path}")
    
    # ==========================================
    # PART A: Extract Top Genes for Enrichment
    # ==========================================
    def get_top_edges(matrix, names, pct):
        # Handle NaNs safely and ignore self-loops
        mat_clean = np.nan_to_num(matrix, nan=0.0)
        np.fill_diagonal(mat_clean, 0.0) 
        
        # Determine number of edges to extract correctly
        if pct < 1.0:
            num_edges = int(pct * mat_clean.size)
        else:
            num_edges = int(pct)
            
        # Sort and get the indices of the top N largest values
        flat_indices = np.argsort(mat_clean.flatten())[-num_edges:][::-1]
        row_idx, col_idx = np.unravel_index(flat_indices, mat_clean.shape)
        
        edges = {}
        for r, c in zip(row_idx, col_idx):
            if mat_clean[r, c] > 0:
                edge_name = f"{names[r]} -> {names[c]}"
                edges[edge_name] = mat_clean[r, c]
        return edges

    # Get top edges from the FULL original matrices for enrichment
    py_top_enrichment_edges = get_top_edges(py_matrix, py_gene_names, enrichment_top_pct)
    r_top_enrichment_edges = get_top_edges(r_matrix, r_gene_names, enrichment_top_pct)
    
    # Extract unique genes
    def extract_unique_genes(edges_dict):
        genes = set()
        for edge in edges_dict.keys():
            g1, g2 = edge.split(" -> ")
            genes.add(g1)
            genes.add(g2)
        return sorted(list(genes))
        
    py_enrichment_genes = extract_unique_genes(py_top_enrichment_edges)
    r_enrichment_genes = extract_unique_genes(r_top_enrichment_edges)
    
    # Save to files
    py_genes_path = os.path.join(output_dir, f"enrichment_genes_{cell_name}_py.txt")
    r_genes_path = os.path.join(output_dir, f"enrichment_genes_{cell_name}_r.txt")
    
    with open(py_genes_path, "w") as f:
        f.write("\n".join(py_enrichment_genes))
        
    with open(r_genes_path, "w") as f:
        f.write("\n".join(r_enrichment_genes))
        
    print(f"[{cell_name}] Exported {len(py_enrichment_genes)} Python genes and {len(r_enrichment_genes)} R genes for enrichment.")

    # ==========================================
    # PART B: Topology Analysis (Scale-Free & Modularity)
    # ==========================================
    def compute_topology(matrix, names_list, edge_threshold_pct=0.005):
        mat_clean = np.nan_to_num(matrix, nan=0.0)
        np.fill_diagonal(mat_clean, 0.0)
        
        if edge_threshold_pct < 1.0:
            num_edges = int(edge_threshold_pct * mat_clean.size)
        else:
            num_edges = int(edge_threshold_pct)
            
        # Fast threshold value extraction using np.partition
        flat = mat_clean.flatten()
        if num_edges > 0 and num_edges < len(flat):
            threshold_val = np.partition(flat, -num_edges)[-num_edges]
        else:
            threshold_val = 0
            
        binary_mat = (mat_clean >= threshold_val).astype(int)
        
        # 1. Scale-Free Fit (Degree Distribution)
        degree = np.sum(binary_mat, axis=1) # Out-degree
        valid_degrees = degree[degree > 0]
        unique_degrees, counts = np.unique(valid_degrees, return_counts=True)
        
        if len(unique_degrees) >= 2:
            log_k = np.log10(unique_degrees)
            log_Pk = np.log10(counts / np.sum(counts))
            slope, intercept, r_value, p_value, std_err = linregress(log_k, log_Pk)
            r2 = r_value**2
        else:
            log_k, log_Pk, slope, intercept, r2 = np.array([]), np.array([]), 0, 0, 0
            
        # 2. Modularity (Q-score) and Communities using Louvain
        G = nx.from_numpy_array(binary_mat, create_using=nx.Graph)
        
        # Map generic node indices back to actual Gene Names
        mapping = {i: names_list[i] for i in range(len(names_list))}
        G = nx.relabel_nodes(G, mapping)
        
        try:
            communities = nx_comm.louvain_communities(G)
            modularity = nx_comm.modularity(G, communities)
        except Exception:
            modularity = 0.0
            communities = []
            
        return log_k, log_Pk, slope, intercept, r2, modularity, communities

    print(f"[{cell_name}] Computing Topology Metrics...")
    
    # Evaluate topology on top 1% edges of the full networks
    py_log_k, py_log_Pk, py_m, py_c, py_r2, py_mod, py_comms = compute_topology(py_matrix, py_gene_names, 0.01)
    r_log_k, r_log_Pk, r_m, r_c, r_r2, r_mod, r_comms = compute_topology(r_matrix, r_gene_names, 0.01)
    
    # ---> EXPORT MODULARITY CLUSTERS FOR ENRICHMENT <---
    modules_dir = os.path.join(output_dir, "modules")
    os.makedirs(modules_dir, exist_ok=True)
    
    def export_communities(communities, prefix):
        # Sort communities by size (largest module gets index 0)
        sorted_comms = sorted(communities, key=len, reverse=True)
        exported_count = 0
        for i, comm in enumerate(sorted_comms):
            if len(comm) >= 10:  # Exclude tiny clusters (<10 genes) which break enrichment tools
                mod_path = os.path.join(modules_dir, f"module_{i}_{cell_name}_{prefix}.txt")
                with open(mod_path, "w") as f:
                    f.write("\n".join(sorted(list(comm))))
                exported_count += 1
        return exported_count
        
    py_mod_count = export_communities(py_comms, "py")
    r_mod_count = export_communities(r_comms, "r")
    print(f"[{cell_name}] Exported {py_mod_count} Python modules and {r_mod_count} R modules to '{modules_dir}'")
    
    # ---> PLOT MODULARITY HEATMAPS <---
    print(f"[{cell_name}] Generating Modularity Heatmaps...")
    heatmap_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    plot_modularity_heatmap(py_matrix, py_gene_names, py_comms, 
                            title=f"Modularity Edge Matrix: type-specific wScReNI ({cell_name})", 
                            output_path=os.path.join(heatmap_dir, f"heatmap_{cell_name}_py.png"),
                            edge_threshold_pct=0.01)
                            
    plot_modularity_heatmap(r_matrix, r_gene_names, r_comms, 
                            title=f"Modularity Edge Matrix: gwScReNI ({cell_name})", 
                            output_path=os.path.join(heatmap_dir, f"heatmap_{cell_name}_r.png"),
                            edge_threshold_pct=0.01)
    
    # Plot Scale-Free Topology comparison
    plt.figure(figsize=(8, 6))
    
    # Cell-Specific Plot
    plt.scatter(py_log_k, py_log_Pk, label=f'type-specific wScReNI (R²={py_r2:.2f})', alpha=0.7, color='#2ca02c')
    if len(py_log_k) > 0:
        plt.plot(py_log_k, py_m * py_log_k + py_c, linestyle='--', color='#2ca02c')
        
    # Reference Plot
    plt.scatter(r_log_k, r_log_Pk, label=f'gwScReNI ref (R²={r_r2:.2f})', alpha=0.7, color='#ff7f0e')
    if len(r_log_k) > 0:
        plt.plot(r_log_k, r_m * r_log_k + r_c, linestyle='--', color='#ff7f0e')
        
    plt.xlabel("log(k) [Degree]")
    plt.ylabel("log(P(k)) [Frequency]")
    plt.title(f"Scale-Free Topology Fit: {cell_name}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scalefree_{cell_name}.png"), dpi=300)
    plt.close()

    # ==========================================
    # PART C: Weighted Jaccard Similarity on Shared Subspace
    # ==========================================
    shared_genes = sorted(list(common_genes))
    
    if len(shared_genes) == 0:
        print(f"WARNING: No shared genes found between Python and R datasets for {cell_name}.")
        return 0.0, [], [], py_mod, r_mod
        
    # Get indices for subsetting
    py_gene_to_idx = {g: i for i, g in enumerate(py_gene_names)}
    r_gene_to_idx = {g: i for i, g in enumerate(r_gene_names)}
    
    py_indices = [py_gene_to_idx[g] for g in shared_genes]
    r_indices = [r_gene_to_idx[g] for g in shared_genes]
    
    # Subset matrices to the shared genes
    py_submat = py_matrix[np.ix_(py_indices, py_indices)]
    r_submat = r_matrix[np.ix_(r_indices, r_indices)]
    
    py_submat = np.nan_to_num(py_submat, nan=0.0)
    r_submat = np.nan_to_num(r_submat, nan=0.0)
    
    py_submat = np.clip(py_submat, 0, None)
    r_submat = np.clip(r_submat, 0, None)
    
    np.fill_diagonal(py_submat, 0.0)
    np.fill_diagonal(r_submat, 0.0)
    
    # Calculate Weighted Jaccard Similarity
    min_mat = np.minimum(py_submat, r_submat)
    max_mat = np.maximum(py_submat, r_submat)
    
    min_sum = np.sum(min_mat)
    max_sum = np.sum(max_mat)
    
    weighted_jaccard = min_sum / max_sum if max_sum > 0 else 0.0
    
    # Extract top 1% discrepancies within this shared subspace
    py_top_edges = get_top_edges(py_submat, shared_genes, top_pct)
    r_top_edges = get_top_edges(r_submat, shared_genes, top_pct)
    
    py_set = set(py_top_edges.keys())
    r_set = set(r_top_edges.keys())
    
    py_only = py_set - r_set
    r_only = r_set - py_set
    
    top_10_py_only = sorted([(e, py_top_edges[e]) for e in py_only], key=lambda x: x[1], reverse=True)[:10]
    top_10_r_only = sorted([(e, r_top_edges[e]) for e in r_only], key=lambda x: x[1], reverse=True)[:10]
    
    # Write Results to File
    output_filename = f"network_comparison_{cell_name}_shared_genes.txt"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, "w") as f:
        f.write("=" * 65 + "\n")
        f.write(f"Global Network Comparison (Target: {cell_name})\n")
        f.write(f"Subspace: {len(shared_genes)} Shared Genes\n")
        f.write("=" * 65 + "\n")
        f.write(f"Weighted Jaccard Similarity: {weighted_jaccard:.4f}\n")
        f.write(f"Scale-Free Topology R²: type-specific = {py_r2:.3f}, gwScReNI = {r_r2:.3f}\n")
        f.write(f"Network Modularity (Q): type-specific = {py_mod:.3f}, gwScReNI = {r_mod:.3f}\n\n")
        
        f.write(f"Top 10 edges chosen by Python wScReNI but NOT by R (in shared top {top_pct*100}%):\n")
        for i, (edge, weight) in enumerate(top_10_py_only, 1):
            f.write(f"  {i:2d}. {edge:<25} (Mean Weight: {weight:.6f})\n")
            
        f.write(f"\nTop 10 edges chosen by R wScReNI but NOT by Python (in shared top {top_pct*100}%):\n")
        for i, (edge, weight) in enumerate(top_10_r_only, 1):
            f.write(f"  {i:2d}. {edge:<25} (Mean Weight: {weight:.6f})\n")
        f.write("=" * 65 + "\n")
    
    return weighted_jaccard, top_10_py_only, top_10_r_only, py_mod, r_mod


# ==========================================
# EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    
    # --- 1. SET FILE PATHS ---
    py_cache_path = "output/comparison/cache/wScReNI_networks_type.npz"
    r_cache_path  = "output/comparison/cache/wScReNI_networks.npz" 
    
    py_genes_path = "output/comparison/cache/wScReNI_gene_names_type.txt"
    r_genes_path  = "output/comparison/cache/wScReNI_gene_names.txt" 
    
    rna_data_path = "data/processed/retinal_rna_sub_type.h5ad"
    
    if not os.path.exists(py_cache_path) or not os.path.exists(r_cache_path):
        print("ERROR: Could not find one of the network .npz files. Check the paths.")
        exit(1)
        
    print("Loading networks from cache...")
    py_networks = np.load(py_cache_path, allow_pickle=False)
    r_networks = np.load(r_cache_path, allow_pickle=False)
    
    # --- 2. LOAD GENE NAMES ---
    print("Loading gene names...")
    with open(py_genes_path, "r") as f:
        py_gene_names_list = [line.strip() for line in f if line.strip()]
        
    with open(r_genes_path, "r") as f:
        r_gene_names_list = [line.strip() for line in f if line.strip()]

    # --- 3. FIND MATCHING CELLS ---
    common_cells = [cell for cell in py_networks.files if cell in r_networks.files]
    print(f"Found {len(common_cells)} common cells between Python and R datasets.")
    
    if len(common_cells) == 0:
        print("ERROR: No matching cell names found between the Python and R datasets.")
        exit(1)

    # --- 4. LOAD ANNOTATIONS & HVGs ---
    print("Loading cell type annotations and specific HVGs...")
    if not os.path.exists(rna_data_path):
        print(f"ERROR: Cannot find RNA data at {rna_data_path} to determine cell types.")
        exit(1)
        
    rna = ad.read_h5ad(rna_data_path)
    cell_type_dict = dict(zip(rna.obs_names, rna.obs['cell_type']))
    cell_types = rna.obs['cell_type'].unique()
    
    # --- 4.5. EXPORT BACKGROUND GENES ---
    print("Exporting background gene lists for enrichment...")
    bg_dir = "output/comparison"
    os.makedirs(bg_dir, exist_ok=True)
    
    # Export Global Background (R reference)
    bg_r_path = os.path.join(bg_dir, "background_genes_global.txt")
    with open(bg_r_path, "w") as f:
        f.write("\n".join(r_gene_names_list))
    print(f"  -> Saved {len(r_gene_names_list)} global HVGs to {bg_r_path}")

    # Export Cell-Specific Backgrounds (Python run)
    if 'cell_type_hvgs' in rna.uns:
        for ct in cell_types:
            if ct in rna.uns['cell_type_hvgs']:
                ct_hvgs = list(rna.uns['cell_type_hvgs'][ct])
                bg_py_path = os.path.join(bg_dir, f"background_genes_{ct}_py.txt")
                with open(bg_py_path, "w") as f:
                    f.write("\n".join(ct_hvgs))
                print(f"  -> Saved {len(ct_hvgs)} cell-specific HVGs for '{ct}' to {bg_py_path}")
            else:
                print(f"  -> WARNING: No specific HVG list found in rna.uns for cell type '{ct}'")
    else:
        print("  -> WARNING: 'cell_type_hvgs' not found in rna.uns. Cannot export cell-type specific backgrounds.")

    # --- 5. HELPER TO COMPUTE MEAN NETWORK ---
    def compute_mean_network(networks_npz, cells):
        matrix_shape = networks_npz[cells[0]].shape 
        
        stacked = np.empty((len(cells), matrix_shape[0], matrix_shape[1]), dtype=np.float32)
        for i, cell in enumerate(cells):
            stacked[i] = networks_npz[cell]
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_matrix = np.nanmean(stacked, axis=0) 
            
        return mean_matrix

    # --- 6. SAFETY CHECK: TRUNCATE IF NECESSARY ---
    sample_py = py_networks[common_cells[0]]
    sample_r = r_networks[common_cells[0]]
    
    if len(py_gene_names_list) != sample_py.shape[0]:
        print(f"WARNING: Python gene list ({len(py_gene_names_list)}) doesn't match matrix shape ({sample_py.shape[0]}). Truncating.")
        py_gene_names_list = py_gene_names_list[:sample_py.shape[0]]
        
    if len(r_gene_names_list) != sample_r.shape[0]:
        print(f"WARNING: R gene list ({len(r_gene_names_list)}) doesn't match matrix shape ({sample_r.shape[0]}). Truncating.")
        r_gene_names_list = r_gene_names_list[:sample_r.shape[0]]

    # Create GRN Output Directory
    grn_out_dir = os.path.join("output/comparison", "grn_plots")
    os.makedirs(grn_out_dir, exist_ok=True)

    # Prepare collection array for Modularity Grouped Bar Chart
    modularity_data = {"CellType": [], "cell-specific wScReNI": [], "gwScReNI": []}

    # --- 7. RUN COMPARISON FOR ALL CELLS (GLOBAL MEAN) ---
    print("\n--- Processing Global Mean (All Cells) ---")
    py_mean_matrix = compute_mean_network(py_networks, common_cells)
    r_mean_matrix = compute_mean_network(r_networks, common_cells)
    
    jaccard, py_only, r_only, py_mod, r_mod = compare_wscreni_networks(
        py_matrix=py_mean_matrix, 
        r_matrix=r_mean_matrix, 
        py_gene_names=py_gene_names_list,    
        r_gene_names=r_gene_names_list,     
        cell_name="Mean_All_Cells",
        top_pct=0.1,
        enrichment_top_pct=150,
        output_dir="output/comparison"
    )
    
    # Plot Mean GRNs
    mat_clean_temp = np.nan_to_num(py_mean_matrix, nan=0.0)
    np.fill_diagonal(mat_clean_temp, 0.0)
    num_edges_opt=int (mat_clean_temp.size*0.0005)
    plot_grn(py_mean_matrix, py_gene_names_list, title="GRN: type-specific wScReNI (Global Mean)", 
             output_path=os.path.join(grn_out_dir, "grn_Mean_All_Cells_py.png"), top_pct=0.0005,num_edges_opt=num_edges_opt)
    plot_grn(r_mean_matrix, r_gene_names_list, title="GRN: gwScReNI (Global Mean)", 
             output_path=os.path.join(grn_out_dir, "grn_Mean_All_Cells_r.png"), top_pct=0.00005,num_edges_opt=num_edges_opt)
    
    # Log modularity
    modularity_data["CellType"].append("All_Cells")
    modularity_data["cell-specific wScReNI"].append(py_mod)
    modularity_data["gwScReNI"].append(r_mod)
    
    # --- 8. RUN COMPARISON PER CELL TYPE ---
    for ct in cell_types:
        print(f"\n--- Processing Cell Type: {ct} ---")
        
        # Find which common cells belong to this cell type
        ct_cells = [c for c in common_cells if cell_type_dict.get(c) == ct]
        
        if not ct_cells:
            print(f"  No common cells found for cell type '{ct}'. Skipping.")
            continue
            
        print(f"  Found {len(ct_cells)} cells for '{ct}'. Calculating means...")
        py_ct_mean_matrix = compute_mean_network(py_networks, ct_cells)
        r_ct_mean_matrix = compute_mean_network(r_networks, ct_cells)
        
        jaccard, py_only, r_only, py_mod, r_mod = compare_wscreni_networks(
            py_matrix=py_ct_mean_matrix, 
            r_matrix=r_ct_mean_matrix, 
            py_gene_names=py_gene_names_list,    
            r_gene_names=r_gene_names_list,     
            cell_name=ct,
            top_pct=0.1,
            enrichment_top_pct=150,
            output_dir="output/comparison"
        )
        
        # Plot Mean GRNs
        plot_grn(py_ct_mean_matrix, py_gene_names_list, title=f"GRN: type-specific wScReNI ({ct})", 
                 output_path=os.path.join(grn_out_dir, f"grn_{ct}_py.png"), top_pct=0.0005,num_edges_opt=num_edges_opt)
        plot_grn(r_ct_mean_matrix, r_gene_names_list, title=f"GRN: gwScReNI ({ct})", 
                 output_path=os.path.join(grn_out_dir, f"grn_{ct}_r.png"), top_pct=0.000005,num_edges_opt=num_edges_opt)
        
        # Log modularity
        modularity_data["CellType"].append(ct)
        modularity_data["cell-specific wScReNI"].append(py_mod)
        modularity_data["gwScReNI"].append(r_mod)
        
    # --- 9. PLOT MODULARITY COMPARISON BAR CHART ---
    print("\nGenerating Global Modularity Comparison Graph...")
    df_mod = pd.DataFrame(modularity_data)
    
    x = np.arange(len(df_mod["CellType"]))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, df_mod["cell-specific wScReNI"], width, label='type-specific wScReNI', color='#2ca02c')
    rects2 = ax.bar(x + width/2, df_mod["gwScReNI"], width, label='gwScReNI', color='#ff7f0e')
    
    ax.set_ylabel('Network Modularity (Q-score)')
    ax.set_title('Network Modularity Comparison (Higher is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(df_mod["CellType"], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join("output/comparison", "modularity_comparison_bar.png"), dpi=300)
    plt.close()
    
    print("\nAll comparisons, topology analysis, GRN plotting, and gene exports completed successfully!")