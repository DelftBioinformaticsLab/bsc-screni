import numpy as np
import scanpy as sc
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score

def calculate_scNetwork_degree(sc_networks, top, cell_type_annotation, ntype):
    """
    Translates the calculate_scNetwork_degree R function to Python.
    
    Parameters:
    -----------
    sc_networks : dict
        Dictionary of network lists, e.g., {'CSN': [mat1, mat2...], 'wScReNI': [mat1, mat2, ...], ...}
        where each matrix is a numpy array of shape (genes, genes).
    top : list or 1D numpy array
        Number of top regulation pairs to keep for each cell.
    cell_type_annotation : list or 1D numpy array
        The true ground-truth labels for the cells.
    ntype : int
        The expected number of cell types (used for hierarchical clustering tree cutting).
        
    Returns:
    --------
    degree_all : dict
        A nested dictionary containing the AnnData objects and ARI scores.
    """
    
    # ---------------------------------------------------------
    # SECTION 1: Setup & Initialization
    # ---------------------------------------------------------
    degree_all = {}
    network_types = list(sc_networks.keys())
    
    for net_name in network_types:
        print(f"Processing network: {net_name}")
        sc_net_list = sc_networks[net_name]
        cell_num = len(sc_net_list)
        gene_num = sc_net_list[0].shape[0]
        
        # Initialize the degree matrices (Genes x Cells to match R logic initially)
        indegree = np.zeros((gene_num, cell_num))
        outdegree = np.zeros((gene_num, cell_num))
        
        # ---------------------------------------------------------
        # SECTION 2: Network Thresholding & Degree Calculation
        # ---------------------------------------------------------
        for j in range(cell_num):
            # Create a copy of the cell's network matrix to avoid mutating original data
            weights = sc_net_list[j].copy()
            
            if net_name != 'CSN':
                # Flatten the matrix to rank all edges globally (like R's order() does)
                flat_weights = weights.flatten()
                
                # Find the indices of the top N strongest edges
                top_k = top[j]
                # argsort is ascending, so we take the last 'top_k' elements
                top_indices = np.argsort(flat_weights)[-top_k:] 
                
                # Create a binary 1/0 matrix based on those top edges
                binary_flat = np.zeros_like(flat_weights)
                binary_flat[top_indices] = 1
                binary_weights = binary_flat.reshape(weights.shape)
                
                # Calculate degrees (mimicking R's rowSums and colSums)
                indegree[:, j] = np.sum(binary_weights, axis=1)
                outdegree[:, j] = np.sum(binary_weights, axis=0)
            else:
                # CSN is already binary/formatted, so just sum
                indegree[:, j] = np.sum(weights, axis=1)
                outdegree[:, j] = np.sum(weights, axis=0)
                
        # Package the degrees for the clustering loop
        degree_dict = {'in.degree': indegree, 'out.degree': outdegree}
        results = {}
        
        # ---------------------------------------------------------
        # SECTION 3: Scanpy Object Creation & Preprocessing
        # ---------------------------------------------------------
        for deg_type, deg_data in degree_dict.items():
            print(f"  Running clustering on {deg_type}...")
            
            # Create Scanpy AnnData object
            # Note: R adds 1. Scanpy requires (Cells x Genes), so we transpose (.T)
            adata = sc.AnnData(X=(deg_data + 1).T)
            adata.obs['True_Label'] = cell_type_annotation
            
            # Seurat equivalent Preprocessing
            sc.pp.normalize_total(adata, target_sum=10000)
            sc.pp.log1p(adata) # Needed before scaling/HVG in Python
            
            # Find variable features (capping at 4000, or max genes if less than 4000)
            n_features = min(4000, adata.n_vars)
            sc.pp.highly_variable_genes(adata, n_top_genes=n_features, flavor='seurat')
            
            sc.pp.scale(adata)
            
            # ---------------------------------------------------------
            # SECTION 4: UMAP Dimensionality Reduction & Clustering
            # ---------------------------------------------------------
            # Run PCA
            sc.tl.pca(adata, use_highly_variable=True)
            
            # Find Neighbors (dim=1:10 equivalent in Scanpy)
            n_pcs_to_use = min(10, adata.obsm['X_pca'].shape[1])
            sc.pp.neighbors(adata, n_pcs=n_pcs_to_use)
            
            # Run UMAP and Leiden Clustering (Scanpy's preferred alternative to Louvain)
            sc.tl.umap(adata)
            sc.tl.leiden(adata, resolution=0.5)
            
            # ---------------------------------------------------------
            # SECTION 5: Hierarchical Clustering
            # ---------------------------------------------------------
            # Equivalent to: dist(cor(log(degree_data + 1)))
            # np.log1p is log(x + 1). We transpose because we want distance between cells.
            log_deg = np.log1p(deg_data)
            
            # pdist with metric='correlation' calculates 1 - Pearson correlation
            dist_matrix = pdist(log_deg.T, metric='correlation')
            
            # Perform hierarchical/agglomerative clustering ('complete' is R's hclust default)
            hc_linkage = linkage(dist_matrix, method='complete')
            
            # Cut the tree into 'ntype' clusters (equivalent to cutree)
            cluster_hclust = fcluster(hc_linkage, t=ntype, criterion='maxclust')
            
            # ---------------------------------------------------------
            # SECTION 6: Evaluation (ARI) & Storage
            # ---------------------------------------------------------
            # Extract Scanpy/UMAP clusters
            cluster_umap = adata.obs['leiden'].values
            
            # Calculate Adjusted Rand Index
            ari_umap = adjusted_rand_score(cell_type_annotation, cluster_umap)
            ari_hclust = adjusted_rand_score(cell_type_annotation, cluster_hclust)
            
            # Store in the results dictionary
            results[f'{deg_type}.adata'] = adata
            results[f'{deg_type}.umap.ARI'] = ari_umap
            results[f'{deg_type}.hclust.ARI'] = ari_hclust
            
        # Add the completed network to the master dictionary
        degree_all[net_name] = results
        
    return degree_all