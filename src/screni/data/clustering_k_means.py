import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list

def process_and_cluster_heatmap_data(RNA1, K1=1, ColumnGroup1=None, Scale1="row",
                                     Range1=(-np.inf, np.inf), Reorder1=True, 
                                     RevOrder1=[-1], NAcolum1=None):
    """
    Translates the custom R K-means/hclust heatmap preparation function to Python.
    
    Parameters:
    -----------
    RNA1 : pandas.DataFrame or numpy.ndarray
        Gene expression matrix (Rows = Genes, Columns = Cells).
    K1 : int
        Number of K-means clusters to split the genes into.
    ColumnGroup1 : list or 1D array
        Labels to group columns (cells) by, used to insert visual blanks.
    Scale1 : str
        If 'row', standardizes the expression of each gene across all cells.
    Range1 : tuple
        (min, max) limits to clip extreme outlier values.
    Reorder1 : bool
        Whether to perform hierarchical clustering inside each K-means module.
    RevOrder1 : list
        List of cluster IDs whose hierarchical leaf order should be reversed.
    NAcolum1 : list
        Indices of columns to exclude during the K-means step.
    """

    # ---------------------------------------------------------
    # SECTION 1: Setup & Initialization
    # ---------------------------------------------------------
    # Keep track of row/col names if a DataFrame is passed
    is_df = isinstance(RNA1, pd.DataFrame)
    gene_names = RNA1.index if is_df else np.arange(RNA1.shape[0])
    cell_names = RNA1.columns if is_df else np.arange(RNA1.shape[1])
    
    # Extract underlying numpy array for math operations
    mat = RNA1.values.copy() if is_df else RNA1.copy()
    
    K2 = K1 if isinstance(K1, int) else len(K1)

    # ---------------------------------------------------------
    # SECTION 2: Scaling & Cleaning
    # ---------------------------------------------------------
    if Scale1 == "row":
        # sklearn.scale with axis=1 standardizes across columns (cells) for each row (gene)
        # This mirrors R's t(scale(t(RNA1)))
        mat = scale(mat, axis=1)
        
    # Replace NAs and NaNs with 0
    mat = np.nan_to_num(mat, nan=0.0)
    
    # Track the "clean" matrix for K-means calculation
    RNA1_clean = mat.copy()

    # ---------------------------------------------------------
    # SECTION 3: Column (Cell) Separation for Heatmap Gaps
    # ---------------------------------------------------------
    # Fix for R's NULL bug: explicitly set blank width to 1 column
    num_column_blank = 1 
    
    if ColumnGroup1 is not None:
        print("Separate columns according to variable ColumnGroup1")
        u_groups = np.unique(ColumnGroup1)
        
        if len(u_groups) > 1:
            new_cols = []
            blank_col = np.zeros((mat.shape[0], num_column_blank))
            
            for i, g in enumerate(u_groups):
                # Grab all cells belonging to this group
                group_data = mat[:, np.array(ColumnGroup1) == g]
                new_cols.append(group_data)
                
                # Append a column of zeros to act as a visual separator
                if i < len(u_groups) - 1:
                    new_cols.append(blank_col)
                    
            mat = np.hstack(new_cols)
            
    # RNA21 represents the display matrix (potentially with blank columns added)
    RNA21 = mat

    # ---------------------------------------------------------
    # SECTION 4: K-Means Clustering
    # ---------------------------------------------------------
    print("Perform k-means")
    # (Note: The impossible 'else' block from R for RowGroup1 is omitted for clean Python)
    
    # Handle NAcolum1 exclusion
    if NAcolum1 is not None:
        mask = np.ones(RNA1_clean.shape[1], dtype=bool)
        mask[NAcolum1] = False
        kmeans_input = RNA1_clean[:, mask]
    else:
        kmeans_input = RNA1_clean
        
    if K2 == 1:
        clusters = np.ones(kmeans_input.shape[0], dtype=int)
    else:
        # R's kmeans defaults to 10 starts. Sklearn defaults to 10 as well.
        kmeans = KMeans(n_clusters=K2, random_state=42, n_init=10)
        # Add 1 to match R's 1-indexed cluster naming convention
        clusters = kmeans.fit_predict(kmeans_input) + 1

    # ---------------------------------------------------------
    # SECTION 5: Sort Genes Within Modules via Hierarchical Clustering
    # ---------------------------------------------------------
    print("Sort genes")
    sorted_rows = []
    cluster_labels = []
    global_sorted_idx = [] # To keep track of original gene names
    
    for i in range(1, K2 + 1):
        # Extract rows belonging to the current K-means cluster
        idx = np.where(clusters == i)[0]
        cluster_mat = RNA21[idx, :]
        
        if Reorder1 and len(idx) > 1:
            # Calculate correlation distance. 
            # pdist calculates 1 - Pearson correlation, perfectly matching R's (1-cor)/2 math.
            dist_mat = pdist(cluster_mat, metric='correlation')
            dist_mat = np.nan_to_num(dist_mat, nan=0.0) # Failsafe for 0-variance genes
            
            # Hierarchical clustering (complete linkage is R's hclust default)
            link = linkage(dist_mat, method='complete')
            leaf_order = leaves_list(link)
            
            # Reverse order if requested
            if i in RevOrder1:
                leaf_order = leaf_order[::-1]
                
            sorted_idx = idx[leaf_order]
        else:
            sorted_idx = idx
            
        # Append the sorted block
        sorted_rows.append(RNA21[sorted_idx, :])
        cluster_labels.extend([i] * len(sorted_idx))
        global_sorted_idx.extend(sorted_idx)

    # Reconstruct the matrix block-by-block
    RNA22 = np.vstack(sorted_rows)

    # ---------------------------------------------------------
    # SECTION 6: Outlier Revision & Final Formatting
    # ---------------------------------------------------------
    print("Revise outlier")
    
    # Recreate the exact logging output from the R script
    num_under = np.sum(RNA22 < Range1[0])
    num_over = np.sum(RNA22 > Range1[1])
    print(f"Number of outlier: [{num_under}, {num_over}]")
    
    # Clip extreme values (squashes anything outside the min/max range)
    RNA23 = np.clip(RNA22, a_min=Range1[0], a_max=Range1[1])
    
    # Build the final pandas DataFrame
    final_df = pd.DataFrame(RNA23)
    final_df.insert(0, "KmeansGroup", cluster_labels)
    
    # Restore original row names (genes) now that they are sorted
    if is_df:
        final_df.index = gene_names[global_sorted_idx]
        
    return final_df