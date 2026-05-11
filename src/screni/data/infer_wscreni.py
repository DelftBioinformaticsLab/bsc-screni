"""
wScReNI - weighted Single-Cell Regulatory Network Inference
Python translation of the original R implementation.

"""

import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


# ---------------------------------------------------------------------------
# Helper dataclass to replicate the R S4 object `gene_peak_overlap_labs`
# ---------------------------------------------------------------------------

class GenePeakOverlapLabs:
    """
    Mirrors the S4 object used in the R code.

    Attributes
    ----------
    genes  : list[str]  - gene name for each entry (parallel to peaks/TFs/labels)
    peaks  : list[str]  - peak name for each entry
    TFs    : list[str]  - semicolon-separated TF names associated with each entry
    labels : list[str]  - 'TF' or '' / other label for each entry
    """
    def __init__(self, genes, peaks, TFs, labels):
        self.genes  = list(genes)
        self.peaks  = list(peaks)
        self.TFs    = list(TFs)
        self.labels = list(labels)


# ---------------------------------------------------------------------------
# Inner function — random-forest importance → regulatory weight matrix
# ---------------------------------------------------------------------------

def gene_peak_random_forest(
    expr_matrix: pd.DataFrame,          # genes × cells
    gene_peak_overlap_matrix: pd.DataFrame,  # genes × cells (peaks as rows)
    gene_peak_overlap_labs: GenePeakOverlapLabs,
    K: str | int = "sqrt",
    nb_trees: int = 100,
    n_jobs: int = 1,
    importance_measure: str = "IncNodePurity",
    seed: int | None = None,
) -> np.ndarray:
    """
    Compute a genes × genes regulatory weight matrix for a single cell
    (or a KNN-pooled pseudo-cell) using Random Forest feature importances.

    Parameters
    ----------
    expr_matrix            : np.ndarray, shape (n_genes, n_cells)
    gene_peak_overlap_matrix: np.ndarray, shape (n_peaks_as_genes_rows, n_cells)
    gene_peak_overlap_labs : GenePeakOverlapLabs
    K                      : 'sqrt' | 'all' | int  - mtry equivalent - max number of features used in RF
    nb_trees               : number of trees per forest
    n_jobs                 : parallel workers (usually 1 inside the outer loop)
    importance_measure     : 'IncNodePurity' (impurity) or '%IncMSE' (permutation)
    seed                   : random seed

    Returns
    -------
    weights : np.ndarray, shape (n_genes, n_genes)
        weights[i, j] = regulatory importance of gene j → gene i
        Diagonal entries encode self-regulatory (TF→own-target) scores.
    """
    if importance_measure not in ("IncNodePurity", "%IncMSE"):
        raise ValueError('importance_measure must be "IncNodePurity" or "%IncMSE"')

    rng = np.random.default_rng(seed)

    # Transpose: rows = cells (samples), columns = features
    X_genes = expr_matrix.T          # shape: (n_cells, n_genes)
    X_peaks = gene_peak_overlap_matrix.T  # shape: (n_cells, n_peaks)

    n_samples, n_genes = X_genes.shape
    n_peaks = X_peaks.shape[1]

    gene_names = (
        list(expr_matrix.index)
        if isinstance(expr_matrix, pd.DataFrame)
        else [f"gene_{i}" for i in range(n_genes)]
    )
    peak_names = (
        list(gene_peak_overlap_matrix.index)
        if isinstance(gene_peak_overlap_matrix, pd.DataFrame)
        else [f"peak_{i}" for i in range(n_peaks)]
    )

    X_genes = np.array(X_genes)
    X_peaks = np.array(X_peaks)

    # Pre-build lookup sets for speed
    # gene → set of peaks that overlap it
    gene_to_peaks: dict[str, list] = {g: [] for g in gene_names}
    for g, p in zip(gene_peak_overlap_labs.genes, gene_peak_overlap_labs.peaks):
        if g in gene_to_peaks:
            gene_to_peaks[g].append(p)

    # gene → set of TFs that bind near it
    gene_to_TFs: dict[str, set] = {g: set() for g in gene_names}
    for g, tf_str in zip(gene_peak_overlap_labs.genes, gene_peak_overlap_labs.TFs):
        if g in gene_to_TFs and tf_str:
            gene_to_TFs[g].update(
                tf.strip() for tf in tf_str.split(";") if tf.strip()
            )

    # gene → label ('TF' or other)
    gene_to_label: dict[str, str] = {}
    for g, lbl in zip(gene_peak_overlap_labs.genes, gene_peak_overlap_labs.labels):
        gene_to_label[g] = lbl

    # Column index maps for fast numpy slicing
    gene_col = {g: i for i, g in enumerate(gene_names)}
    peak_col = {p: i for i, p in enumerate(peak_names)}

    def _fit_one_gene(target_idx: int) -> np.ndarray:
        """Return a 1-D weight vector of length n_genes for one target gene."""
        target_name = gene_names[target_idx]

        # Predictors: all other genes + all peaks
        other_idxs = [i for i in range(n_genes) if i != target_idx]
        other_names = [gene_names[i] for i in other_idxs]

        X_other = X_genes[:, other_idxs]                 # (n_cells, n_genes-1)
        X_input = np.hstack([X_other, X_peaks])           # (n_cells, n_genes-1+n_peaks)

        y = X_genes[:, target_idx]                        # (n_cells,)

        weight_vector = np.zeros(n_genes)

        if y.sum() == 0:
            # All-zero target → no signal, return zero weights
            return weight_vector

        # mtry (max_features)
        n_input_vars = len(other_idxs) + n_peaks
        if isinstance(K, int):
            mtry = K
        elif K == "sqrt":
            mtry = max(1, round(np.sqrt(n_input_vars)))
        elif K == "all":
            mtry = n_input_vars
        else:
            raise ValueError('K must be "sqrt", "all", or an integer')

        # Fit Random Forest
        use_permutation = (importance_measure == "%IncMSE")
        rf = RandomForestRegressor(
            n_estimators=nb_trees,
            max_features=mtry,
            min_samples_leaf=1,       # fully grown trees (nodesize=1)
            bootstrap=True,
            oob_score=True,
            n_jobs=n_jobs,
            random_state=rng.integers(0, 2**31)
        )

        if not use_permutation:
            # IncNodePurity branch: normalise target
            y_fit = y / (y.std(ddof=1) + 1e-12)
            rf.fit(X_input, y_fit)
        else:
            rf.fit(X_input, y)

        # sklearn always exposes impurity-based importances via .feature_importances_
        # For %IncMSE you would need permutation_importance;

        if use_permutation:
            result = permutation_importance(
                rf, X_input, y, n_repeats=5, random_state=int(rng.integers(0, 2 ** 31)), n_jobs=1
            )
            importances = result.importances_mean
        else:
            importances = rf.feature_importances_ # length: n_genes-1 + n_peaks

        # Map importances back to named features
        feature_names = other_names + peak_names
        im: dict[str, float] = dict(zip(feature_names, importances))

        # Decompose importances into regulatory weights
        # Peaks near the TARGET gene
        target_peaks = gene_to_peaks.get(target_name, [])
        y_peak_coef = sum(im.get(p, 0.0) for p in target_peaks)

        # TFs that bind near the target gene
        TFs_of_target = gene_to_TFs.get(target_name, set())

        # Is the target gene itself labelled as a TF?
        target_label = gene_to_label.get(target_name, "")
        if target_label == "TF":
            weight_vector[target_idx] = y_peak_coef   # self-regulatory TF score
        else:
            weight_vector[target_idx] = 0.0

        # Score every other gene j
        for j, gene_j in zip(other_idxs, other_names):
            peaks_j = gene_to_peaks.get(gene_j, [])
            peakj_coef = sum(im.get(p, 0.0) for p in peaks_j)
            gene_coef  = im.get(gene_j, 0.0)

            if gene_j in TFs_of_target:
                # gene_j is a TF that regulates the target → include target peak signal
                weight_vector[j] = gene_coef + y_peak_coef + peakj_coef
            else:
                weight_vector[j] = gene_coef + peakj_coef

        return weight_vector

    # Run over all target genes (sequential inside one pseudo-cell)
    columns = [_fit_one_gene(idx) for idx in range(n_genes)]

    # Stack: each column = importances onto that target gene
    weights = np.column_stack(columns)          # shape: (n_genes, n_genes)
    weights /= n_samples                        # normalize by number of pooled cells

    return weights


# ---------------------------------------------------------------------------
# Outer function — infer per-cell networks
# ---------------------------------------------------------------------------

def infer_wScReNI_sc_networks(
    expr_matrix,                   # pd.DataFrame or np.ndarray, genes × cells
    gene_peak_overlap_matrix,      # pd.DataFrame or np.ndarray, genes × cells
    gene_peak_overlap_labs: GenePeakOverlapLabs,
    nearest_neighbors_idx: np.ndarray,   # shape (n_cells, K)  — 0-based indices
    network_path: str,
    data_name: str,
    cell_index=None,               # list/array of 0-based cell indices to process
    nthread: int = 50,
    max_cell_per_batch: int = 10,
) -> list[np.ndarray]:
    """
    Build single-cell regulatory weight matrices with wScReNI.

    For each cell i:
      1. Pool expression of cell i + its K nearest neighbors.
      2. Run gene_peak_random_forest on the pooled matrix.
      3. Save the genes × genes weight matrix to disk.

    Parameters
    ----------
    expr_matrix              : genes × cells expression (RNA)
    gene_peak_overlap_matrix : genes × cells peak accessibility (ATAC)
    gene_peak_overlap_labs   : GenePeakOverlapLabs object
    nearest_neighbors_idx    : (n_cells, K) array of neighbor cell indices (0-based)
    network_path             : root directory to save results
    data_name                : identifier string (informational)
    cell_index               : subset of 0-based cell indices; None = all cells
    nthread                  : total parallel threads
    max_cell_per_batch       : cells processed per parallel batch

    Returns
    -------
    list of np.ndarray, one weight matrix per requested cell
    """
    # Convert inputs to numpy if dataframe
    if isinstance(expr_matrix, pd.DataFrame):
        cell_names = list(expr_matrix.columns)
        expr_np    = expr_matrix.values.astype(float)   # genes × cells
    else:
        expr_np    = expr_matrix.astype(float)
        cell_names = [str(i) for i in range(expr_np.shape[1])]

    if isinstance(gene_peak_overlap_matrix, pd.DataFrame):
        peak_np = gene_peak_overlap_matrix.values.astype(float)
    else:
        peak_np = gene_peak_overlap_matrix.astype(float)

    gene_names_list = list(expr_matrix.index)
    peak_names_list = list(gene_peak_overlap_matrix.index)

    # Align peak matrix columns (cells) to expression matrix
    # (mirrors: colnames(gene_peak_overlap_matrix) <- colnames(exprMatrix))
    assert peak_np.shape[1] == expr_np.shape[1], (
        "expr_matrix and gene_peak_overlap_matrix must have the same number of cells"
    )

    n_cells = expr_np.shape[1]
    print(f"Total number of cells: {n_cells}")

    # Resolve cell range
    if cell_index is None:
        cells_to_process = list(range(n_cells))   # 0-based
    else:
        cells_to_process = list(cell_index)        # caller supplies 0-based indices

    n_target_cells = len(cells_to_process)
    n_batches = int(np.ceil(n_target_cells / max_cell_per_batch))

    # Create output directories
    out_dir = os.path.join(network_path, "wScReNI")
    os.makedirs(out_dir, exist_ok=True)

    # Worker: process one cell
    def _process_cell(i: int):
        """
        i : 0-based cell index into expr_np / peak_np.
        Returns (i, cell_name, weight_matrix).
        """
        # Pool cell i with its K nearest neighbors
        neighbor_idxs  = nearest_neighbors_idx[i]          # shape: (K,)
        pooled_cols    = np.array([i, *neighbor_idxs])     # cell + neighbors

        # wnn_expr = expr_np[:, pooled_cols]   # genes × (1+K)
        # wnn_peak = peak_np[:, pooled_cols]   # genes × (1+K)

        wnn_expr = pd.DataFrame(expr_np[:, pooled_cols], index=gene_names_list)
        wnn_peak = pd.DataFrame(peak_np[:, pooled_cols], index=peak_names_list)

        weights = gene_peak_random_forest(
            wnn_expr, wnn_peak, gene_peak_overlap_labs, n_jobs=1
        )

        cell_name = cell_names[i]
        out_file  = os.path.join(out_dir, f"{i}.{cell_name}.network.txt")
        pd.DataFrame(weights, index=gene_names_list, columns=gene_names_list).to_csv(
            out_file, sep="\t"
        )
        print(f"  Saved: {out_file}")
        return weights

    # Batch loop with joblib parallelism
    all_networks: list[np.ndarray] = []

    for batch_idx in range(n_batches):
        start = batch_idx * max_cell_per_batch
        end   = min(start + max_cell_per_batch, n_target_cells)
        batch_cells = cells_to_process[start:end]

        print(f"Cell {batch_cells[0]} to cell {batch_cells[-1]}")

        # Cap threads to ~1.5× batch size (mirrors the R logic)
        n_workers = min(nthread, max(1, int(len(batch_cells) * 1.5)))

        batch_results = Parallel(n_jobs=n_workers)(
            delayed(_process_cell)(i) for i in batch_cells
        )
        all_networks.extend(batch_results)

    return all_networks


# ---------------------------------------------------------------------------
# Load preprocessed retinal data and run wScReNI
# ---------------------------------------------------------------------------
# Expected files in data/processed/ (adjust BASE path as needed):
#   retinal_rna_sub.h5ad           (400 cells, 500 HVGs)
#   retinal_atac_sub.h5ad          (400 cells, 10000 peaks)  — upstream use only
#   retinal_knn_indices.npy        (400, 20)  0-based KNN from Harmony embedding
#   retinal_triplets.csv           columns: target_gene | peak | spearman_r | TF
#   retinal_gene_labels.csv        columns: gene | type | associated_peaks | associated_TFs
#   retinal_peak_overlap_matrix.npz  key "peak_matrix", shape (400, 217)

if __name__ == "__main__":
    import anndata as ad
    import scipy.sparse as sp

    # Paths
    # Adjust BASE to point to your data/processed/ directory
    DATASET = "retinal"   # swap to "pbmc", "seaad_paired", "seaad_unpaired"
    BASE    = f"../../../data/processed/{DATASET}"

    # 1. Load files
    rna      = ad.read_h5ad(f"{BASE}_rna_sub.h5ad")       # (400 cells, 500 genes)
    knn      = np.load(f"{BASE}_knn_indices.npy")          # (400, 20)  0-based
    triplets = pd.read_csv(f"{BASE}_triplets.csv")         # target_gene | peak | spearman_r | TF
    labels   = pd.read_csv(f"{BASE}_gene_labels.csv")      # gene | type | associated_peaks | associated_TFs

    peak_matrix   = np.load(f"../../../data/processed/peak_matrix.npy")
    # peak_matrix = peak_data["peak_matrix"]                 # (400, 217)
    # 2. expr_matrix: genes × cells
    # AnnData is cells × genes; wScReNI expects genes × cells — transpose
    X = rna.X.toarray() if sp.issparse(rna.X) else rna.X  # (400, 500)
    expr_df = pd.DataFrame(
        X.T,                        # → (500 genes, 400 cells)
        index   = rna.var_names,    # 500 gene names
        columns = rna.obs['_original_rna_cell'],    # 400 cell barcodes
    )

    # 3. peak_df: peaks × cells
    # peak_matrix is (400 cells, 217 peaks) — transpose to (217 peaks, 400 cells)
    # Peak names come from unique peaks in triplets (217 unique values)
    peak_names = triplets["peak"].unique()                 # 217 peak names
    peak_df = pd.DataFrame(
        peak_matrix.T,              # → (217 peaks, 400 cells)
        index   = peak_names,
        columns = rna.obs_names,
    )

    # 4. Build GenePeakOverlapLabs from gene_labels.csv
    # gene_labels already has associated_peaks and associated_TFs per gene
    # (comma-separated strings). Expand to one row per (gene, peak) pair.
    # The "type" column holds "TF" or "target" — maps to R's labels slot.
    rows = []
    for _, row in labels.iterrows():
        gene  = row["gene"]
        gtype = row["type"]          # "TF" or "target"

        if pd.isna(row["associated_peaks"]):
            # Remove empt peak rows
            continue
        else:
            peaks_for_gene = str(row["associated_peaks"]).split(",")
            TF_str = (
                ";".join(str(row["associated_TFs"]).split(","))
                if pd.notna(row["associated_TFs"]) else ""
            )
            for peak in peaks_for_gene:
                rows.append({"gene": gene, "peak": peak.strip(), "TF": TF_str, "label": gtype})

    labs_df = pd.DataFrame(rows)

    labs = GenePeakOverlapLabs(
        genes  = labs_df["gene"].tolist(),
        peaks  = labs_df["peak"].tolist(),
        TFs    = labs_df["TF"].tolist(),      # semicolon-separated TF names
        labels = labs_df["label"].tolist(),   # "TF" or "target"
    )

    # # Sanity checks
    # assert expr_df.shape == (500, 400), f"Unexpected expr shape: {expr_df.shape}"
    # assert peak_df.shape == (217, 400), f"Unexpected peak shape: {peak_df.shape}"
    # assert knn.shape     == (400, 20),  f"Unexpected KNN shape: {knn.shape}"

    print(f"expr_matrix : {expr_df.shape}  (genes × cells)")
    print(f"peak_df     : {peak_df.shape}  (peaks × cells)")
    print(f"knn         : {knn.shape}  (cells × neighbors)")
    print(f"labs entries: {len(labs.genes)} gene-peak rows")
    print(f"TF genes    : {(labels['type'] == 'TF').sum()}, "
          f"target genes: {(labels['type'] == 'target').sum()}")

    # Run wScReNI
    networks = infer_wScReNI_sc_networks(
        expr_matrix              = expr_df,
        gene_peak_overlap_matrix = peak_df,
        gene_peak_overlap_labs   = labs,
        nearest_neighbors_idx    = knn,       # (400, 20), 0-based
        network_path             = "output/networks",
        data_name                = DATASET,
        cell_index               = None,      # None = all 400 cells
        nthread                  = 8,
        max_cell_per_batch       = 10,
    )

    # 7. Output
    # networks is a plain list of length 400
    # networks[i] is a np.ndarray of shape (500, 500) for cell i
    # networks[i][row, col] = regulatory weight of gene col → gene row in cell i
    #
    # Files are also saved to: output/networks/wScReNI/{i}.{cell_barcode}.network.txt

    # Optional: convert to dict keyed by cell barcode for easier lookup
    cell_barcodes = rna.obs_names.tolist()
    networks_dict = {cell_barcodes[i]: networks[i] for i in range(len(networks))}

    # Optional: save results back into AnnData
    rna.uns["wScReNI_networks"] = networks
    rna.write_h5ad(f"output/{DATASET}_with_networks.h5ad")

    print(f"\nDone. {len(networks)} networks inferred.")
    print(f"Network shape per cell : {networks[0].shape}")   # (500, 500)
    print(f"Output dir             : output/networks/wScReNI/")