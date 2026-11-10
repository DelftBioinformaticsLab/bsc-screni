"""Run Cell2Map or Tangram for one (donor, subclass) and read out positions.

Cell2Map (https://github.com/shaoqiangzhang/Cell2Map, vendored as a git
submodule in external/Cell2Map) was written for multi-cell spots. Four
adaptations make it run on single-cell MERFISH, all done here so the
upstream code stays untouched:

1. ``map_utils`` and ``common`` import ``Celloc`` and ``datatable``, which
   the mapping never calls; stub modules are registered before import. ``loss.py`` uses ``nn`` and ``F`` without
   importing them; they are provided while the package is imported.
   PyG's ``knn_graph`` needs the compiled ``pyg-lib``; a scikit-learn
   version with the same output replaces it. A singular covariance in the
   Mahalanobis loss term falls back to the pseudo-inverse.
2. The spatial graph is hard-coded as a 5-unit radius graph. MERFISH
   coordinates are in micrometres and a subclass subset is sparse, so the
   spatial graph is replaced by a k-nearest-neighbour graph (default k=7)
   or a radius graph with a radius you choose.
3. ``map_cell_to_space`` cannot take a custom density vector (its
   ``cell_count_prior == "cell_counts"`` check fails on arrays). The same
   steps are run here with a uniform density (one cell per MERFISH "spot"),
   which is what the Cell2Map paper uses when no cell counts are known.

All sections of a donor are mapped together, so the argmax also picks the
section (as Tangram did in the thesis). Each section is shifted along x so
that the spatial graph never links cells from different sections; the
original coordinates are kept in ``obsm['spatial_orig']``.
"""

from __future__ import annotations

import logging
import sys
import time
import types
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[3]
CELL2MAP_DIR = REPO / "external" / "Cell2Map"
SECTION_GAP_UM = 10_000.0


def _cell2map_modules():
    import builtins

    import torch.nn as nn
    import torch.nn.functional as F

    for unused in ("Celloc", "datatable"):  # imported upstream, never called here
        sys.modules.setdefault(unused, types.ModuleType(unused))
    if str(CELL2MAP_DIR) not in sys.path:
        sys.path.insert(0, str(CELL2MAP_DIR))
    # Upstream loss.py uses nn and F without importing them; make them
    # visible only while the package is imported.
    added = [k for k in ("nn", "F") if not hasattr(builtins, k)]
    builtins.nn, builtins.F = nn, F
    try:
        from Cell2Map import autoencoder as au
        from Cell2Map import map_optimizer as mo
        from Cell2Map import utils as ut
    finally:
        for k in added:
            delattr(builtins, k)
    au.knn_graph = _knn_graph
    return au, mo, ut


def _knn_graph(x, k: int, loop: bool = False, flow: str = "source_to_target", **_):
    """Drop-in for torch_geometric's knn_graph, which needs the compiled
    pyg-lib package. Same edge_index layout, computed with scikit-learn."""
    import torch
    from sklearn.neighbors import NearestNeighbors

    X = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    n = len(X)
    kk = min(n, k + (0 if loop else 1))
    idx = NearestNeighbors(n_neighbors=kk).fit(X).kneighbors(X, return_distance=False)
    centre = np.repeat(np.arange(n), kk)
    nbr = idx.ravel()
    if not loop:
        keep = centre != nbr
        centre, nbr = centre[keep], nbr[keep]
    edge = (centre, nbr) if flow == "target_to_source" else (nbr, centre)
    return torch.as_tensor(np.vstack(edge), dtype=torch.long)


def stack_sections(sp_a: ad.AnnData, section_col: str = "Section") -> ad.AnnData:
    """Shift each section along x so no two sections overlap."""
    sp_a = sp_a.copy()
    xy = np.asarray(sp_a.obsm["spatial"], dtype=np.float64)
    sp_a.obsm["spatial_orig"] = xy.copy()
    sections = sorted(sp_a.obs[section_col].unique())
    width = (xy[:, 0].max() - xy[:, 0].min()) + SECTION_GAP_UM
    shifted = xy.copy()
    for i, s in enumerate(sections):
        m = (sp_a.obs[section_col] == s).to_numpy()
        shifted[m, 0] += i * width
    sp_a.obsm["spatial"] = shifted
    return sp_a


def preprocess(sc_a: ad.AnnData, sp_a: ad.AnnData) -> tuple[ad.AnnData, ad.AnnData]:
    """Cell2Map's own preprocessing (``process_adatas``) on raw counts.

    Shared genes, CPM + log1p on both sides, PCA on the snRNA side (used for
    its expression graph). PCA components are capped for small inputs.
    """
    sc_a, sp_a = sc_a.copy(), sp_a.copy()
    sc.pp.filter_genes(sc_a, min_cells=1)
    sc.pp.filter_genes(sp_a, min_cells=1)
    genes = [g for g in sc_a.var_names if g in set(sp_a.var_names)]
    sc_a, sp_a = sc_a[:, genes].copy(), sp_a[:, genes].copy()
    for a in (sc_a, sp_a):
        sc.pp.normalize_total(a, target_sum=1e6)
        sc.pp.log1p(a)
    n_comps = int(min(50, sc_a.n_obs - 1, len(genes) - 1))
    sc.tl.pca(sc_a, n_comps=n_comps)
    return sc_a, sp_a


def run_cell2map(sc_a: ad.AnnData, sp_a: ad.AnnData, *, k_sc: int = 7,
                 sp_graph: str = "knn", k_sp: int = 7, radius_um: float = 50.0,
                 epochs: int = 1500, lr: float = 0.005,
                 device: str = "cuda") -> np.ndarray:
    """Mapping matrix (nuclei x MERFISH cells) from Cell2Map.

    ``sc_a`` and ``sp_a`` hold raw counts on the same genes; ``sp_a`` has
    stacked coordinates in ``obsm['spatial']``.
    """
    au, mo, ut = _cell2map_modules()
    sc_a, sp_a = preprocess(sc_a, sp_a)
    sc_a.obsm.pop("spatial", None)  # snRNA graph must come from expression

    original_radius = au.Cal_Spatial_Net_Radius
    if sp_graph == "knn":
        au.Cal_Spatial_Net_Radius = lambda adata, radius_cutoff, **kw: au.Cal_Spatial_Net(
            adata, k_cutoff=k_sp, **kw)
    else:
        au.Cal_Spatial_Net_Radius = lambda adata, radius_cutoff, **kw: original_radius(
            adata, radius_cutoff=radius_um, **kw)
    try:
        sc_a, sp_a = au.embedding_feature(sc_a, sp_a, k_cutoff=k_sc)
    finally:
        au.Cal_Spatial_Net_Radius = original_radius

    # map_cell_to_space, with a uniform density over MERFISH cells
    S = sc_a.obsm["embedding"]
    G = sp_a.obsm["embedding"]
    a = np.ones(S.shape[0])
    b = np.ones(G.shape[0]) / S.shape[0]
    d = np.full(G.shape[0], 1.0 / G.shape[0])
    sc_X = ut.to_dense_array(sc_a.X) + 0.01
    sp_X = ut.to_dense_array(sp_a.X) + 0.01
    D = ut.kl_divergence_backend(sc_X, sp_X) + ut.pcc_distances(S, G)
    mapper = mo.Mapper(S=S, G=G, d=d, a=a, b=b, D=D, device=device,
                       lambda_d=0.8, lambda_g1=1.5, lambda_g2=0.8, alpha=2,
                       lambda_mahalanobis=0.7, lambda_distance=0.01)
    # Upstream inverts the embedding covariance with torch.inverse, which fails
    # when it is singular (a handful of cells, e.g. small hold-out sets); fall
    # back to the pseudo-inverse in that case only.
    import torch

    original_inverse = torch.inverse

    def safe_inverse(A, *a, **kw):
        try:
            return original_inverse(A, *a, **kw)
        except torch._C._LinAlgError:
            return torch.linalg.pinv(A)

    torch.inverse = safe_inverse
    try:
        return np.asarray(mapper.train(num_epochs=epochs, learning_rate=lr, print_each=None))
    finally:
        torch.inverse = original_inverse


def run_tangram(sc_a: ad.AnnData, sp_a: ad.AnnData, *, epochs: int = 1000,
                seed: int = 42, device: str = "cuda") -> np.ndarray:
    """Mapping matrix from Tangram, set up as in the thesis (cells mode,
    uniform density), on the same CPM + log1p input as Cell2Map."""
    import tangram as tg

    sc_a, sp_a = preprocess(sc_a, sp_a)
    tg.pp_adatas(sc_a, sp_a, genes=list(sc_a.var_names))
    amap = tg.map_cells_to_space(sc_a, sp_a, mode="cells", density_prior="uniform",
                                 num_epochs=epochs, device=device,
                                 random_state=seed, verbose=False)
    M = amap.X
    return M.toarray() if sp.issparse(M) else np.asarray(M)


def run_method(method: str, sc_a: ad.AnnData, sp_a: ad.AnnData, *, k_sp: int = 7,
               epochs: int = 1500, epochs_tangram: int = 1000, seed: int = 42,
               device: str = "cuda") -> tuple[np.ndarray, float]:
    """Run one method; ``epochs`` is Cell2Map's, ``epochs_tangram`` Tangram's."""
    t0 = time.time()
    if method == "cell2map":
        M = run_cell2map(sc_a, sp_a, k_sp=k_sp, epochs=epochs, device=device)
    elif method == "tangram":
        M = run_tangram(sc_a, sp_a, epochs=epochs_tangram, seed=seed, device=device)
    elif method == "random":
        # baseline: each cell goes to a random MERFISH cell of its subclass
        M = np.random.default_rng(seed).random((sc_a.n_obs, sp_a.n_obs), dtype=np.float32)
    else:
        raise ValueError(method)
    return M, time.time() - t0


def readout(M: np.ndarray, sc_a: ad.AnnData, sp_a: ad.AnnData,
            carry: list[str]) -> pd.DataFrame:
    """Argmax readout: each nucleus takes its top MERFISH cell's position."""
    idx = M.argmax(axis=1)
    xy = np.asarray(sp_a.obsm.get("spatial_orig", sp_a.obsm["spatial"]))[idx]
    out = pd.DataFrame({
        "x": xy[:, 0],
        "y": xy[:, 1],
        "prob_max": M[np.arange(len(idx)), idx],
        "matched_merfish_cell": sp_a.obs_names[idx].to_numpy(),
    }, index=sc_a.obs_names)
    for col in carry:
        if col in sp_a.obs:
            out["mapped_" + col.lower().replace(" ", "_")] = sp_a.obs[col].to_numpy()[idx]
    return out
