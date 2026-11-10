"""Load SEA-AD MERFISH and snRNA-seq data for donor-matched mapping.

MERFISH: 1.89M cells x 180 features (140 genes + 40 ``Blank-*`` control
probes), 27 donors, 69 sections. Every section has its own coordinate frame.

snRNA: Allen's ``SEAAD_MTG_RNAseq_final-nuclei`` file (1.38M nuclei x 36,601
genes, 89 donors). It is ~34 GB, so it is read in row chunks with h5py and
only the MERFISH donors and the 140 MERFISH genes are kept. Raw counts come
from ``layers/UMIs``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

SHARED = Path("/tudelft.net/staff-umbrella/ScReNI/bsc-screni/data")
MERFISH_PATH = SHARED / "seaad_spatial" / "SEAAD_MTG_MERFISH.2024-12-11.h5ad"
RNA_PATH = SHARED / "seaad" / "SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad"

DONOR = "Donor ID"
SUBCLASS = "Subclass"
SECTION = "Section"
COORDS_KEY = "spatial"

# obs columns carried from MERFISH onto each mapped nucleus
MERFISH_CARRY = ["Section", "Supertype", "Depth from pia", "Normalized depth from pia",
                 "Layer annotation"]


def _read_obs(f: h5py.File) -> pd.DataFrame:
    """Read obs from an open h5ad, including the pre-0.8 categorical layout."""
    from anndata.io import read_elem

    g = f["obs"]
    if "encoding-type" in g.attrs:
        return read_elem(g)
    # Legacy layout: obs is a compound/structured dataset or a group of
    # code arrays with categories in obs/__categories/<col>.
    if isinstance(g, h5py.Dataset):
        df = pd.DataFrame(g[:])
    else:
        df = pd.DataFrame({k: g[k][:] for k in g.keys() if k != "__categories"})
    idx_col = g.attrs.get("_index", "index")
    if isinstance(idx_col, bytes):
        idx_col = idx_col.decode()
    cats = g["__categories"] if isinstance(g, h5py.Group) and "__categories" in g else (
        f["uns"]["__categories"] if "__categories" in f.get("uns", {}) else None)
    if cats is not None:
        for col in cats.keys():
            if col in df:
                labels = [c.decode() if isinstance(c, bytes) else c for c in cats[col][:]]
                df[col] = pd.Categorical.from_codes(df[col].astype(int), labels)
    for col in df.columns:
        if df[col].dtype == object and len(df) and isinstance(df[col].iloc[0], bytes):
            df[col] = df[col].str.decode("utf-8")
    if idx_col in df:
        df = df.set_index(idx_col)
    df.index = df.index.astype(str)
    return df


def _var_names(f: h5py.File) -> list[str]:
    g = f["var"]
    if isinstance(g, h5py.Dataset):
        names = g[:][g.attrs.get("_index", "index")]
    else:
        names = g[g.attrs.get("_index", "_index")][:]
    return [n.decode() if isinstance(n, bytes) else str(n) for n in names]


def merfish_genes(path: Path = MERFISH_PATH) -> list[str]:
    """The 140 measured MERFISH genes (Blank-* control probes dropped)."""
    with h5py.File(path, "r") as f:
        return [g for g in _var_names(f) if not g.startswith("Blank")]


def load_merfish(path: Path = MERFISH_PATH, donors: list[str] | None = None) -> ad.AnnData:
    """MERFISH cells x 140 genes, optionally restricted to some donors.

    X is returned as stored (see ``check_merfish_scale``); coordinates are in
    ``obsm['spatial']`` (one frame per section).
    """
    with h5py.File(path, "r") as f:
        obs = _read_obs(f)
        names = _var_names(f)
        keep_g = np.array([not g.startswith("Blank") for g in names])
        rows = np.ones(len(obs), bool)
        if donors is not None:
            rows = obs[DONOR].astype(str).isin(donors).to_numpy()
        ridx = np.flatnonzero(rows)
        X = np.empty((len(ridx), keep_g.sum()), dtype=np.float32)
        # read in contiguous blocks to keep memory flat
        for start in range(0, len(ridx), 200_000):
            sel = ridx[start:start + 200_000]
            block = f["X"][sel.min():sel.max() + 1]
            X[start:start + len(sel)] = block[sel - sel.min()][:, keep_g]
        coords = f["obsm"][COORDS_KEY][:][ridx]
    a = ad.AnnData(X=X, obs=obs.iloc[ridx].copy(),
                   var=pd.DataFrame(index=[g for g, k in zip(names, keep_g) if k]))
    a.obsm["spatial"] = coords.astype(np.float64)
    for col in (DONOR, SUBCLASS, SECTION):
        a.obs[col] = a.obs[col].astype(str)
    return a


def check_merfish_scale(a: ad.AnnData) -> str:
    """Return 'log' if X looks log-transformed, else 'linear'.

    The file's uns says ln(spots per 10K + 1), but on 2026-11-10 the values
    reached 268 (ln-scale values stay below ~10) and were not all whole
    numbers, so X is linear and probably volume-scaled. Cell2Map applies CPM
    + log1p itself, so linear input is what it needs; log input would be
    expm1-transformed first.
    """
    return "log" if float(np.max(a.X)) < 20 else "linear"


def load_rna(donors: list[str], genes: list[str], path: Path = RNA_PATH,
             layer: str = "UMIs", chunk: int = 50_000) -> ad.AnnData:
    """snRNA nuclei from ``donors`` x ``genes``, raw counts from ``layer``.

    Streams the CSR matrix in row chunks so the 34 GB file never sits in
    memory. Keeps every assay (10Xv3, 10Xv3.1 and the 10xMulti nuclei).
    """
    with h5py.File(path, "r") as f:
        obs = _read_obs(f)
        names = _var_names(f)
        col_of = {g: i for i, g in enumerate(names)}
        missing = [g for g in genes if g not in col_of]
        if missing:
            raise KeyError(f"{len(missing)} genes not in snRNA: {missing[:10]}")
        cols = np.array([col_of[g] for g in genes])
        keep = obs[DONOR].astype(str).isin(donors).to_numpy()
        g = f["layers"][layer] if layer else f["X"]
        indptr = g["indptr"][:]
        n_vars = len(names)
        parts = []
        for r0 in range(0, len(obs), chunk):
            r1 = min(r0 + chunk, len(obs))
            m = keep[r0:r1]
            if not m.any():
                continue
            s, e = indptr[r0], indptr[r1]
            block = sp.csr_matrix(
                (g["data"][s:e], g["indices"][s:e], indptr[r0:r1 + 1] - s),
                shape=(r1 - r0, n_vars))
            parts.append(block[m][:, cols])
            logger.info("  rows %d-%d: kept %d nuclei", r0, r1, int(m.sum()))
    X = sp.vstack(parts).tocsr().astype(np.float32)
    a = ad.AnnData(X=X, obs=obs[keep].copy(), var=pd.DataFrame(index=genes))
    for col in (DONOR, SUBCLASS):
        a.obs[col] = a.obs[col].astype(str)
    return a
