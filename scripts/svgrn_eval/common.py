"""Shared helpers for the SVGRN evaluation on SVGRN's own simulated data.

Each dataset directory (e.g. g110_c2k_n01, from the SVGRN README's Google
Drive link) holds:
    expression_loc_cluster_wlayout.csv  cells x (genes, x, y, ClusterID); SVGRN's input
    allcells_GRN.csv                    tissue GRN, rows = genes, columns = TFs
    cell_specific_GRN/<cell>.csv        each cell's true GRN, same layout
Gene and TF names are numbers stored as strings ("1".."110").

All predicted networks in this evaluation are stored as sender x receiver
matrices (row = TF/regulator, column = target), so they can be scored the
same way whatever the method's own orientation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INPUT_CSV = "expression_loc_cluster_wlayout.csv"
NON_GENE_COLS = ["x", "y", "ClusterID"]


def load(ds: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Return (expression cells x genes, positions with x, y, ClusterID, genes, TFs)."""
    df = pd.read_csv(ds / INPUT_CSV, index_col=0)
    df.columns = df.columns.astype(str)
    genes = [c for c in df.columns if c not in NON_GENE_COLS]
    meta = df[[c for c in NON_GENE_COLS if c in df.columns]]
    tfs = list(pd.read_csv(ds / "allcells_GRN.csv", index_col=0).columns.astype(str))
    return df[genes], meta, genes, tfs


def truth(ds: Path, cell: str, genes: list[str]) -> np.ndarray:
    """Binary sender x receiver matrix of the cell's true GRN."""
    g = pd.read_csv(ds / "cell_specific_GRN" / f"{cell}.csv", index_col=0)
    g.index, g.columns = g.index.astype(str), g.columns.astype(str)
    idx = {n: i for i, n in enumerate(genes)}
    T = np.zeros((len(genes), len(genes)), dtype=np.int8)
    rec, send = np.nonzero(g.to_numpy() != 0)
    T[[idx[g.columns[s]] for s in send], [idx[g.index[r]] for r in rec]] = 1
    np.fill_diagonal(T, 0)
    return T


def eval_mask(genes: list[str], tfs: list[str]) -> np.ndarray:
    """Pairs that are scored: TF -> any other gene (as in SVGRN's eval.py)."""
    M = np.zeros((len(genes), len(genes)), dtype=bool)
    for t in tfs:
        if t in genes:
            M[genes.index(t), :] = True
    np.fill_diagonal(M, False)
    return M


def score(pred: np.ndarray, T: np.ndarray, mask: np.ndarray) -> dict:
    """Edge-recovery scores of one predicted sender x receiver network.

    auprc / auroc rank edges by |weight| (standard). auprc_signed ranks by
    the signed weight, as SVGRN's eval.py does, which ranks strong negative
    (repressive) edges as non-edges. ep is the share of true edges among the
    top-k |weights| (k = number of true edges); epr divides by the random
    expectation, so 1 = random.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score  # only scoring needs it

    y, s = T[mask].astype(int), pred[mask].astype(float)
    a = np.abs(s)
    k = int(y.sum())
    if k == 0 or k == len(y):
        return {}
    top = np.argsort(-a, kind="stable")[:k]
    ep = y[top].mean()
    prevalence = k / len(y)
    return {
        "auprc": average_precision_score(y, a),
        "auprc_ratio": average_precision_score(y, a) / prevalence,
        "auroc": roc_auc_score(y, a),
        "auprc_signed": average_precision_score(y, s),
        "auprc_ratio_signed": average_precision_score(y, s) / prevalence,  # SVGRN eval.py's AUPRC ratio
        "ep": ep,
        "epr": ep / prevalence,
        "n_true": k,
    }


def sample_targets(meta: pd.DataFrame, n: int, seed: int = 0) -> list[str]:
    """n target cells, stratified by GRN cluster (proportional, at least 1 each)."""
    rng = np.random.default_rng(seed)
    groups = meta.groupby("ClusterID").groups
    out = []
    for cid, cells in sorted(groups.items()):
        k = max(1, round(n * len(cells) / len(meta)))
        out += list(rng.choice(np.asarray(cells), min(k, len(cells)), replace=False))
    return sorted(out[:n] if len(out) > n else out, key=lambda c: int(c.removeprefix("cell")))


def save_preds(path: Path, preds: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **preds)


def load_preds(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
