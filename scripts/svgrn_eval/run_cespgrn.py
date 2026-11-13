"""CeSpGRN (external/CeSpGRN) baselines on one simulated dataset, per target cell.

Two versions:
    svgrn_setup  as in SVGRN's benchmark/cespgrn/run_cespgrn.py: spatial
                 Gaussian kernel, bandwidth 0.1, lamb 0.1, 30 neighbours, no
                 prior, one setting. SVGRN's script calls
                 kernel.calc_kernel_distance, which is not in any CeSpGRN
                 release; we use a Gaussian kernel on the Euclidean cell
                 distances with CeSpGRN's own bandwidth rule (calc_kernel).
    tf_avg       CeSpGRN's TF prior (TF=..., edges must touch a TF) and the
                 average |network| over 8 settings: bandwidth {0.1, 1} x
                 lamb {0.01, 0.1} x neighbours {30, 100}, as the CeSpGRN
                 paper averages over settings instead of picking one.

Only the target cells are solved (CeSpGRN's cells are independent when its
smoothing term beta = 0, the default), so the empirical covariances are built
for the targets only, with CeSpGRN's own weighted Kendall tau.

Writes <out>/cespgrn/preds_{svgrn_setup,tf_avg}.npz (sender x receiver,
edge weight = CeSpGRN's partial correlation, the matrices train() returns).
CeSpGRN's PSD helpers use torch.eig, removed in torch 2, so the PSD check and
correction are redone here with torch.linalg.eigh (same rule: eigenvalues
<= 0 set to 1e-3).

Usage:
    python scripts/svgrn_eval/run_cespgrn.py --data <dir>/g110_c2k_n01 \
        --targets <out>/targets.txt --out <out>
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external" / "CeSpGRN" / "src"))
import g_admm as CeSpGRN  # noqa: E402


def preprocess(counts: np.ndarray) -> np.ndarray:
    # as in SVGRN's run_cespgrn.py and CeSpGRN's demo
    libsize = np.median(counts.sum(1))
    return np.log1p(counts / counts.sum(1)[:, None] * libsize)


def kernel_rows(xy: np.ndarray, rows: np.ndarray, bandwidth: float, n_neigh: int) -> np.ndarray:
    """Truncated, row-normalised Gaussian kernel for the target rows
    (CeSpGRN calc_kernel's bandwidth rule on Euclidean distances)."""
    D_all = cdist(xy, xy)
    D_all = D_all / D_all.max()
    mdis = 0.5 * bandwidth * np.median(D_all)
    K = np.exp(-(D_all[rows] ** 2) / mdis)
    kth = -np.partition(-K, n_neigh - 1, axis=1)[:, n_neigh - 1:n_neigh]
    K = K * (K >= kth)
    return K / K.sum(1, keepdims=True)


def est_cov_rows(X: np.ndarray, K_trun: np.ndarray) -> np.ndarray:
    """CeSpGRN's est_cov(weighted_kt=True), for the given kernel rows only."""
    Xt = torch.FloatTensor(X)
    n_genes = X.shape[1]
    out = np.zeros((len(K_trun), n_genes, n_genes))
    for t, row in enumerate(K_trun):
        w = torch.FloatTensor(row)
        sel = w > 0
        cov = torch.eye(n_genes)
        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                kt = CeSpGRN.weighted_kendall_tau(Xt[sel, i], Xt[sel, j], w[sel])
                cov[i, j] = cov[j, i] = torch.sin(np.pi / 2 * kt)
        S, U = torch.linalg.eigh(cov)
        if bool((S <= 1e-5).any()):
            S = torch.where(S <= 0, torch.full_like(S, 1e-3), S)
            cov = U @ torch.diag(S) @ U.T
        out[t] = cov.numpy()
    return out


def solve(cov: np.ndarray, X_rows: np.ndarray, lamb: float, tf_idx: list[int] | None) -> np.ndarray:
    n = len(cov)
    model = CeSpGRN.G_admm_minibatch(X=X_rows[:, None, :], K=np.eye(n), TF=tf_idx,
                                     pre_cov=torch.FloatTensor(cov), batchsize=min(120, n))
    return model.train(max_iters=1000, n_intervals=100, lamb=lamb)


def to_edges(G: np.ndarray) -> np.ndarray:
    E = np.asarray(G, float).copy()
    np.fill_diagonal(E, 0)
    return E


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--versions", nargs="+", default=["svgrn_setup", "tf_avg"])
    args = p.parse_args()

    _, meta, genes, tfs = c.load(args.data)
    counts = pd.read_csv(args.data / "raw_count.csv", index_col=0)
    # raw_count.csv's gene names are broken in the published data (101
    # missing, 105 twice, last six named gene105..gene110). SVGRN's own
    # CeSpGRN script reads it by position, so do the same, after checking the
    # columns line up with the expression SVGRN uses.
    if counts.shape[1] != len(genes):
        raise SystemExit(f"raw_count.csv has {counts.shape[1]} genes, expected {len(genes)}")
    counts.columns = genes
    counts = counts.loc[meta.index]
    X = preprocess(counts.to_numpy(float))
    expr, _, _, _ = c.load(args.data)
    r = [np.corrcoef(X[:, j], expr.loc[meta.index].iloc[:, j])[0, 1] for j in range(len(genes))]
    print(f"raw vs SVGRN-input expression, per-gene correlation: median {np.nanmedian(r):.3f}, min {np.nanmin(r):.3f}")
    if np.nanmedian(r) < 0.5:
        raise SystemExit("raw_count.csv columns do not line up with the expression file")
    xy = meta[["x", "y"]].to_numpy(float)
    targets = args.targets.read_text().split()
    rows = meta.index.get_indexer(targets)
    tf_idx = [genes.index(t) for t in tfs]

    settings = {"svgrn_setup": [(0.1, 0.1, 30, None)],
                "tf_avg": [(b, l, k, tf_idx) for b, l, k in product([0.1, 1.0], [0.01, 0.1], [30, 100])]}
    covs = {}
    for version in args.versions:
        acc = np.zeros((len(targets), len(genes), len(genes)))
        for bw, lamb, k, tf in settings[version]:
            t0 = time.time()
            if (bw, k) not in covs:
                covs[(bw, k)] = est_cov_rows(X, kernel_rows(xy, rows, bw, k))
            Gs = solve(covs[(bw, k)], X[rows], lamb, tf)
            acc += np.abs(np.stack([to_edges(th) for th in Gs])) if version == "tf_avg" \
                else np.stack([to_edges(th) for th in Gs])
            print(f"{version}: bandwidth {bw}, lamb {lamb}, neighbours {k}: {time.time() - t0:.0f} s", flush=True)
        acc /= len(settings[version])
        c.save_preds(args.out / "cespgrn" / f"preds_{version}.npz", dict(zip(targets, acc)))


if __name__ == "__main__":
    main()
