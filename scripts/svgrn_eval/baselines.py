"""Simple baselines for the SVGRN evaluation, per target cell.

    random          random weights
    pearson_global  Pearson correlation over all cells (same network for every cell)
    pearson_local   Pearson correlation weighted by the same RBF kernel SVGRN's
                    stage 2 uses (sigma = W = 6 in the raw coordinates), so it
                    sees the same neighbourhood as SVGRN
    target_mean     no gene-gene information at all: every TF -> gene edge
    target_var      scores the target gene's mean (or variance) over all cells.
                    Survives shuffling each gene across cells, so it tests
                    whether a score comes from per-gene properties alone.

All are scored on the same TF -> gene pairs as SVGRN, so they get the same TF
prior. Writes <out>/baselines/preds_<method>.npz.

Usage:
    python scripts/svgrn_eval/baselines.py --data <dir>/g110_c2k_n01 \
        --targets <out>/targets.txt --out <out>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402


def weighted_corr(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = w / w.sum()
    mu = w @ X
    Xc = X - mu
    cov = (Xc * w[:, None]).T @ Xc
    sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    R = cov / np.outer(sd, sd)
    np.fill_diagonal(R, 0)
    return R


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sigma", type=float, default=6.0, help="RBF bandwidth, as SVGRN's W")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    X, meta, genes, _ = c.load(args.data)
    X = X.to_numpy(float)
    xy = meta[["x", "y"]].to_numpy(float)
    targets = args.targets.read_text().split()
    pos = {n: i for i, n in enumerate(meta.index)}
    rng = np.random.default_rng(args.seed)
    g = weighted_corr(X, np.ones(len(X)))

    preds = {"random": {}, "pearson_global": {}, "pearson_local": {},
             "target_mean": {}, "target_var": {}}
    n = len(genes)
    by_mean = np.tile(X.mean(0), (n, 1))  # row = sender, column = target
    by_var = np.tile(X.var(0), (n, 1))
    for t in targets:
        d2 = ((xy - xy[pos[t]]) ** 2).sum(1)
        w = np.exp(-d2 / (2 * args.sigma ** 2))
        preds["random"][t] = rng.random((len(genes), len(genes)))
        preds["pearson_global"][t] = g
        preds["pearson_local"][t] = weighted_corr(X, w)
        preds["target_mean"][t] = by_mean
        preds["target_var"][t] = by_var
    for m, d in preds.items():
        c.save_preds(args.out / "baselines" / f"preds_{m}.npz", d)
    print(f"baselines for {len(targets)} targets written to {args.out / 'baselines'}")


if __name__ == "__main__":
    main()
