"""Hold-out validation of a mapping method on one donor's MERFISH cells.

Same design as the thesis (Tangram, 06_validate.py): hold out 5% of a
subclass's MERFISH cells, map them as if they were nuclei onto the rest,
and measure how far the argmax lands from their true position.

Two scopes:
    section  reference = the rest of the same section (the thesis setup;
             gives localisation error within the right frame)
    donor    reference = all of the donor's sections stacked (as in the real
             mapping; also measures how often the right section is picked)

Errors are in micrometres. Writes <out>/<donor>_cells.csv (one row per
held-out cell) and <out>/<donor>_summary.csv.

Note: held-out MERFISH cells are the same modality as the reference, so this
is an easier task than mapping real snRNA nuclei. It ranks methods; it does
not give the true error for nuclei.

Usage:
    python scripts/cell2map/validate_holdout.py --donor H20.33.025 --methods cell2map tangram
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from screni.cell2map import data as d
from screni.cell2map import mapping as mp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def split(obs: pd.DataFrame, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean held-out mask, stratified by section."""
    held = np.zeros(len(obs), bool)
    for s in obs[d.SECTION].unique():
        idx = np.flatnonzero((obs[d.SECTION] == s).to_numpy())
        k = max(1, int(round(frac * len(idx))))
        held[rng.choice(idx, k, replace=False)] = True
    return held


def evaluate(method: str, query: ad.AnnData, ref: ad.AnnData, args, device: str) -> pd.DataFrame:
    q = query.copy()
    true_xy = np.asarray(q.obsm["spatial"]).copy()
    true_sec = q.obs[d.SECTION].to_numpy()
    del q.obsm["spatial"]
    ref = mp.stack_sections(ref)
    M, secs = mp.run_method(method, q, ref, k_sp=args.k_sp, epochs=args.epochs_cell2map,
                            epochs_tangram=args.epochs_tangram, seed=args.seed, device=device)
    out = mp.readout(M, q, ref, [d.SECTION])
    hit = out["mapped_section"].to_numpy() == true_sec
    err = np.linalg.norm(out[["x", "y"]].to_numpy() - true_xy, axis=1)
    return pd.DataFrame({"true_section": true_sec, "section_correct": hit,
                         "error_um": np.where(hit, err, np.nan), "seconds_total": secs},
                        index=q.obs_names)


def summarise(cells: pd.DataFrame) -> pd.DataFrame:
    def f(g):
        e = g["error_um"].dropna()
        return pd.Series({
            "n_cells": len(g),
            "section_acc_pct": 100 * g["section_correct"].mean(),
            "median_um": e.median(),
            "mean_um": e.mean(),
            "within_500_pct": 100 * (e <= 500).mean(),
            "within_1000_pct": 100 * (e <= 1000).mean(),
        })
    by_sub = cells.groupby(["method", "scope", "subclass"]).apply(f).reset_index()
    overall = cells.groupby(["method", "scope"]).apply(f).reset_index()
    overall["subclass"] = "ALL"
    return pd.concat([overall, by_sub], ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--donor", required=True)
    p.add_argument("--inputs", type=Path, default=Path("data/cell2map"))
    p.add_argument("--out", type=Path, default=Path("output/cell2map/validation"))
    p.add_argument("--methods", nargs="+", default=["cell2map", "tangram"])
    p.add_argument("--scopes", nargs="+", default=["section", "donor"], choices=["section", "donor"])
    p.add_argument("--subclasses", nargs="*", default=None)
    p.add_argument("--holdout", type=float, default=0.05)
    p.add_argument("--min-cells", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs-cell2map", type=int, default=1500)
    p.add_argument("--epochs-tangram", type=int, default=1000)
    p.add_argument("--k-sp", type=int, default=7)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mer = ad.read_h5ad(args.inputs / "merfish" / f"{args.donor}.h5ad")
    subclasses = args.subclasses or sorted(mer.obs[d.SUBCLASS].unique())
    rng = np.random.default_rng(args.seed)
    frames = []
    for sub in subclasses:
        m = mer[mer.obs[d.SUBCLASS] == sub].copy()
        if m.n_obs < args.min_cells:
            continue
        held = split(m.obs, args.holdout, rng)
        jobs = []
        if "donor" in args.scopes:
            jobs.append(("donor", m[held], m[~held]))
        if "section" in args.scopes:
            for s in m.obs[d.SECTION].unique():
                ins = (m.obs[d.SECTION] == s).to_numpy()
                if (ins & ~held).sum() >= args.min_cells:
                    jobs.append(("section", m[held & ins], m[~held & ins]))
        for method in args.methods:
            for scope, q, ref in jobs:
                r = evaluate(method, q, ref.copy(), args, device)
                r["method"], r["scope"], r["subclass"] = method, scope, sub
                frames.append(r)
            logger.info("  %-22s %-9s done (%d held out)", sub, method, int(held.sum()))
    cells = pd.concat(frames)
    cells.index.name = "merfish_cell"
    cells.to_csv(args.out / f"{args.donor}_cells.csv")
    summary = summarise(cells)
    summary.to_csv(args.out / f"{args.donor}_summary.csv", index=False)
    with pd.option_context("display.width", 160, "display.max_rows", 20):
        print(summary[summary["subclass"] == "ALL"].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
