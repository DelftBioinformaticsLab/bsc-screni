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
DEPTH = "Depth from pia"


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
    out = mp.readout(M, q, ref, [d.SECTION, d.SUBCLASS, DEPTH])
    hit = out["mapped_section"].to_numpy() == true_sec
    err = np.linalg.norm(out[["x", "y"]].to_numpy() - true_xy, axis=1)
    # depth part of the error (distance from pia, µm) and the rest, which runs
    # along the cortical sheet: err^2 = depth^2 + along^2 (approximately)
    derr = np.abs(out["mapped_" + DEPTH.lower().replace(" ", "_")].to_numpy(float)
                  - q.obs[DEPTH].to_numpy(float))
    along = np.sqrt(np.clip(err ** 2 - derr ** 2, 0, None))
    return pd.DataFrame({"true_section": true_sec, "section_correct": hit,
                         "error_um": np.where(hit, err, np.nan),
                         "depth_error_um": np.where(hit, derr, np.nan),
                         "along_error_um": np.where(hit, along, np.nan),
                         "subclass_correct": out["mapped_subclass"].to_numpy() == q.obs[d.SUBCLASS].to_numpy(),
                         "seconds_total": secs,
                         "true_x": true_xy[:, 0], "true_y": true_xy[:, 1],
                         "mapped_x": out["x"].to_numpy(), "mapped_y": out["y"].to_numpy(),
                         "n_ref": ref.n_obs},
                        index=q.obs_names)


def _pearson_within_section(g: pd.DataFrame, axis: str) -> float:
    """Pearson r between true and mapped coordinate, after centring both on the
    true section's mean, over cells mapped to the right section. Centring keeps
    the section offsets from inflating r."""
    h = g[g["section_correct"]]
    if len(h) < 3:
        return np.nan
    mu = h.groupby("true_section")[f"true_{axis}"].transform("mean")
    t, m = h[f"true_{axis}"] - mu, h[f"mapped_{axis}"] - mu
    return float(np.corrcoef(t, m)[0, 1]) if t.std() > 0 and m.std() > 0 else np.nan


def summarise(cells: pd.DataFrame) -> pd.DataFrame:
    def f(g):
        e = g["error_um"].dropna()
        rx, ry = _pearson_within_section(g, "x"), _pearson_within_section(g, "y")
        return pd.Series({
            "n_cells": len(g),
            "section_acc_pct": 100 * g["section_correct"].mean(),
            "median_um": e.median(),
            "mean_um": e.mean(),
            "within_500_pct": 100 * (e <= 500).mean(),
            "within_1000_pct": 100 * (e <= 1000).mean(),
            "median_depth_um": g["depth_error_um"].median() if "depth_error_um" in g else np.nan,
            "median_along_um": g["along_error_um"].median() if "along_error_um" in g else np.nan,
            "subclass_acc_pct": 100 * g["subclass_correct"].mean() if "subclass_correct" in g else np.nan,
            "pearson_x": rx, "pearson_y": ry,
            "r2_x": rx ** 2, "r2_y": ry ** 2,
        })
    by_sub = cells.groupby(["method", "scope", "subclass"])[cells.columns.drop(["method", "scope", "subclass"]).tolist()].apply(f).reset_index()
    overall = cells.groupby(["method", "scope"])[cells.columns.drop(["method", "scope"]).tolist()].apply(f).reset_index()
    overall["subclass"] = "ALL"
    return pd.concat([overall, by_sub], ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--donor")
    g.add_argument("--donor-index", type=int, help="Line in <inputs>/donors.txt (0-based)")
    p.add_argument("--inputs", type=Path, default=Path("data/cell2map"))
    p.add_argument("--out", type=Path, default=Path("output/cell2map/validation"))
    p.add_argument("--methods", nargs="+", default=["cell2map", "tangram", "random"],
                   choices=["cell2map", "tangram", "random", "nearest"],
                   help="random = a random reference cell; nearest = the reference cell "
                        "with the most correlated expression (no optimisation)")
    p.add_argument("--match", choices=["subclass", "none"], default="subclass",
                   help="subclass: map each subclass onto its own cells (the real setup); "
                        "none: map all cell types together (as in the Cell2Map benchmarks)")
    p.add_argument("--scopes", nargs="+", default=["section", "donor"], choices=["section", "donor"])
    p.add_argument("--subclasses", nargs="*", default=None)
    p.add_argument("--holdout", type=float, default=0.05)
    p.add_argument("--min-cells", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs-cell2map", type=int, default=1500)
    p.add_argument("--epochs-tangram", type=int, default=1000)
    p.add_argument("--k-sp", type=int, default=7)
    args = p.parse_args()
    args.donor = args.donor or (args.inputs / "donors.txt").read_text().split()[args.donor_index]
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # held-out sets depend only on --seed and the subclass order, so runs with
    # different --methods evaluate the same cells and can be compared

    mer = ad.read_h5ad(args.inputs / "merfish" / f"{args.donor}.h5ad")
    group_col = d.SUBCLASS
    if args.match == "none":
        mer.obs["_group"] = "all types"
        group_col = "_group"
    subclasses = args.subclasses or sorted(mer.obs[group_col].unique())
    rng = np.random.default_rng(args.seed)
    frames = []
    for sub in subclasses:
        m = mer[mer.obs[group_col] == sub].copy()
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
    print("Cell2Map covariance-inverse fallbacks (epochs):", mp.INVERSE_FALLBACKS)


if __name__ == "__main__":
    main()
