"""Score every prediction file against each target cell's true GRN.

Finds <out>/**/preds_*.npz, scores each target cell (see common.score) and
writes <out>/scores_cells.csv and <out>/scores_summary.csv (mean and SD over
target cells, plus the share of cells where the method beats SVGRN stage 2).

Usage:
    python scripts/svgrn_eval/score.py --data <dir>/g110_c2k_n01 --out <out>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

REF = "published/stage2"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    _, _, genes, tfs = c.load(args.data)
    mask = c.eval_mask(genes, tfs)
    rows, truths = [], {}
    for f in sorted(args.out.rglob("preds_*.npz")):
        method = f"{f.parent.relative_to(args.out)}/{f.stem.removeprefix('preds_')}"
        for cell, pred in c.load_preds(f).items():
            if cell not in truths:
                truths[cell] = c.truth(args.data, cell, genes)
            s = c.score(pred, truths[cell], mask)
            if s:
                rows.append({"method": method, "cell": cell, **s})
    cells = pd.DataFrame(rows)
    cells.to_csv(args.out / "scores_cells.csv", index=False)

    metrics = ["auprc_ratio", "auroc", "epr", "auprc_signed"]
    summ = cells.groupby("method")[metrics].agg(["mean", "std"])
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ.insert(0, "n_cells", cells.groupby("method").size())
    if REF in set(cells["method"]):
        # paired over the same target cells: method - SVGRN published (stage 2)
        ref = cells[cells["method"] == REF].set_index("cell")["auprc_ratio"]
        wins, diff, tval = {}, {}, {}
        for m, g in cells.groupby("method"):
            d = (g.set_index("cell")["auprc_ratio"].reindex(ref.index) - ref).dropna()
            wins[m] = 100 * (d > 0).mean()
            diff[m] = d.mean()
            se = d.std(ddof=1) / len(d) ** 0.5 if len(d) > 1 else float("nan")
            tval[m] = d.mean() / se if se and se > 0 else float("nan")
        tag = REF.replace("/", "_")
        summ[f"pct_cells_beating_{tag}"] = pd.Series(wins)
        summ[f"paired_diff_vs_{tag}"] = pd.Series(diff)
        summ[f"paired_t_vs_{tag}"] = pd.Series(tval)
    summ = summ.sort_values("auprc_ratio_mean", ascending=False)
    summ.to_csv(args.out / "scores_summary.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
