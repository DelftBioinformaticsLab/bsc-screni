"""Score SVGRN stage-2 output with SVGRN's own eval.py (unmodified), per cell.

Uses external/SVGRN/eval.py's get_evaluation_matrix, get_truth_edges,
evaluate (EPR) and get_AUPRC_AUROC (AUPRC ratio on signed weights), the
functions behind the paper's Fig. 3, on <run>/cells/<cell>/RN_<epochs>.csv
(transposed to sender rows, as eval_scGRN_SVGRN does). Also scores the
same networks on |weights| with common.score for comparison.

Writes <run>/paper_eval_cells.csv and prints the means.

Usage:
    python scripts/svgrn_eval/eval_paper.py --data <dir>/g110_c2k_n01 \
        --run <out>/paper_w4 --targets <out>/targets_paper.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as c  # noqa: E402

spec = importlib.util.spec_from_file_location("svgrn_eval_py", HERE.parents[1] / "external" / "SVGRN" / "eval.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=150)
    args = p.parse_args()

    _, _, genes, tfs = c.load(args.data)
    mask_paper = ev.get_evaluation_matrix(tfs, genes)
    mask = c.eval_mask(genes, tfs)
    rows = []
    for cell in args.targets.read_text().split():
        f = args.run / "cells" / cell / f"RN_{args.epochs}.csv"
        if not f.exists():
            rows.append({"cell": cell, "missing": True})
            continue
        A = pd.read_csv(f).to_numpy().T  # sender rows, as eval_scGRN_SVGRN
        truth = ev.get_truth_edges(str(args.data / "cell_specific_GRN" / f"{cell}.csv"))
        tp, ep, epr = ev.evaluate(A, truth, Evaluate_Mask=mask_paper, cutoff_factor=1)
        auprc, auprc_ratio, auroc = ev.get_AUPRC_AUROC(A, truth, Evaluate_Mask=mask_paper)
        ours = c.score(A, c.truth(args.data, cell, genes), mask)
        rows.append({"cell": cell, "missing": False, "paper_epr": epr, "paper_auprc_ratio": auprc_ratio,
                     "paper_auroc": auroc, "abs_auprc_ratio": ours["auprc_ratio"], "abs_epr": ours["epr"]})
    df = pd.DataFrame(rows)
    df.to_csv(args.run / "paper_eval_cells.csv", index=False)
    ok = df[~df["missing"]]
    print(f"{len(ok)} of {len(df)} cells scored ({int(df['missing'].sum())} missing)")
    for col in ["paper_epr", "paper_auprc_ratio", "paper_auroc", "abs_auprc_ratio", "abs_epr"]:
        v = ok[col].astype(float)
        print(f"{col:18s} mean {v.mean():.4f}  SD {v.std():.4f}  SE {v.std() / np.sqrt(len(v)):.4f}")


if __name__ == "__main__":
    main()
