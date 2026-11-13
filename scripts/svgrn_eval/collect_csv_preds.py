"""Collect per-cell network CSVs (sender rows x receiver columns, gene names as
labels) into one preds_<name>.npz in the gene order of the dataset.

Usage:
    python scripts/svgrn_eval/collect_csv_preds.py --data <dir>/g110_c2k_n01 \
        --cells <out>/kscreni/cells --out <out>/kscreni/preds_kscreni.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--data", type=Path, required=True)
p.add_argument("--cells", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
_, _, genes, _ = c.load(args.data)
preds = {}
for f in sorted(args.cells.glob("*.csv")):
    m = pd.read_csv(f, index_col=0)
    m.index, m.columns = m.index.astype(str), m.columns.astype(str)
    preds[f.stem] = m.loc[genes, genes].to_numpy(float)
c.save_preds(args.out, preds)
print(f"{len(preds)} networks -> {args.out}")
