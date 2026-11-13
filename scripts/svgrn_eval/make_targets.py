"""Pick the target cells for one dataset (stratified by GRN cluster).

Usage:
    python scripts/svgrn_eval/make_targets.py --data <dir>/g110_c2k_n01 --n 30 --out <out>/targets.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--data", type=Path, required=True)
p.add_argument("--n", type=int, default=30)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
_, meta, _, _ = c.load(args.data)
targets = c.sample_targets(meta, args.n, args.seed)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text("\n".join(targets) + "\n")
print(f"{len(targets)} targets -> {args.out}")
