"""Join the per-donor mapping CSVs into one table per method.

Writes <maps>/<method>_all_donors.csv and prints nuclei and sections per donor.

Usage:
    python scripts/cell2map/merge_maps.py --maps output/cell2map/maps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--maps", type=Path, default=Path("output/cell2map/maps"))
    p.add_argument("--methods", nargs="+", default=["cell2map"])
    args = p.parse_args()
    for method in args.methods:
        files = sorted(f for f in (args.maps / method).glob("*.csv") if not f.stem.endswith("_runs"))
        df = pd.concat([pd.read_csv(f, index_col=0) for f in files])
        df.to_csv(args.maps / f"{method}_all_donors.csv")
        per = df.groupby("Donor ID").agg(nuclei=("x", "size"), sections=("mapped_section", "nunique"))
        print(f"{method}: {len(df):,} nuclei from {len(files)} donors")
        print(per.to_string())


if __name__ == "__main__":
    main()
