"""Combine the per-donor hold-out validation results (step 3) into one table.

Reads <dir>/<donor>_cells.csv for every donor and writes
    <dir>/all_donors_cells.csv     every held-out cell, with a donor column
    <dir>/all_donors_summary.csv   per donor and pooled ("ALL"), per method and scope
and prints the pooled table plus the spread over donors.

Usage:
    python scripts/cell2map/summarise_validation.py --dir output/cell2map/validation_all
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "validate_holdout", Path(__file__).with_name("validate_holdout.py"))
vh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vh)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", type=Path, default=Path("output/cell2map/validation_all"))
    args = p.parse_args()
    files = sorted(f for f in args.dir.glob("*_cells.csv") if not f.name.startswith("all_"))
    cells = pd.concat([pd.read_csv(f).assign(donor=f.name.removesuffix("_cells.csv")) for f in files])
    cells.to_csv(args.dir / "all_donors_cells.csv", index=False)

    per_donor = pd.concat([vh.summarise(g).assign(donor=d) for d, g in cells.groupby("donor")])
    pooled = vh.summarise(cells).assign(donor="ALL")
    summary = pd.concat([pooled, per_donor], ignore_index=True)
    summary.to_csv(args.dir / "all_donors_summary.csv", index=False)

    cols = ["n_cells", "section_acc_pct", "median_um", "within_500_pct",
            "within_1000_pct", "pearson_x", "pearson_y", "r2_x", "r2_y"]
    print(f"{len(files)} donors, {len(cells):,} held-out cell evaluations\n")
    print("Pooled over donors:")
    print(pooled[pooled["subclass"] == "ALL"][["method", "scope"] + cols].round(3).to_string(index=False))
    print("\nPer-donor median [min, max]:")
    d = per_donor[per_donor["subclass"] == "ALL"]
    spread = d.groupby(["method", "scope"])[["section_acc_pct", "median_um", "within_500_pct", "pearson_y"]].agg(
        lambda s: f"{s.median():.1f} [{s.min():.1f}, {s.max():.1f}]")
    print(spread.to_string())
    # paired per-donor comparison: in how many donors does Cell2Map beat each baseline
    piv = d[d["scope"] == "section"].pivot(index="donor", columns="method", values="median_um")
    for other in [m for m in piv.columns if m != "cell2map"]:
        if "cell2map" in piv:
            print(f"\nCell2Map median error lower than {other} in "
                  f"{(piv['cell2map'] < piv[other]).sum()} of {len(piv)} donors")


if __name__ == "__main__":
    main()
