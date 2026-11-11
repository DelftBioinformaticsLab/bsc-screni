"""Map one donor's snRNA nuclei onto that donor's MERFISH sections.

Mapping is subclass-matched: the nuclei of each subclass are mapped onto the
MERFISH cells of the same subclass, across all of the donor's sections at
once. The argmax MERFISH cell gives each nucleus its section and (x, y).

Writes <out>/<method>/<donor>.csv (one row per nucleus) and
<out>/<method>/<donor>_runs.csv (one row per subclass: sizes, runtime).

Usage:
    python scripts/cell2map/map_donor.py --donor H20.33.025 --max-nuclei 300
    python scripts/cell2map/map_donor.py --donor-index $SLURM_ARRAY_TASK_ID
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

RNA_CARRY = ["Donor ID", "Subclass", "Supertype", "method", "Braak",
             "Continuous Pseudo-progression Score", "Cognitive Status",
             "Overall AD neuropathological Change"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--donor")
    g.add_argument("--donor-index", type=int, help="Line in <inputs>/donors.txt (0-based)")
    p.add_argument("--inputs", type=Path, default=Path("data/cell2map"))
    p.add_argument("--out", type=Path, default=Path("output/cell2map/maps"))
    p.add_argument("--methods", nargs="+", default=["cell2map"], choices=["cell2map", "tangram", "random"])
    p.add_argument("--subclasses", nargs="*", default=None, help="Default: all 24")
    p.add_argument("--max-nuclei", type=int, default=None,
                   help="Subsample at most this many nuclei per subclass (pilot)")
    p.add_argument("--min-cells", type=int, default=20,
                   help="Skip a subclass with fewer nuclei or MERFISH cells than this")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs-cell2map", type=int, default=1500)
    p.add_argument("--epochs-tangram", type=int, default=1000)
    p.add_argument("--k-sp", type=int, default=7, help="kNN for the MERFISH spatial graph")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    donor = args.donor or (args.inputs / "donors.txt").read_text().split()[args.donor_index]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Donor %s, methods %s, device %s (%s)", donor, args.methods, device,
                torch.cuda.get_device_name(0) if device == "cuda" else "-")
    if device != "cuda" and "cell2map" in args.methods:
        raise SystemExit("Cell2Map's embedding step needs a GPU (it hard-codes cuda).")

    mer = ad.read_h5ad(args.inputs / "merfish" / f"{donor}.h5ad")
    rna = ad.read_h5ad(args.inputs / "rna" / f"{donor}.h5ad")
    subclasses = args.subclasses or sorted(set(rna.obs[d.SUBCLASS]) & set(mer.obs[d.SUBCLASS]))
    rng = np.random.default_rng(args.seed)

    for method in args.methods:
        (args.out / method).mkdir(parents=True, exist_ok=True)
        rows, runs = [], []
        for sub in subclasses:
            r = rna[rna.obs[d.SUBCLASS] == sub]
            m = mer[mer.obs[d.SUBCLASS] == sub]
            if r.n_obs < args.min_cells or m.n_obs < args.min_cells:
                logger.info("  %-22s skipped: %d nuclei, %d MERFISH cells", sub, r.n_obs, m.n_obs)
                runs.append({"subclass": sub, "n_rna": r.n_obs, "n_merfish": m.n_obs, "status": "skipped"})
                continue
            if args.max_nuclei and r.n_obs > args.max_nuclei:
                r = r[np.sort(rng.choice(r.n_obs, args.max_nuclei, replace=False))]
            r, m = r.copy(), mp.stack_sections(m.copy())
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            M, secs = mp.run_method(method, r, m, k_sp=args.k_sp, epochs=args.epochs_cell2map,
                                    epochs_tangram=args.epochs_tangram, seed=args.seed, device=device)
            out = mp.readout(M, r, m, d.MERFISH_CARRY)
            for col in RNA_CARRY:
                if col in r.obs:
                    out[col] = r.obs[col].to_numpy()
            rows.append(out)
            gpu_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else np.nan
            runs.append({"subclass": sub, "n_rna": r.n_obs, "n_merfish": m.n_obs,
                         "n_sections": m.obs[d.SECTION].nunique(), "seconds": round(secs, 1),
                         "gpu_peak_gb": round(gpu_gb, 2), "status": "ok"})
            logger.info("  %-22s %5d nuclei -> %6d MERFISH cells  %6.1fs  GPU %.2f GB",
                        sub, r.n_obs, m.n_obs, secs, gpu_gb)
        res = pd.concat(rows)
        res.index.name = "nucleus_id"
        res.to_csv(args.out / method / f"{donor}.csv")
        pd.DataFrame(runs).to_csv(args.out / method / f"{donor}_runs.csv", index=False)
        logger.info("%s: wrote %d nuclei for %s", method, len(res), donor)


if __name__ == "__main__":
    main()
