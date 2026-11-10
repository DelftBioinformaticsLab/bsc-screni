"""Step 0: write per-donor MERFISH and snRNA inputs for Cell2Map mapping.

For every MERFISH donor (27) this writes
    <out>/merfish/<donor>.h5ad   MERFISH cells x 140 genes, coords in obsm['spatial']
    <out>/rna/<donor>.h5ad       snRNA nuclei x 140 genes, raw UMI counts
and, over all donors,
    <out>/donors.txt                 donors with both modalities (SLURM array index)
    <out>/pilot_donor.txt            donor with the most snRNA nuclei
    <out>/counts_donor_subclass.csv  cells per donor x subclass, both modalities

snRNA comes from Allen's raw file (all assays: 10Xv3, 10Xv3.1, 10xMulti),
not from the processed unpaired file, so the multiome nuclei stay in.

Usage:
    python scripts/cell2map/prepare_inputs.py --out data/cell2map
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from screni.cell2map import data as d

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("data/cell2map"))
    args = p.parse_args()
    (args.out / "merfish").mkdir(parents=True, exist_ok=True)
    (args.out / "rna").mkdir(parents=True, exist_ok=True)

    genes = d.merfish_genes()
    logger.info("MERFISH genes (blanks dropped): %d", len(genes))

    mer = d.load_merfish()
    scale = d.check_merfish_scale(mer)
    logger.info("MERFISH: %d cells, %d donors, %d sections, X scale=%s (max %.1f)",
                mer.n_obs, mer.obs[d.DONOR].nunique(), mer.obs[d.SECTION].nunique(),
                scale, float(mer.X.max()))
    if scale == "log":
        mer.X = np.expm1(mer.X)
    donors = sorted(mer.obs[d.DONOR].unique())

    rna = d.load_rna(donors, genes)
    logger.info("snRNA: %d nuclei in %d MERFISH donors; assays %s",
                rna.n_obs, rna.obs[d.DONOR].nunique(),
                rna.obs["method"].astype(str).value_counts().to_dict())

    sub_m = set(mer.obs[d.SUBCLASS]); sub_r = set(rna.obs[d.SUBCLASS])
    if sub_m != sub_r:
        logger.warning("Subclass names differ. MERFISH only: %s; snRNA only: %s",
                       sorted(sub_m - sub_r), sorted(sub_r - sub_m))

    both = []
    for donor in donors:
        m = mer[mer.obs[d.DONOR] == donor].copy()
        r = rna[rna.obs[d.DONOR] == donor].copy()
        if r.n_obs == 0:
            logger.warning("%s: no snRNA nuclei, skipped", donor)
            continue
        m.write_h5ad(args.out / "merfish" / f"{donor}.h5ad", compression="gzip")
        r.write_h5ad(args.out / "rna" / f"{donor}.h5ad", compression="gzip")
        both.append(donor)
        logger.info("%s: %d MERFISH cells in %d sections, %d nuclei",
                    donor, m.n_obs, m.obs[d.SECTION].nunique(), r.n_obs)

    (args.out / "donors.txt").write_text("\n".join(both) + "\n")
    nuc = rna.obs[d.DONOR].value_counts()
    pilot = nuc[nuc.index.isin(both)].idxmax()
    (args.out / "pilot_donor.txt").write_text(pilot + "\n")

    cm = pd.crosstab(mer.obs[d.DONOR], mer.obs[d.SUBCLASS]).stack().rename("merfish_cells")
    cr = pd.crosstab(rna.obs[d.DONOR], rna.obs[d.SUBCLASS]).stack().rename("rna_nuclei")
    counts = pd.concat([cm, cr], axis=1).fillna(0).astype(int)
    counts["matrix_entries"] = counts["merfish_cells"] * counts["rna_nuclei"]
    counts.index.names = ["donor", "subclass"]
    counts.to_csv(args.out / "counts_donor_subclass.csv")

    logger.info("Donors with both modalities: %d; pilot donor: %s (%d nuclei)",
                len(both), pilot, int(nuc[pilot]))
    logger.info("Largest mapping matrix: %.1fM entries", counts["matrix_entries"].max() / 1e6)


if __name__ == "__main__":
    main()
