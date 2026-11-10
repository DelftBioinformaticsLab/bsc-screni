"""Map SEA-AD snRNA-seq nuclei onto MERFISH sections (Cell2Map, Tangram baseline).

Modules:
    data     -- load MERFISH and the raw Allen snRNA file, filtered to the
                MERFISH donors and the 140 measured MERFISH genes.
    mapping  -- run Cell2Map or Tangram for one (donor, subclass) pair and
                read out a position per nucleus.

Scripts in scripts/cell2map/ and jobs in slurm/cell2map_*.sh drive these.
"""
