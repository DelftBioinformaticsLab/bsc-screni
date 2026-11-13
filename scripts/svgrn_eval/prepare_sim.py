"""Turn a simulate.R output folder into SVGRN's input files.

Same steps as external/SVGRN/scMultiSim_DataPrepare/sim_prepare.py:
    GRN_cluster_ids.csv, cell_loc_GRNcluster4.csv  KMeans (4 clusters) on the flattened cell GRNs
    allcells_GRN.csv                               mean of the cell GRNs
    normalized_count.csv                           log1p(counts / total * 1e4 + 1), as SVGRN
    expression_loc_cluster_wlayout.csv             normalized counts + x, y, ClusterID
Differences: KMeans gets random_state=0 (SVGRN's has none) and cells are read
in cell order (SVGRN uses os.listdir order; this only affects cluster labels).

Usage:
    python scripts/svgrn_eval/prepare_sim.py <dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def main() -> None:
    d = Path(sys.argv[1])
    loc = pd.read_csv(d / "cell_loc.csv", index_col=0)
    cells = [f"cell{i}" for i in range(1, len(loc) + 1)]
    grns = [pd.read_csv(d / "cell_specific_GRN" / f"{c}.csv", index_col=0) for c in cells]
    flat = np.stack([g.to_numpy().ravel() for g in grns])

    cid = KMeans(n_clusters=4, random_state=0, n_init=10).fit_predict(flat)
    pd.DataFrame({"ClusterID": cid}, index=cells).to_csv(d / "GRN_cluster_ids.csv")
    loc.index = loc.index.astype(str)
    if not loc.index.str.startswith("cell").all():
        loc.index = cells
    loc_c = loc.join(pd.DataFrame({"ClusterID": cid}, index=cells))
    loc_c.to_csv(d / "cell_loc_GRNcluster4.csv")

    avg = pd.DataFrame(flat.mean(0).reshape(grns[0].shape),
                       index=[str(i) for i in grns[0].index], columns=[str(c) for c in grns[0].columns])
    avg.to_csv(d / "allcells_GRN.csv")

    raw = pd.read_csv(d / "raw_count.csv", index_col=0)
    norm = np.log1p(raw.div(raw.sum(1), axis=0) * 10000 + 1)
    norm.columns = [str(i) for i in range(1, raw.shape[1] + 1)]
    norm.to_csv(d / "normalized_count.csv")
    norm.join(loc_c[["x", "y", "ClusterID"]]).to_csv(d / "expression_loc_cluster_wlayout.csv")
    print(f"{d}: {len(cells)} cells, {raw.shape[1]} genes, clusters {np.bincount(cid).tolist()}, "
          f"TFs {list(avg.columns)}")


if __name__ == "__main__":
    main()
