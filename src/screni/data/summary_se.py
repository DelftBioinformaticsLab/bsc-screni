"""summarySE: group-wise summary statistics with confidence intervals.

Python port of the R helper ``summarySE`` from
``R/Precision_recall_affiliated_functions.R``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def summary_se(
    data: pd.DataFrame,
    measurevar: str,
    groupvars: Sequence[str] | None = None,
    na_rm: bool = False,
    conf_interval: float = 0.95,
) -> pd.DataFrame:
    """Group-wise count, mean, sd, se, and confidence interval.

    Parameters
    ----------
    data
        Input DataFrame.
    measurevar
        Name of the column to summarise.
    groupvars
        Column names to group by. If ``None`` or empty, the entire DataFrame
        is summarised as a single group.
    na_rm
        If True, NaNs are ignored when computing N, mean, and sd
        (matches R's ``na.rm=TRUE``).
    conf_interval
        Confidence-interval coverage; default is 0.95.

    Returns
    -------
    DataFrame with columns ``[*groupvars, "N", measurevar, "sd", "se", "ci"]``.
    The mean column is renamed to ``measurevar`` (matching the R behaviour).
    """
    groupvars = list(groupvars) if groupvars else []

    def _n(s: pd.Series) -> int:
        return int(s.notna().sum()) if na_rm else int(len(s))

    def _mean(s: pd.Series) -> float:
        return s.mean(skipna=na_rm)

    def _sd(s: pd.Series) -> float:
        return s.std(ddof=1, skipna=na_rm)

    if groupvars:
        agg = (
            data.groupby(groupvars, dropna=False, observed=True)[measurevar]
            .agg(N=_n, mean=_mean, sd=_sd)
            .reset_index()
        )
    else:
        col = data[measurevar]
        agg = pd.DataFrame([{"N": _n(col), "mean": _mean(col), "sd": _sd(col)}])

    agg["se"] = agg["sd"] / np.sqrt(agg["N"])
    # qt(conf_interval/2 + 0.5, N - 1)
    ci_mult = stats.t.ppf(conf_interval / 2 + 0.5, agg["N"] - 1)
    agg["ci"] = agg["se"] * ci_mult

    agg = agg.rename(columns={"mean": measurevar})
    cols = [*groupvars, "N", measurevar, "sd", "se", "ci"]
    return agg[cols].reset_index(drop=True)
