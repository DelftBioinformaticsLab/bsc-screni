"""Aggregate precision/recall across top-K cutoffs.

Python port of ``Calculate_scNetwork_precision_recall_top`` from
``R/Calculate_scNetwork_precision_recall_top.R``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Sequence

import pandas as pd

from .summary_se import summary_se


def calculate_scnetwork_precision_recall_top(
    scNet_precision_recall: Mapping[int, pd.DataFrame],
    top_number: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise precision and recall by network type across cutoffs.

    Parameters
    ----------
    scNet_precision_recall
        Output of :func:`calculate_scnetwork_precision_recall`.
    top_number
        Cutoffs to include in the summary; each must be a key of
        ``scNet_precision_recall``.

    Returns
    -------
    Tuple ``(precision_summary, recall_summary)``. Each is the output of
    :func:`summary_se` with ``groupvars=["scNetwork_type", "top_number"]``.

    Notes
    -----
    The R source indexes ``scNet_precision_recall`` positionally via
    ``[[i]]``, which mislabels rows whenever the input dict carries
    keys outside ``top_number`` (e.g. the implicit ``0`` bucket). This
    implementation looks up by cutoff value instead.
    """
    parts = []
    for k in top_number:
        df = scNet_precision_recall[int(k)].copy()
        df.insert(0, "top_number", int(k))
        parts.append(df)

    combined = pd.concat(parts, ignore_index=True)
    combined["top_number"] = combined["top_number"].astype("category")
    combined["scNetwork_type"] = combined["scNetwork_type"].astype("category")
    combined["precision"] = pd.to_numeric(combined["precision"])
    combined["recall"] = pd.to_numeric(combined["recall"])

    precision_top = summary_se(
        combined,
        measurevar="precision",
        groupvars=["scNetwork_type", "top_number"],
    )
    recall_top = summary_se(
        combined,
        measurevar="recall",
        groupvars=["scNetwork_type", "top_number"],
    )
    return precision_top, recall_top
