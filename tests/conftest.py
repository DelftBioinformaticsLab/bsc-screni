"""Shared pytest setup for the precision/recall test suite.

The lower-level helper ``calculate_precision_recall`` is the subject of
issue #8. While that PR is open we install a local Python port of the R
helper into ``sys.modules`` so the higher-level functions can be exercised.
Once #8 lands, the real implementation is imported and the stub is skipped.
"""

from __future__ import annotations

import sys
import types

import numpy as np


def _calculate_precision_recall_port(
    scNetwork_weights,
    TF_target_pair,
    top_number=1000,
    gene_id_gene_name_pair=None,
    gene_name_type=None,
):
    """Local Python port of the R ``calculate_precision_recall``.

    Restricted to the ``gene_id_gene_name_pair=None`` branch (sufficient for
    every test in this directory).
    """
    if gene_id_gene_name_pair is not None:
        raise NotImplementedError("test stub only covers the NULL branch")

    M = np.asarray(scNetwork_weights, dtype=float)
    rows = (
        list(scNetwork_weights.index)
        if hasattr(scNetwork_weights, "index")
        else list(range(M.shape[0]))
    )
    cols = (
        list(scNetwork_weights.columns)
        if hasattr(scNetwork_weights, "columns")
        else list(range(M.shape[1]))
    )

    rr, cc = np.meshgrid(np.arange(M.shape[0]), np.arange(M.shape[1]), indexing="ij")
    weights = M.ravel()
    mask = ~np.isnan(weights)
    rr, cc, weights = rr.ravel()[mask], cc.ravel()[mask], weights[mask]

    order = np.argsort(-weights, kind="stable")
    rr, cc = rr[order], cc[order]

    keys = [f"{rows[i]}_{cols[j]}" for i, j in zip(rr, cc)]
    tf_set = set(TF_target_pair)
    true_state = np.fromiter((k in tf_set for k in keys), dtype=bool, count=len(keys))

    K = int(top_number)
    numerator = int(true_state[:K].sum())
    precision = numerator / K if K else 0.0
    rec_denom = int(true_state.sum())
    recall = numerator / rec_denom if rec_denom else 0.0
    return precision, recall


try:
    from screni.data import calculate_precision_recall as _real  # noqa: F401
except ImportError:
    _stub = types.ModuleType("screni.data.calculate_precision_recall")
    _stub.calculate_precision_recall = _calculate_precision_recall_port
    sys.modules["screni.data.calculate_precision_recall"] = _stub
