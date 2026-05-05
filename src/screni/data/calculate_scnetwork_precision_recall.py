"""Precision/recall of cell-specific networks against a TF-target gold standard.

Python port of ``Calculate_scNetwork_precision_recall`` from
``R/Calculate_scNetwork_precision_recall.R``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .calculate_precision_recall import calculate_precision_recall


def calculate_scnetwork_precision_recall(
    scNetworks: Mapping[str, Sequence],
    TF_target_pair: Iterable[str],
    top_number: Sequence[int] = (1000, 2000, 4000, 6000, 8000, 10000, 20000),
    gene_id_gene_name_pair: pd.DataFrame | None = None,
    gene_name_type: str | None = None,
) -> dict[int, pd.DataFrame]:
    """Compute precision/recall for each cell-specific network at each cutoff.

    Parameters
    ----------
    scNetworks
        Mapping from method name (e.g. ``"CSN"``, ``"wScReNI"``, ``"kScReNI"``)
        to a per-cell collection of weighted adjacency matrices. Cell counts
        must be equal across methods. Each per-cell collection may be a
        sequence or a Mapping; mappings preserve cell labels in the output.
    TF_target_pair
        Iterable of ground-truth pairs encoded as ``"TF_TARGET"`` strings.
    top_number
        Top-K cutoffs at which to compute precision/recall. The cutoff ``0``
        is added implicitly: at ``0``, every method (CSN included) is scored
        using a per-cell K equal to the number of non-zero pairs in that
        cell's CSN.
    gene_id_gene_name_pair, gene_name_type
        Forwarded to :func:`calculate_precision_recall`.

    Returns
    -------
    Dict keyed by cutoff (int). Each value is a DataFrame with columns
    ``["scNetwork_type", "precision", "recall"]`` and one row per
    (network type, cell) pair. The index holds cell labels.
    """
    if "CSN" not in scNetworks:
        raise KeyError(
            "scNetworks must contain a 'CSN' entry: it provides the per-cell "
            "non-zero edge count used as the reference cutoff for top_number=0."
        )

    network_types = list(scNetworks.keys())
    cell_num = len(scNetworks[network_types[0]])

    csn_nonzero = np.array(
        [int(np.count_nonzero(np.asarray(net))) for net in _values(scNetworks["CSN"])],
        dtype=int,
    ).reshape(-1, 1)

    # NB: deviation from the R source, which uses substring grep('CSN', ...).
    # Exact match avoids accidentally excluding entries like 'CSN_v2'.
    non_csn_types = [t for t in network_types if t != "CSN"]

    cutoffs: list[int] = sorted({0, *(int(k) for k in top_number)})

    out: dict[int, pd.DataFrame] = {}
    for k in cutoffs:
        if k == 0:
            types_to_eval = network_types
            top_per_cell = csn_nonzero
        else:
            types_to_eval = non_csn_types
            top_per_cell = np.full((cell_num, 1), k, dtype=int)

        rows: list[tuple[str, float, float]] = []
        index_labels: list = []
        for net_type in types_to_eval:
            net_list = scNetworks[net_type]
            for j, (cell_label, scNet) in enumerate(_enumerate(net_list)):
                k_for_cell = int(top_per_cell[j, 0])
                precision, recall = calculate_precision_recall(
                    scNet,
                    TF_target_pair,
                    k_for_cell,
                    gene_id_gene_name_pair,
                    gene_name_type,
                )
                rows.append((net_type, float(precision), float(recall)))
                index_labels.append(cell_label)

        df = pd.DataFrame(rows, columns=["scNetwork_type", "precision", "recall"])
        df.index = pd.Index(index_labels)
        df["scNetwork_type"] = df["scNetwork_type"].astype("category")
        out[k] = df

    return out


def _values(net_list):
    if isinstance(net_list, Mapping):
        return list(net_list.values())
    return list(net_list)


def _enumerate(net_list):
    """Yield ``(cell_label, network)`` pairs from a sequence or mapping."""
    if isinstance(net_list, Mapping):
        return list(net_list.items())
    return list(enumerate(net_list))
