"""Precision–recall evaluation orchestration for single-cell regulatory networks.

Matches the original R functions:

    ``Calculate_scNetwork_precision_recall()``
        → :func:`calculate_network_precision_recall`
          (``Calculate_scNetwork_precision_recall.R``)

    ``Calculate_scNetwork_precision_recall_top()``
        → :func:`calculate_network_precision_recall_top`
          (``Calculate_scNetwork_precision_recall_top.R``)

    ``summarySE()``
        → :func:`summary_se`
          (``Precision_recall_affiliated_functions.R``, line 85)

These three functions are the orchestration layer that calls Leo's per-cell
helper :func:`~screni.data.precision_recall.calculate_precision_recall` and
produces the full precision–recall evaluation results across all network types,
cells, and top-N thresholds.

Typical usage
-------------
::

    from screni.data.precision_recall import deal_gene_information
    from screni.data.evaluation import (
        calculate_network_precision_recall,
        calculate_network_precision_recall_top,
    )

    # Step 3 setup (mirrors tutorial R code)
    gene_map = deal_gene_information(gtf_regions, gene_name_type="symbol")
    tf_target_pair = load_chip_atlas("mmp9.TSV.5kb_TF_target.df.txt")

    # Evaluate all network types across thresholds
    top_number = [1000, 2000, 4000, 6000, 8000, 10000, 20000]
    all_results = calculate_network_precision_recall(
        sc_networks={"CSN": csn_nets, "kScReNI": k_nets, "wScReNI": w_nets},
        tf_target_pair=tf_target_pair,
        top_number=top_number,
        gene_id_gene_name_pair=gene_map,
        gene_name_type="symbol",
    )

    # Aggregate: skip the CSN-matched threshold (key 0), pass only top_number
    non_csn_results = {k: v for k, v in all_results.items() if k != 0}
    precision_summary, recall_summary = calculate_network_precision_recall_top(
        non_csn_results, top_number
    )
"""

from __future__ import annotations

import logging

from joblib import Parallel, delayed
from typing import Collection, Optional

import numpy as np
import pandas as pd
from scipy import stats

from screni.data.combine import ScReniNetworks
from screni.data.precision_recall import calculate_precision_recall

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# summary_se
# ---------------------------------------------------------------------------


def summary_se(
    data: pd.DataFrame,
    measurevar: str,
    groupvars: Optional[list[str]] = None,
    na_rm: bool = False,
    conf_interval: float = 0.95,
) -> pd.DataFrame:
    """Compute grouped descriptive statistics with standard error and CI.

    Python equivalent of ``summarySE()`` from the original R code
    (``Precision_recall_affiliated_functions.R``, line 85), which uses
    ``plyr::ddply`` internally.

    For each group defined by ``groupvars``, computes:

    * **N** — number of observations (excluding NaN when ``na_rm=True``)
    * **mean** — arithmetic mean of ``measurevar``
    * **sd** — sample standard deviation (ddof=1, matching R's ``sd()``)
    * **se** — standard error of the mean (``sd / sqrt(N)``)
    * **ci** — half-width of a two-sided confidence interval at level
      ``conf_interval``, using a t-distribution with ``N − 1`` degrees of
      freedom (matching R's ``qt(conf_interval/2 + 0.5, N-1)``)

    Parameters
    ----------
    data
        Input DataFrame.
    measurevar
        Name of the numeric column to summarise.
    groupvars
        Column name(s) to group by.  ``None`` summarises the entire column.
    na_rm
        If ``True``, NaN values are excluded before computing statistics,
        matching R's ``na.rm=TRUE``.
    conf_interval
        Confidence level for the CI (default 0.95 → 95 % CI).

    Returns
    -------
    DataFrame with one row per group and columns
    ``[*groupvars, measurevar, "N", "sd", "se", "ci"]``.
    The ``measurevar`` column holds the group mean (R renames ``mean`` →
    ``measurevar`` via ``plyr::rename``).

    Examples
    --------
    >>> df = pd.DataFrame({"method": ["A","A","B","B"], "precision": [0.1,0.2,0.3,0.4]})
    >>> summary_se(df, "precision", groupvars=["method"])
      method  precision  N        sd        se        ci
    0      A       0.15  2  0.070711  0.050000  0.635310
    1      B       0.35  2  0.070711  0.050000  0.635310
    """
    if groupvars is None:
        groupvars = []

    def _agg(x: pd.Series) -> pd.Series:
        vals = x.dropna() if na_rm else x
        n = len(vals)
        mean_val = vals.mean(skipna=na_rm) if n > 0 else float("nan")
        sd_val = vals.std(ddof=1) if n > 1 else float("nan")
        se_val = sd_val / np.sqrt(n) if n > 0 else float("nan")
        # t critical value matching R's qt(conf_interval/2 + 0.5, df=N-1)
        ci_val = (
            se_val * stats.t.ppf(conf_interval / 2 + 0.5, df=n - 1)
            if n > 1
            else float("nan")
        )
        return pd.Series(
            {"N": n, measurevar: mean_val, "sd": sd_val, "se": se_val, "ci": ci_val}
        )

    if groupvars:
        # Apply aggregation per group, then unstack the inner Series index into columns.
        # groupby(...)[col].apply(fn) in pandas 2.x returns a Series with a MultiIndex
        # (group_keys × stat_names), so we unstack the last level to get columns.
        grouped = data.groupby(groupvars, observed=True)[measurevar].apply(_agg)
        # `grouped` has group keys in outer levels and stat names (N, measurevar, …)
        # in the innermost index level — unstack that level into columns.
        result = grouped.unstack(level=-1).reset_index()
    else:
        agg = _agg(data[measurevar])
        result = agg.to_frame().T.reset_index(drop=True)

    # Ensure column order matches R output: groupvars, measurevar, N, sd, se, ci
    col_order = groupvars + [measurevar, "N", "sd", "se", "ci"]
    result = result[[c for c in col_order if c in result.columns]]
    result["N"] = result["N"].astype(int)
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# calculate_network_precision_recall
# ---------------------------------------------------------------------------


def calculate_network_precision_recall(
    sc_networks: dict[str, "ScReniNetworks"],
    tf_target_pair: "Collection[str]",
    top_number: "list[int]" = None,
    gene_id_gene_name_pair: "Optional[pd.DataFrame]" = None,
    gene_name_type: "Optional[str]" = None,
    n_jobs: int = 1,
) -> "dict[int, pd.DataFrame]":
    """Evaluate precision and recall across network types, cells, and thresholds.

    Python equivalent of ``Calculate_scNetwork_precision_recall()``
    (``Calculate_scNetwork_precision_recall.R``).

    This is the top-level evaluation loop.  It iterates over every combination
    of network type, cell, and top-N threshold and delegates to
    :func:`~screni.data.precision_recall.calculate_precision_recall` for
    the per-cell computation.

    CSN special case
    ----------------
    The CSN method produces binary adjacency matrices whose nonzero edge count
    varies per cell.  To make comparisons fair, threshold ``0`` is always
    prepended to ``top_number``.  At threshold ``0``:

    * **All** network types (including CSN) are evaluated.
    * Each cell uses the nonzero edge count from *that cell's* CSN matrix as
      its individual ``top_number``.  This ensures every method is judged on
      the same number of edges as CSN predicted for that cell.

    For all other thresholds (``> 0``):

    * CSN is **excluded** (its fixed nonzero structure makes a fixed-N
      comparison misleading).
    * All other methods use the shared fixed ``top_number``.

    Parameters
    ----------
    sc_networks
        Mapping of network-type name → :class:`~screni.data.combine.ScReniNetworks`
        (which is itself a ``dict[cell_name, weight_matrix]``).  Must contain
        a ``"CSN"`` key; all other keys are treated as variable-threshold
        methods.  Cell ordering must be consistent across all network types
        (same cell names in the same order).
    tf_target_pair
        Set of known TF → target ground-truth pairs encoded as
        ``"TF_symbol_Target_symbol"`` strings, as produced by
        :func:`load_chip_atlas`.
    top_number
        List of integer thresholds for the number of top edges to evaluate.
        Defaults to ``[1000, 2000, 4000, 6000, 8000, 10000, 20000]`` (matching
        the tutorial). ``0`` is automatically prepended for the CSN comparison.
    gene_id_gene_name_pair
        Gene-ID ↔ symbol lookup from
        :func:`~screni.data.precision_recall.deal_gene_information`.
        Required when ``gene_name_type`` is not ``None``.
    gene_name_type
        ``'symbol'``, ``'id'``, or ``None`` — passed through to
        :func:`~screni.data.precision_recall.calculate_precision_recall`.
    n_jobs
        Number of parallel workers for the per-cell precision/recall loop.
        ``1`` (default) runs serially; ``-1`` uses all available CPUs.
        Uses a thread pool (``prefer="threads"``) because
        :func:`~screni.data.precision_recall.calculate_precision_recall`
        releases the GIL during its numpy-heavy inner loop.

    Returns
    -------
    dict mapping each threshold value → DataFrame with columns
    ``["scNetwork_type", "precision", "recall"]`` and one row per
    (network type, cell) combination.  The key ``0`` corresponds to the
    CSN-matched threshold.

    Notes
    -----
    Cell ordering across network types must be consistent: the function pairs
    cells from different methods by position, matching R's behaviour where
    ``scNet_list[[j]]`` and ``nonzero_num1[j, ]`` are both positionally
    indexed.

    Examples
    --------
    >>> results = calculate_network_precision_recall(
    ...     sc_networks={"CSN": csn_nets, "wScReNI": w_nets},
    ...     tf_target_pair=chip_atlas_pairs,
    ...     top_number=[1000, 5000, 10000],
    ... )
    >>> results[1000].head()
      scNetwork_type  precision    recall
    0        wScReNI      0.012  0.034...
    """
    if top_number is None:
        top_number = [1000, 2000, 4000, 6000, 8000, 10000, 20000]

    # R: top_number <- unique(c(0, top_number))  — prepend the CSN-matched slot
    all_thresholds = list(dict.fromkeys([0] + list(top_number)))

    network_types = list(sc_networks.keys())
    non_csn_types = [t for t in network_types if t != "CSN"]

    # --- compute per-cell nonzero counts from CSN ---
    if "CSN" not in sc_networks:
        raise ValueError(
            "'CSN' key must be present in sc_networks. "
            "The CSN nonzero counts are required for threshold=0 evaluation."
        )
    csn_nets = sc_networks["CSN"]
    csn_cells = list(csn_nets.keys())  # ordered cell names
    cell_num = len(csn_cells)

    # nonzero_num[i] = number of nonzero entries in cell i's CSN matrix
    nonzero_counts: dict[str, int] = {
        cell: int(np.count_nonzero(csn_nets[cell])) for cell in csn_cells
    }

    # Build once; frozenset is immutable and passes the isinstance check
    # in calculate_precision_recall to skip re-wrapping.
    tf_set = frozenset(tf_target_pair)
    results: dict[int, pd.DataFrame] = {}

    for threshold in all_thresholds:
        logger.info(f"Evaluating top_number={threshold} ...")

        if threshold == 0:
            # All methods, per-cell threshold from CSN nonzero count
            types_to_eval = network_types
            nets_to_eval = sc_networks
        else:
            # All methods except CSN, fixed threshold
            types_to_eval = non_csn_types
            nets_to_eval = {t: sc_networks[t] for t in non_csn_types}

        # Build a flat work list: one item per (net_type, cell) pair.
        # This lets joblib parallelise across ALL network types × cells in one
        # Parallel call, minimising per-worker overhead.
        work_items: list[tuple[str, str, object, int]] = []
        for net_type in types_to_eval:
            net   = nets_to_eval[net_type]
            cells = list(net.keys())
            gnames = sc_networks[net_type].gene_names or None
            for j, cell in enumerate(cells):
                if threshold == 0:
                    csn_ref_cell = csn_cells[j] if j < cell_num else cell
                    cell_top_n   = nonzero_counts.get(csn_ref_cell,
                                                      nonzero_counts.get(cell, 1))
                else:
                    cell_top_n = threshold
                work_items.append((net_type, cell, net[cell], cell_top_n, gnames))

        def _eval_one(net_type, cell, weight_matrix, cell_top_n, gnames):
            p, r = calculate_precision_recall(
                scnetwork_weights=weight_matrix,
                tf_target_pair=tf_set,
                top_number=cell_top_n,
                gene_id_gene_name_pair=gene_id_gene_name_pair,
                gene_name_type=gene_name_type,
                gene_names=gnames,
            )
            return {"scNetwork_type": net_type, "cell": cell,
                    "precision": p, "recall": r}

        # prefer="processes": np.char.add string ops hold the GIL,
        # so threading gives no parallelism for the recall denominator.
        rows: list[dict] = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_eval_one)(nt, c, wm, tn, gn)
            for nt, c, wm, tn, gn in work_items
        )

        df = pd.DataFrame(rows)
        df["scNetwork_type"] = df["scNetwork_type"].astype("category")
        df["precision"] = pd.to_numeric(df["precision"], errors="coerce")
        df["recall"] = pd.to_numeric(df["recall"], errors="coerce")
        results[threshold] = df
        logger.info(
            f"  threshold={threshold}: {len(df)} rows "
            f"({df['scNetwork_type'].nunique()} network type(s), "
            f"{df.groupby('scNetwork_type', observed=True).size().to_dict()})"
        )

    return results


# ---------------------------------------------------------------------------
# calculate_network_precision_recall_top
# ---------------------------------------------------------------------------


def calculate_network_precision_recall_top(
    sc_net_precision_recall: "dict[int, pd.DataFrame]",
    top_number: "list[int]",
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Aggregate per-cell precision–recall results into grouped summary tables.

    Python equivalent of ``Calculate_scNetwork_precision_recall_top()``
    (``Calculate_scNetwork_precision_recall_top.R``).

    Stacks the per-threshold DataFrames from
    :func:`calculate_network_precision_recall` (excluding the ``0`` / CSN-matched
    threshold), adds a ``top_number`` column, then calls :func:`summary_se`
    to compute per-group mean ± SE ± CI for both precision and recall.

    Parameters
    ----------
    sc_net_precision_recall
        Result dict from :func:`calculate_network_precision_recall`, with the
        ``0`` key (CSN-matched threshold) already **removed** by the caller —
        matching the tutorial's ``scNetworks_precision_recall_noCSN <-
        scNetworks_precision_recall[-1]`` step.

        .. code-block:: python

            all_results = calculate_network_precision_recall(...)
            non_csn = {k: v for k, v in all_results.items() if k != 0}
            prec, rec = calculate_network_precision_recall_top(non_csn, top_number)

    top_number
        The same list of integer thresholds passed to
        :func:`calculate_network_precision_recall`.  Must match the keys of
        ``sc_net_precision_recall`` in order.

    Returns
    -------
    ``(precision_summary, recall_summary)`` — two DataFrames with one row per
    ``(scNetwork_type, top_number)`` combination and columns
    ``["scNetwork_type", "top_number", "<measurevar>", "N", "sd", "se", "ci"]``.

    Notes
    -----
    ``top_number`` is stored as a categorical column (matching R's
    ``as.factor(top_number)``), which preserves order in downstream plots.

    Examples
    --------
    >>> prec_summary, rec_summary = calculate_network_precision_recall_top(
    ...     non_csn_results, top_number=[1000, 5000, 10000]
    ... )
    >>> prec_summary.columns
    Index(['scNetwork_type', 'top_number', 'precision', 'N', 'sd', 'se', 'ci'])
    """
    # Stack all per-threshold DataFrames, adding a top_number column.
    # R: for(i in 1:length(top_number)) cbind(top_number[i], scNet_precision_recall[[i]])
    stacked_parts: list[pd.DataFrame] = []
    for i, tn in enumerate(top_number):
        # Access by list position (matching R's [[i]]) with key fallback
        if tn in sc_net_precision_recall:
            df_slice = sc_net_precision_recall[tn].copy()
        else:
            keys = list(sc_net_precision_recall.keys())
            if i < len(keys):
                df_slice = sc_net_precision_recall[keys[i]].copy()
                logger.warning(
                    f"top_number[{i}]={tn} not found in results dict; "
                    f"using key {keys[i]} by position"
                )
            else:
                logger.warning(f"top_number[{i}]={tn} not found; skipping")
                continue

        df_slice.insert(0, "top_number", tn)
        stacked_parts.append(df_slice)

    if not stacked_parts:
        empty = pd.DataFrame(
            columns=["scNetwork_type", "top_number", "precision", "N", "sd", "se", "ci"]
        )
        return empty, empty.rename(columns={"precision": "recall"})

    stacked = pd.concat(stacked_parts, ignore_index=True)

    # R: as.factor(top_number) — categorical preserves ordering in plots
    stacked["top_number"] = pd.Categorical(
        stacked["top_number"], categories=top_number, ordered=True
    )
    stacked["scNetwork_type"] = stacked["scNetwork_type"].astype("category")
    stacked["precision"] = pd.to_numeric(stacked["precision"], errors="coerce")
    stacked["recall"] = pd.to_numeric(stacked["recall"], errors="coerce")

    precision_summary = summary_se(
        stacked,
        measurevar="precision",
        groupvars=["scNetwork_type", "top_number"],
    )
    recall_summary = summary_se(
        stacked,
        measurevar="recall",
        groupvars=["scNetwork_type", "top_number"],
    )

    return precision_summary, recall_summary


# ---------------------------------------------------------------------------
# load_chip_atlas  (helper for building tf_target_pair)
# ---------------------------------------------------------------------------


def load_chip_atlas(filepath: str) -> set[str]:
    """Load TF–target pairs from a ChIP-Atlas TSV file.

    Replicates the two-line R tutorial snippet::

        TF_target <- read.table(chip_atlas_path, sep="\\t", header=TRUE, row.names=1)
        TF_target_pair <- unique(paste(TF_target[,1], TF_target[,2], sep="_"))

    The ChIP-Atlas file format (mmp9.TSV.5kb_TF_target.df.txt) has:
    - Header row: ``"TF"  "Target_genes"``
    - Data rows:  ``"row_number"  "TF_symbol"  "Target_symbol"``
    
    We skip the row number column (index_col=0) and extract the TF and Target
    columns to create ``"TF_Target"`` pair strings.

    Parameters
    ----------
    filepath
        Path to the ChIP-Atlas TSV file (e.g. ``mmp9.TSV.5kb_TF_target.df.txt``).

    Returns
    -------
    Set of unique ``"TF_symbol_Target_symbol"`` strings, ready to pass to
    :func:`calculate_network_precision_recall`.

    Examples
    --------
    >>> tf_pairs = load_chip_atlas("refer/mmp9.TSV.5kb_TF_target.df.txt")
    >>> "Acss2_Cldn34d" in tf_pairs  # First real data row
    True
    """
    # ChIP-Atlas file format:
    #   Header row: "TF"  "Target_genes"
    #   Data rows:  "row_number"  "TF_symbol"  "Target_symbol"
    # 
    # We use index_col=0 to treat the row number as index, then extract
    # columns 0 (TF) and 1 (Target_genes) for the pairs.
    df = pd.read_csv(filepath, sep="\t", header=0, index_col=0)
    
    # Create TF_Target pairs from the first two data columns
    pairs = df.iloc[:, 0].astype(str) + "_" + df.iloc[:, 1].astype(str)
    return set(pairs.unique())
