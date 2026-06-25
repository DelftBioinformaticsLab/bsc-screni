"""Precision–recall evaluation helpers for single-cell regulatory networks.

Matches the original R functions from ``Precision_recall_affiliated_functions.R``:

    ``Deal_gene_information()``     →  :func:`deal_gene_information`
    ``calculate_precision_recall()`` → :func:`calculate_precision_recall`

Usage example
-------------
Build the gene-id <-> symbol lookup from a GTF DataFrame::

    gene_map = deal_gene_information(gtf_df, gene_name_type="symbol")
    # gene_map is indexed by gene symbol; columns: gene_id, gene_name

Evaluate one cell's inferred network against ChIP-Atlas ground truth::

    precision, recall = calculate_precision_recall(
        scnetwork_weights=weight_matrix,   # (n_genes, n_genes) ndarray
        tf_target_pair=chip_atlas_pairs,   # set of "TF_Target" strings
        top_number=1000,
        gene_id_gene_name_pair=gene_map,
        gene_name_type="symbol",
        gene_names=gene_list,
    )
"""

from __future__ import annotations

import logging
import re
from typing import Collection, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# deal_gene_information
# ---------------------------------------------------------------------------


def deal_gene_information(
    gtf_regions: pd.DataFrame,
    gene_name_type: str = "symbol",
) -> pd.DataFrame:
    """Filter GTF annotations to a clean gene_id <-> gene_name mapping table.

    Matches ``Deal_gene_information()`` from the original R code
    (``Precision_recall_affiliated_functions.R``, line 2).

    Steps
    -----
    1. Keep only protein-coding genes (requires a ``gene_type`` column).
    2. Take unique ``(gene_id, gene_name)`` pairs.
    3. Strip Ensembl version suffixes from ``gene_id``
       (e.g. ``ENSG00000139618.18`` → ``ENSG00000139618``).
    4. Remove PAR_Y pseudo-autosomal entries (``gene_id`` contains ``_PAR_Y``).
    5. Resolve one-gene-name → many-gene-id ambiguities:

       * ``gene_name_type='symbol'`` — **removes ALL entries** for any gene
         name that maps to more than one gene ID.  Guarantees a 1-to-1
         symbol → ID mapping in the result.  This is the mode used when
         the network matrices are labelled with gene symbols.
       * ``gene_name_type='id'`` — removes only the *first* gene_id per
         duplicated group (replicating R's exact behaviour, which leaves
         n − 1 entries for a group of n).  This mode is used when matrices
         are labelled with Ensembl IDs.

    6. Set the DataFrame index to ``gene_name`` (symbol mode) or ``gene_id``
       (id mode) so that downstream row-lookups work identically to R's
       ``gene_id_gene_name_pair[symbol, "gene_id"]`` syntax.

    Parameters
    ----------
    gtf_regions
        DataFrame from a parsed GTF file.  Required columns: ``gene_id``,
        ``gene_name``.  Optional but strongly recommended: ``gene_type``
        (used to filter to protein-coding genes only; if absent a warning is
        logged and the filter is skipped).
    gene_name_type
        ``'symbol'`` (default) or ``'id'``.

    Returns
    -------
    DataFrame with columns ``gene_id`` and ``gene_name``, indexed by
    ``gene_name`` (symbol mode) or ``gene_id`` (id mode).

    Raises
    ------
    ValueError
        If ``gene_name_type`` is neither ``'symbol'`` nor ``'id'``.

    Examples
    --------
    >>> gene_map = deal_gene_information(gtf_df, gene_name_type="symbol")
    >>> gene_map.loc["BRCA1", "gene_id"]
    'ENSG00000012048'
    """
    if gene_name_type not in ("symbol", "id"):
        raise ValueError(
            f"gene_name_type must be 'symbol' or 'id', got {gene_name_type!r}"
        )

    # ---- Step 1: protein-coding filter ----
    df = gtf_regions.copy()
    if "gene_type" in df.columns:
        n_before = len(df)
        df = df[df["gene_type"] == "protein_coding"]
        logger.info(f"  Protein-coding filter: {n_before} → {len(df)} rows")
    else:
        logger.warning(
            "  'gene_type' column not found in gtf_regions; "
            "skipping protein-coding filter"
        )

    # ---- Step 2: unique (gene_id, gene_name) pairs ----
    df = df[["gene_id", "gene_name"]].drop_duplicates().reset_index(drop=True)

    # ---- Step 3: strip version suffixes ----
    # R: gsub("[.][0-9]+", "", gene_id)  — removes .N suffix
    df["gene_id"] = df["gene_id"].str.replace(r"\.[0-9]+$", "", regex=True)

    # ---- Step 4: remove _PAR_Y entries ----
    n_before = len(df)
    df = df[~df["gene_id"].str.contains("_PAR_Y", na=False)].reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed:
        logger.info(f"  Removed {n_removed} _PAR_Y entries")

    # ---- Step 5: resolve one-name → many-ID ambiguities ----
    # R uses duplicated() which marks only non-first occurrences, so
    # duplicated_gene_names contains gene names seen ≥2 times.
    # duplicated_gene.df then contains ALL rows for those names.
    duplicated_names_mask = df["gene_name"].duplicated(keep=False)
    dup_df = df[duplicated_names_mask].copy()

    if len(dup_df) > 0:
        unique_dup_names = dup_df["gene_name"].unique()
        logger.info(
            f"  Found {len(unique_dup_names)} gene name(s) mapping to "
            f"multiple IDs — resolving via gene_name_type='{gene_name_type}'"
        )

        if gene_name_type == "symbol":
            # R adds gene_name (not gene_id) to the remove list for each group,
            # then filters out ALL rows where gene_name is in that list.
            # Net effect: every entry for any ambiguous gene_name is removed.
            to_remove: set[str] = set(unique_dup_names)
            df = df[~df["gene_name"].isin(to_remove)].reset_index(drop=True)
            logger.info(
                f"  Removed all {len(to_remove)} ambiguous gene name(s) "
                f"({len(df)} entries remain)"
            )

        else:  # 'id'
            # R adds the *first* gene_id per group to the remove list,
            # then filters out only those specific IDs.
            # This keeps n-1 entries per duplicated group — faithful replication
            # of the R behaviour even though it is asymmetric with 'symbol' mode.
            first_ids_per_group = (
                dup_df.groupby("gene_name", sort=False)["gene_id"].first()
            )
            to_remove_ids: set[str] = set(first_ids_per_group.values)
            df = df[~df["gene_id"].isin(to_remove_ids)].reset_index(drop=True)
            logger.info(
                f"  Removed {len(to_remove_ids)} gene IDs "
                f"(first entry of each duplicate group; {len(df)} remain)"
            )

    # ---- Step 6: set index ----
    if gene_name_type == "symbol":
        df = df.set_index("gene_name")
    else:
        df = df.set_index("gene_id")

    logger.info(
        f"deal_gene_information: returning {len(df)} "
        f"gene_id <-> gene_name pairs"
    )
    return df


# ---------------------------------------------------------------------------
# calculate_precision_recall
# ---------------------------------------------------------------------------


def calculate_precision_recall(
    scnetwork_weights: "np.ndarray | pd.DataFrame",
    tf_target_pair: "Collection[str]",
    top_number: int = 1000,
    gene_id_gene_name_pair: "Optional[pd.DataFrame]" = None,
    gene_name_type: "Optional[str]" = None,
    gene_names: "Optional[list[str]]" = None,
) -> tuple[float, float]:
    
    # ---- resolve matrix + gene labels ----
    if isinstance(scnetwork_weights, pd.DataFrame):
        mat = scnetwork_weights.to_numpy(dtype=float)
        labels: list[str] = scnetwork_weights.index.tolist()
    else:
        mat = np.asarray(scnetwork_weights, dtype=float)
        if gene_names is None:
            raise ValueError(
                "gene_names must be provided when scnetwork_weights is a numpy array"
            )
        labels = list(gene_names)

    if gene_name_type is not None:
        if gene_id_gene_name_pair is None:
            raise ValueError(
                "gene_id_gene_name_pair must be provided when gene_name_type is set"
            )
        if gene_name_type not in ("symbol", "id"):
            raise ValueError(
                f"gene_name_type must be 'symbol' or 'id', got {gene_name_type!r}"
            )

    n = len(labels)
    labels_arr = np.asarray(labels)

    # ---- rank edges by weight, keep only what we need ---------------
    flat = mat.ravel()

    # NaN mask (na.rm=TRUE in R)
    valid_mask = ~np.isnan(flat)
    if not np.all(valid_mask):
        valid_idx   = np.where(valid_mask)[0]
        flat_valid  = flat[valid_idx]
    else:
        valid_idx   = np.arange(len(flat))
        flat_valid  = flat

    total_valid = len(flat_valid)
    k = min(top_number, total_valid)

    if k < total_valid:
        part = np.argpartition(-flat_valid, k - 1)[:k]
        top_local = part[np.argsort(-flat_valid[part], kind="stable")]
    else:
        top_local = np.argsort(-flat_valid, kind="stable")

    top_flat_idx = valid_idx[top_local]
    rows_top = top_flat_idx // n
    cols_top = top_flat_idx % n
    weights_top = flat[top_flat_idx]

    from_genes_arr = labels_arr[rows_top]
    to_genes_arr   = labels_arr[cols_top]
    pair_keys_top  = np.char.add(np.char.add(from_genes_arr, "_"), to_genes_arr)

    link = pd.DataFrame(
        {
            "from_gene": from_genes_arr,
            "to_gene":   to_genes_arr,
            "im":        weights_top,
            "pair_key":  pair_keys_top,
        }
    )

    # ---- build pair keys ----
    if gene_id_gene_name_pair is None:
        pass 
    elif gene_name_type == "symbol":
        lookup_col = "gene_id"
        link["from_gene_id"] = gene_id_gene_name_pair[lookup_col].reindex(link["from_gene"]).values
        link["to_gene_id"] = gene_id_gene_name_pair[lookup_col].reindex(link["to_gene"]).values
        link = link.dropna(subset=["from_gene_id", "to_gene_id"]).reset_index(drop=True)
        link["pair_key"] = link["from_gene"] + "_" + link["to_gene"]
    else:  
        lookup_col = "gene_name"
        link["from_symbol"] = gene_id_gene_name_pair[lookup_col].reindex(link["from_gene"]).values
        link["to_symbol"] = gene_id_gene_name_pair[lookup_col].reindex(link["to_gene"]).values
        link = link.dropna(subset=["from_symbol", "to_symbol"]).reset_index(drop=True)
        link["pair_key"] = link["from_symbol"] + "_" + link["to_symbol"]

    # ---- mark ground-truth true positives ----
    # Make sure tf_set is assigned unconditionally so it is safely in scope
    tf_set = (tf_target_pair
              if isinstance(tf_target_pair, (set, frozenset))
              else set(tf_target_pair))

    # ── Precision: TPs in the top-N rows already in `link` ──────────────────
    pair_key_arr_top = link["pair_key"].to_numpy(dtype=str)
    tp_mask_top = np.array([k in tf_set for k in pair_key_arr_top], dtype=bool)
    numerator   = int(tp_mask_top[:top_number].sum())

    precision_denominator = top_number
    precision             = numerator / precision_denominator

    # ── Recall denominator: TPs over the FULL edge list ─────────────────────
    if gene_id_gene_name_pair is None:
        # OPTIMIZED PATH: Iterate over the small tf_set rather than generating N^2 edge strings
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        recall_denominator = 0
        for pair in tf_set:
            try:
                tf, tg = pair.split('_')
                if tf in label_to_idx and tg in label_to_idx:
                    # Increment if the edge exists and is not padded with NaN
                    if not np.isnan(mat[label_to_idx[tf], label_to_idx[tg]]):
                        recall_denominator += 1
            except ValueError:
                pass
    else:
        all_rows_arr = valid_idx // n
        all_cols_arr = valid_idx % n
        all_from_raw = labels_arr[all_rows_arr]
        all_to_raw   = labels_arr[all_cols_arr]

        if gene_name_type == "symbol":
            lookup_col = "gene_id"
            all_from_id = gene_id_gene_name_pair[lookup_col].reindex(all_from_raw).values
            all_to_id   = gene_id_gene_name_pair[lookup_col].reindex(all_to_raw).values
            valid_both  = ~(pd.isna(all_from_id) | pd.isna(all_to_id))
            all_keys = np.char.add(np.char.add(all_from_raw[valid_both], "_"),
                                   all_to_raw[valid_both])
        else:  
            lookup_col = "gene_name"
            all_from_sym = gene_id_gene_name_pair[lookup_col].reindex(all_from_raw).values
            all_to_sym   = gene_id_gene_name_pair[lookup_col].reindex(all_to_raw).values
            valid_both   = ~(pd.isna(all_from_sym) | pd.isna(all_to_sym))
            all_keys = np.char.add(np.char.add(all_from_sym[valid_both].astype(str), "_"),
                                   all_to_sym[valid_both].astype(str))

        recall_denominator = int(sum(1 for k in all_keys if k in tf_set))

    if recall_denominator == 0:
        recall = float("nan")
    else:
        recall = numerator / recall_denominator

    return precision, recall
