"""Regulator enrichment analysis for single-cell regulatory networks.

Translates R functions from ScReNI:

    ``network_analysis()``              →  :func:`network_analysis`
      (``network_analysis.R``)

    ``Identify_enriched_scRegulators()``→  :func:`identify_enriched_scregulators`
      (``Identify_enriched_scRegulators.R``)

Internal helpers from ``reconstruct_network_part.R``:

    ``get_Enriched_TFs()``              →  :func:`_get_enriched_tfs`
    ``get_regulation_of_TFs_to_modules()`` → :func:`_get_regulation_of_tfs_to_modules`
    ``get_partial_regulations()``       →  :func:`_get_partial_regulations`
    ``merge_Module_Regulations()``      →  :func:`_merge_module_regulations`

Pipeline position
-----------------
These functions sit after clustering and network inference (rows 23-25) and
consume the k-means clustering result plus regulatory relationships::

    ScReniNetworks + clustering_kmeans
        → network_analysis()
            → TFs_list (enriched regulators, module regulations, networks)

Method overview
---------------
The regulator enrichment pipeline:

1. **Enrichment testing** (:func:`_get_enriched_tfs`):
   For each gene-module combination, performs hypergeometric tests to identify
   transcription factors (TFs) significantly enriched in regulating genes
   within that module.

2. **TF-module mapping** (:func:`_get_regulation_of_tfs_to_modules`):
   Maps which enriched TFs regulate which modules with positive or negative
   regulation.

3. **Network filtering** (:func:`_get_partial_regulations`):
   Extracts edges where both the TF and target are themselves enriched TFs.

4. **Inter-module analysis** (:func:`_merge_module_regulations`):
   Analyzes regulatory relationships between modules using hypergeometric
   tests.

Returns
-------
The main output is a ``TFs_list`` dictionary containing:

- ``Cor_TFs``: All TFs with correlation statistics
- ``Cor_EnTFs``: Enriched TFs only
- ``FOSF_RegMTF_Cor_EnTFs``: Regulatory edges involving enriched TFs
- ``FOSF_RegMTF_Cor_EnTFsTarg``: Edges where both TF and target are enriched
- ``FOSF_RegMTF_Cor_EnTFsTargM``: Within-module edges (both in same module)
- ``TF_list``: List of enriched TF names
- ``TF_module_regulation``: TF-to-module regulation summary
- ``TF_network``: Filtered network of enriched TF relationships
- ``intramodular_network``: Significant inter-module regulations
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------------------------


def _get_enriched_tfs(
    regulatory_relationships: pd.DataFrame,
    kmeans_result: pd.DataFrame,
    tf_fdr_thr: float = 10.0,
) -> dict:
    """Identify transcription factors enriched in regulating each module.

    For each TF and module combination, performs a hypergeometric test to
    determine if the TF is significantly enriched in regulating genes within
    that module (separately for positive and negative correlations).

    Matches R's ``get_Enriched_TFs()`` from ``reconstruct_network_part.R``.

    Parameters
    ----------
    regulatory_relationships
        DataFrame with columns:
        - ``TF``: Transcription factor gene name
        - ``Target``: Target gene name
        - ``Correlation``: Regulatory weight/correlation
        Must have row index matching ``kmeans_result`` row index.
    kmeans_result
        DataFrame with columns:
        - ``KmeansGroup``: Module assignment (integer)
        - ``Symbol``: Gene symbol (optional)
        Row index must be gene names matching TF/Target in regulatory_relationships.
    tf_fdr_thr
        -log10(FDR) threshold for calling a TF enriched. Default 10 (-log10(1e-10)).

    Returns
    -------
    dict
        With keys:
        - ``Cor_TFs``: DataFrame of all TFs with enrichment statistics
        - ``Cor_EnTFs``: DataFrame of significantly enriched TFs
        - ``FOSF_RegMTF_Cor_EnTFs``: Regulatory edges where TF is enriched
        - ``FOSF_RegMTF_Cor_EnTFsTarg``: Edges where both TF and target are enriched
        - ``FOSF_RegMTF_Cor_EnTFsTargM``: Within-module edges
    """
    # Validate inputs
    if "Correlation" not in regulatory_relationships.columns:
        raise ValueError("regulatory_relationships must contain 'Correlation' column")
    if "TF" not in regulatory_relationships.columns:
        raise ValueError("regulatory_relationships must contain 'TF' column")
    if "Target" not in regulatory_relationships.columns:
        raise ValueError("regulatory_relationships must contain 'Target' column")
    if "KmeansGroup" not in kmeans_result.columns:
        raise ValueError("kmeans_result must contain 'KmeansGroup' column")

    # Check that all TFs and Targets are in kmeans_result
    missing_tfs = set(regulatory_relationships["TF"]) - set(kmeans_result.index)
    if missing_tfs:
        raise ValueError(
            f"kmeans_result row index must contain all TFs. Missing: {missing_tfs}"
        )
    missing_targets = set(regulatory_relationships["Target"]) - set(kmeans_result.index)
    if missing_targets:
        raise ValueError(
            f"kmeans_result row index must contain all Targets. Missing: {missing_targets}"
        )

    # Split into positive and negative correlations
    gene_cor = regulatory_relationships.copy()
    gene_cor["TF"] = gene_cor["TF"].astype("category")
    gene_cor_p = gene_cor[gene_cor["Correlation"] > 0].copy()
    gene_cor_n = gene_cor[gene_cor["Correlation"] < 0].copy()

    # Count edges per TF
    tf_all = gene_cor["TF"].value_counts()
    tf_zeros = pd.Series(0, index=tf_all.index)
    tf_p_counts = gene_cor_p["TF"].value_counts()
    tf_n_counts = gene_cor_n["TF"].value_counts()

    tf_p = tf_zeros.copy()
    tf_p.update(tf_p_counts)
    tf_n = tf_zeros.copy()
    tf_n.update(tf_n_counts)

    # Get unique modules
    u_group = sorted(kmeans_result["KmeansGroup"].unique())

    # Build enrichment table
    results_list = []

    for module_i in u_group:
        # Get genes in this module
        module_genes = kmeans_result.index[kmeans_result["KmeansGroup"] == module_i].tolist()

        # Test positive and negative separately
        for reg_type_idx, (gene_cor_pn, tf_counts) in enumerate(
            [(gene_cor_p, tf_p), (gene_cor_n, tf_n)]
        ):
            # Edges targeting this module
            gene_cor_module = gene_cor_pn[gene_cor_pn["Target"].isin(module_genes)]

            # Count edges per TF in this module
            tf_module_counts = gene_cor_module["TF"].value_counts()
            tf_module = tf_zeros.copy()
            tf_module.update(tf_module_counts)

            # Build contingency table for hypergeometric test
            # For each TF:
            #   x = edges from TF to module
            #   M = total edges from TF
            #   N = total edges NOT from TF
            #   n = total edges to module
            tf_table = pd.DataFrame({
                "x": tf_module,
                "M": tf_counts,
                "N": len(gene_cor_pn) - tf_counts,
                "n": len(gene_cor_module),
            })

            # Hypergeometric test: P(X >= x) where X ~ Hypergeom(M, N, n)
            p_values = []
            for idx in tf_table.index:
                row = tf_table.loc[idx]
                x, M, N, n = row["x"], row["M"], row["N"], row["n"]

                # Filter low counts (< 5 and < 2% of module size)
                if x == 0 or (x < 5 and x < 0.02 * n):
                    p_val = 1.0
                else:
                    # phyper(x, M, N, n, lower.tail=FALSE) = P(X > x) = 1 - P(X <= x)
                    # In scipy: sf(x-1) = P(X > x-1) = P(X >= x)
                    p_val = hypergeom.sf(x - 1, M + N, M, n)

                p_values.append(p_val)

            # FDR correction
            from statsmodels.stats.multitest import fdrcorrection
            _, fdr_values = fdrcorrection(p_values, alpha=0.05, method="indep")

            # Store results
            reg_type = "P" if reg_type_idx == 0 else "N"
            results_list.append({
                "module": module_i,
                "type": reg_type,
                "counts": tf_table.apply(lambda r: f"{int(r['x'])};{int(r['M'])};{int(r['N'])};{int(r['n'])}", axis=1),
                "p": pd.Series(p_values, index=tf_table.index),
                "fdr": pd.Series(fdr_values, index=tf_table.index),
            })

    # Combine results into wide format
    all_tfs = tf_all.index.tolist()
    result_df = pd.DataFrame(index=all_tfs)

    # Add module assignment for each TF
    result_df = result_df.join(kmeans_result[["KmeansGroup"]], how="left")
    if "Symbol" in kmeans_result.columns:
        result_df = result_df.join(kmeans_result[["Symbol"]], how="left")

    # Add enrichment statistics for each module
    for res in results_list:
        module = res["module"]
        reg_type = res["type"]
        prefix = f"{reg_type}fdr{module}"

        # Add columns: Pnum, Pp, Pfdr OR Nnum, Np, Nfdr
        result_df[f"{reg_type}num{module}"] = res["counts"]
        result_df[f"{reg_type}p{module}"] = res["p"]
        result_df[prefix] = res["fdr"]

    # Compute minimum FDR and identify enriched modules
    fdr_cols = [c for c in result_df.columns if any(c.endswith(f"fdr{m}") for m in u_group)]
    fdr_vals = result_df[fdr_cols].values
    neg_log_fdr = -np.log10(fdr_vals + 1e-300)  # avoid log(0)

    result_df["TFMinNlogfdr"] = neg_log_fdr.max(axis=1)
    max_idx = neg_log_fdr.argmax(axis=1)
    result_df["TFMinGroup"] = [fdr_cols[i] for i in max_idx]

    # Extract group type (P or N) and module number
    def parse_min_group(col_name):
        # Format: Pfdr1 or Nfdr2
        reg = col_name[0]  # P or N
        mod = col_name[4:]  # module number
        return f"{reg}{mod}"

    result_df["TFMinGroup"] = result_df["TFMinGroup"].apply(parse_min_group)

    # Identify significantly activated/repressed modules
    def get_sig_modules(row, threshold):
        p_modules = []
        n_modules = []
        for i, module in enumerate(u_group):
            p_col = f"Pfdr{module}"
            n_col = f"Nfdr{module}"
            if p_col in row.index and -np.log10(row[p_col] + 1e-300) > threshold:
                p_modules.append(str(i + 1))
            if n_col in row.index and -np.log10(row[n_col] + 1e-300) > threshold:
                n_modules.append(str(i + 1))
        return ";".join(p_modules) if p_modules else "NA", ";".join(n_modules) if n_modules else "NA"

    sig_act, sig_rep = zip(*result_df.apply(lambda row: get_sig_modules(row, tf_fdr_thr), axis=1))
    result_df["SigActModules"] = sig_act
    result_df["SigRepModules"] = sig_rep

    # Identify enriched TFs
    enriched_mask = result_df["TFMinNlogfdr"] > tf_fdr_thr
    enriched_tfs = result_df[enriched_mask].copy()

    logger.info(f"Total TFs: {len(result_df)}")
    logger.info(f"Enriched TFs: {len(enriched_tfs)}")

    # Annotate regulatory relationships with TF enrichment info
    gene_cor_annotated = gene_cor.copy()

    # Add TF enrichment columns
    tf_enrich_cols = ["TFMinNlogfdr", "TFMinGroup", "SigActModules", "SigRepModules"]
    for col in tf_enrich_cols:
        gene_cor_annotated[f"TF{col}"] = gene_cor_annotated["TF"].map(result_df[col])

    # Add target module info
    gene_cor_annotated["TargetGroup"] = gene_cor_annotated["Target"].map(
        kmeans_result["KmeansGroup"]
    )
    gene_cor_annotated["TFGroup"] = gene_cor_annotated["TF"].map(
        kmeans_result["KmeansGroup"]
    )
    if "Symbol" in kmeans_result.columns:
        gene_cor_annotated["TFSymbol"] = gene_cor_annotated["TF"].map(
            kmeans_result["Symbol"]
        )
        gene_cor_annotated["TargetSymbol"] = gene_cor_annotated["Target"].map(
            kmeans_result["Symbol"]
        )

    # Add regulation type
    gene_cor_annotated["Regulation"] = gene_cor_annotated["Correlation"].apply(
        lambda x: "Positive" if x > 0 else "Negative"
    )

    # Filter to edges involving enriched TFs
    enriched_tf_names = enriched_tfs.index.tolist()
    en_tf_reg = gene_cor_annotated[gene_cor_annotated["TF"].isin(enriched_tf_names)].copy()

    # Edges where both TF and target are enriched
    en_tf_targ = gene_cor_annotated[
        gene_cor_annotated["TF"].isin(enriched_tf_names)
        & gene_cor_annotated["Target"].isin(enriched_tf_names)
    ].copy()

    # Within-module edges (both TF and target in same module)
    en_tf_targ_m = en_tf_targ[en_tf_targ["TFGroup"] == en_tf_targ["TargetGroup"]].copy()

    return {
        "Cor_TFs": result_df,
        "Cor_EnTFs": enriched_tfs,
        "FOSF_RegMTF_Cor_EnTFs": en_tf_reg,
        "FOSF_RegMTF_Cor_EnTFsTarg": en_tf_targ,
        "FOSF_RegMTF_Cor_EnTFsTargM": en_tf_targ_m,
    }


def _get_regulation_of_tfs_to_modules(
    tfs_list: dict,
    threshold: float = 2.0,
) -> dict:
    """Extract TF-to-module regulatory relationships.

    For each enriched TF, identifies which modules it regulates
    (positive or negative) based on FDR threshold.

    Matches R's ``get_regulation_of_TFs_to_modules()`` from
    ``reconstruct_network_part.R``.

    Parameters
    ----------
    tfs_list
        Dictionary from :func:`_get_enriched_tfs` containing ``Cor_EnTFs``.
    threshold
        -log10(FDR) threshold for significant regulation. Default 2.

    Returns
    -------
    dict
        Updated ``tfs_list`` with added keys:
        - ``TF_list``: List of enriched TF names
        - ``TF_module_regulation``: DataFrame of TF-module regulations
    """
    enriched_tfs = tfs_list["Cor_EnTFs"]

    # Extract FDR columns (format: Pfdr1, Nfdr2, etc.)
    fdr_pattern = r"^[PN]fdr\d+$"
    fdr_cols = [c for c in enriched_tfs.columns if pd.Series([c]).str.match(fdr_pattern).iloc[0]]

    # Build TF-module regulation table
    regulations = []

    for tf_idx, row in enriched_tfs.iterrows():
        tf_symbol = row.get("Symbol", tf_idx)
        tf_group = row.get("KmeansGroup", "NA")

        for col in fdr_cols:
            fdr = row[col]
            if fdr == 0:
                neg_log_fdr = np.inf
            else:
                neg_log_fdr = -np.log10(fdr)

            if neg_log_fdr > threshold:
                # Parse column name: Pfdr1 -> regulation=Positive, module=1
                reg_type = "Positive" if col.startswith("P") else "Negative"
                module = col[4:]  # Extract module number

                regulations.append({
                    "TF": tf_idx,
                    "TFSymbol": tf_symbol,
                    "TFGroup": tf_group,
                    "TargetModule": f"Group{module}",
                    "TargetGroup": module,
                    "Regulation": reg_type,
                    "Nlogfdr": neg_log_fdr,
                })

    tf_module_reg = pd.DataFrame(regulations)

    # Get unique TF list
    tf_list = enriched_tfs.index.tolist()

    tfs_list["TF_list"] = tf_list
    tfs_list["TF_module_regulation"] = tf_module_reg

    return tfs_list


def _get_partial_regulations(tfs_list: dict) -> dict:
    """Filter to regulatory edges where both TF and target are enriched.

    Matches R's ``get_partial_regulations()`` from ``reconstruct_network_part.R``.

    Parameters
    ----------
    tfs_list
        Dictionary from previous steps containing:
        - ``FOSF_RegMTF_Cor_EnTFs``: All edges with enriched TFs
        - ``TF_list``: List of enriched TF names

    Returns
    -------
    dict
        Updated ``tfs_list`` with added key:
        - ``TF_network``: DataFrame of TF-TF edges
    """
    edges = tfs_list["FOSF_RegMTF_Cor_EnTFs"]
    enriched_tfs = tfs_list["TF_list"]

    # Filter to rows where both TF and Target are in enriched list
    tf_network = edges[
        edges["TF"].isin(enriched_tfs) & edges["Target"].isin(enriched_tfs)
    ].copy()

    tfs_list["TF_network"] = tf_network

    return tfs_list


def _merge_module_regulations(
    tfs_list: dict,
    kmeans_result: pd.DataFrame,
    module_fdr: float = 0.05,
) -> dict:
    """Analyze inter-module regulatory relationships.

    For each pair of modules, performs hypergeometric tests to identify
    significant regulatory relationships (positive and negative).

    Matches R's ``merge_Module_Regulations()`` from ``reconstruct_network_part.R``.

    Parameters
    ----------
    tfs_list
        Dictionary from previous steps containing ``FOSF_RegMTF_Cor_EnTFsTarg``.
    kmeans_result
        K-means clustering result with ``KmeansGroup`` column.
    module_fdr
        FDR threshold for significant inter-module regulations. Default 0.05.

    Returns
    -------
    dict
        Updated ``tfs_list`` with added key:
        - ``intramodular_network``: DataFrame of significant module-module regulations
    """
    regulation = tfs_list["FOSF_RegMTF_Cor_EnTFsTarg"].copy()
    regulation["Correlation"] = pd.to_numeric(regulation["Correlation"])

    # Get unique modules
    modules = sorted(kmeans_result["KmeansGroup"].unique())

    # Split by regulation type
    reg_pos = regulation[regulation["Regulation"] == "Positive"]
    reg_neg = regulation[regulation["Regulation"] == "Negative"]

    # Build inter-module statistics
    results = []

    for i, mod_i in enumerate(modules):
        # Edges from module i
        reg_from_i = regulation[regulation["TFGroup"] == mod_i]
        reg_from_i_pos = reg_from_i[reg_from_i["Regulation"] == "Positive"]
        reg_from_i_neg = reg_from_i[reg_from_i["Regulation"] == "Negative"]

        # Edges to module i
        reg_to_i_pos = reg_pos[reg_pos["TargetGroup"] == mod_i]
        reg_to_i_neg = reg_neg[reg_neg["TargetGroup"] == mod_i]

        for j, mod_j in enumerate(modules[i:], start=i):
            # Edges from module j
            reg_from_j = regulation[regulation["TFGroup"] == mod_j]
            reg_from_j_pos = reg_from_j[reg_from_j["Regulation"] == "Positive"]
            reg_from_j_neg = reg_from_j[reg_from_j["Regulation"] == "Negative"]

            # Edges to module j
            reg_to_j_pos = reg_pos[reg_pos["TargetGroup"] == mod_j]
            reg_to_j_neg = reg_neg[reg_neg["TargetGroup"] == mod_j]

            # i->j edges
            reg_ij_pos = reg_from_i_pos[reg_from_i_pos["TargetGroup"] == mod_j]
            reg_ij_neg = reg_from_i_neg[reg_from_i_neg["TargetGroup"] == mod_j]

            # j->i edges
            reg_ji_pos = reg_from_j_pos[reg_from_j_pos["TargetGroup"] == mod_i]
            reg_ji_neg = reg_from_j_neg[reg_from_j_neg["TargetGroup"] == mod_i]

            # Build contingency stats for i->j positive
            ij_pos_stats = [
                mod_i, mod_j, "Positive",
                reg_ij_pos["Correlation"].mean() if len(reg_ij_pos) > 0 else np.nan,
                len(reg_ij_pos), len(reg_to_j_pos),
                len(reg_pos) - len(reg_to_j_pos), len(reg_from_i_pos)
            ]

            # i->j negative
            ij_neg_stats = [
                mod_i, mod_j, "Negative",
                reg_ij_neg["Correlation"].mean() if len(reg_ij_neg) > 0 else np.nan,
                len(reg_ij_neg), len(reg_to_j_neg),
                len(reg_neg) - len(reg_to_j_neg), len(reg_from_i_neg)
            ]

            # j->i positive
            ji_pos_stats = [
                mod_j, mod_i, "Positive",
                reg_ji_pos["Correlation"].mean() if len(reg_ji_pos) > 0 else np.nan,
                len(reg_ji_pos), len(reg_to_i_pos),
                len(reg_pos) - len(reg_to_i_pos), len(reg_from_j_pos)
            ]

            # j->i negative
            ji_neg_stats = [
                mod_j, mod_i, "Negative",
                reg_ji_neg["Correlation"].mean() if len(reg_ji_neg) > 0 else np.nan,
                len(reg_ji_neg), len(reg_to_i_neg),
                len(reg_neg) - len(reg_to_i_neg), len(reg_from_j_neg)
            ]

            if i == j:
                # Same module, only add i->j
                results.append(ij_pos_stats)
                results.append(ij_neg_stats)
            else:
                # Different modules, add both directions
                results.append(ij_pos_stats)
                results.append(ij_neg_stats)
                results.append(ji_pos_stats)
                results.append(ji_neg_stats)

    # Create DataFrame
    result_df = pd.DataFrame(
        results,
        columns=[
            "TFGroup", "TargetGroup", "Regulation", "Correlation",
            "x", "M", "N", "n"
        ]
    )

    # Remove rows with NaN correlation (no edges)
    result_df = result_df.dropna(subset=["Correlation"])

    # Hypergeometric test: P(X >= x | M, N, n)
    p_values = []
    for _, row in result_df.iterrows():
        x, M, N, n = int(row["x"]), int(row["M"]), int(row["N"]), int(row["n"])

        if x < 4:
            p_val = 1.0
        else:
            p_val = hypergeom.sf(x - 1, M + N, M, n)

        p_values.append(p_val)

    # FDR correction
    from statsmodels.stats.multitest import fdrcorrection
    _, fdr_values = fdrcorrection(p_values, alpha=0.05, method="indep")

    # Add results
    result_df["NumberRegulation"] = [
        f"{int(row['x'])};{int(row['M'])};{int(row['N'])};{int(row['n'])}"
        for _, row in result_df.iterrows()
    ]
    result_df["Pvalue"] = p_values
    result_df["NlogFdr"] = -np.log10(fdr_values + 1e-300)

    # Filter to significant regulations
    sig_regs = result_df[result_df["NlogFdr"] > -np.log10(module_fdr)].copy()
    sig_regs = sig_regs.sort_values(by=["TFGroup", "TargetGroup"])

    logger.info(f"Significant regulations: {len(sig_regs)}")

    tfs_list["intramodular_network"] = sig_regs

    return tfs_list


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


def network_analysis(
    regulatory_relationships: pd.DataFrame,
    kmeans_result: pd.DataFrame,
    tf_fdr_1: float = 10.0,
    tf_fdr_2: float = 10.0,
    module_fdr: float = 0.05,
) -> dict:
    """Identify enriched regulators and analyze regulatory structure.

    Main function that orchestrates the complete regulator enrichment pipeline.

    Matches R's ``network_analysis()`` from ``network_analysis.R``.

    Parameters
    ----------
    regulatory_relationships
        DataFrame with TF-target edges. Required columns:
        - ``TF``: Transcription factor gene name
        - ``Target``: Target gene name
        - ``Correlation``: Regulatory weight
    kmeans_result
        K-means clustering result from :func:`~screni.data.clustering.clustering_kmeans`.
        Required columns:
        - ``KmeansGroup``: Module assignment
        Row index must match gene names in regulatory_relationships.
    tf_fdr_1
        -log10(FDR) threshold for TF enrichment. Default 10.
    tf_fdr_2
        -log10(FDR) threshold for TF-module regulation. Default 10.
    module_fdr
        FDR threshold for inter-module regulations. Default 0.05.

    Returns
    -------
    dict
        ``TFs_list`` containing:

        - ``Cor_TFs``: All TFs with enrichment statistics
        - ``Cor_EnTFs``: Enriched TFs only
        - ``FOSF_RegMTF_Cor_EnTFs``: Edges with enriched TFs as regulators
        - ``FOSF_RegMTF_Cor_EnTFsTarg``: Edges where both TF and target are enriched
        - ``FOSF_RegMTF_Cor_EnTFsTargM``: Within-module enriched TF edges
        - ``TF_list``: List of enriched TF names
        - ``TF_module_regulation``: TF-to-module regulation summary
        - ``TF_network``: Filtered network of enriched TF relationships
        - ``intramodular_network``: Significant inter-module regulations

    Examples
    --------
    >>> # After running clustering_kmeans and building regulatory network
    >>> tfs_list = network_analysis(
    ...     regulatory_relationships=network_edges,
    ...     kmeans_result=kmeans_result,
    ...     tf_fdr_1=10,
    ...     tf_fdr_2=10,
    ...     module_fdr=0.05
    ... )
    >>> enriched_tfs = tfs_list["Cor_EnTFs"]
    >>> tf_network = tfs_list["TF_network"]
    """
    # Validate inputs
    if "Correlation" not in regulatory_relationships.columns:
        raise ValueError("regulatory_relationships must contain 'Correlation' column")
    if "TF" not in regulatory_relationships.columns:
        raise ValueError("regulatory_relationships must contain 'TF' column")
    if "Target" not in regulatory_relationships.columns:
        raise ValueError("regulatory_relationships must contain 'Target' column")
    if "KmeansGroup" not in kmeans_result.columns:
        raise ValueError("kmeans_result must contain 'KmeansGroup' column")

    # Check that all TFs and targets are in kmeans_result
    missing_tfs = set(regulatory_relationships["TF"]) - set(kmeans_result.index)
    if missing_tfs:
        raise ValueError(
            f"kmeans_result must contain all TFs from regulatory_relationships. "
            f"Missing: {list(missing_tfs)[:5]}..."
        )

    missing_targets = set(regulatory_relationships["Target"]) - set(kmeans_result.index)
    if missing_targets:
        raise ValueError(
            f"kmeans_result must contain all Targets from regulatory_relationships. "
            f"Missing: {list(missing_targets)[:5]}..."
        )

    # Step 1: Identify enriched TFs
    logger.info("Step 1: Identifying enriched TFs...")
    tfs_list = _get_enriched_tfs(regulatory_relationships, kmeans_result, tf_fdr_1)

    # Step 2: Map TF-to-module regulations
    logger.info("Step 2: Mapping TF-to-module regulations...")
    tfs_list = _get_regulation_of_tfs_to_modules(tfs_list, tf_fdr_2)

    # Step 3: Extract TF-TF network
    logger.info("Step 3: Extracting TF-TF network...")
    tfs_list = _get_partial_regulations(tfs_list)

    # Step 4: Analyze inter-module regulations
    logger.info("Step 4: Analyzing inter-module regulations...")
    tfs_list = _merge_module_regulations(tfs_list, kmeans_result, module_fdr)

    return tfs_list


def identify_enriched_scregulators(
    regulatory_network: pd.DataFrame,
    kmeans_clustering: pd.DataFrame,
    tf_fdr_1: float = 10.0,
    tf_fdr_2: float = 10.0,
) -> dict:
    """Identify regulators enriched in single-cell regulatory networks.

    Wrapper around :func:`network_analysis` that matches the original R function
    ``Identify_enriched_scRegulators()``.

    Parameters
    ----------
    regulatory_network
        DataFrame with TF-target edges (same as ``regulatory_relationships``
        in :func:`network_analysis`).
    kmeans_clustering
        K-means clustering result (same as ``kmeans_result`` in
        :func:`network_analysis`).
    tf_fdr_1
        -log10(FDR) threshold for TF enrichment. Default 10.
    tf_fdr_2
        -log10(FDR) threshold for TF-module regulation. Default 10.

    Returns
    -------
    dict
        ``TFs_list`` from :func:`network_analysis`.

    Examples
    --------
    >>> tfs_list = identify_enriched_scregulators(
    ...     regulatory_network=edges_df,
    ...     kmeans_clustering=kmeans_result,
    ...     tf_fdr_1=10,
    ...     tf_fdr_2=10
    ... )
    """
    return network_analysis(
        regulatory_relationships=regulatory_network,
        kmeans_result=kmeans_clustering,
        tf_fdr_1=tf_fdr_1,
        tf_fdr_2=tf_fdr_2,
    )
