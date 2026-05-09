"""Differential edge analysis for SQ1.

Compares per-cell wScReNI weight matrices between two donor groups
(control vs ad) and produces a ranked TF->target edge table per cell type.

Pipeline
--------
1. :func:`pseudobulk_per_donor` averages each donor's per-cell weight
   matrices into one (n_genes, n_genes) pseudobulk matrix per donor.
2. :func:`differential_edges` runs the per-edge two-group test over donor
   pseudobulks, with covariate adjustment, and returns a long-form
   DataFrame with q-values.

Test choice (covariate-adjusted regression)
-------------------------------------------
For each candidate edge (TF, target) we fit::

    weight ~ condition + age + sex + LATE_present + LBD_present

via OLS over the donor pseudobulks.  The condition coefficient's
two-sided t-statistic and p-value are recorded; BH-FDR is applied
across edges.

Why this and not Wilcoxon: the SeaAD case-control cohort has a known
age imbalance (controls median ~10 years younger).  Plain Wilcoxon
would either ignore that confounder or force us to age-match (losing
control donors we already barely have).  Covariate-adjusted regression
is the standard pseudobulk-DE approach (limma/DESeq2 use the same
pattern) and explicitly accounts for age, sex, and co-pathology.
See ``progress_log.md`` for the full rationale.

The test is implemented as a swappable callable so a Wilcoxon variant
can be plugged in for sensitivity analysis.

Edge universe
-------------
The differential test runs only on the candidate TF->target edges
recovered by Phase 3 (the ``triplets`` DataFrame), not on all
gene x gene pairs.  This is what keeps the FDR penalty manageable
(a few thousand edges instead of 250k).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from screni.data.combine import ScReniNetworks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class DonorPseudobulks:
    """Per-donor pseudobulk weight matrices for one cell type.

    Attributes
    ----------
    weights
        ``{donor_id: (n_genes, n_genes) ndarray}`` — averaged per-cell
        wScReNI weights.  Matrix orientation matches wScReNI: ``[i, j]`` =
        regulatory weight of gene ``j`` -> gene ``i``.
    gene_names
        Ordered gene labels shared by every matrix.
    metadata
        DataFrame indexed by donor_id with at least the columns
        ``condition`` (str), ``age`` (float), ``sex`` (str),
        ``LATE_present`` (bool), ``LBD_present`` (bool), ``n_cells`` (int).
    """

    weights: dict[str, np.ndarray]
    gene_names: list[str]
    metadata: pd.DataFrame

    @property
    def donor_ids(self) -> list[str]:
        return list(self.weights.keys())


# ---------------------------------------------------------------------------
# Pseudobulking
# ---------------------------------------------------------------------------


def pseudobulk_per_donor(
    networks: ScReniNetworks,
    cell_to_donor: dict[str, str] | pd.Series,
    donor_metadata: pd.DataFrame,
    aggregator: str = "mean",
) -> DonorPseudobulks:
    """Average per-cell wScReNI weight matrices to one matrix per donor.

    Parameters
    ----------
    networks
        ``{cell_name: (n_genes, n_genes) ndarray}`` from
        :func:`screni.data.inference.infer_wscreni_networks` or
        :func:`screni.data.combine.combine_wscreni_networks`.
    cell_to_donor
        Mapping ``cell_name -> donor_id``.  Must cover every cell in
        ``networks``; cells without a mapping raise.
    donor_metadata
        DataFrame indexed by donor_id with columns required by
        :func:`differential_edges` (see ``DonorPseudobulks.metadata``).
        Donors with no cells in ``networks`` are dropped from the result.
    aggregator
        ``'mean'`` or ``'median'`` across cells per donor.

    Returns
    -------
    DonorPseudobulks with one matrix per donor that has at least one cell.

    Raises
    ------
    ValueError if any cell has no donor mapping, or if a donor's cells
    yield zero usable weight matrices, or if matrices have inconsistent
    shapes.
    """
    if isinstance(cell_to_donor, pd.Series):
        cell_to_donor = cell_to_donor.to_dict()

    missing = [c for c in networks.keys() if c not in cell_to_donor]
    if missing:
        raise ValueError(
            f"{len(missing)} cells in networks have no donor mapping "
            f"(e.g. {missing[:3]})"
        )

    if aggregator not in ("mean", "median"):
        raise ValueError("aggregator must be 'mean' or 'median'")

    # Group cell names by donor
    donor_cells: dict[str, list[str]] = {}
    for cell, donor in cell_to_donor.items():
        if cell in networks:
            donor_cells.setdefault(donor, []).append(cell)

    gene_names = list(getattr(networks, "gene_names", []))
    weights: dict[str, np.ndarray] = {}
    for donor, cells in donor_cells.items():
        stack = np.stack([np.asarray(networks[c]) for c in cells], axis=0)
        if stack.ndim != 3:
            raise ValueError(f"Donor {donor}: expected 3D stack, got {stack.shape}")
        if not gene_names:
            n = stack.shape[1]
            gene_names = [f"gene_{i}" for i in range(n)]
        if stack.shape[1] != len(gene_names) or stack.shape[2] != len(gene_names):
            raise ValueError(
                f"Donor {donor}: matrix shape {stack.shape[1:]} "
                f"!= ({len(gene_names)}, {len(gene_names)})"
            )
        weights[donor] = (
            stack.mean(axis=0) if aggregator == "mean" else np.median(stack, axis=0)
        )

    meta = donor_metadata.copy()
    if meta.index.name != "donor_id" and "donor_id" in meta.columns:
        meta = meta.set_index("donor_id")
    meta = meta.loc[meta.index.intersection(weights.keys())]
    if meta.empty:
        raise ValueError(
            "No donors in donor_metadata also have cells in networks; "
            "check that index labels match."
        )

    # Keep only donors with both data + metadata
    weights = {d: w for d, w in weights.items() if d in meta.index}
    meta = meta.loc[list(weights.keys())]

    logger.info(
        f"  pseudobulk: {len(weights)} donors, "
        f"{meta['condition'].value_counts().to_dict() if 'condition' in meta else 'no condition col'}"
    )
    return DonorPseudobulks(weights=weights, gene_names=gene_names, metadata=meta)


# ---------------------------------------------------------------------------
# Per-edge tests (swappable)
# ---------------------------------------------------------------------------


EdgeTest = Callable[[np.ndarray, pd.DataFrame], "EdgeTestResult"]


@dataclass
class EdgeTestResult:
    """Per-edge regression output."""

    coef: float        # condition coefficient (ad - control on the modelled scale)
    stderr: float
    t_stat: float
    p_value: float
    n_donors: int


def ols_with_covariates(
    edge_values: np.ndarray,
    metadata: pd.DataFrame,
    covariates: Iterable[str] = ("age", "sex", "LATE_present", "LBD_present"),
) -> EdgeTestResult:
    """OLS fit of ``weight ~ condition + covariates``.

    ``condition`` is expected to be coded as 'control' / 'ad'; the
    coefficient is for the ad-vs-control contrast (after dummy-coding).
    Categorical covariates ('sex') are dummy-coded automatically.
    Boolean covariates are cast to int.

    Returns
    -------
    EdgeTestResult.  If the design matrix is rank-deficient (e.g. sex
    constant on the donor pool), the offending covariate is dropped
    silently with a debug log message.
    """
    if "condition" not in metadata.columns:
        raise KeyError("metadata is missing 'condition' column")
    if len(edge_values) != len(metadata):
        raise ValueError(
            f"edge_values length {len(edge_values)} != n donors {len(metadata)}"
        )

    df = metadata.copy()
    df["_y"] = edge_values
    df["_condition_ad"] = (df["condition"].astype(str) == "ad").astype(int)

    cols = ["_condition_ad"]
    for c in covariates:
        if c not in df.columns:
            continue
        ser = df[c]
        if ser.dtype == bool or ser.dtype == "boolean":
            df[f"_cov_{c}"] = ser.astype(int)
            cols.append(f"_cov_{c}")
        elif ser.dtype.kind in "biufc":
            df[f"_cov_{c}"] = ser.astype(float)
            cols.append(f"_cov_{c}")
        else:
            # Categorical: dummy code (drop_first=True)
            dummies = pd.get_dummies(ser, prefix=f"_cov_{c}", drop_first=True)
            for dc in dummies.columns:
                df[dc] = dummies[dc].astype(int)
                cols.append(dc)

    # Drop covariate columns with zero variance (rank-deficient)
    keep = []
    for c in cols:
        if c == "_condition_ad" or df[c].nunique(dropna=False) > 1:
            keep.append(c)
        else:
            logger.debug(f"  dropping constant covariate {c}")
    cols = keep

    X = df[cols].astype(float).values
    X = sm.add_constant(X, has_constant="add")
    y = df["_y"].values.astype(float)
    try:
        fit = sm.OLS(y, X).fit()
    except Exception as e:
        logger.debug(f"  OLS failed: {e}")
        return EdgeTestResult(np.nan, np.nan, np.nan, np.nan, len(df))

    # condition_ad is column index 1 (0 = const, 1 = first design col)
    return EdgeTestResult(
        coef=float(fit.params[1]),
        stderr=float(fit.bse[1]),
        t_stat=float(fit.tvalues[1]),
        p_value=float(fit.pvalues[1]),
        n_donors=len(df),
    )


def wilcoxon_unadjusted(
    edge_values: np.ndarray,
    metadata: pd.DataFrame,
) -> EdgeTestResult:
    """Mann-Whitney U on ad vs control donors.  Sensitivity-analysis only.

    Provided as a swappable test for comparison with the covariate-adjusted
    OLS.  Reports the U-test p-value with a placeholder coef = mean(ad) -
    mean(control); ``stderr`` and ``t_stat`` are NaN.
    """
    is_ad = metadata["condition"].astype(str).values == "ad"
    a = edge_values[is_ad]
    b = edge_values[~is_ad]
    if len(a) < 1 or len(b) < 1:
        return EdgeTestResult(np.nan, np.nan, np.nan, np.nan, len(edge_values))
    try:
        u_res = stats.mannwhitneyu(a, b, alternative="two-sided")
        p = float(u_res.pvalue)
    except ValueError:
        p = np.nan
    return EdgeTestResult(
        coef=float(np.mean(a) - np.mean(b)),
        stderr=np.nan,
        t_stat=np.nan,
        p_value=p,
        n_donors=len(edge_values),
    )


# ---------------------------------------------------------------------------
# Multiple testing
# ---------------------------------------------------------------------------


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """BH FDR correction. NaNs are preserved at their positions and ignored
    in the ranking."""
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    if not np.any(valid):
        return q
    pv = p[valid]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    q_ranked = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity (running min from the right)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q_valid = np.empty(n)
    q_valid[order] = q_ranked
    q[valid] = q_valid
    return q


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def differential_edges(
    pseudobulks: DonorPseudobulks,
    edges: pd.DataFrame,
    test: EdgeTest = ols_with_covariates,
    cell_type: str | None = None,
) -> pd.DataFrame:
    """Per-edge differential test on donor pseudobulks.

    Parameters
    ----------
    pseudobulks
        Output of :func:`pseudobulk_per_donor`.
    edges
        DataFrame of candidate edges with columns ``TF`` and ``target_gene``
        (the Phase 3 triplets table works directly).  Both names must exist
        in ``pseudobulks.gene_names``; missing edges are skipped with a log.
    test
        Per-edge test callable.  Default :func:`ols_with_covariates`.
        Swap in :func:`wilcoxon_unadjusted` for a sensitivity check.
    cell_type
        Optional label written into the output table.

    Returns
    -------
    DataFrame with one row per tested edge, columns:
    ``cell_type, TF, target, mean_control, mean_ad, log2FC,
    coef, stderr, t_stat, p_value, q_value, n_donors``.
    Sorted by ``q_value`` ascending.
    """
    if not pseudobulks.weights:
        raise ValueError("pseudobulks contains no donors")
    if "TF" not in edges.columns or "target_gene" not in edges.columns:
        raise KeyError("edges must have 'TF' and 'target_gene' columns")

    gene_to_idx = {g: i for i, g in enumerate(pseudobulks.gene_names)}
    donors = list(pseudobulks.weights.keys())
    meta = pseudobulks.metadata.loc[donors]
    is_ad = meta["condition"].astype(str).values == "ad"

    # Stack weights into a single (n_donors, n_genes, n_genes) tensor for
    # vectorised mean/log2FC computation.  Then call the per-edge test
    # in a loop — that part is cheap.
    stack = np.stack([pseudobulks.weights[d] for d in donors], axis=0)

    rows = []
    skipped = 0
    for _, edge in edges[["TF", "target_gene"]].drop_duplicates().iterrows():
        tf = edge["TF"]
        target = edge["target_gene"]
        if tf not in gene_to_idx or target not in gene_to_idx:
            skipped += 1
            continue
        # wScReNI: weights[i, j] = j -> i (regulator j onto target i)
        i = gene_to_idx[target]
        j = gene_to_idx[tf]
        ev = stack[:, i, j]
        result = test(ev, meta)

        ctrl = ev[~is_ad]
        ad_arr = ev[is_ad]
        mean_ctrl = float(np.mean(ctrl)) if len(ctrl) else np.nan
        mean_ad = float(np.mean(ad_arr)) if len(ad_arr) else np.nan
        # log2FC with pseudocount so we don't blow up at zero
        eps = 1e-12
        log2fc = float(
            np.log2(max(abs(mean_ad), eps) / max(abs(mean_ctrl), eps))
            * np.sign(mean_ad - mean_ctrl)
            if not np.isnan(mean_ad) and not np.isnan(mean_ctrl)
            else np.nan
        )

        rows.append({
            "cell_type": cell_type,
            "TF": tf,
            "target": target,
            "mean_control": mean_ctrl,
            "mean_ad": mean_ad,
            "log2FC": log2fc,
            "coef": result.coef,
            "stderr": result.stderr,
            "t_stat": result.t_stat,
            "p_value": result.p_value,
            "n_donors": result.n_donors,
        })

    if skipped:
        logger.info(f"  skipped {skipped} edges with TF or target outside gene set")

    if not rows:
        logger.warning("No edges tested; returning empty DataFrame.")
        return pd.DataFrame(columns=[
            "cell_type", "TF", "target", "mean_control", "mean_ad",
            "log2FC", "coef", "stderr", "t_stat", "p_value", "q_value",
            "n_donors",
        ])

    df = pd.DataFrame(rows)
    df["q_value"] = benjamini_hochberg(df["p_value"].values)
    df = df.sort_values("q_value", kind="mergesort").reset_index(drop=True)
    logger.info(
        f"  differential_edges: {len(df)} tested, "
        f"{int((df['q_value'] < 0.05).sum())} with q<0.05"
    )
    return df


# ---------------------------------------------------------------------------
# Convenience: load networks + metadata from a driver run directory
# ---------------------------------------------------------------------------


def load_run_dir(run_dir: Path | str) -> tuple[ScReniNetworks, pd.DataFrame, pd.DataFrame]:
    """Load wScReNI outputs + donor metadata + triplets from one ``run_seaad_inference.py`` cell-type dir.

    Returns ``(networks, donor_metadata, triplets)``.
    """
    from screni.data.combine import combine_wscreni_networks

    run_dir = Path(run_dir)
    # Pull prefix from one of the *_donor_metadata.csv files
    meta_files = list(run_dir.glob("*_donor_metadata.csv"))
    if not meta_files:
        raise FileNotFoundError(f"No *_donor_metadata.csv in {run_dir}")
    meta_path = meta_files[0]
    prefix = meta_path.name.replace("_donor_metadata.csv", "")

    donor_metadata = pd.read_csv(meta_path).set_index("donor_id")
    triplets = pd.read_csv(run_dir / f"{prefix}_triplets.csv")

    rna_sub = (run_dir / f"{prefix}_rna_sub.h5ad")
    import anndata as ad
    adata = ad.read_h5ad(rna_sub)
    cell_names = list(adata.obs_names)

    networks = combine_wscreni_networks(
        cell_names=cell_names, network_dir=run_dir
    )
    return networks, donor_metadata, triplets
