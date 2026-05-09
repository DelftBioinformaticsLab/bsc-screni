"""Synthetic-fixture tests for ``screni.data.differential``.

The tests construct a small toy dataset where the ground truth is known:

* Two designed "differential" edges whose weight is a strong function of
  ``condition`` (one up in AD, one down in AD).
* All other candidate edges drawn from one common distribution, so they
  should NOT be detected.
* Donor-level age, sex, LATE, LBD added as covariates with no real effect
  on the differential edges, just to exercise the design matrix.

The headline assertion is that BH-FDR < 0.05 selects only the seeded
edges with high probability — verifying that the pipeline's signal
detection actually works end-to-end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screni.data.combine import ScReniNetworks
from screni.data.differential import (
    DonorPseudobulks,
    benjamini_hochberg,
    differential_edges,
    ols_with_covariates,
    pseudobulk_per_donor,
    wilcoxon_unadjusted,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_donors(
    n_genes: int = 20,
    n_control: int = 6,
    n_ad: int = 16,
    n_cells_per_donor: int = 30,
    diff_edges: list[tuple[int, int, float]] | None = None,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, str]]:
    """Synthesise per-cell weight matrices + donor metadata.

    Each cell's weight matrix is drawn from N(mu, 0.1) where ``mu`` is a
    random per-edge baseline.  For each ``(target, tf, delta)`` in
    ``diff_edges``, AD donors get an additional ``+delta`` shift on the
    edge ``(target, tf)``.

    Returns
    -------
    networks_dict, donor_metadata, cell_to_donor
    """
    rng = np.random.default_rng(seed)
    diff_edges = diff_edges or []

    baseline = rng.normal(loc=0.0, scale=0.05, size=(n_genes, n_genes))

    donors_ctrl = [f"D_ctrl_{i:02d}" for i in range(n_control)]
    donors_ad = [f"D_ad_{i:02d}" for i in range(n_ad)]
    all_donors = donors_ctrl + donors_ad

    # Donor metadata.  Confounder vars are random and uncorrelated with
    # condition apart from a deliberate small age offset (to mimic SeaAD).
    rows = []
    for d in donors_ctrl:
        rows.append({
            "donor_id": d,
            "condition": "control",
            "age": float(rng.normal(80, 5)),
            "sex": rng.choice(["Male", "Female"]),
            "LATE_present": bool(rng.choice([True, False])),
            "LBD_present": bool(rng.choice([True, False])),
            "n_cells": n_cells_per_donor,
        })
    for d in donors_ad:
        rows.append({
            "donor_id": d,
            "condition": "ad",
            "age": float(rng.normal(91, 5)),  # AD donors slightly older
            "sex": rng.choice(["Male", "Female"]),
            "LATE_present": bool(rng.choice([True, False])),
            "LBD_present": bool(rng.choice([True, False])),
            "n_cells": n_cells_per_donor,
        })
    donor_metadata = pd.DataFrame(rows).set_index("donor_id")

    networks: dict[str, np.ndarray] = {}
    cell_to_donor: dict[str, str] = {}
    for d in all_donors:
        is_ad = donor_metadata.loc[d, "condition"] == "ad"
        for c in range(n_cells_per_donor):
            cell = f"{d}_c{c:03d}"
            mat = baseline + rng.normal(scale=0.05, size=baseline.shape)
            if is_ad:
                for tgt, tf, delta in diff_edges:
                    mat[tgt, tf] += delta
            networks[cell] = mat
            cell_to_donor[cell] = d

    return networks, donor_metadata, cell_to_donor


def _make_networks_obj(networks_dict: dict[str, np.ndarray], n_genes: int) -> ScReniNetworks:
    n = ScReniNetworks(
        gene_names=[f"gene_{i:03d}" for i in range(n_genes)]
    )
    for k, v in networks_dict.items():
        n[k] = v
    return n


def _gene_name(i: int) -> str:
    return f"gene_{i:03d}"


# ---------------------------------------------------------------------------
# benjamini_hochberg
# ---------------------------------------------------------------------------


def test_benjamini_hochberg_simple():
    p = np.array([0.001, 0.01, 0.02, 0.5])
    q = benjamini_hochberg(p)
    # Smallest p-value: q = 0.001 * 4 / 1 = 0.004
    assert pytest.approx(q[0], abs=1e-6) == 0.004
    # All q's monotone non-decreasing in p-rank
    assert q[0] <= q[1] <= q[2] <= q[3]
    # All q's <= 1
    assert (q <= 1.0).all()


def test_benjamini_hochberg_handles_nan():
    p = np.array([0.01, np.nan, 0.5])
    q = benjamini_hochberg(p)
    assert np.isnan(q[1])
    assert not np.isnan(q[0])
    assert q[0] <= 0.05  # 0.01 * 2 / 1 = 0.02


def test_benjamini_hochberg_all_nan_returns_all_nan():
    p = np.array([np.nan, np.nan, np.nan])
    q = benjamini_hochberg(p)
    assert np.all(np.isnan(q))


# ---------------------------------------------------------------------------
# pseudobulk_per_donor
# ---------------------------------------------------------------------------


def test_pseudobulk_averages_match():
    networks_dict, donor_metadata, cell_to_donor = _make_synthetic_donors(
        n_genes=5, n_control=2, n_ad=2, n_cells_per_donor=10
    )
    net_obj = _make_networks_obj(networks_dict, n_genes=5)
    pb = pseudobulk_per_donor(net_obj, cell_to_donor, donor_metadata)

    assert set(pb.donor_ids) == set(donor_metadata.index)
    # Each pseudobulk is the mean of its donor's cells
    for d in pb.donor_ids:
        cells = [c for c, dd in cell_to_donor.items() if dd == d]
        expected = np.mean(np.stack([networks_dict[c] for c in cells]), axis=0)
        np.testing.assert_allclose(pb.weights[d], expected)


def test_pseudobulk_drops_donors_without_metadata():
    networks_dict, donor_metadata, cell_to_donor = _make_synthetic_donors(
        n_genes=3, n_control=2, n_ad=2, n_cells_per_donor=4
    )
    net_obj = _make_networks_obj(networks_dict, n_genes=3)
    # Drop one donor from metadata
    drop = donor_metadata.index[0]
    metadata_short = donor_metadata.drop(index=drop)
    pb = pseudobulk_per_donor(net_obj, cell_to_donor, metadata_short)
    assert drop not in pb.donor_ids
    assert len(pb.donor_ids) == len(metadata_short)


def test_pseudobulk_raises_on_missing_cell_mapping():
    networks_dict, donor_metadata, cell_to_donor = _make_synthetic_donors(
        n_genes=3, n_control=1, n_ad=1, n_cells_per_donor=3
    )
    net_obj = _make_networks_obj(networks_dict, n_genes=3)
    # Inject a cell with no donor entry
    net_obj["orphan_cell"] = np.zeros((3, 3))
    with pytest.raises(ValueError, match="no donor mapping"):
        pseudobulk_per_donor(net_obj, cell_to_donor, donor_metadata)


# ---------------------------------------------------------------------------
# ols_with_covariates per-edge test
# ---------------------------------------------------------------------------


def test_ols_finds_clear_signal():
    rng = np.random.default_rng(0)
    n = 22
    # Strong condition effect, no other association.
    df = pd.DataFrame({
        "condition": ["control"] * 6 + ["ad"] * 16,
        "age": rng.normal(85, 5, size=n),
        "sex": rng.choice(["Male", "Female"], size=n),
        "LATE_present": rng.choice([True, False], size=n),
        "LBD_present": rng.choice([True, False], size=n),
    })
    y = np.where(df["condition"] == "ad", 1.0, 0.0) + rng.normal(scale=0.05, size=n)
    res = ols_with_covariates(y, df)
    assert res.coef > 0.5  # AD shifts up by ~1
    assert res.p_value < 1e-6


def test_ols_returns_high_p_for_pure_noise():
    rng = np.random.default_rng(1)
    n = 22
    df = pd.DataFrame({
        "condition": ["control"] * 6 + ["ad"] * 16,
        "age": rng.normal(85, 5, size=n),
        "sex": rng.choice(["Male", "Female"], size=n),
        "LATE_present": rng.choice([True, False], size=n),
        "LBD_present": rng.choice([True, False], size=n),
    })
    y = rng.normal(size=n)
    res = ols_with_covariates(y, df)
    assert res.p_value > 0.05


# ---------------------------------------------------------------------------
# differential_edges end-to-end
# ---------------------------------------------------------------------------


def test_differential_edges_recovers_seeded_signal():
    """The headline test — a strong signal on two edges should survive FDR
    while the other ~30 candidate edges do not."""
    rng = np.random.default_rng(42)
    n_genes = 20
    diff_edges = [(2, 5, 1.0), (7, 12, -0.8)]   # (target, tf, delta)
    networks_dict, donor_metadata, cell_to_donor = _make_synthetic_donors(
        n_genes=n_genes,
        n_control=6,
        n_ad=16,
        n_cells_per_donor=30,
        diff_edges=diff_edges,
        seed=42,
    )
    net_obj = _make_networks_obj(networks_dict, n_genes=n_genes)
    pb = pseudobulk_per_donor(net_obj, cell_to_donor, donor_metadata)

    # Build a candidate edge list: include the two seeded edges + 30 random.
    seeded = [(_gene_name(t), _gene_name(tf)) for (t, tf, _d) in diff_edges]
    candidates = list(seeded)
    while len(candidates) < 32:
        i = rng.integers(0, n_genes)
        j = rng.integers(0, n_genes)
        if i == j:
            continue
        pair = (_gene_name(i), _gene_name(j))
        if pair not in candidates and (pair[0], pair[1]) not in seeded:
            candidates.append(pair)
    edges = pd.DataFrame(
        [{"target_gene": t, "TF": tf} for (t, tf) in candidates]
    )

    out = differential_edges(pb, edges, cell_type="MockType")
    assert (out["cell_type"] == "MockType").all()
    assert "q_value" in out.columns

    # Seeded edges should be at the top after sort
    top2 = out.head(2)
    seeded_set = {(t, tf) for t, tf in seeded}
    found = {(row.target, row.TF) for row in top2.itertuples()}
    assert found == seeded_set, (
        f"Expected top-2 to be {seeded_set}, got {found}\n{out.head(5)}"
    )
    # Seeded edges should be FDR-significant
    assert (top2["q_value"] < 0.05).all()


def test_differential_edges_skips_unknown_genes():
    rng = np.random.default_rng(7)
    n_genes = 5
    networks_dict, donor_metadata, cell_to_donor = _make_synthetic_donors(
        n_genes=n_genes, n_control=3, n_ad=4, n_cells_per_donor=5, seed=7
    )
    net_obj = _make_networks_obj(networks_dict, n_genes=n_genes)
    pb = pseudobulk_per_donor(net_obj, cell_to_donor, donor_metadata)
    edges = pd.DataFrame([
        {"target_gene": "gene_001", "TF": "gene_002"},     # in set
        {"target_gene": "GHOST_GENE", "TF": "gene_002"},   # not in set
    ])
    out = differential_edges(pb, edges)
    assert len(out) == 1
    assert out.iloc[0]["target"] == "gene_001"


def test_wilcoxon_alternative_runs():
    """Sensitivity check: the swappable Wilcoxon test should also run."""
    rng = np.random.default_rng(3)
    n_genes = 6
    networks_dict, donor_metadata, cell_to_donor = _make_synthetic_donors(
        n_genes=n_genes, n_control=4, n_ad=4, n_cells_per_donor=6,
        diff_edges=[(0, 1, 1.5)], seed=3,
    )
    net_obj = _make_networks_obj(networks_dict, n_genes=n_genes)
    pb = pseudobulk_per_donor(net_obj, cell_to_donor, donor_metadata)
    edges = pd.DataFrame([
        {"target_gene": "gene_000", "TF": "gene_001"},
        {"target_gene": "gene_002", "TF": "gene_003"},
    ])
    out = differential_edges(pb, edges, test=wilcoxon_unadjusted)
    assert len(out) == 2
    # NaN stderr for Wilcoxon
    assert out["stderr"].isna().all()
    # Seeded edge should still rank above noise edge
    assert out.iloc[0]["target"] == "gene_000"
