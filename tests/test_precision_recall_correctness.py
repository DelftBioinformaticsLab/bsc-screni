"""Independent correctness tests for the precision/recall functions.

The parity test suite asserts the Python implementations agree with the
original R code. These tests are complementary: they assert correctness
against (a) hand-computed numbers and (b) sklearn's ``precision_score`` /
``recall_score`` as an independent reference. Together with the parity
suite this gives three sources of truth.

The shared ``conftest.py`` handles the issue #8 stub.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screni.data.calculate_scnetwork_precision_recall import (
    calculate_scnetwork_precision_recall,
)
from screni.data.summary_se import summary_se


# ----------------------------------------------------------------------
# Hand-computed unit tests
# ----------------------------------------------------------------------


@pytest.fixture
def hand_fixture():
    """Tiny 3-gene fixture with hand-computable expected outputs.

    ``mymethod`` matrix (rows = from, cols = to)::

             a    b    c
          a 0.0  0.9  0.1
          b 0.5  0.0  0.7
          c 0.3  0.2  0.0

    Edges sorted by descending weight:
        rank 1: a_b (0.9)   <- in gold
        rank 2: b_c (0.7)
        rank 3: b_a (0.5)
        rank 4: c_a (0.3)   <- in gold
        rank 5: c_b (0.2)
        rank 6: a_c (0.1)

    Gold standard = {a_b, c_a}, so total positives = 2.

    CSN matrix has exactly 4 non-zero entries::

             a    b    c
          a 0.0  0.5  0.0
          b 0.0  0.0  0.4
          c 0.3  0.2  0.0

    Edges sorted: a_b (0.5), b_c (0.4), c_a (0.3), c_b (0.2).
    """
    genes = ["a", "b", "c"]
    M_method = pd.DataFrame(
        [[0.0, 0.9, 0.1], [0.5, 0.0, 0.7], [0.3, 0.2, 0.0]],
        index=genes, columns=genes,
    )
    M_csn = pd.DataFrame(
        [[0.0, 0.5, 0.0], [0.0, 0.0, 0.4], [0.3, 0.2, 0.0]],
        index=genes, columns=genes,
    )
    sc_networks = {"CSN": [M_csn], "mymethod": [M_method]}
    TF_target_pair = ["a_b", "c_a"]
    return sc_networks, TF_target_pair


def test_hand_computed_k2(hand_fixture):
    """Top-2 edges of mymethod = (a_b, b_c). One in gold; precision=1/2, recall=1/2."""
    sc_networks, gold = hand_fixture
    res = calculate_scnetwork_precision_recall(sc_networks, gold, top_number=[2])
    df = res[2]
    assert len(df) == 1  # one cell, one method (CSN excluded for k>0)
    row = df.iloc[0]
    assert row["scNetwork_type"] == "mymethod"
    assert row["precision"] == 0.5
    assert row["recall"] == 0.5


def test_hand_computed_k4(hand_fixture):
    """Top-4 edges of mymethod = (a_b, b_c, b_a, c_a). Two in gold; P=2/4, R=2/2."""
    sc_networks, gold = hand_fixture
    res = calculate_scnetwork_precision_recall(sc_networks, gold, top_number=[4])
    row = res[4].iloc[0]
    assert row["precision"] == 0.5
    assert row["recall"] == 1.0


def test_hand_computed_k0_uses_csn_nonzero_count(hand_fixture):
    """k=0 evaluates every method at K = per-cell CSN nonzero count = 4 here."""
    sc_networks, gold = hand_fixture
    res = calculate_scnetwork_precision_recall(sc_networks, gold, top_number=[2])
    # k=0 bucket: CSN included, K = number of CSN non-zero edges = 4.
    df0 = res[0].sort_values("scNetwork_type").reset_index(drop=True)
    assert df0["scNetwork_type"].astype(str).tolist() == ["CSN", "mymethod"]

    # CSN top 4: (a_b, b_c, c_a, c_b). a_b and c_a in gold => 2 hits.
    csn = df0[df0["scNetwork_type"] == "CSN"].iloc[0]
    assert csn["precision"] == 0.5  # 2/4
    assert csn["recall"] == 1.0     # 2/2

    # mymethod top 4 same as test_hand_computed_k4
    mm = df0[df0["scNetwork_type"] == "mymethod"].iloc[0]
    assert mm["precision"] == 0.5
    assert mm["recall"] == 1.0


# ----------------------------------------------------------------------
# Independent cross-check vs sklearn
# ----------------------------------------------------------------------


def _sklearn_topk_pr(M: pd.DataFrame, gold: set[str], K: int) -> tuple[float, float]:
    """Reference implementation using sklearn's precision_score/recall_score."""
    from sklearn.metrics import precision_score, recall_score

    rows, cols = list(M.index), list(M.columns)
    keys = []
    scores = []
    for i, gi in enumerate(rows):
        for j, gj in enumerate(cols):
            keys.append(f"{gi}_{gj}")
            scores.append(M.iloc[i, j])
    scores = np.asarray(scores, dtype=float)
    y_true = np.array([k in gold for k in keys], dtype=int)

    order = np.argsort(-scores, kind="stable")
    y_pred = np.zeros_like(y_true)
    y_pred[order[:K]] = 1

    p = precision_score(y_true, y_pred, zero_division=0.0)
    r = recall_score(y_true, y_pred, zero_division=0.0)
    return float(p), float(r)


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
@pytest.mark.parametrize("K", [3, 5, 10, 25])
def test_vs_sklearn_random_networks(seed, K):
    """Random networks: production code must match sklearn's metric."""
    rng = np.random.default_rng(seed)
    n_genes = 8
    genes = [f"g{i}" for i in range(n_genes)]
    M_method = pd.DataFrame(rng.random((n_genes, n_genes)), index=genes, columns=genes)
    np.fill_diagonal(M_method.values, 0.0)

    M_csn = M_method.copy()
    M_csn.values[M_csn.values < 0.7] = 0.0

    sc = {"CSN": [M_csn], "method_x": [M_method]}

    n_gold = max(1, int(0.2 * n_genes * n_genes))
    all_pairs = [f"{a}_{b}" for a in genes for b in genes if a != b]
    gold = list(rng.choice(all_pairs, size=n_gold, replace=False))
    gold_set = set(gold)

    res = calculate_scnetwork_precision_recall(sc, gold, top_number=[K])
    py_row = res[K][res[K]["scNetwork_type"] == "method_x"].iloc[0]
    p_py, r_py = float(py_row["precision"]), float(py_row["recall"])

    p_sk, r_sk = _sklearn_topk_pr(M_method, gold_set, K)

    assert p_py == pytest.approx(p_sk, abs=1e-12), f"precision: prod={p_py} sk={p_sk}"
    assert r_py == pytest.approx(r_sk, abs=1e-12), f"recall: prod={r_py} sk={r_sk}"


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------


def test_k_exceeds_total_edges():
    """When K > number of distinct edges, precision_denominator is still K.

    R extends the ranked list with NAs and still divides by K. We match.
    """
    genes = ["a", "b"]
    M = pd.DataFrame([[0.0, 0.5], [0.3, 0.0]], index=genes, columns=genes)
    sc = {"CSN": [M], "method": [M]}
    gold = ["a_b"]

    # Total edges = 4 (2x2), only 2 are non-zero. K = 100 is huge.
    res = calculate_scnetwork_precision_recall(sc, gold, top_number=[100])
    row = res[100].iloc[0]
    # Numerator: 1 (the gold pair a_b is in there). Denominator: 100. So P = 1/100.
    assert row["precision"] == pytest.approx(1 / 100)
    # Recall: 1 hit / 1 total positive = 1.0
    assert row["recall"] == 1.0


def test_empty_gold_standard_recall_is_zero():
    """When the gold standard is empty, recall denominator is 0 -> recall = 0."""
    genes = ["a", "b"]
    M = pd.DataFrame([[0.0, 0.5], [0.3, 0.0]], index=genes, columns=genes)
    sc = {"CSN": [M], "method": [M]}

    res = calculate_scnetwork_precision_recall(sc, [], top_number=[2])
    row = res[2].iloc[0]
    assert row["precision"] == 0.0  # no hits possible
    assert row["recall"] == 0.0     # division by zero -> 0.0 by stub convention


def test_missing_csn_key_raises():
    """The function relies on CSN for the k=0 bucket; absence is a hard error."""
    genes = ["a", "b"]
    M = pd.DataFrame([[0.0, 0.5], [0.3, 0.0]], index=genes, columns=genes)
    with pytest.raises(KeyError, match="CSN"):
        calculate_scnetwork_precision_recall({"method": [M]}, ["a_b"], top_number=[2])


# ----------------------------------------------------------------------
# summary_se edge cases
# ----------------------------------------------------------------------


def test_summary_se_n_equals_one_gives_nan_ci():
    """A single-row group has df=0 for the t distribution; ci is NaN.

    R returns Inf in this case (qt with df=0); we accept scipy's NaN as
    a deliberate, documented divergence — both are "no usable CI".
    """
    df = pd.DataFrame({"g": ["a"], "x": [3.14]})
    out = summary_se(df, measurevar="x", groupvars=["g"])
    assert out.loc[0, "N"] == 1
    assert out.loc[0, "x"] == 3.14
    assert np.isnan(out.loc[0, "sd"])  # sd of one value
    assert np.isnan(out.loc[0, "se"])
    assert np.isnan(out.loc[0, "ci"])


def test_summary_se_na_rm_true_drops_nans():
    """na_rm=True ignores NaNs in N, mean, and sd."""
    df = pd.DataFrame({"g": ["a", "a", "a"], "x": [1.0, np.nan, 3.0]})
    out = summary_se(df, measurevar="x", groupvars=["g"], na_rm=True)
    assert out.loc[0, "N"] == 2
    assert out.loc[0, "x"] == 2.0  # mean of [1, 3]
    assert out.loc[0, "sd"] == pytest.approx(np.sqrt(2.0))  # sample sd of [1, 3]


def test_summary_se_na_rm_false_propagates_nan():
    """na_rm=False keeps NaNs in the count and propagates them through mean/sd."""
    df = pd.DataFrame({"g": ["a", "a", "a"], "x": [1.0, np.nan, 3.0]})
    out = summary_se(df, measurevar="x", groupvars=["g"], na_rm=False)
    assert out.loc[0, "N"] == 3
    assert np.isnan(out.loc[0, "x"])
    assert np.isnan(out.loc[0, "sd"])


def test_summary_se_no_groupvars():
    """Without groupvars the entire frame is one group."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    out = summary_se(df, measurevar="x")
    assert len(out) == 1
    assert out.loc[0, "N"] == 4
    assert out.loc[0, "x"] == 2.5
    # sample sd of [1,2,3,4] = sqrt(5/3)
    assert out.loc[0, "sd"] == pytest.approx(np.sqrt(5.0 / 3.0))
