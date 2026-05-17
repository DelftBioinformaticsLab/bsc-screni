"""Tests for the SQ1 helpers added to ``screni.data.loading_seaad``.

Verifies condition / co-pathology / eligibility / subsampling functions on
a synthetic AnnData fixture without touching the 12 GB SeaAD files.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from screni.data.loading_seaad import (
    add_condition_column,
    add_copathology_columns,
    select_eligible_donors,
    subsample_cells_per_donor,
)


def _make_synthetic_seaad(n_per_donor: int = 60) -> ad.AnnData:
    donors = [
        ("D_NotAD", "Not AD", 78, "Male"),
        ("D_Low", "Low", 82, "Female"),
        ("D_Int", "Intermediate", 90, "Male"),
        ("D_High", "High", 91, "Female"),
        ("D_NotAD_small", "Not AD", 80, "Female"),
        ("D_garbage", "Some unrecognised label", 85, "Male"),
    ]
    rows = []
    for donor, adnc, age, sex in donors:
        n = 5 if donor == "D_NotAD_small" else n_per_donor
        for k in range(n):
            cell_type = "Microglia-PVM" if k % 2 == 0 else "L2/3 IT"
            late = "LATE Stage 1" if k % 3 == 0 else "Not Identified"
            lbd = "Limbic (Transitional)" if k % 4 == 0 else "Not Identified (olfactory bulb not assessed)"
            rows.append({
                "Donor ID": donor,
                "Overall AD neuropathological Change": adnc,
                "Age at Death": float(age),
                "Sex": sex,
                "LATE": late,
                "Highest Lewy Body Disease": lbd,
                "Subclass": cell_type,
                "cell_type": cell_type,
            })
    obs = pd.DataFrame(rows)
    obs.index = [f"cell_{i:04d}" for i in range(len(obs))]
    return ad.AnnData(obs=obs, X=np.zeros((len(obs), 5)))


def test_add_condition_column_maps_correctly():
    adata = _make_synthetic_seaad()
    add_condition_column(adata)
    cond = adata.obs["condition"]
    assert (cond[adata.obs["Donor ID"] == "D_NotAD"] == "control").all()
    assert (cond[adata.obs["Donor ID"] == "D_Low"] == "control").all()
    assert (cond[adata.obs["Donor ID"] == "D_Int"] == "ad").all()
    assert (cond[adata.obs["Donor ID"] == "D_High"] == "ad").all()
    assert cond[adata.obs["Donor ID"] == "D_garbage"].isna().all()


def test_add_condition_column_raises_on_missing_col():
    adata = _make_synthetic_seaad()
    del adata.obs["Overall AD neuropathological Change"]
    with pytest.raises(KeyError, match="ADNC"):
        add_condition_column(adata)


def test_copathology_present_flags():
    adata = _make_synthetic_seaad()
    add_copathology_columns(adata)
    obs = adata.obs
    assert obs.loc[obs["LATE"] == "LATE Stage 1", "LATE_present"].all()
    assert not obs.loc[obs["LATE"] == "Not Identified", "LATE_present"].any()
    assert obs.loc[
        obs["Highest Lewy Body Disease"] == "Limbic (Transitional)", "LBD_present"
    ].all()
    assert not obs.loc[
        obs["Highest Lewy Body Disease"]
        == "Not Identified (olfactory bulb not assessed)",
        "LBD_present",
    ].any()


def test_copathology_handles_missing_columns():
    adata = _make_synthetic_seaad()
    del adata.obs["LATE"]
    del adata.obs["Highest Lewy Body Disease"]
    add_copathology_columns(adata)
    assert (adata.obs["LATE_present"] == False).all()
    assert (adata.obs["LBD_present"] == False).all()


def test_select_eligible_donors_filters():
    adata = _make_synthetic_seaad(n_per_donor=60)
    add_condition_column(adata)
    add_copathology_columns(adata)
    df = select_eligible_donors(
        adata, cell_type="Microglia-PVM", min_cells_per_donor=10
    )
    assert "D_garbage" not in df["donor_id"].values
    assert "D_NotAD_small" not in df["donor_id"].values
    assert {"D_NotAD", "D_Low", "D_Int", "D_High"}.issubset(set(df["donor_id"]))
    md = df.set_index("donor_id")
    assert md.loc["D_NotAD", "condition"] == "control"
    assert md.loc["D_High", "condition"] == "ad"


def test_select_eligible_donors_empty_when_celltype_absent():
    adata = _make_synthetic_seaad(n_per_donor=20)
    add_condition_column(adata)
    add_copathology_columns(adata)
    df = select_eligible_donors(adata, cell_type="Astrocyte", min_cells_per_donor=10)
    assert df.empty


def test_subsample_caps_per_donor():
    adata = _make_synthetic_seaad(n_per_donor=100)
    add_condition_column(adata)
    sub = subsample_cells_per_donor(
        adata[adata.obs["condition"].notna()].copy(),
        n_per_donor=10, seed=0,
    )
    counts = sub.obs.groupby("Donor ID", observed=True).size()
    assert (counts <= 10).all()


def test_subsample_keeps_all_when_below_cap():
    adata = _make_synthetic_seaad(n_per_donor=100)
    add_condition_column(adata)
    sub = subsample_cells_per_donor(
        adata[adata.obs["condition"].notna()].copy(),
        n_per_donor=50, seed=0,
    )
    small = sub.obs[sub.obs["Donor ID"] == "D_NotAD_small"]
    assert len(small) == 5


def test_subsample_with_celltype_filter():
    adata = _make_synthetic_seaad(n_per_donor=40)
    add_condition_column(adata)
    sub = subsample_cells_per_donor(
        adata[adata.obs["condition"].notna()].copy(),
        n_per_donor=5, cell_type="Microglia-PVM", seed=0,
    )
    assert (sub.obs["cell_type"] == "Microglia-PVM").all()


def test_subsample_reproducible():
    adata = _make_synthetic_seaad(n_per_donor=40)
    add_condition_column(adata)
    src = adata[adata.obs["condition"].notna()].copy()
    a = subsample_cells_per_donor(src, n_per_donor=5, seed=123)
    b = subsample_cells_per_donor(src, n_per_donor=5, seed=123)
    assert list(a.obs_names) == list(b.obs_names)
