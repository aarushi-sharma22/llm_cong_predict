"""Tests for the cleaning block (option-b subset: everything except clean_ncds).

These verify MECHANICS on synthetic frames shaped like the real cleaned data. They
do NOT validate numbers against real NCDS data or against R (deferred:
VALIDATION_CHECKLIST V3 for factor parity, V5 for aspirations). The polychoric
factors are asserted to be *deferred*, not computed, which is the intended honest
behaviour until V3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from llm_cong_predict.cleaning.aspirations import create_aspirations
from llm_cong_predict.cleaning.assemble import (
    find_essay_teacher_genetics_overlap,
    find_full_overlap,
    get_complete_ncds,
)
from llm_cong_predict.cleaning.factors import (
    FACTOR_DEFINITIONS,
    create_factors,
)


# ------------------------------------------------------------- create_aspirations --

def _camsis_frame():
    df = pd.DataFrame(
        {
            "co1970": [1.0, 2.0, 3.0],
            "mcamsis": [70.0, 50.0, 40.0],
            "fcamsis": [65.0, 55.0, 45.0],
            "stdempst": [0.0, 0.0, 0.0],   # all "unknown employment status" -> kept
        }
    )
    df.attrs["value_labels"] = {"co1970": {1: "Teacher", 2: "Farmer", 3: "Fisher"}}
    return df


def _ncds_for_aspirations():
    df = pd.DataFrame({"ncdsid": ["1", "2"], "n2771": [1, 2], "n622": [1, 2]})
    df.attrs["value_labels"] = {
        "n2771": {1: "Teacher", 2: "Farmer"},   # aspiration label matches mapping
        "n622": {1: "Male", 2: "Female"},
    }
    return df


def test_create_aspirations_sex_specific_score():
    camsis = _camsis_frame()
    mapping = pd.DataFrame(
        {"aspiration_n2771": ["Teacher", "Farmer"], "occupation_1970": ["Teacher", "Farmer"]}
    )
    ncds = _ncds_for_aspirations()
    out = create_aspirations(ncds, camsis, mapping)

    assert list(out.columns) == [
        "ncdsid", "s2_co_aspiration_camsis_male",
        "s2_co_aspiration_camsis_female", "s2_co_aspiration_camsis",
    ]
    row1 = out[out["ncdsid"] == "1"].iloc[0]  # sex label "Male" (not "1") -> female branch
    row2 = out[out["ncdsid"] == "2"].iloc[0]
    # FAITHFUL-QUIRK CHECK: R compares character sex to literal 1; "Male"/"Female"
    # never equal "1", so BOTH rows take the female score. We reproduce that exactly.
    assert row1["s2_co_aspiration_camsis"] == row1["s2_co_aspiration_camsis_female"]
    assert row2["s2_co_aspiration_camsis"] == row2["s2_co_aspiration_camsis_female"]


def test_create_aspirations_filters_employment_status():
    camsis = _camsis_frame()
    camsis.loc[0, "stdempst"] = 2.0  # occupation 1 (Teacher) now excluded
    mapping = pd.DataFrame(
        {"aspiration_n2771": ["Teacher", "Farmer"], "occupation_1970": ["Teacher", "Farmer"]}
    )
    ncds = _ncds_for_aspirations()
    out = create_aspirations(ncds, camsis, mapping)
    # respondent 1 aspired to Teacher, which was filtered out -> no camsis score
    row1 = out[out["ncdsid"] == "1"].iloc[0]
    assert pd.isna(row1["s2_co_aspiration_camsis_male"])


# ------------------------------------------------------------- find_full_overlap --

def test_find_full_overlap_complete_case():
    df = pd.DataFrame(
        {"ncdsid": ["1", "2", "3"], "a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan], "c": [9, 9, 9]}
    )
    out = find_full_overlap(df, ["a", "b"])
    assert list(out.columns) == ["ncdsid", "a", "b"]
    assert list(out["ncdsid"]) == ["1"]   # only row 1 is complete on a AND b


def test_find_full_overlap_ignores_duplicate_ncdsid_in_varlist():
    df = pd.DataFrame({"ncdsid": ["1"], "a": [1.0]})
    out = find_full_overlap(df, ["ncdsid", "a"])
    assert list(out.columns) == ["ncdsid", "a"]  # ncdsid not duplicated


# ------------------------------------------------------------- get_complete_ncds --

def test_get_complete_ncds_left_joins():
    cleaned = pd.DataFrame({"ncdsid": ["1", "2"], "v": [10, 20]})
    factors = pd.DataFrame({"ncdsid": ["1"], "f": [0.5]})
    aspir = pd.DataFrame({"ncdsid": ["2"], "asp": [66.0]})
    essay = pd.DataFrame({"ncdsid": ["1", "2"], "nwords": [100, 200]})
    out = get_complete_ncds(cleaned, factors, aspir, essay)
    assert set(out["ncdsid"]) == {"1", "2"}          # left frame preserved
    assert out.set_index("ncdsid").loc["1", "nwords"] == 100
    assert bool(pd.isna(out.set_index("ncdsid").loc["2", "f"]))   # no factor for id 2


def test_get_complete_ncds_optional_gene_join():
    cleaned = pd.DataFrame({"ncdsid": ["1"], "v": [10]})
    empty = pd.DataFrame({"ncdsid": ["1"]})
    gene = pd.DataFrame({"ncdsid": ["1"], "pgs": [1.23]})
    out = get_complete_ncds(cleaned, empty, empty, empty, ncds_gene=gene)
    assert "pgs" in out.columns


# ---------------------------------------------------- find_essay_teacher_genetics --

def test_find_essay_teacher_genetics_overlap_inner_joins_and_labels():
    codes = ["n876", "n877", "n878", "n879", "n880", "n881", "n882", "n883", "n884", "n885"]
    raw = pd.DataFrame({"ncdsid": ["1", "2"], **{c: [1, 2] for c in codes}})
    raw.attrs["value_labels"] = {c: {1: "Good", 2: "Dont know"} for c in codes}

    ability = ["s2_co_factor_ability", "s2_co_verbal_ability", "s2_co_nonverbal_ability",
               "s2_co_reading_ability", "s2_co_mathematics_ability"]
    genetics = pd.DataFrame({"ncdsid": ["1", "2"], **{a: [0.1, 0.2] for a in ability}})
    essays = pd.DataFrame({"ncdsid": ["1", "2"], "text": ["a", "b"]})

    out = find_essay_teacher_genetics_overlap(genetics, essays, raw)
    # row 2's teacher codes are all "Dont know" -> set NA -> dropped from extended
    # teacher frame -> inner join keeps only id 1.
    assert list(out["ncdsid"]) == ["1"]


# --------------------------------------------------------------- create_factors --

def test_create_factors_computes_pearson_defers_polychoric():
    # Build synthetic data with the columns the ability factor needs.
    rng = np.random.default_rng(0)
    n = 100
    ability_vars = FACTOR_DEFINITIONS["s2_co_factor_ability"]["vars"]
    base = rng.normal(size=n)
    data = {"ncdsid": [str(i) for i in range(n)]}
    for v in ability_vars:
        data[v] = base + rng.normal(scale=0.3, size=n)  # correlated -> a real factor
    df = pd.DataFrame(data)

    res = create_factors(df, include_polychoric=False)
    # Pearson ability factor computed; the three poly factors deferred (V3).
    assert res.computed == ["s2_co_factor_ability"]
    assert set(res.deferred) == {
        "s3_co_factor_scholastic_motivation",
        "s3_te_factor_internalizing",
        "s3_te_factor_externalizing",
    }
    assert "s2_co_factor_ability" in res.scores.columns
    assert len(res.scores) == n
    # the ability factor should correlate strongly with the shared latent `base`
    r = np.corrcoef(res.scores["s2_co_factor_ability"], base)[0, 1]
    assert abs(r) > 0.8   # |corr| because factor sign is arbitrary


def test_create_factors_polychoric_raises_not_guesses():
    # Asking for polychoric before V3 must raise, not emit unverified numbers.
    # Provide columns for ALL factors so we reach the polychoric path (not a
    # missing-column error on an earlier factor).
    cols = {}
    for spec in FACTOR_DEFINITIONS.values():
        for v in spec["vars"]:
            cols[v] = [1, 2, 3, 4]
    df = pd.DataFrame({"ncdsid": ["1", "2", "3", "4"], **cols})
    with pytest.raises(NotImplementedError):
        create_factors(df, include_polychoric=True)
