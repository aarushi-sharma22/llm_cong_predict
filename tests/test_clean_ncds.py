"""clean_ncds (brief Task 2.1, fact F2), on synthetic data.

The variable table is the public ``data/variables.xlsx``. The NCDS values are
synthetic (IDs SYN000001...), generated from a seed; they encode nothing real.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llm_cong_predict.cleaning.clean_ncds import (
    MISSING_LABELS,
    TEACHER_COLUMNS,
    blank_missing_labels,
    clean_ncds,
    combine_parent_education,
    recode_nssec,
    recode_parent_education,
    variables_table,
)
from llm_cong_predict.config import VARIABLES_XLSX
from llm_cong_predict.io.labels import VALUE_LABELS_KEY
from llm_cong_predict.io.readers import read_datalist

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mapping() -> pd.DataFrame:
    return read_datalist(str(VARIABLES_XLSX))


def _combined(mapping: pd.DataFrame, n: int = 40, seed: int = 0, drop: tuple = ()) -> pd.DataFrame:
    """Synthetic combined NCDS frame: every code of the table, labelled values."""
    rng = np.random.default_rng(seed)
    data = {"ncdsid": [f"SYN{i:06d}" for i in range(1, n + 1)]}
    labels = {}
    for code in mapping["variable"].str.lower():
        if code in drop:
            continue
        if code == "n622":
            data[code] = rng.choice([1.0, 2.0], size=n)
            labels[code] = {1.0: "Male", 2.0: "Female"}
        elif code in ("n2396", "n2397"):
            data[code] = rng.integers(1, 13, size=n).astype(float)
        elif code == "n2snssec":
            data[code] = rng.choice([1.0, 2.0, 3.2, 5.0, 6.5, 8.1, 13.5], size=n)
        else:
            data[code] = rng.integers(1, 6, size=n).astype(float)
            labels[code] = {1.0: "Very low", 2.0: "Low", 3.0: "Mid", 4.0: "High", 5.0: "Very high",
                            8.0: "Dont know"}
    df = pd.DataFrame(data)
    df.attrs[VALUE_LABELS_KEY] = labels
    return df


# --------------------------------------------------------------- table (F1) --

def test_read_datalist_gives_63_named_rows(mapping):
    """R: llm_paper/R/functions.R:L22–24 read with assigned names (PORTING_NOTES A1)."""
    assert len(mapping) == 63
    assert list(mapping.columns) == ["sweep", "type", "respondent", "question", "label",
                                     "new_varname", "variable"]
    assert variables_table(mapping).loc[0, "full_name"] == "s0_mo_birthweight"


def test_all_74_r_names_resolve(mapping):
    """Brief F1: every s<sweep>_<co|te|pa|mo>_... name in _targets.R and functions.R is a
    full_name of the table or created by the R; the only exception is
    s2_co_total_ability, which the R only removes (functions.R:L202)."""
    names = set(json.loads((REPO / "docs/reference/r_variable_names.json").read_text())["names"])
    assert len(names) == 74
    table = set(variables_table(mapping)["full_name"])
    created = {
        "s0_co_male",  # functions.R:L151
        "s3_pa_mother_edu", "s3_pa_father_edu", "s3_pa_edu",  # L116–145
        "s2_co_factor_ability", "s3_co_factor_scholastic_motivation",  # L277–297
        "s3_te_factor_internalizing", "s3_te_factor_externalizing",
        "s2_co_aspiration_camsis", "s2_co_aspiration_camsis_male",  # L266–269
        "s2_co_aspiration_camsis_female",
    }
    assert names - table - created == {"s2_co_total_ability"}


# ------------------------------------------------------- step 2: labels only --

def test_missing_strings_apply_only_through_labels():
    """R: llm_paper/R/functions.R:L80–90 — to_character(x) %in% <list>: a value becomes NA
    only if its LABEL is in the verbatim list (add.non.labelled = FALSE, so unlabelled
    values never match; sjlabelled/R/as_character.R:L14)."""
    df = pd.DataFrame({
        "ncdsid": ["SYN000001", "SYN000002", "SYN000003", "SYN000004"],
        "a": [1.0, 9.0, 7.0, 6.0],   # 9 "Dont know", 7 unlabelled, 6 " Cant say,inappl"
        "b": [9.0, 9.0, 1.0, 5.0],   # 9 labelled "Nine", 5 "Cant say,inappl" (no leading space)
        "c": [9.0, 1.0, 2.0, 3.0],   # no labels at all
    })
    df.attrs[VALUE_LABELS_KEY] = {"a": {1.0: "Yes", 9.0: "Dont know", 6.0: " Cant say,inappl"},
                                  "b": {9.0: "Nine", 5.0: "Cant say,inappl"}}
    out = blank_missing_labels(df)
    assert out["a"].isna().tolist() == [False, True, False, True]
    assert out["b"].isna().tolist() == [False, False, False, False]
    assert out["c"].isna().tolist() == [False, False, False, False]
    assert len(MISSING_LABELS) == 20 and MISSING_LABELS.count("Inapplicable") == 2


def test_no_labels_remain_after_step_2(mapping):
    """ifelse() returns plain vectors (r-source/src/library/base/R/ifelse.R:L46–55): after
    step 2 no column carries value labels, and none are re-applied."""
    df = _combined(mapping)
    assert VALUE_LABELS_KEY not in blank_missing_labels(df).attrs
    out = clean_ncds(df, mapping)
    assert VALUE_LABELS_KEY not in out.attrs


# ----------------------------------------------------------------- parents --

def test_nssec_boundaries_endpoints_and_gaps():
    """R: llm_paper/R/functions.R:L104–115 — closed ranges; values in the gaps are NA."""
    endpoints = [1, 2, 3.1, 3.4, 4.1, 6, 7.1, 7.4, 8.1, 9.2, 10, 11.2, 12.1, 12.7, 13.1, 13.5]
    expected = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
    np.testing.assert_array_equal(recode_nssec(pd.Series(endpoints, dtype=float)), expected)
    gaps = [0.5, 2.5, 3.0, 3.5, 6.5, 7.5, 9.5, 11.5, 12.8, 13.8, np.nan]
    assert recode_nssec(pd.Series(gaps)).isna().all()


def test_parent_education_bounds():
    """R: llm_paper/R/functions.R:L116–129 — <4, <6, <8, <11, else NA; ordered levels."""
    fac = recode_parent_education(pd.Series([3.9, 4, 5.9, 6, 7.9, 8, 10.9, 11, np.nan]))
    assert list(fac.categories) == ["No Qualifications", "Lower Secondary", "Upper Secondary", "Degree"]
    assert fac.ordered
    assert [None if pd.isna(v) else v for v in fac] == [
        "No Qualifications", "Lower Secondary", "Lower Secondary", "Upper Secondary",
        "Upper Secondary", "Degree", "Degree", None, None]


def test_parent_education_truth_table():
    """R: llm_paper/R/functions.R:L130–144: all 16 ordered (mother, father) pairs plus a
    missing parent, via pmin/pmax (NA if either is NA)."""
    levels = ["No Qualifications", "Lower Secondary", "Upper Secondary", "Degree"]
    pairs = [(m, f) for m in levels for f in levels] + [(None, "Degree"), ("Degree", None), (None, None)]
    mother = pd.Categorical([p[0] for p in pairs], categories=levels, ordered=True)
    father = pd.Categorical([p[1] for p in pairs], categories=levels, ordered=True)
    rank = {lv: i for i, lv in enumerate(levels)}

    def expected(m, f):
        if m is None or f is None:
            return np.nan
        lo, hi = sorted((rank[m], rank[f]))
        table = {(0, 0): 1, (0, 1): 2, (1, 1): 3, (0, 2): 4, (1, 2): 4, (2, 2): 5,
                 (0, 3): 6, (1, 3): 6, (2, 3): 6, (3, 3): 7}
        return table[(lo, hi)]

    out = combine_parent_education(mother, father)
    np.testing.assert_array_equal(np.asarray(out, dtype=float), [expected(m, f) for m, f in pairs])


def test_pedu_level_position_when_a_level_is_absent():
    """factor(s3_pa_edu) (functions.R:L145) has the sorted OBSERVED values as levels, so
    as.numeric() gives the level position: with values {1, 3, 7} observed, 3 -> 2 and
    7 -> 3 (brief F2; used by _targets.R:L208, L368)."""
    levels = ["No Qualifications", "Lower Secondary", "Upper Secondary", "Degree"]
    mother = pd.Categorical(["No Qualifications", "Lower Secondary", "Degree", None],
                            categories=levels, ordered=True)
    father = pd.Categorical(["No Qualifications", "Lower Secondary", "Degree", "Degree"],
                            categories=levels, ordered=True)
    out = combine_parent_education(mother, father)
    assert list(out.categories) == [1.0, 3.0, 7.0]
    np.testing.assert_array_equal(out.codes + 1, [1, 2, 3, 0])  # 0 = missing (code -1)


# ----------------------------------------------------------------- teacher --

def test_teacher_rank_coding_with_gap(mapping):
    """R: llm_paper/R/functions.R:L190–191 — labels are gone, so haven::as_factor is
    factor(x) and ifelse returns its codes: the rank among the column's sorted distinct
    observed values over the whole data, starting at 1 (a gap in the values closes)."""
    df = _combined(mapping, n=5)
    df["n876"] = [1.0, 3.0, 3.0, 5.0, np.nan]  # 2 and 4 never observed
    out = clean_ncds(df, mapping)
    np.testing.assert_array_equal(out["s2_te_general_knowledge"], [1.0, 2.0, 2.0, 3.0, np.nan])


def test_sex_is_true_false_or_missing(mapping):
    """R: llm_paper/R/functions.R:L151 — ifelse(s0_co_sex == 1, TRUE, FALSE), NA stays NA."""
    df = _combined(mapping, n=3)
    df["n622"] = [1.0, 2.0, np.nan]
    out = clean_ncds(df, mapping)
    assert out["s0_co_male"].tolist()[:2] == [True, False] and pd.isna(out["s0_co_male"].iloc[2])


# ------------------------------------------------------- ability and output --

def test_s2_co_total_ability_absent_is_handled(mapping):
    """R: llm_paper/R/functions.R:L202 — select(-s2_co_total_ability) errors because the
    table never creates that name; the reconstruction drops it only when present."""
    out = clean_ncds(_combined(mapping), mapping)
    assert "s2_co_total_ability" not in out.columns
    assert "s2_co_verbal_ability" in out.columns


def test_collision_check_reports_identical_and_different(mapping):
    """Nine teacher columns also appear in the ability or behavior block (brief F2). The
    teacher (rank-coded) version is kept, as plyr keeps the first occurrence; the check
    says whether it equals the raw version, which holds when the observed values are
    exactly 1..max."""
    df = _combined(mapping, n=6)
    df["n876"] = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]  # general_knowledge: 1..3 -> identical
    df["n880"] = [2.0, 4.0, 4.0, 2.0, 5.0, 5.0]  # poor_hand_control: ranks 1,2,2,1,3,3 -> differ
    out = clean_ncds(df, mapping)
    report = out.attrs["clean_ncds_collisions"]
    assert set(report) == set(TEACHER_COLUMNS[:9])
    assert report["s2_te_general_knowledge"] == {"other_block": "ability", "identical": True,
                                                 "cells_differing": 0}
    assert report["s2_te_poor_hand_control"]["identical"] is False
    assert report["s2_te_poor_hand_control"]["other_block"] == "behavior"
    np.testing.assert_array_equal(out["s2_te_poor_hand_control"], [1, 2, 2, 1, 3, 3])


def test_one_row_per_ncdsid_and_column_order(mapping):
    """R: llm_paper/R/functions.R:L228–239 — ncdsid, then the columns of sex,
    birthweight, height, teacher, parents, personality, behavior, ability, motivation
    and highest_edu, each column once at its first position."""
    df = _combined(mapping, n=25, seed=3)
    out = clean_ncds(df, mapping)
    v = variables_table(mapping)

    def of_type(t):
        return list(v.loc[v["type"] == t, "full_name"])

    blocks = [["s0_co_male"], ["s0_mo_birthweight"], ["s3_co_height"], list(TEACHER_COLUMNS),
              of_type("parental class") + ["s3_pa_mother_edu", "s3_pa_father_edu", "s3_pa_edu"],
              of_type("personality"), of_type("behavior"), of_type("ability"),
              of_type("academic_motivation"), of_type("education")]
    expected = list(dict.fromkeys(["ncdsid"] + [c for b in blocks for c in b]))
    assert list(out.columns) == expected
    assert len(expected) == 66  # computed from the public table
    assert out["ncdsid"].is_unique and len(out) == 25
    assert list(out["ncdsid"]) == list(df["ncdsid"])


def test_absent_codes_are_logged_not_fatal(mapping, caplog):
    """one_of drops absent names with a warning (tidyselect/R/helpers.R:L124–127)."""
    df = _combined(mapping, drop=("nd8ext",))
    out = clean_ncds(df, mapping)
    assert out.attrs["clean_ncds_absent_codes"] == ["nd8ext"]
    assert "s8_co_extraversion" not in out.columns
    assert any("absent from the data" in r.getMessage() for r in caplog.records)
