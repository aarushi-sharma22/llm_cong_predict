"""Tests for the IO layer.

These verify reader MECHANICS (lower-casing, column selection, missing-code recode,
label carrying, essay parsing, outer join) against synthetic fixtures, and exercise
the readers that CAN touch real shipped files (CAMSIS .dta, the two xlsx files).
They do not and cannot validate against the real NCDS data — that is deferred
(VALIDATION_CHECKLIST). The fixtures are synthetic and encode no real values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from llm_cong_predict.config import OCCUPATION_ASPIRATION_XLSX, VARIABLES_XLSX
from llm_cong_predict.io.labels import as_factor, get_value_labels, set_na_range
from llm_cong_predict.io.readers import (
    combine_ncds,
    read_camsis,
    read_datalist,
    read_essays,
    read_gene_data,
    read_ncds,
    read_occupation_aspiration_mapping,
)

from fixtures.synthetic_data import (
    make_synthetic_camsis_dta,
    make_synthetic_essays,
    make_synthetic_ncds_dta,
)


# ----------------------------------------------------------------- read_ncds --

def test_read_ncds_lowercases_selects_and_sets_na(tmp_path):
    dta = tmp_path / "wave.dta"
    labels = make_synthetic_ncds_dta(str(dta))
    df = read_ncds(str(dta), varlist=["n622", "n876"])

    # column names lower-cased; selection in requested order; ncdsid first
    assert list(df.columns) == ["ncdsid", "n622", "n876"]
    # -1 code in N622 became missing; -1 in n876 became missing
    assert bool(pd.isna(df.loc[3, "n622"]))
    assert bool(pd.isna(df.loc[2, "n876"]))
    # non-missing values preserved
    assert df.loc[0, "n622"] == 1
    # value labels carried on attrs (lower-cased key)
    vl = get_value_labels(df)
    assert "n622" in vl and vl["n622"][1] == "Male"
    assert labels["N622"][2] == "Female"  # fixture sanity


def test_read_ncds_ignores_absent_requested_columns(tmp_path):
    # dplyr::one_of ignores names not present (no error).
    dta = tmp_path / "wave.dta"
    make_synthetic_ncds_dta(str(dta))
    df = read_ncds(str(dta), varlist=["n876", "n99999_absent"])
    assert list(df.columns) == ["ncdsid", "n876"]


def test_read_ncds_n2771_missing_code_99(tmp_path):
    dta = tmp_path / "wave.dta"
    make_synthetic_ncds_dta(str(dta))
    df = read_ncds(str(dta), varlist=["n2771"])
    # -99 recoded to missing; 10/47/20 preserved
    assert bool(pd.isna(df.loc[3, "n2771"]))
    assert set(df["n2771"].dropna().astype(int)) == {10, 47, 20}


# ---------------------------------------------------------------- read_camsis --

def test_read_camsis_synthetic_with_labels(tmp_path):
    dta = tmp_path / "camsis.dta"
    make_synthetic_camsis_dta(str(dta))
    df = read_camsis(str(dta))
    assert {"co1970", "stdempst", "mcamsis", "fcamsis"}.issubset(df.columns)
    # value label on co1970 carried and applyable via as_factor
    fac = as_factor(df, "co1970")
    assert "Fishermen" in list(fac.astype(str))


def test_read_camsis_real_shipped_file():
    # The real CAMSIS .dta is in the repo, so this reader runs on real data.
    from llm_cong_predict.config import CAMSIS_FILE

    if not CAMSIS_FILE.exists():
        pytest.skip("real CAMSIS file not present")
    df = read_camsis(str(CAMSIS_FILE))
    assert "co1970" in df.columns
    assert len(df) > 0


# --------------------------------------------------------------- read_essays --

def test_read_essays_parses_format(tmp_path):
    folder = tmp_path / "essays"
    expected = make_synthetic_essays(str(folder))
    df = read_essays(str(folder))

    assert list(df.columns) == ["doc_id", "ncdsid", "text", "words"]
    assert len(df) == len(expected)
    # ID prefix stripped; text and words parsed
    row = df[df["ncdsid"] == "0001"].iloc[0]
    assert row["text"] == "When i grow up i want to be a teacher."
    assert row["words"] == "8"
    assert row["ncdsid"] == "0001"  # no "ID: " prefix remaining


# ------------------------------------------------------------- xlsx readers --

def test_read_occupation_aspiration_mapping_real_file():
    if not OCCUPATION_ASPIRATION_XLSX.exists():
        pytest.skip("occupation mapping xlsx not present")
    df = read_occupation_aspiration_mapping(str(OCCUPATION_ASPIRATION_XLSX))
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    # the real file has these columns (see PORTING_NOTES / repo inspection)
    assert "aspiration_n2771" in df.columns
    assert "occupation_1970" in df.columns


def test_read_datalist_real_file_returns_frame():
    # Faithful thin reader. We only assert it returns a non-empty frame; the
    # schema mismatch (PORTING_NOTES A1) is a cleaning-layer concern, not here.
    if not VARIABLES_XLSX.exists():
        pytest.skip("variables.xlsx not present")
    df = read_datalist(str(VARIABLES_XLSX))
    assert isinstance(df, pd.DataFrame) and len(df) > 0


# ------------------------------------------------------------ read_gene_data --

def test_read_gene_data_raises():
    with pytest.raises(NotImplementedError):
        read_gene_data()


# ------------------------------------------------------------- combine_ncds --

def test_combine_ncds_full_outer_join():
    a = pd.DataFrame({"ncdsid": ["1", "2", "3"], "x": [10, 20, 30]})
    b = pd.DataFrame({"ncdsid": ["2", "3", "4"], "y": [200, 300, 400]})
    out = combine_ncds(a, b)
    # union of ids (full join)
    assert set(out["ncdsid"]) == {"1", "2", "3", "4"}
    # id "1" has no y -> NaN; id "4" has no x -> NaN
    assert bool(pd.isna(out.set_index("ncdsid").loc["1", "y"]))
    assert bool(pd.isna(out.set_index("ncdsid").loc["4", "x"]))


def test_combine_ncds_coalesces_column_collision():
    # Overlapping non-key column should coalesce, not suffix.
    a = pd.DataFrame({"ncdsid": ["1", "2"], "shared": [1, np.nan]})
    b = pd.DataFrame({"ncdsid": ["1", "2"], "shared": [np.nan, 2]})
    out = combine_ncds(a, b)
    assert "shared" in out.columns and "shared__r" not in out.columns
    vals = out.set_index("ncdsid")["shared"]
    assert vals.loc["1"] == 1 and vals.loc["2"] == 2
    assert out.attrs.get("combine_ncds_collisions") == ["shared"]


def test_combine_ncds_carries_merged_value_labels():
    a = pd.DataFrame({"ncdsid": ["1"], "x": [1]})
    a.attrs["value_labels"] = {"x": {1: "one"}}
    b = pd.DataFrame({"ncdsid": ["1"], "y": [2]})
    b.attrs["value_labels"] = {"y": {2: "two"}}
    out = combine_ncds(a, b)
    vl = get_value_labels(out)
    assert vl.get("x") == {1: "one"} and vl.get("y") == {2: "two"}


# -------------------------------------------------------------------- labels --

def test_set_na_range_helper():
    df = pd.DataFrame({"a": [5, -1, -50, 3], "b": ["x", "y", "z", "w"]})
    out = set_na_range(df, -99, -1)
    assert bool(pd.isna(out.loc[1, "a"])) and bool(pd.isna(out.loc[2, "a"]))
    assert out.loc[0, "a"] == 5 and out.loc[3, "a"] == 3
    assert list(out["b"]) == ["x", "y", "z", "w"]  # strings untouched


def test_as_factor_maps_codes_to_labels():
    df = pd.DataFrame({"c": [1, 2, 1]})
    df.attrs["value_labels"] = {"c": {1: "yes", 2: "no"}}
    fac = as_factor(df, "c")
    assert list(fac.astype(str)) == ["yes", "no", "yes"]
