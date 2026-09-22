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


def test_read_gene_data_reads_the_placeholder_format(tmp_path):
    """The optional polygenic score table: ncdsid first, then numeric
    scores, because the R takes them as colnames(gene_data)[-1]
    (R: llm_paper/_targets.R:L132)."""
    path = tmp_path / "pgs.csv"
    pd.DataFrame({"ncdsid": ["SYN000001", "SYN000002"], "syn_pgs_1": [0.1, -0.3],
                  "syn_pgs_2": [1.0, 2.0]}).to_csv(path, index=False)
    frame = read_gene_data(str(path))
    assert list(frame.columns) == ["ncdsid", "syn_pgs_1", "syn_pgs_2"]
    assert frame["ncdsid"].tolist() == ["SYN000001", "SYN000002"]

    other = tmp_path / "bad.csv"
    pd.DataFrame({"id": ["SYN000001"], "syn_pgs_1": [0.1]}).to_csv(other, index=False)
    with pytest.raises(ValueError, match="first column must be ncdsid"):
        read_gene_data(str(other))
    text = tmp_path / "text.csv"
    pd.DataFrame({"ncdsid": ["SYN000001"], "syn_pgs_1": ["high"]}).to_csv(text, index=False)
    with pytest.raises(ValueError, match="must be numeric"):
        read_gene_data(str(text))


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


def test_combine_ncds_collision_keeps_first_frame_like_plyr():
    """R pkg: plyr/R/join.r:L127–130 and rbind-fill.r:L70–71, L80 (1.8.9): a column in
    both frames is not duplicated and not coalesced; rows of the first frame keep the
    first frame's value, even when it is NA."""
    a = pd.DataFrame({"ncdsid": ["1", "2"], "shared": [1, np.nan]})
    b = pd.DataFrame({"ncdsid": ["1", "2"], "shared": [np.nan, 2]})
    out = combine_ncds(a, b)
    assert list(out.columns) == ["ncdsid", "shared"]
    vals = out.set_index("ncdsid")["shared"]
    assert vals.loc["1"] == 1 and bool(pd.isna(vals.loc["2"]))  # NOT filled from b
    (rec,) = out.attrs["combine_ncds_collisions"]
    assert rec["column"] == "shared" and rec["cells_differing"] == 2 and rec["frame"] == 1


def test_combine_ncds_right_only_rows_take_right_value_in_plyr_order():
    """R pkg: plyr/R/join.r:L125–130: matched = every x row (x order) with its y match;
    unmatched y rows are appended and bring their own values, also for shared columns."""
    a = pd.DataFrame({"ncdsid": ["2", "1"], "shared": [np.nan, 1.0], "x": [5, 6]})
    b = pd.DataFrame({"ncdsid": ["3", "2"], "shared": [30.0, 20.0], "y": [7, 8]})
    out = combine_ncds(a, b)
    assert list(out["ncdsid"]) == ["2", "1", "3"]
    assert list(out.columns) == ["ncdsid", "shared", "x", "y"]
    assert pd.isna(out.loc[0, "shared"]) and out.loc[1, "shared"] == 1.0 and out.loc[2, "shared"] == 30.0
    assert out.loc[0, "y"] == 8 and pd.isna(out.loc[2, "x"])


def test_combine_ncds_strict_mode_raises_on_collision():
    """Opt-in strict mode (not in the R) refuses shared columns."""
    from llm_cong_predict.io.readers import ColumnCollisionError

    a = pd.DataFrame({"ncdsid": ["1"], "shared": [1]})
    b = pd.DataFrame({"ncdsid": ["1"], "shared": [2]})
    with pytest.raises(ColumnCollisionError, match="shared"):
        combine_ncds(a, b, strict=True)
    assert combine_ncds(a, pd.DataFrame({"ncdsid": ["1"], "other": [3]}), strict=True).shape == (1, 3)


def test_combine_ncds_collided_column_keeps_first_frame_labels():
    """rbind.fill takes a column's attributes from its first occurrence
    (R pkg: plyr/R/rbind-fill.r:L70–71), so the first frame's labels win."""
    a = pd.DataFrame({"ncdsid": ["1"], "v": [1]})
    a.attrs["value_labels"] = {"v": {1: "first"}}
    b = pd.DataFrame({"ncdsid": ["1"], "v": [1]})
    b.attrs["value_labels"] = {"v": {1: "second"}}
    assert get_value_labels(combine_ncds(a, b))["v"] == {1: "first"}


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


# ------------------------------------------ haven / sjlabelled label semantics --

def test_read_ncds_drops_labels_of_recoded_missing_codes(tmp_path):
    """R pkg: sjlabelled/R/set_na.R:L268–272 (1.2.0): set_na(na = -99:-1) removes the
    value labels of the values it sets to NA, so they never become factor levels."""
    import pyreadstat

    df = pd.DataFrame({"ncdsid": ["SYN000001", "SYN000002"], "n876": [2.0, -1.0]})
    pyreadstat.write_dta(df, str(tmp_path / "w.dta"), variable_value_labels={
        "n876": {-1: "Not answered", 1: "Very poor", 2: "Poor"}})
    out = read_ncds(str(tmp_path / "w.dta"), ["n876"])
    assert bool(pd.isna(out.loc[1, "n876"]))
    assert set(get_value_labels(out)["n876"]) == {1, 2}


def test_as_factor_keeps_unlabelled_values_like_haven_default():
    """R pkg: haven/R/as_factor.R:L62–84 (2.5.5), levels = "default": an unlabelled value
    keeps its value as a level (it is not set to missing); levels are all
    labels plus observed unlabelled values, sorted by value, unobserved labels included."""
    df = pd.DataFrame({"c": [2.0, 47.0, 20.5, np.nan]})
    df.attrs["value_labels"] = {"c": {1: "one", 2: "two", 50: "fifty"}}
    fac = as_factor(df, "c")
    assert list(fac.categories) == ["one", "two", "20.5", "47", "fifty"]
    assert list(fac.astype(object)[:3]) == ["two", "47", "20.5"] and pd.isna(fac[3])


def test_as_factor_unlabelled_numeric_uses_r_number_strings():
    """forcats::as_factor.numeric = factor(x) (R pkg: forcats/R/as_factor.R:L52–54):
    levels are the sorted distinct values printed as R prints them (1, not 1.0)."""
    df = pd.DataFrame({"c": [3.0, 1.0, 3.0]})
    assert list(as_factor(df, "c").categories) == ["1", "3"]


def test_haven_codes_differ_from_observed_rank_codes_when_a_label_is_unobserved():
    """Owner decision C9. find_essay_teacher_genetics_overlap (functions.R:L332–336)
    codes a still-labelled column by its position in haven's level set, which includes
    labels that never occur (haven/R/as_factor.R:L74–82). clean_ncds' teacher block
    (functions.R:L190–191) codes a column whose labels are gone by its rank among the
    observed values (forcats factor(x)). With label 1 unobserved the two differ."""
    from llm_cong_predict.io.labels import labelled_factor_codes, observed_rank_codes

    df = pd.DataFrame({"n876": [2.0, 3.0, 3.0, np.nan]})
    df.attrs["value_labels"] = {"n876": {1: "Very poor", 2: "Poor", 3: "Average"}}
    haven = labelled_factor_codes(df, "n876")
    rank = observed_rank_codes(df["n876"])
    np.testing.assert_array_equal(haven, [2.0, 3.0, 3.0, np.nan])
    np.testing.assert_array_equal(rank, [1.0, 2.0, 2.0, np.nan])
    assert not np.array_equal(haven[:3], rank[:3])


def test_labelled_factor_codes_sets_dont_know_to_missing():
    """ifelse(x == "Dont know", NA, x) on the factor (functions.R:L335)."""
    from llm_cong_predict.io.labels import labelled_factor_codes

    df = pd.DataFrame({"c": [1.0, 8.0, 9.0]})  # 9 is unlabelled
    df.attrs["value_labels"] = {"c": {1: "Good", 8: "Dont know"}}
    np.testing.assert_array_equal(labelled_factor_codes(df, "c", ("Dont know",)), [1.0, np.nan, 3.0])


# ---------------------------------------- read_essays: tidyr::separate semantics --

def _write_essays(folder, contents: dict[str, bytes]):
    folder.mkdir(parents=True, exist_ok=True)
    for name, raw in contents.items():
        (folder / name).write_bytes(raw)


_SEP = "\n----------------------\n"


def test_read_essays_malformed_files_follow_tidyr_separate(tmp_path, caplog):
    """R: llm_paper/R/functions.R:L27–30 with tidyr::separate(extra = "warn",
    fill = "warn") (R pkg: tidyr/R/separate.R:L170–201, src/simplifyPieces.cpp; 1.3.2):
    pieces after a second separator are discarded, a missing piece is NA. The warning
    carries only counts: no file name, ID or text."""
    _write_essays(tmp_path / "e", {
        "a.txt": f"ID: SYN000001{_SEP}good essay  Words: 2".encode(),
        "b.txt": f"ID: SYN000002{_SEP}first  Words: 3  Words: 99".encode(),  # extra
        "c.txt": f"ID: SYN000003{_SEP}second part{_SEP}third  Words: 4".encode(),  # extra
        "d.txt": b"ID: SYN000004 no separator at all",  # missing (both)
        "e.txt": f"ID: SYN000005{_SEP}no word count".encode(),  # missing words separator
    })
    # c.txt counts twice: its second ID separator is discarded together with the
    # "  Words: " part behind it, so the words piece is then missing as well.
    with pytest.warns(RuntimeWarning, match=r"2 file\(s\) had additional pieces .* 3 file\(s\) had missing"):
        df = read_essays(str(tmp_path / "e")).set_index("doc_id")
    assert (df.loc["b.txt", "text"], df.loc["b.txt", "words"]) == ("first", "3")
    assert df.loc["c.txt", "text"] == "second part" and pd.isna(df.loc["c.txt", "words"])
    assert df.loc["d.txt", "ncdsid"] == "SYN000004 no separator at all"
    assert pd.isna(df.loc["d.txt", "text"]) and pd.isna(df.loc["d.txt", "words"])
    assert df.loc["e.txt", "text"] == "no word count" and pd.isna(df.loc["e.txt", "words"])
    assert df.attrs["read_essays_malformed"] == {"files_with_extra_pieces": 2,
                                                 "files_with_missing_pieces": 3}
    for record in caplog.records:
        for secret in ("SYN00000", ".txt", "good essay", "second part", "no separator"):
            assert secret not in record.getMessage()


def test_read_essays_reads_text_like_readtext(tmp_path):
    """R pkg: readtext/R/get-functions.R:L2–4 (0.92.1): paste(readLines(con), collapse =
    "\\n"): CRLF and CR line endings become "\\n" and the final line terminator is
    dropped, so the word count is "8", not "8\\n"."""
    _write_essays(tmp_path / "e", {
        "a.txt": b"ID: SYN000001\r\n----------------------\r\nline one\r\nline two  Words: 8\r\n",
    })
    row = read_essays(str(tmp_path / "e")).iloc[0]
    assert (row["ncdsid"], row["text"], row["words"]) == ("SYN000001", "line one\nline two", "8")


def test_r_as_numeric_matches_r():
    """Expected values computed with R 4.6.1:
    as.numeric(c("123","123\\n"," 45 ","1e3","1_000","abc","0x1A","Inf","inf","NaN","NA",
                 "1.5e-2","+7",".5","5.")) (NaN and NA are both missing here)."""
    from llm_cong_predict.io.labels import r_as_numeric

    got = r_as_numeric(["123", "123\n", " 45 ", "1e3", "1_000", "abc", "0x1A", "Inf", "inf",
                        "NaN", "NA", "1.5e-2", "+7", ".5", "5."]).tolist()
    expected = [123, 123, 45, 1000, np.nan, np.nan, 26, np.inf, np.inf, np.nan, np.nan,
                0.015, 7, 0.5, 5]
    np.testing.assert_array_equal(got, expected)


def test_check_essay_format_prints_only_counts(tmp_path):
    """scripts/check_essay_format.py prints aggregate
    counts and never essay text, file names or IDs."""
    import subprocess
    import sys
    from pathlib import Path

    _write_essays(tmp_path / "e", {
        "file_alpha.txt": f"ID: SYN000001{_SEP}secret words here  Words: 3".encode(),
        "file_beta.txt": f"ID: SYN000002{_SEP}more secret text  Words: three".encode(),
        "file_gamma.txt": f"ID: SYN000003{_SEP}x{_SEP}y  Words: 1".encode(),
        "file_delta.txt": b"ID: SYN000004 secret without separators",
        "file_eps.txt": b"\xff\xfe\xfa not utf-8",
    })
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_essay_format.py"
    result = subprocess.run([sys.executable, str(script), "--essays", str(tmp_path / "e")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    # gamma: extra ID separator -> its "  Words: " part is discarded -> missing, no count
    assert result.stdout.splitlines() == [
        "files found: 5", "matching format: 1", "missing separators: 2",
        "extra separators: 1", "word count unparsable: 3", "unreadable files: 1"]
    output = result.stdout + result.stderr
    for secret in ("SYN0000", "file_", "secret", ".txt", str(tmp_path)):
        assert secret not in output
