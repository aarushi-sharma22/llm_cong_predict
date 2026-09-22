"""The export guard.

Every file written outside ``$LCP_DATA_ROOT`` goes through one function, which refuses
a table with an ID-like column, a free-text column, one row per person, or a count
below the minimum cell size — and refuses everything until that minimum is set.
"""

from __future__ import annotations

import pandas as pd
import pytest

from llm_cong_predict import config
from llm_cong_predict.export import (
    ExportRefused,
    MinimumCellSizeNotSet,
    check_export,
    export_table,
    export_tables,
)

N_PEOPLE = 500


@pytest.fixture
def cell_size(monkeypatch):
    """A confirmed minimum cell size, so the other checks can be reached."""
    monkeypatch.setattr(config, "MINIMUM_CELL_SIZE", 10)
    return 10


def _aggregate() -> pd.DataFrame:
    """An aggregate table of the kind the reporting module builds."""
    return pd.DataFrame({
        "target": ["essay_superlearner", "teacher_superlearner"],
        "var": ["s2_co_verbal_ability", "s2_co_verbal_ability"],
        "name": ["Verbal Ability (Age 11)", "Verbal Ability (Age 11)"],
        "type": ["Prediction based on ~250 Word Essay", "Prediction based on Teacher Evaluation"],
        "mean_r2": [0.31, 0.42], "n": [480, 480],
    })


# --------------------------------------------------- the minimum cell size ----

def test_the_configured_minimum_cell_size():
    """10. The UK Data Service's handling guide gives 3
    as the baseline threshold and advises 10 where several outputs come from the same
    source, which is the case here. Still to be confirmed with UKDS."""
    assert config.MINIMUM_CELL_SIZE == 10


def test_nothing_is_exported_when_no_minimum_cell_size_is_set(monkeypatch, tmp_path):
    """the minimum comes from the config and the code never picks one;
    with it unset the guard raises, pointing at the UK Data Service's output rules."""
    monkeypatch.setattr(config, "MINIMUM_CELL_SIZE", None)
    with pytest.raises(MinimumCellSizeNotSet, match="UK Data Service"):
        export_table(_aggregate(), tmp_path / "fig_2_data.csv", n_people=N_PEOPLE)
    assert list(tmp_path.iterdir()) == []


def test_a_nonsense_minimum_cell_size_is_refused(monkeypatch):
    monkeypatch.setattr(config, "MINIMUM_CELL_SIZE", 0)
    with pytest.raises(MinimumCellSizeNotSet, match="positive integer"):
        check_export(_aggregate(), n_people=N_PEOPLE)


# ------------------------------------------------------------ each refusal ----

@pytest.mark.parametrize("column", ["ncdsid", "NCDSID", "id"])
def test_a_table_with_an_id_like_column_is_refused(cell_size, column):
    table = _aggregate()
    table[column] = ["SYN000001", "SYN000002"]
    with pytest.raises(ExportRefused, match="ID-like column"):
        check_export(table, n_people=N_PEOPLE)


def test_a_table_with_a_free_text_column_is_refused(cell_size):
    """A long value is text; the labels the R's tables carry are not."""
    table = _aggregate()
    table["essay_text"] = ["synthetic essay " * 20, "another synthetic essay " * 20]
    with pytest.raises(ExportRefused, match="free-text column"):
        check_export(table, n_people=N_PEOPLE)
    check_export(_aggregate(), n_people=N_PEOPLE)  # the label columns are fine


def test_a_column_named_like_text_is_refused_however_short(cell_size):
    table = _aggregate()
    table["text"] = ["ok", "ok"]
    with pytest.raises(ExportRefused, match="free-text column"):
        check_export(table, n_people=N_PEOPLE)


def test_a_table_with_one_row_per_person_is_refused(cell_size):
    """A row count equal to the number of people in the analysis is what a per-person
    table looks like, whatever its columns are called."""
    table = pd.DataFrame({"prediction": range(N_PEOPLE), "fold": [1] * N_PEOPLE})
    with pytest.raises(ExportRefused, match="number of people"):
        check_export(table, n_people=N_PEOPLE)
    # several sample sizes: any of them refuses
    with pytest.raises(ExportRefused, match="number of people"):
        check_export(table, n_people=[120, N_PEOPLE, 980])
    check_export(table.iloc[:-1], n_people=N_PEOPLE)  # one row fewer is not that table


def test_a_count_below_the_minimum_cell_size_is_refused(cell_size):
    """a count column may hold no value below the minimum."""
    table = pd.DataFrame({"aspired_job": ["Teacher", "Fisherman"], "n": [120, 3]})
    with pytest.raises(ExportRefused, match="below the minimum cell size 10"):
        check_export(table, n_people=N_PEOPLE)
    at_the_limit = pd.DataFrame({"aspired_job": ["Teacher", "Fisherman"], "n": [120, 10]})
    check_export(at_the_limit, n_people=N_PEOPLE)  # exactly the minimum is allowed


def test_proportions_are_not_counts(cell_size):
    """appendix D5 holds proportions, not counts: a value below the minimum there is a
    fraction, not a small cell."""
    table = pd.DataFrame({"name": ["General Knowledge"] * 2, "value": ["Above average", "Average"],
                          "n": [0.02, 0.98]})
    check_export(table, n_people=N_PEOPLE)


# --------------------------------------------------------- what gets written ----

def test_a_clean_aggregate_table_is_written(cell_size, tmp_path):
    path = export_table(_aggregate(), tmp_path / "out" / "fig_2_data.csv", n_people=N_PEOPLE)
    assert path.exists()
    pd.testing.assert_frame_equal(pd.read_csv(path), _aggregate())


def test_a_refused_table_leaves_no_file_behind(cell_size, tmp_path):
    table = _aggregate()
    table["ncdsid"] = ["SYN000001", "SYN000002"]
    with pytest.raises(ExportRefused):
        export_table(table, tmp_path / "fig_2_data.csv", n_people=N_PEOPLE)
    assert not (tmp_path / "fig_2_data.csv").exists()


def test_export_tables_reports_each_refusal_and_writes_the_rest(cell_size, tmp_path):
    """The point is to say which tables may leave and which may not, so one refusal
    does not hide the others."""
    tables = {
        "fig_2_data": _aggregate(),
        "appendix_D8_data": pd.DataFrame({"aspired_job": ["Teacher", "Fisherman"], "n": [120, 2]}),
        "appendix_D4_data": pd.DataFrame({"ncdsid": ["SYN000001"], "words": [250]}),
    }
    written, refused = export_tables(tables, tmp_path, n_people=N_PEOPLE, prefix="SMOKE_")
    assert [p.name for p in written] == ["SMOKE_fig_2_data.csv"]
    assert set(refused) == {"appendix_D8_data", "appendix_D4_data"}
    assert "minimum cell size" in refused["appendix_D8_data"]
    assert "ID-like column" in refused["appendix_D4_data"]
    assert not (tmp_path / "SMOKE_appendix_D8_data.csv").exists()


def test_export_tables_stops_entirely_when_no_minimum_is_set(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MINIMUM_CELL_SIZE", None)
    with pytest.raises(MinimumCellSizeNotSet):
        export_tables({"fig_2_data": _aggregate()}, tmp_path, n_people=N_PEOPLE)
    assert list(tmp_path.iterdir()) == []
