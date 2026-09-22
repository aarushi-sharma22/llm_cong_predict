"""Tests for the feature layer.

Covers what CAN be verified here (pure dataframe logic): SALAT/spelling ingestion,
essay-variable assembly + filter, the GPT reshaper, that the readability
external-tool boundary raises rather than substituting, and RoBERTa pooling/batching
on a tiny randomly initialised model (no download). Embeddings from the real
roberta-base weights, and GPT embeddings (an external API, gated off), are not
produced here.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llm_cong_predict.features.essay_variables import create_essay_variables
from llm_cong_predict.features.readability import (
    calculate_readability_metrics,
    ingest_readability_metrics,
    tokenize_essays,
)
from llm_cong_predict.features.salat import (
    SpellingMetricsError,
    get_salat_metrics,
    get_spelling_error_metrics,
)
from llm_cong_predict.features.embeddings import gpt_embeddings


# --------------------------------------------------------------- SALAT ingest --

def test_get_salat_metrics_joins_and_drops_filename(tmp_path):
    essays = pd.DataFrame({"doc_id": ["e1.txt", "e2.txt"], "ncdsid": ["1", "2"]})
    taaled = tmp_path / "taaled.csv"
    taales = tmp_path / "taales.csv"
    seance = tmp_path / "seance.csv"
    pd.DataFrame({"filename": ["e1.txt", "e2.txt"], "mtld": [50.0, 60.0]}).to_csv(taaled, index=False)
    # TAALES uses capital 'Filename' -> renamed to 'filename' by the port
    pd.DataFrame({"Filename": ["e1.txt", "e2.txt"], "aoa": [3.1, 4.2]}).to_csv(taales, index=False)
    pd.DataFrame({"filename": ["e1.txt", "e2.txt"], "sentiment": [0.5, -0.2]}).to_csv(seance, index=False)

    out = get_salat_metrics(essays, [str(taaled)], [str(taales)], [str(seance)])
    assert "filename" not in out.columns
    assert set(["ncdsid", "mtld", "aoa", "sentiment"]).issubset(out.columns)
    row1 = out[out["ncdsid"] == "1"].iloc[0]
    assert row1["mtld"] == 50.0 and row1["aoa"] == 3.1 and row1["sentiment"] == 0.5


# ------------------------------------------------------------ spelling ingest --

def test_get_spelling_error_metrics_pivots_fills_and_sums(tmp_path):
    # Fixture extended in Phase 1 (owner decision C8): the R stops unless all nine
    # categories occur somewhere (functions.R:L410-414) and at least one essay has no
    # error (L415), so essay 3 carries the other seven categories and essay 4 none.
    essays = pd.DataFrame({"ncdsid": ["1", "2", "3", "4"], "words": ["100", "50", "10", "20"]})
    others = ["typographical", "locale-violation", "duplication", "style", "whitespace",
              "uncategorized", "inconsistency"]
    spelling = pd.DataFrame(
        {
            "ncdsid": ["1", "1", "2"] + ["3"] * len(others),
            "rule_issue_type": ["misspelling", "grammar", "misspelling"] + others,
        }
    )
    path = tmp_path / "spelling.csv"
    spelling.to_csv(path, index=False)

    out = get_spelling_error_metrics(essays, str(path))
    r1 = out[out["ncdsid"] == "1"].iloc[0]
    # essay 1: 1 misspelling / 100 words = 0.01 ; 1 grammar / 100 = 0.01
    assert r1["misspelling"] == pytest.approx(0.01)
    assert r1["grammar"] == pytest.approx(0.01)
    # a category with no occurrences is filled to 0
    assert r1["typographical"] == 0
    # total sums the nine categories
    assert r1["total"] == pytest.approx(0.02)
    # essay 2: 1 misspelling / 50 = 0.02
    r2 = out[out["ncdsid"] == "2"].iloc[0]
    assert r2["misspelling"] == pytest.approx(0.02)


# ------------------------------------------------------- create_essay_variables --

def test_create_essay_variables_joins_and_filters():
    # salat has a good column and a zero-variance column; readability adds one;
    # spelling one; embeddings two. Filter should drop zero-variance + any-NA cols.
    salat = pd.DataFrame(
        {"filename": ["e1", "e2", "e3"], "ncdsid": ["1", "2", "3"],
         "good": [1.0, 2.0, 3.0], "constant": [5.0, 5.0, 5.0]}
    )
    readability = pd.DataFrame({"filename": ["e1", "e2", "e3"], "ncdsid": ["1", "2", "3"], "read": [0.1, 0.2, 0.3]})
    spelling = pd.DataFrame({"ncdsid": ["1", "2", "3"], "spell": [0.0, 0.1, 0.2]})
    emb = pd.DataFrame({"id": ["1", "2", "3"], "e_1": [1.0, 2.0, 3.0], "e_2": [9.0, 8.0, 7.0]})

    out = create_essay_variables(salat, readability, spelling, emb)
    assert "ncdsid" in out.columns
    assert "filename" not in out.columns
    assert "good" in out.columns and "read" in out.columns and "e_1" in out.columns
    # zero-variance column dropped by the select_if filter
    assert "constant" not in out.columns
    assert len(out) == 3


def test_create_essay_variables_drops_columns_with_na():
    salat = pd.DataFrame({"ncdsid": ["1", "2"], "has_na": [1.0, np.nan], "ok": [1.0, 2.0]})
    empty = pd.DataFrame({"ncdsid": ["1", "2"]})
    emb = pd.DataFrame({"id": ["1", "2"], "e_1": [1.0, 2.0]})
    out = create_essay_variables(salat, empty, empty, emb)
    assert "has_na" not in out.columns  # column with an NA is excluded
    assert "ok" in out.columns


# --------------------------------------------------------------- gpt reshaper --

def test_gpt_embeddings_reshapes_saved_file(tmp_path):
    saved = pd.DataFrame(
        {"ncdsid": ["1", "2"], "embedding_1": [0.1, 0.2], "embedding_2": [0.3, 0.4]}
    )
    path = tmp_path / "emb.csv"   # CSV to avoid a hard pyarrow dependency in tests
    saved.to_csv(path, index=False)
    out = gpt_embeddings(str(path))
    assert "id" in out.columns and "ncdsid" not in out.columns
    assert list(out.columns) == ["id", "embedding_1", "embedding_2"]


def test_gpt_embeddings_rejects_missing_ncdsid(tmp_path):
    bad = pd.DataFrame({"wrong": [1, 2]})
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError):
        gpt_embeddings(str(path))


# ------------------------------------------------- readability boundary raises --

def test_tokenize_essays_raises_external_boundary():
    with pytest.raises(NotImplementedError):
        tokenize_essays(pd.DataFrame({"text": ["hello"]}))


def test_calculate_readability_metrics_raises_external_boundary():
    with pytest.raises(NotImplementedError):
        calculate_readability_metrics(pd.DataFrame({"doc_id": ["e1"], "ncdsid": ["1"]}))


def test_ingest_readability_metrics_reads_what_the_korpus_script_writes(tmp_path):
    """The file r/readability.R writes (Task 2.5), the R's
    calculate_readability_metrics output (R: llm_paper/R/functions.R:L384-387):
    filename, ncdsid, then one column per koRpus index. The R writes the values as
    text; pandas reads numbers back where it can, and create_essay_variables coerces
    either way (functions.R:L324). One row per essay, in essay order; an essay with no
    row gets NaN, which create_essay_variables then drops (G5). The script itself has
    not been run here."""
    essays = pd.DataFrame({"doc_id": ["e1.txt", "e2.txt", "e3.txt"],
                           "ncdsid": ["SYN000001", "SYN000002", "SYN000003"]})
    path = tmp_path / "readability_metrics.csv"
    pd.DataFrame({"filename": ["e2.txt", "e1.txt"], "ncdsid": ["SYN000002", "SYN000001"],
                  "Flesch.Kincaid": ["8.1", "10.4"], "ARI": ["9.2", "11.0"]}).to_csv(path, index=False)

    out = ingest_readability_metrics(essays, str(path))
    assert list(out.columns) == ["filename", "ncdsid", "Flesch.Kincaid", "ARI"]
    assert out["filename"].tolist() == ["e1.txt", "e2.txt", "e3.txt"]  # essay order
    assert float(out.loc[out["ncdsid"] == "SYN000001", "Flesch.Kincaid"].iloc[0]) == 10.4
    assert pd.isna(out.loc[out["ncdsid"] == "SYN000003", "ARI"].iloc[0])  # no row for e3


def test_ingest_readability_metrics_path(tmp_path):
    # the provided ingestion path should join a pre-computed readability CSV
    essays = pd.DataFrame({"doc_id": ["e1", "e2"], "ncdsid": ["1", "2"]})
    read_csv = tmp_path / "read.csv"
    pd.DataFrame({"filename": ["e1", "e2"], "flesch": [70.0, 65.0]}).to_csv(read_csv, index=False)
    out = ingest_readability_metrics(essays, str(read_csv))
    assert "flesch" in out.columns
    assert out[out["ncdsid"] == "1"].iloc[0]["flesch"] == 70.0


# -------------------------------------------------- spelling (brief F8, C8) --

_NINE = ["grammar", "misspelling", "typographical", "locale-violation", "duplication",
         "style", "whitespace", "uncategorized", "inconsistency"]


def _spelling(tmp_path, essays, rows):
    path = tmp_path / "spelling.csv"
    pd.DataFrame(rows, columns=["ncdsid", "rule_issue_type"]).to_csv(path, index=False)
    return get_spelling_error_metrics(essays, str(path))


def test_spelling_keeps_essays_without_errors_with_zeros(tmp_path):
    """Brief F8 reproducer: essays A, B, C with error rows only for A and B. In R the
    left join (functions.R:L399–400) keeps C with rule type NA, pivot_wider makes an
    "NA" column, and C's categories are filled with 0 (L405–408); the old port lost C."""
    essays = pd.DataFrame({"ncdsid": ["A", "B", "C"], "words": ["10", "20", "30"]})
    rows = [("A", c) for c in _NINE[:5]] + [("B", c) for c in _NINE[5:]]
    out = _spelling(tmp_path, essays, rows)
    assert list(out["ncdsid"]) == ["A", "B", "C"]
    c = out.set_index("ncdsid").loc["C"]
    assert (c[_NINE] == 0).all() and c["total"] == 0 and c["other"] == 0
    assert out.set_index("ncdsid").loc["A", "total"] == pytest.approx(5 / 10)
    assert "NA" not in out.columns


def test_spelling_columns_follow_first_appearance_with_sorted_types(tmp_path):
    """pivot_wider orders columns by first appearance (L404); count() returns each
    essay's rule types sorted (L394–396), and the join keeps essay order (L399–400)."""
    essays = pd.DataFrame({"ncdsid": ["A", "B", "C"], "words": ["10", "20", "30"]})
    rows = [("A", "style"), ("A", "grammar")] + [("B", c) for c in _NINE]
    out = _spelling(tmp_path, essays, rows)
    rest = sorted(set(_NINE) - {"grammar", "style"})
    assert list(out.columns) == ["ncdsid", "grammar", "style"] + rest + ["total", "other"]


def test_spelling_raises_like_r_when_no_essay_is_error_free(tmp_path):
    """Owner decision C8: with no error-free essay, pivot_wider makes no "NA" column and
    R's select(-"NA") stops (functions.R:L415)."""
    essays = pd.DataFrame({"ncdsid": ["A", "B"], "words": ["10", "20"]})
    rows = [("A", c) for c in _NINE] + [("B", "grammar")]
    with pytest.raises(SpellingMetricsError, match=r"select\(-\"NA\"\).*L415"):
        _spelling(tmp_path, essays, rows)


def test_spelling_raises_like_r_when_a_category_never_occurs(tmp_path):
    """Owner decision C8: rowSums(select(., grammar, ..., inconsistency)) names every
    category, so R stops when one never occurs (functions.R:L410–414)."""
    essays = pd.DataFrame({"ncdsid": ["A", "B"], "words": ["10", "20"]})
    rows = [("A", c) for c in _NINE[:-1]]
    with pytest.raises(SpellingMetricsError, match=r"\['inconsistency'\].*L410-414"):
        _spelling(tmp_path, essays, rows)


# ------------------------------------------------------ SALAT joins (brief F8) --

def test_salat_natural_join_uses_shared_metric_as_key(tmp_path):
    """R: llm_paper/R/functions.R:L440–442: left_join without `by` joins on every shared
    column (dplyr join_by_common). A metric reported by two tools (here nwords) becomes a
    join key, and an essay whose values differ gets NA in all of the later tool's columns."""
    essays = pd.DataFrame({"doc_id": ["e1.txt", "e2.txt"], "ncdsid": ["1", "2"]})
    paths = {}
    for tool, frame in {
        "taaled": pd.DataFrame({"filename": ["e1.txt", "e2.txt"], "mtld": [50.0, 60.0], "nwords": [100, 200]}),
        "taales": pd.DataFrame({"Filename": ["e1.txt", "e2.txt"], "aoa": [3.1, 4.2], "nwords": [100, 199]}),
        "seance": pd.DataFrame({"filename": ["e1.txt", "e2.txt"], "sentiment": [0.5, -0.2]}),
    }.items():
        paths[tool] = tmp_path / f"{tool}.csv"
        frame.to_csv(paths[tool], index=False)
    out = get_salat_metrics(essays, [str(paths["taaled"])], [str(paths["taales"])], [str(paths["seance"])])
    assert out.attrs["salat_join_keys"] == {
        "taaled": ["filename"], "taales": ["filename", "nwords"], "seance": ["filename"]}
    rows = out.set_index("ncdsid")
    assert rows.loc["1", "aoa"] == 3.1
    assert pd.isna(rows.loc["2", "aoa"])  # nwords 200 vs 199: TAALES columns lost
    assert rows.loc["2", "sentiment"] == -0.2


# -------------------------------------------------------- RoBERTa (brief F8) --
#
# These tests run in a SUBPROCESS. torch bundles its own OpenMP runtime
# (install name /opt/llvm-openmp/lib/libomp.dylib) while xgboost loads Homebrew's
# (/opt/homebrew/opt/libomp/lib/libomp.dylib); once torch is imported, a later xgboost
# fit in the same process segfaults (observed with torch 2.14.0 and xgboost 3.3.0 on
# macOS). The pytest process therefore never imports torch (PORTING_NOTES G2).

_SRC = str(Path(__file__).resolve().parents[1] / "src")
_TORCH_MISSING = (importlib.util.find_spec("torch") is None
                  or importlib.util.find_spec("transformers") is None)
_needs_torch = pytest.mark.skipif(
    _TORCH_MISSING, reason="torch/transformers not installed (pip install -e '.[embeddings]')")

_TINY_MODEL = """
import numpy as np, pandas as pd, torch, transformers
torch.manual_seed(0)
cfg = transformers.RobertaConfig(vocab_size=100, hidden_size=16, num_hidden_layers=2,
                                 num_attention_heads=2, intermediate_size=32,
                                 max_position_embeddings=40)
model = transformers.RobertaModel(cfg).eval()
"""


def _run_isolated(body: str) -> None:
    code = _TINY_MODEL + textwrap.dedent(body) + "\nprint('ISOLATED-OK')\n"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={"PYTHONPATH": _SRC, "PATH": "/usr/bin:/bin"})
    assert "ISOLATED-OK" in result.stdout, result.stdout + result.stderr


@_needs_torch
def test_roberta_pool_batched_equals_unbatched():
    """R: llm_paper/R/functions.R:L480–482: mean of the last hidden state over all
    positions, no attention mask. Rows are independent, so batching must give identical
    numbers (brief F8). Tiny randomly initialised model; nothing is downloaded."""
    _run_isolated("""
        from llm_cong_predict.features.embeddings import roberta_pool
        ids = torch.randint(3, cfg.vocab_size, (7, 20))
        ids[:, 15:] = cfg.pad_token_id  # padding positions are included in the mean, as in R
        full = roberta_pool(ids, model)
        assert full.shape == (7, 16)
        for bs in (1, 2, 3):
            batched = np.concatenate([roberta_pool(ids[i:i + bs], model) for i in range(0, 7, bs)])
            np.testing.assert_array_equal(batched, full)
    """)


@_needs_torch
def test_roberta_embeddings_batch_size_does_not_change_output():
    """roberta_embeddings runs the model in batches; the frame is identical for any
    batch size and has id + roberta_dim_* columns (functions.R:L487–489)."""
    _run_isolated("""
        from llm_cong_predict.features.embeddings import roberta_embeddings

        class Tok:  # stands in for the roberta-base tokenizer (no download)
            def __call__(self, texts, max_length, truncation, padding, return_tensors):
                ids = torch.full((len(texts), max_length), cfg.pad_token_id)
                for i, t in enumerate(texts):
                    toks = [3 + (ord(ch) % 90) for ch in t][:max_length]
                    ids[i, :len(toks)] = torch.tensor(toks)
                return {"input_ids": ids}

        essays = pd.DataFrame({"ncdsid": ["SYN000001", "SYN000002", "SYN000003"],
                               "text": ["a short essay", "another one here", "x"]})
        one = roberta_embeddings(essays, max_len=12, batch_size=1, model=model, tokenizer=Tok())
        three = roberta_embeddings(essays, max_len=12, batch_size=3, model=model, tokenizer=Tok())
        pd.testing.assert_frame_equal(one, three)
        assert list(one.columns[:2]) == ["id", "roberta_dim_1"] and one.shape == (3, 17)
    """)


# ------------------------------------ process isolation and the RoBERTa step --

def _run_plain(code: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": _SRC, "PATH": "/usr/bin:/bin", **(env_extra or {})}
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True,
                          text=True, env=env)


@_needs_torch
def test_xgboost_learner_refuses_after_torch_is_imported():
    """docs/ORCHESTRATION.md (owner decision, Checkpoint B): torch and xgboost never share
    a process. With torch loaded, building SL.xgboost.hist raises before xgboost is
    imported, instead of segfaulting."""
    result = _run_plain("""
        import sys, torch
        from llm_cong_predict.isolation import ProcessIsolationError
        from llm_cong_predict.models.base_learners import _make_xgboost_hist
        try:
            _make_xgboost_hist(1, 3)
        except ProcessIsolationError as exc:
            print("REFUSED", "xgboost" in sys.modules)
    """)
    assert "REFUSED False" in result.stdout, result.stdout + result.stderr


def test_roberta_generation_refuses_after_xgboost_is_imported():
    """The other direction: with xgboost loaded, RoBERTa generation raises before torch
    is imported."""
    result = _run_plain("""
        import sys, xgboost
        import pandas as pd
        from llm_cong_predict.isolation import ProcessIsolationError
        from llm_cong_predict.features.embeddings import roberta_embeddings
        try:
            roberta_embeddings(pd.DataFrame({"ncdsid": ["SYN000001"], "text": ["x"]}))
        except ProcessIsolationError:
            print("REFUSED", "torch" in sys.modules)
    """)
    assert "REFUSED False" in result.stdout, result.stdout + result.stderr


@_needs_torch
def test_roberta_step_writes_derived_file_that_reads_back_exactly(tmp_path):
    """The separate RoBERTa step writes $LCP_DATA_ROOT/derived/roberta_embeddings.csv;
    read_roberta_embeddings returns exactly the same float64 values, in this
    (torch-free) process."""
    from llm_cong_predict.features.embeddings import read_roberta_embeddings

    out = tmp_path / "derived" / "roberta_embeddings.csv"
    out.parent.mkdir()
    _run_isolated(f"""
        from llm_cong_predict.features.roberta_step import write_roberta_embeddings

        class Tok:
            def __call__(self, texts, max_length, truncation, padding, return_tensors):
                ids = torch.full((len(texts), max_length), cfg.pad_token_id)
                for i, t in enumerate(texts):
                    toks = [3 + (ord(ch) % 90) for ch in t][:max_length]
                    ids[i, :len(toks)] = torch.tensor(toks)
                return {{"input_ids": ids}}

        essays = pd.DataFrame({{"ncdsid": ["SYN000001", "SYN000002"], "text": ["one", "two words"]}})
        emb = write_roberta_embeddings(essays, {str(out)!r}, max_len=10, model=model, tokenizer=Tok())
        emb.to_pickle({str(tmp_path / "expected.pkl")!r})
    """)
    got = read_roberta_embeddings(str(out))
    expected = pd.read_pickle(tmp_path / "expected.pkl")
    pd.testing.assert_frame_equal(got, expected)
    assert list(got["id"]) == ["SYN000001", "SYN000002"]


def test_roberta_step_refuses_without_data_root():
    """The step reads essays only from $LCP_DATA_ROOT (brief Section 2.3)."""
    result = _run_plain("""
        from llm_cong_predict.features.roberta_step import main
        from llm_cong_predict.config import DataRootError
        try:
            main([])
        except DataRootError as exc:
            print("REFUSED")
    """)
    assert "REFUSED" in result.stdout, result.stdout + result.stderr


def test_read_roberta_embeddings_checks_columns(tmp_path):
    from llm_cong_predict.features.embeddings import read_roberta_embeddings

    bad = tmp_path / "bad.csv"
    pd.DataFrame({"id": ["SYN000001"], "roberta_dim_2": [0.1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="roberta_dim_1"):
        read_roberta_embeddings(str(bad))
