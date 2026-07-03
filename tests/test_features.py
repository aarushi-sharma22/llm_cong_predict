"""Tests for the feature layer.

Covers what CAN be verified here (pure dataframe logic): SALAT/spelling ingestion,
essay-variable assembly + filter, the GPT reshaper, and that the readability
external-tool boundary raises rather than substituting. RoBERTa generation and GPT
generation need model weights / API keys / real essays and are NOT run here
(validation deferred).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from llm_cong_predict.features.essay_variables import create_essay_variables
from llm_cong_predict.features.readability import (
    calculate_readability_metrics,
    ingest_readability_metrics,
    tokenize_essays,
)
from llm_cong_predict.features.salat import get_salat_metrics, get_spelling_error_metrics
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
    # essays with word counts
    essays = pd.DataFrame({"ncdsid": ["1", "2"], "words": ["100", "50"]})
    # LanguageTool-style per-issue rows
    spelling = pd.DataFrame(
        {
            "ncdsid": ["1", "1", "2"],
            "rule_issue_type": ["misspelling", "grammar", "misspelling"],
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


def test_ingest_readability_metrics_path(tmp_path):
    # the provided ingestion path should join a pre-computed readability CSV
    essays = pd.DataFrame({"doc_id": ["e1", "e2"], "ncdsid": ["1", "2"]})
    read_csv = tmp_path / "read.csv"
    pd.DataFrame({"filename": ["e1", "e2"], "flesch": [70.0, 65.0]}).to_csv(read_csv, index=False)
    out = ingest_readability_metrics(essays, str(read_csv))
    assert "flesch" in out.columns
    assert out[out["ncdsid"] == "1"].iloc[0]["flesch"] == 70.0
