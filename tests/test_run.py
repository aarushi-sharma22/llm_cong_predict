"""Model runner, data preparation, gene skipping and the run log.

Synthetic data only (IDs SYN000001...), generated from a seed, with a planted linear
signal.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from llm_cong_predict import config
from llm_cong_predict.metrics.cv_metrics import lm_metrics, superlearner_metrics
from llm_cong_predict.models.data_prep import apply_data_prep, r_as_numeric_column
from llm_cong_predict.models.run import NonNumericPredictorError, fit_model, save_predictions
from llm_cong_predict.pipeline.model_spec import DATA_PREP, model_specs, split_gene_dependent
from llm_cong_predict.pipeline.run_log import skipped_targets_entry, write_run_log
from llm_cong_predict.rbridge import r_unavailable_reason


def _planted(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2, x3 = rng.normal(size=(3, n))
    df = pd.DataFrame({
        "ncdsid": [f"SYN{i:06d}" for i in range(1, n + 1)],
        "x1": x1, "x2": x2, "x3": x3,
        "noise_col": rng.normal(size=n),
        "y": 2.0 * x1 - 1.0 * x2 + rng.normal(scale=0.5, size=n),  # planted signal
    })
    df.loc[[3, 17], "x1"] = np.nan  # predictor missing -> row dropped
    df.loc[[5], "y"] = np.nan  # outcome missing -> row dropped
    df.loc[[8, 9], "noise_col"] = np.nan  # not selected -> rows kept
    return df


# --------------------------------------------------------------- fit_model --

def test_fit_model_lm_recovers_planted_signal_and_applies_na_omit():
    """R: llm_paper/R/functions.R:L526–548 (get_lm_cv_model): select outcome and
    predictors, na.omit, CV.SuperLearner with mean + lm; returns (fit, var)."""
    df = _planted()
    fit, var = fit_model("y", ["x1", "x2", "x3"], df, method="lm")
    assert var == "y" and fit.outcome_var == "y"
    assert len(fit.Y) == 200 - 3  # rows 3, 17 (x1) and 5 (y) dropped; noise_col ignored
    assert fit.library_names == ["SL.mean_All", "SL.lm_All"]
    assert set(fit.ids) == set(df["ncdsid"]) - {"SYN000004", "SYN000018", "SYN000006"}
    assert lm_metrics(fit)["mean_r2"] > 0.8  # planted R^2 ~ 0.95
    assert len(fit.folds) == 10


def test_fit_model_refuses_non_numeric_predictors():
    """after the documented data preparation every predictor is
    numeric; anything else raises instead of being coerced."""
    df = _planted()
    df["flag"] = df["x3"] > 0  # bool
    df["cat"] = pd.Categorical(np.where(df["x3"] > 0, "a", "b"))
    df["text"] = "t"
    for col in ("flag", "cat", "text"):
        with pytest.raises(NonNumericPredictorError, match=col):
            fit_model("y", ["x1", col], df, method="lm")
    df["ybool"] = df["y"] > 0
    with pytest.raises(NonNumericPredictorError, match="outcome Y must be a numeric vector"):
        fit_model("ybool", ["x1"], df, method="lm")


def test_fit_model_missing_column_raises():
    with pytest.raises(KeyError, match="not_there"):
        fit_model("y", ["x1", "not_there"], _planted(), method="lm")


@pytest.mark.slow
def test_fit_model_superlearner_recovers_planted_signal():
    """R: llm_paper/R/functions.R:L496–524 — the full 6-learner library, 10/5 folds."""
    fit, _ = fit_model("y", ["x1", "x2", "x3"], _planted(n=150), method="superlearner")
    assert fit.library_names[0] == "SL.mean_All" and len(fit.library_names) == 6
    assert fit.failures == []
    assert superlearner_metrics(fit)["mean_r2"] > 0.8


@pytest.mark.skipif(r_unavailable_reason(("SuperLearner", "nnls")) is not None,
                    reason=r_unavailable_reason(("SuperLearner", "nnls")) or "")
def test_fit_model_r_backend_uses_the_same_folds():
    """backend="r" gets the native default outer and inner folds, so the lm library
    agrees with the native one to rounding."""
    df = _planted()
    native, _ = fit_model("y", ["x1", "x2", "x3"], df, method="lm")
    r, _ = fit_model("y", ["x1", "x2", "x3"], df, method="lm", backend="r")
    np.testing.assert_allclose(r.sl_predict, native.sl_predict, atol=1e-10)
    np.testing.assert_array_equal(r.ids, native.ids)


# ------------------------------------------------------ per-person predictions --

def test_save_predictions_only_under_data_root(monkeypatch, tmp_path):
    """Per-person predictions are participant-level: written only to
    $LCP_DATA_ROOT/fits/."""
    fit, _ = fit_model("y", ["x1", "x2"], _planted(), method="lm")
    monkeypatch.delenv(config.LCP_DATA_ROOT_ENV, raising=False)
    with pytest.raises(config.DataRootError):
        save_predictions(fit, "essay_lm_lm")
    monkeypatch.setenv(config.LCP_DATA_ROOT_ENV, str(tmp_path))
    path = save_predictions(fit, "essay_lm_lm")
    assert path == tmp_path.resolve() / "fits" / "essay_lm_lm__y.csv"
    table = pd.read_csv(path)
    assert list(table.columns) == ["ncdsid", "fold", "Y", "SL.predict", "SL.mean_All", "SL.lm_All"]
    assert len(table) == len(fit.Y) and set(table["fold"]) == set(range(1, 11))


# --------------------------------------------------------- data preparation --

def test_as_numeric_gives_level_positions_and_logical_codes():
    """R's as.numeric (llm_paper/_targets.R:L208, L258): a factor gives its level
    position, a logical 0/1, a number itself."""
    pedu = pd.Series(pd.Categorical([7.0, 1.0, np.nan, 3.0], categories=[1.0, 3.0, 7.0]))
    np.testing.assert_array_equal(r_as_numeric_column(pedu), [3.0, 1.0, np.nan, 2.0])
    male = pd.Series(pd.array([True, False, None], dtype="boolean"))
    np.testing.assert_array_equal(r_as_numeric_column(male), [1.0, 0.0, np.nan])
    edu = pd.Series(pd.Categorical(["Degree", "No Qualifications"],
                                   categories=["No Qualifications", "Lower Secondary",
                                               "Upper Secondary", "Degree"], ordered=True))
    np.testing.assert_array_equal(r_as_numeric_column(edu), [4.0, 1.0])


def test_data_prep_steps_from_the_spec():
    """The recorded steps (pipeline/model_spec.py DATA_PREP): sociological as.numeric
    (L258), RoBERTa inner join on ncdsid = id (L227), GPT-4 drop starts_with("embedding")
    then inner join (L230)."""
    sample = pd.DataFrame({
        "ncdsid": ["SYN000001", "SYN000002", "SYN000003"],
        "s0_co_male": pd.array([True, False, True], dtype="boolean"),
        "embedding_1": [0.1, 0.2, 0.3], "Embedding_2": [1.0, 2.0, 3.0], "nwords": [10, 20, 30],
    })
    lists = {"sociological_variables": ["s0_co_male"]}
    out = apply_data_prep(sample, DATA_PREP["sociological"], variable_lists=lists)
    assert out["s0_co_male"].tolist() == [1.0, 0.0, 1.0]

    gpt4 = pd.DataFrame({"id": ["SYN000003", "SYN000001"], "embedding_1": [9.0, 8.0]})
    out = apply_data_prep(sample, DATA_PREP["gpt4_embeddings"], tables={"gpt4_embeddings": gpt4})
    assert list(out.columns) == ["ncdsid", "s0_co_male", "nwords", "embedding_1"]
    assert out["ncdsid"].tolist() == ["SYN000001", "SYN000003"]  # x order, inner join
    assert out["embedding_1"].tolist() == [8.0, 9.0]

    roberta = pd.DataFrame({"id": ["SYN000002"], "roberta_dim_1": [0.5]})
    out = apply_data_prep(sample, DATA_PREP["roberta_embeddings"], tables={"roberta_embeddings": roberta})
    assert out["ncdsid"].tolist() == ["SYN000002"] and "id" not in out.columns


# ------------------------------------------------------ gene data and run log --

def test_gene_dependent_targets_are_skipped_and_logged(monkeypatch, tmp_path):
    """without gene data, targets whose predictors include
    gene_variables are skipped and listed in the run log (computed: 26 targets,
    166 fits)."""
    runnable, skipped = split_gene_dependent(model_specs(), gene_data_available=False)
    assert len(skipped) == 26 and sum(s.n_fits() for s in skipped) == 166
    assert all(any(p.value == "gene_variables" for p in s.predictors) for s in skipped)
    assert not any(any(p.value == "gene_variables" for p in s.predictors) for s in runnable)
    assert split_gene_dependent(model_specs(), gene_data_available=True)[1] == []

    monkeypatch.setenv(config.LCP_DATA_ROOT_ENV, str(tmp_path))
    path = write_run_log(skipped_targets_entry(skipped, "gene data absent"))
    log = json.loads(path.read_text())
    assert path.parent == tmp_path.resolve() / "logs"
    assert log["n_skipped"] == 26 and "gene_superlearner" in log["skipped_targets"]
