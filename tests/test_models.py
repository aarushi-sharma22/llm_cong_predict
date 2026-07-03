"""Tests for the native model layer.

These verify what CAN be verified without R: fold correctness, the NNLS
meta-learner's known behaviour, nested-CV bookkeeping (no leakage, right shapes,
right library names), and that end-to-end output carries a real signal and flows
into the already-verified metric layer. They do NOT and cannot verify numerical
agreement with R's SuperLearner -- that is the rpy2 oracle's job (V4).
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_cong_predict.metrics.cv_metrics import superlearner_metrics
from llm_cong_predict.models.base_learners import (
    LearnerSpec,
    _make_lm,
    _make_mean,
    _make_ranger,
    lm_library,
    superlearner_library,
)
from llm_cong_predict.models.folds import make_folds, train_indices
from llm_cong_predict.models.meta import nnls_meta
from llm_cong_predict.models.native_superlearner import fit_cv_superlearner


# ------------------------------------------------------------------ folds ----

def test_folds_partition_exactly():
    n, v = 103, 10
    folds = make_folds(n, v, seed=1)
    assert len(folds) == v
    allidx = np.concatenate(folds)
    # every index exactly once (a true partition)
    assert np.array_equal(np.sort(allidx), np.arange(n))
    assert len(allidx) == len(set(allidx.tolist())) == n


def test_folds_deterministic():
    a = make_folds(50, 5, seed=1)
    b = make_folds(50, 5, seed=1)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_train_indices_complement():
    n = 20
    folds = make_folds(n, 4, seed=0)
    for test in folds:
        tr = train_indices(n, test)
        assert set(tr.tolist()).isdisjoint(test.tolist())
        assert len(tr) + len(test) == n


def test_folds_reject_bad_args():
    with pytest.raises(ValueError):
        make_folds(3, 5, seed=1)   # more folds than obs
    with pytest.raises(ValueError):
        make_folds(10, 1, seed=1)  # too few folds


# -------------------------------------------------------------- meta (NNLS) --

def test_nnls_gives_all_weight_to_perfect_learner():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=200)
    noise = rng.normal(size=200)
    # column 0 is exactly Y (perfect); column 1 is pure noise.
    Z = np.column_stack([Y, noise])
    fit = nnls_meta(Z, Y)
    assert fit.coef[0] == pytest.approx(1.0, abs=1e-6)
    assert fit.coef[1] == pytest.approx(0.0, abs=1e-6)
    assert fit.best == 0                      # lowest CV risk is the perfect learner


def test_nnls_weights_are_a_convex_combination():
    rng = np.random.default_rng(1)
    Y = rng.normal(size=100)
    Z = np.column_stack([Y + rng.normal(scale=0.5, size=100),
                         Y + rng.normal(scale=0.5, size=100)])
    fit = nnls_meta(Z, Y)
    assert (fit.coef >= -1e-12).all()         # non-negative
    assert fit.coef.sum() == pytest.approx(1.0, abs=1e-6)  # normalised to sum 1


def test_nnls_cv_risk_matches_manual():
    Y = np.array([1.0, 2.0, 3.0, 4.0])
    Z = np.column_stack([Y, np.full(4, Y.mean())])  # perfect vs mean
    fit = nnls_meta(Z, Y)
    assert fit.cv_risk[0] == pytest.approx(0.0)
    assert fit.cv_risk[1] == pytest.approx(np.mean((Y - Y.mean()) ** 2))


# ------------------------------------------------- nested CV bookkeeping -----

def _small_library() -> list[LearnerSpec]:
    # A fast, deterministic 3-learner library for structural tests.
    return [
        LearnerSpec("SL.mean", _make_mean, None),
        LearnerSpec("SL.lm", _make_lm, None),
        LearnerSpec("SL.ranger", _make_ranger, "screen.glmnet"),
    ]


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(42)
    n, p = 160, 6
    X = rng.normal(size=(n, p))
    beta = np.array([2.0, -1.5, 1.0, 0.0, 0.0, 0.0])  # only first 3 features matter
    y = X @ beta + rng.normal(scale=0.5, size=n)
    return X, y


def test_fit_produces_valid_contract(linear_data):
    X, y = linear_data
    fit = fit_cv_superlearner(X, y, _small_library(), outcome_var="y", seed=1)
    n = len(y)
    # shapes
    assert fit.sl_predict.shape == (n,)
    assert fit.library_predict.shape == (n, 3)
    assert fit.coef.shape == (len(fit.folds), 3)
    # every observation got an out-of-fold prediction (no NaNs left)
    assert not np.isnan(fit.sl_predict).any()
    assert not np.isnan(fit.library_predict).any()
    # library names in the expected order
    assert fit.library_names == ["SL.mean_All", "SL.lm_All", "SL.ranger_screen.glmnet"]
    # per-fold weights are convex
    for row in fit.coef:
        assert (row >= -1e-9).all()
        assert row.sum() == pytest.approx(1.0, abs=1e-6)


def test_lm_library_names_use_All_suffix(linear_data):
    # get_cv_lm_metrics filters on "SL.lm_All"; the lm library must produce that.
    X, y = linear_data
    fit = fit_cv_superlearner(X, y, lm_library(), outcome_var="y", seed=1)
    assert fit.library_names == ["SL.mean_All", "SL.lm_All"]


def test_end_to_end_signal_flows_into_metrics(linear_data):
    # A real behavioural check: with a strong linear signal, the ensemble should
    # explain a large share of variance, and metrics should be finite and sane.
    X, y = linear_data
    fit = fit_cv_superlearner(X, y, _small_library(), outcome_var="y", seed=1)
    row = superlearner_metrics(fit)
    assert row["var"] == "y"
    assert row["n"] == len(y)
    assert np.isfinite(row["mean_r2"])
    assert row["mean_r2"] > 0.7          # strong signal -> high R^2 (not just "runs")
    assert row["mean_rmse"] > 0
    assert row["mean_mse"] == pytest.approx(row["mean_rmse"] ** 2, rel=0.5)  # loose sanity


@pytest.mark.slow
def test_full_six_learner_library_instantiates(linear_data):
    # Confirms the full library (incl. nnet/ksvm/xgboost) runs end-to-end and
    # yields a valid contract. Slower; not a numerical check.
    X, y = linear_data
    fit = fit_cv_superlearner(X, y, superlearner_library(), outcome_var="y", seed=1)
    assert fit.library_names == [
        "SL.mean_All", "SL.ranger_screen.glmnet", "SL.nnet_screen.glmnet",
        "SL.xgboost.hist_screen.glmnet", "SL.ksvm_screen.glmnet", "SL.lm_screen.glmnet",
    ]
    assert not np.isnan(fit.sl_predict).any()
    row = superlearner_metrics(fit)
    assert np.isfinite(row["mean_r2"])
