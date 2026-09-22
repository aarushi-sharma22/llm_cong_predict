"""Golden-value tests for the CV metric port.

The expected numbers are computed by hand from the fold-wise definitions in the
original ``R/functions.R`` (see comments), so a passing test means the Python
port reproduces the original's arithmetic, not merely that it runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_cong_predict.metrics.cv_metrics import (
    CVSuperLearnerFit,
    cv_lm_r2,
    cv_mad,
    cv_mse,
    cv_predictive_r2,
    cv_rmse,
    lm_metrics,
    superlearner_metrics,
    _winsorise_sl_predict,
)


@pytest.fixture
def toy_fit() -> CVSuperLearnerFit:
    """A deterministic 4-observation, 2-fold fit with hand-checkable numbers.

    Y            = [1, 2, 3, 4]
    SL.predict   = [1.1, 1.9, 3.2, 3.8]
    SL.mean_All  = [2.5, 2.5, 2.5, 2.5]   (mean of Y)
    SL.lm_All    = [1.0, 2.0, 3.0, 4.0]   (perfect fit)
    folds        = fold0 -> [0,1], fold1 -> [2,3]
    """
    Y = np.array([1.0, 2.0, 3.0, 4.0])
    sl_predict = np.array([1.1, 1.9, 3.2, 3.8])
    library_predict = np.column_stack(
        [
            np.full(4, 2.5),          # SL.mean_All
            np.array([1.0, 2.0, 3.0, 4.0]),  # SL.lm_All
        ]
    )
    return CVSuperLearnerFit(
        Y=Y,
        sl_predict=sl_predict,
        library_predict=library_predict,
        library_names=["SL.mean_All", "SL.lm_All"],
        folds=[np.array([0, 1]), np.array([2, 3])],
        outcome_var="toy_outcome",
    )


def test_cv_mse(toy_fit):
    # fold0 SL MSE = mean(.1^2, .1^2) = .01 ; fold1 = mean(.2^2, .2^2) = .04
    out = cv_mse(toy_fit)
    assert out["mean_mse"] == pytest.approx(0.025)
    assert out["min_mse"] == pytest.approx(0.01)
    assert out["max_mse"] == pytest.approx(0.04)


def test_cv_rmse(toy_fit):
    # sqrt(.01)=.1 ; sqrt(.04)=.2
    out = cv_rmse(toy_fit)
    assert out["mean_rmse"] == pytest.approx(0.15)
    assert out["min_rmse"] == pytest.approx(0.1)
    assert out["max_rmse"] == pytest.approx(0.2)


def test_cv_mad(toy_fit):
    # fold0 = mean(|.1|,|.1|)=.1 ; fold1 = mean(|.2|,|.2|)=.2
    out = cv_mad(toy_fit)
    assert out["mean_mad"] == pytest.approx(0.15)
    assert out["min_mad"] == pytest.approx(0.1)
    assert out["max_mad"] == pytest.approx(0.2)


def test_cv_predictive_r2(toy_fit):
    # mean-model MSE per fold: fold0 mean(1.5^2,.5^2)=1.25 ; fold1 mean(.5^2,1.5^2)=1.25
    # r2_0 = 1 - .01/1.25 = 0.992 ; r2_1 = 1 - .04/1.25 = 0.968
    out = cv_predictive_r2(toy_fit)
    assert out["mean_r2"] == pytest.approx(0.980)
    assert out["min_r2"] == pytest.approx(0.968)
    assert out["max_r2"] == pytest.approx(0.992)


def test_cv_lm_r2_perfect(toy_fit):
    # SL.lm_All is a perfect fit -> risk_lm = 0 -> r2 = 1 in every fold.
    out = cv_lm_r2(toy_fit)
    assert out["mean_r2"] == pytest.approx(1.0)
    assert out["min_r2"] == pytest.approx(1.0)
    assert out["max_r2"] == pytest.approx(1.0)


def test_superlearner_metrics_row_shape(toy_fit):
    row = superlearner_metrics(toy_fit)
    # No SL prediction here is extreme, so winsorisation is a no-op and the MSE
    # matches test_cv_mse.
    assert row["var"] == "toy_outcome"
    assert row["n"] == 4
    assert row["mean_mse"] == pytest.approx(0.025)
    assert set(row) == {
        "mean_mse", "se_mse", "min_mse", "max_mse",
        "mean_r2", "min_r2", "max_r2",
        "mean_mad", "min_mad", "max_mad",
        "mean_rmse", "min_rmse", "max_rmse",
        "var", "n",
    }


def test_winsorise_only_clamps_beyond_10_sd():
    """Documents a real property of the original formula: because the threshold
    ``10*sd(abs(sl)) + mean(sl)`` is inflated by the outlier itself, a single
    large value is NOT clamped unless it exceeds 10 SDs above the mean. Here the
    SD is large enough that 1000 stays put -- matching the R behaviour exactly.
    """
    sl = np.array([1.0, 1.0, 1.0, 1.0, 1000.0])
    out = _winsorise_sl_predict(sl)
    assert np.allclose(out, sl)  # nothing clamped, by design of the original formula


def test_winsorise_clamps_when_threshold_exceeded():
    """A tight cluster with one value just past 10 SDs IS clamped to the mean."""
    base = np.zeros(100)
    base[:50] = -1.0
    base[50:] = 1.0            # sd(abs)=0 -> threshold == mean; make one point exceed it
    base[0] = 50.0            # a clear outlier relative to a small-SD bulk
    out = _winsorise_sl_predict(base)
    m = base.mean()
    threshold = 10.0 * np.std(np.abs(base), ddof=1) + m
    expected = base.copy()
    expected[np.abs(base) > threshold] = m
    assert np.allclose(out, expected)
    # sanity: at least confirm the clamp path is exercised somewhere
    assert (out != base).any() or (np.abs(base) <= threshold).all()


def test_winsorise_noop_when_no_outliers():
    sl = np.array([1.0, 2.0, 3.0, 4.0])
    out = _winsorise_sl_predict(sl)
    assert np.allclose(out, sl)


# ------------------------------------------- metric rows: clamp order, SE -----

def _clamp_fixture() -> CVSuperLearnerFit:
    """100 observations, 2 folds of 50, one SL prediction far enough out to be clamped.

    SL.predict: 50.0 at index 0, -1.0 at 1..49, 1.0 at 50..99. Threshold
    10*sd(|p|) + mean(p) = 10*4.9 + 0.51 ≈ 49.5 < 50, so index 0 becomes mean(p) = 0.51.
    """
    Y = np.zeros(100)
    sl = np.concatenate([[50.0], np.full(49, -1.0), np.full(50, 1.0)])
    lib = np.column_stack([np.zeros(100), 0.5 * sl])  # SL.mean_All, SL.lm_All
    return CVSuperLearnerFit(
        Y=Y, sl_predict=sl, library_predict=lib, library_names=["SL.mean_All", "SL.lm_All"],
        folds=[np.arange(50), np.arange(50, 100)], outcome_var="y",
    )


def test_superlearner_mse_unclamped_r2_mad_rmse_clamped():
    """R: llm_paper/R/functions.R:L704–707 — summary(cv_fit) (MSE, se) is taken BEFORE
    the outlier clamp on L707; R², MAD and RMSE (L709–713) use the clamped predictions."""
    fit = _clamp_fixture()
    clamped = _winsorise_sl_predict(fit.sl_predict)
    assert clamped[0] == pytest.approx(0.51) and fit.sl_predict[0] == 50.0  # clamp fired

    row = superlearner_metrics(fit)
    fold0, fold1 = fit.folds
    raw_mse = [np.mean((fit.Y[f] - fit.sl_predict[f]) ** 2) for f in (fold0, fold1)]
    cl_mse = [np.mean((fit.Y[f] - clamped[f]) ** 2) for f in (fold0, fold1)]
    assert row["mean_mse"] == pytest.approx(np.mean(raw_mse))  # 25.49 vs 0.9852 clamped
    assert row["max_mse"] == pytest.approx(max(raw_mse))
    assert row["mean_mse"] != pytest.approx(np.mean(cl_mse))
    assert row["mean_rmse"] == pytest.approx(np.mean(np.sqrt(cl_mse)))
    cl_mad = [np.mean(np.abs(fit.Y[f] - clamped[f])) for f in (fold0, fold1)]
    assert row["mean_mad"] == pytest.approx(np.mean(cl_mad))
    # R² here: mean model predicts Y exactly (risk 0) -> use a Y with variance instead
    fit2 = CVSuperLearnerFit(
        Y=np.linspace(-1, 1, 100), sl_predict=fit.sl_predict, library_predict=np.column_stack(
            [np.zeros(100), np.zeros(100)]), library_names=["SL.mean_All", "SL.lm_All"],
        folds=fit.folds,
    )
    row2 = superlearner_metrics(fit2)
    r2 = [1 - np.mean((fit2.Y[f] - clamped[f]) ** 2) / np.mean(fit2.Y[f] ** 2) for f in (fold0, fold1)]
    assert row2["mean_r2"] == pytest.approx(np.mean(r2))


def test_lm_metrics_mse_is_sl_lm_all_fold_mean():
    """R: llm_paper/R/functions.R:L731–732 — get_cv_lm_metrics takes mean/min/max MSE
    from the SL.lm_All row of summary(cv_fit): the linear model alone, unclamped, not
    the ensemble."""
    fit = _clamp_fixture()
    row = lm_metrics(fit)
    lm_col = fit.library_predict[:, 1]
    lm_mse = [np.mean((fit.Y[f] - lm_col[f]) ** 2) for f in fit.folds]
    assert row["mean_mse"] == pytest.approx(np.mean(lm_mse))
    assert row["min_mse"] == pytest.approx(min(lm_mse))
    assert row["max_mse"] == pytest.approx(max(lm_mse))
    ens_mse = [np.mean((fit.Y[f] - fit.sl_predict[f]) ** 2) for f in fit.folds]
    assert row["mean_mse"] != pytest.approx(np.mean(ens_mse))
    assert "n" in row  # R names this column `length(cv_fit$Y)` (L748); the port keeps n


def test_se_mse_hand_computation(toy_fit):
    """R pkg: SuperLearner/R/summary.CV.SuperLearner.R:L43 (2.0-40):
    se = (1/sqrt(n)) * sd(w * (Y - pred)^2), sd with denominator n-1.

    toy_fit: squared errors of SL.predict are [.01, .01, .04, .04]; their sd is
    sqrt(4 * .015^2 / 3) = 0.0173205..., so se = 0.0173205 / sqrt(4) = 0.00866025.
    For SL.lm_All (a perfect fit) every squared error is 0, so se = 0.
    """
    assert superlearner_metrics(toy_fit)["se_mse"] == pytest.approx(0.008660254037844387)
    assert lm_metrics(toy_fit)["se_mse"] == pytest.approx(0.0)
