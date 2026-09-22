"""Super Learner mechanics (brief Task 1.3, fact F4).

Failure handling, shared screening, the weight recomputation after a failed refit,
determinism and n_jobs invariance. Cheap learners are used wherever the rule does not
depend on the learner, so these tests are fast; the full-library parity test is slow.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from llm_cong_predict.metrics.cv_metrics import superlearner_metrics
from llm_cong_predict.models import native_superlearner as nsl
from llm_cong_predict.models.base_learners import (
    LearnerSpec,
    RLinearModel,
    _make_lm,
    _make_mean,
    _make_ranger,
    lm_library,
    superlearner_library,
)
from llm_cong_predict.models.meta import compute_coef, compute_pred


# ------------------------------------------------------------------ helpers --

class _Fails(RegressorMixin, BaseEstimator):
    """Raises in fit when the training set has more than ``max_n`` rows."""

    def __init__(self, max_n: int = -1, zeros: bool = False):
        self.max_n, self.zeros = max_n, zeros

    def fit(self, X, y):
        if len(y) > self.max_n:
            raise RuntimeError("synthetic learner failure")
        self.lm_ = RLinearModel().fit(X, y)
        return self

    def predict(self, X):
        return np.zeros(len(X)) if self.zeros else self.lm_.predict(X)


def _spec(name, max_n, zeros=False, screener=None):
    return LearnerSpec(name, lambda seed, p: _Fails(max_n=max_n, zeros=zeros), screener)


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 4))
    y = X @ np.array([2.0, -1.0, 0.5, 0.0]) + rng.normal(scale=0.5, size=100)
    return X, y


def _fit(X, y, library, **kw):
    kw.setdefault("outer_v", 5)
    kw.setdefault("inner_v", 4)
    return nsl.fit_cv_superlearner(X, y, library, outcome_var="y", seed=1, **kw)


# ----------------------------------------------------------- failure rules --

def test_failing_learner_gets_weight_zero_and_ensemble_has_no_nan(data):
    """R pkg: SuperLearner/R/SuperLearner.R:L143–152, L180–183 (2.0-40): a learner that
    errors leaves NA in its column of Z, the whole column is set to 0, so NNLS gives it
    weight 0; computePred uses only non-zero weights (method.R:L66–75), so its NA
    predictions never reach SL.predict; its cvRisk is NA (L303–305)."""
    X, y = data
    library = [LearnerSpec("SL.mean", _make_mean, None), LearnerSpec("SL.lm", _make_lm, None),
               _spec("SL.broken", max_n=-1)]
    fit = _fit(X, y, library)
    assert (fit.coef[:, 2] == 0).all()
    assert np.isnan(fit.library_predict[:, 2]).all()
    assert np.isnan(fit.cv_risk[:, 2]).all() and not np.isnan(fit.cv_risk[:, :2]).any()
    assert np.isfinite(fit.sl_predict).all()
    assert {f["learner"] for f in fit.failures} == {"SL.broken_All"}
    assert np.isfinite(superlearner_metrics(fit)["mean_r2"])


def test_refit_failure_with_positive_weight_recomputes_weights(data):
    """R pkg: SuperLearner/R/SuperLearner.R:L274–289: a learner that succeeds in the inner
    CV but errors in the refit has positive weight, so its Z column is set to 0 and the
    weights are recomputed. Its cvRisk is NOT set to NA (only CV failures are, L303–305);
    it is recomputed from the zeroed column, i.e. mean(Y^2) over the training rows."""
    X, y = data
    y = y + 10.0  # positive mean, so the remaining mean learner keeps a positive weight
    base = [LearnerSpec("SL.mean", _make_mean, None)]
    # outer train = 80 rows, inner train = 60 rows: max_n = 70 fails only in the refit
    refit_only = _fit(X, y, base + [_spec("SL.good", max_n=70)])
    always = _fit(X, y, base + [_spec("SL.good", max_n=-1)])
    succeeds = _fit(X, y, base + [_spec("SL.good", max_n=10**9)])

    assert (succeeds.coef[:, 1] > 0).all()  # it would have had positive weight
    np.testing.assert_allclose(refit_only.coef, always.coef)  # recomputed without it
    assert (refit_only.coef[:, 1] == 0).all()
    assert np.isnan(refit_only.library_predict[:, 1]).all()
    assert np.isfinite(refit_only.sl_predict).all()
    assert np.isnan(always.cv_risk[:, 1]).all()
    for k, test in enumerate(refit_only.folds):
        train = np.setdiff1d(np.arange(len(y)), test)
        assert refit_only.cv_risk[k, 1] == pytest.approx(np.mean(y[train] ** 2))


def test_refit_failure_with_zero_weight_changes_nothing(data):
    """R pkg: SuperLearner/R/SuperLearner.R:L289–291: if the failed learners' weight is
    already 0, the weights are not recomputed ("Coefficients already 0")."""
    X, y = data
    base = [LearnerSpec("SL.mean", _make_mean, None), LearnerSpec("SL.lm", _make_lm, None)]
    # predicts exactly 0 in CV (zero column -> weight 0), then fails in the refit
    failing = _fit(X, y, base + [_spec("SL.zero", max_n=70, zeros=True)])
    working = _fit(X, y, base + [_spec("SL.zero", max_n=10**9, zeros=True)])
    assert (working.coef[:, 2] == 0).all()
    np.testing.assert_array_equal(failing.coef, working.coef)
    np.testing.assert_array_equal(failing.cv_risk, working.cv_risk)
    np.testing.assert_array_equal(failing.sl_predict, working.sl_predict)


def test_all_learners_failing_stops(data):
    """R pkg: SuperLearner/R/SuperLearner.R:L184–186: stop("All algorithms dropped")."""
    X, y = data
    with pytest.raises(nsl.AllLearnersFailed):
        _fit(X, y, [_spec("SL.a", max_n=-1), _spec("SL.b", max_n=-1)])


def test_compute_pred_ignores_nan_of_zero_weight_learner():
    """R pkg: SuperLearner/R/method.R:L72: crossprod over coef != 0 only."""
    pred = np.array([[1.0, np.nan], [3.0, np.nan]])
    np.testing.assert_array_equal(compute_pred(pred, np.array([1.0, 0.0])), [1.0, 3.0])


def test_compute_coef_normalises_nnls():
    """R pkg: SuperLearner/R/method.R:L44–61: nnls on sqrt(w) Z, NA -> 0, divided by the sum."""
    Y = np.array([1.0, 2.0, 3.0, 4.0])
    Z = np.column_stack([2 * Y, np.full(4, 2.5)])
    coef, risk = compute_coef(Z, Y)
    assert coef.sum() == pytest.approx(1.0) and (coef >= 0).all()
    assert risk[1] == pytest.approx(np.mean((2.5 - Y) ** 2))


# ------------------------------------------------------------- screening -----

def test_screening_runs_once_per_training_set_and_mask_is_shared(data, monkeypatch):
    """R pkg: SuperLearner/R/SuperLearner.R:L119–131, L143 and L222–223: each screener
    runs once per training set (every inner split plus the refit) and its mask is used
    by every learner paired with it."""
    X, y = data
    calls = []
    real = nsl.SCREENERS["screen.glmnet"]

    def counting(Xs, ys, seed):
        calls.append(len(ys))
        return real(Xs, ys, seed=seed)

    monkeypatch.setitem(nsl.SCREENERS, "screen.glmnet", counting)
    library = [LearnerSpec("SL.lm", _make_lm, "screen.glmnet"),
               LearnerSpec("SL.ranger", _make_ranger, "screen.glmnet")]
    _fit(X, y, library, outer_v=5, inner_v=4)
    # 5 outer folds x (4 inner splits + 1 refit) = 25 calls, not 50
    assert len(calls) == 25


def test_screening_failure_keeps_all_columns(data):
    """R pkg: SuperLearner/R/SuperLearner.R:L121–124: a screener that errors is replaced
    by All; glmnet errors on a single column (glmnet.R:L393)."""
    X, y = data
    library = [LearnerSpec("SL.mean", _make_mean, None),
               LearnerSpec("SL.lm", _make_lm, "screen.glmnet")]
    fit = _fit(X[:, :1], y, library)
    assert np.isfinite(fit.library_predict).all()
    stages = {(f["stage"], f["learner"]) for f in fit.failures}
    assert stages == {("screen", "screen.glmnet")}
    masks = nsl._screen_masks(library, X[:, :1], y, 1, 0, 0, [])
    assert masks["screen.glmnet"].tolist() == [True]


# --------------------------------------------------------- determinism --------

def test_repeated_runs_are_identical(data):
    X, y = data
    library = [LearnerSpec("SL.mean", _make_mean, None),
               LearnerSpec("SL.ranger", _make_ranger, "screen.glmnet")]
    a, b = _fit(X, y, library), _fit(X, y, library)
    np.testing.assert_array_equal(a.sl_predict, b.sl_predict)
    np.testing.assert_array_equal(a.coef, b.coef)


def test_n_jobs_does_not_change_output_small_library(data):
    """Seeds depend only on (base seed, outer fold, inner fold, name), so n_jobs=1 and
    n_jobs=2 give bit-identical output."""
    X, y = data
    library = lm_library() + [LearnerSpec("SL.ranger", _make_ranger, "screen.glmnet")]
    one = _fit(X, y, library, n_jobs=1)
    two = _fit(X, y, library, n_jobs=2)
    for attr in ("sl_predict", "library_predict", "discrete_sl_predict", "coef", "cv_risk"):
        np.testing.assert_array_equal(getattr(one, attr), getattr(two, attr))


@pytest.mark.slow
def test_n_jobs_does_not_change_output_full_library():
    """As above with the full 6-learner library (R: llm_paper/R/functions.R:L514–519)."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 4))
    y = X[:, 0] - X[:, 1] + rng.normal(scale=0.5, size=60)
    kw = dict(outer_v=3, inner_v=3, seed=1, outcome_var="y")
    one = nsl.fit_cv_superlearner(X, y, superlearner_library(), n_jobs=1, **kw)
    two = nsl.fit_cv_superlearner(X, y, superlearner_library(), n_jobs=2, **kw)
    assert one.failures == [] and two.failures == []
    for attr in ("sl_predict", "library_predict", "discrete_sl_predict", "coef", "cv_risk"):
        np.testing.assert_array_equal(getattr(one, attr), getattr(two, attr))


def test_fixed_folds_are_used(data):
    """Outer and inner folds can be fixed, for the R oracle comparison (Task 2.2)."""
    X, y = data
    folds = [np.arange(i, 100, 4) for i in range(4)]
    fit = _fit(X, y, lm_library(), folds=folds)
    assert [f.tolist() for f in fit.folds] == [f.tolist() for f in folds]


def test_missing_values_are_refused(data):
    """R pkg: SuperLearner/R/SuperLearner.R:L70–72 stops on missing data."""
    X, y = data
    X = X.copy()
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing data"):
        _fit(X, y, lm_library())
