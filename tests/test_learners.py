"""Learner settings and the glmnet screener (brief Task 1.3, facts F5).

Each test names the R wrapper or package line it encodes. Numerical agreement with R
itself is the job of the oracle tests (Task 2.2); these tests pin the ported rules.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from llm_cong_predict.models import screeners
from llm_cong_predict.models.base_learners import (
    KsvmLike,
    LearnerFailure,
    NnetLike,
    RLinearModel,
    _make_ranger,
    _make_xgboost_hist,
    nnet_weight_count,
    sigest_sigma,
)
from llm_cong_predict.models.screeners import ScreenFailure, lambda_grid, screen_glmnet
from llm_cong_predict.models.seeds import FULL_FIT, derive_seed


# ------------------------------------------------------------------- SL.nnet --

def test_nnet_weight_count_is_2p_plus_5():
    """R pkg: nnet/R/nnet.R:L242–268 (7.3-21): add.net gives every hidden and output
    unit a bias, so size 2 with one output has 2(p+1) + 3 = 2p + 5 weights."""
    assert nnet_weight_count(497) == 999
    assert nnet_weight_count(498) == 1001


def test_nnet_fails_at_498_columns_and_fits_at_497():
    """R pkg: nnet/R/nnet.R:L105–106 stops with "too many weights" when the count
    exceeds MaxNWts = 1000 (L79), i.e. at p = 498 columns after screening."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=30)
    NnetLike(seed=1).fit(rng.normal(size=(30, 497)), y)  # 999 weights: fits
    with pytest.raises(LearnerFailure, match="too many \\(1001\\) weights"):
        NnetLike(seed=1).fit(rng.normal(size=(30, 498)), y)


def test_nnet_uses_maxit_500_and_no_decay():
    """R pkg: SuperLearner/R/SL.nnet.R:L8 (2.0-40) sets maxit = 500 (nnet's own default
    is 100, nnet.R:L78) and leaves decay = 0."""
    rng = np.random.default_rng(1)
    model = NnetLike(seed=1).fit(rng.normal(size=(20, 3)), rng.normal(size=20)).model_
    assert model.max_iter == 500 and model.alpha == 0.0
    assert model.hidden_layer_sizes == (2,) and model.activation == "logistic"


# ------------------------------------------------------------------- SL.ksvm --

def test_ksvm_skips_all_scaling_when_a_column_is_constant():
    """R pkg: kernlab/R/ksvm.R:L127–148 (0.9-33): if any column has zero variance,
    `scaled` becomes FALSE for every column and y is not scaled either; sigma is then
    estimated on the unscaled X (L153–156)."""
    rng = np.random.default_rng(2)
    X = np.column_stack([rng.normal(size=40), np.full(40, 3.0), rng.normal(size=40)])
    y = 5 + 2 * X[:, 0] + rng.normal(size=40)
    model = KsvmLike(seed=7).fit(X, y)
    assert model.scaled_ is False
    assert model.x_center_ is None and (model.y_center_, model.y_scale_) == (0.0, 1.0)
    sigma = sigest_sigma(X, np.random.default_rng(7))
    ref = SVR(kernel="rbf", gamma=sigma, C=1.0, epsilon=0.1, tol=1e-3, cache_size=40).fit(X, y)
    np.testing.assert_array_equal(model.predict(X[:5]), ref.predict(X[:5]))


def test_ksvm_scales_x_and_y_when_no_column_is_constant():
    """R pkg: kernlab/R/ksvm.R:L139–147: X and y are standardised with R's scale()
    (sd with denominator n-1) and predictions are converted back (L2810–2811)."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 2)) * [1.0, 10.0]
    y = 5 + 2 * X[:, 0] + rng.normal(size=40)
    model = KsvmLike(seed=7).fit(X, y)
    assert model.scaled_ is True
    xc, xs = X.mean(0), X.std(0, ddof=1)
    yc, ys = y.mean(), y.std(ddof=1)
    Xs = (X - xc) / xs
    sigma = sigest_sigma(Xs, np.random.default_rng(7))
    ref = SVR(kernel="rbf", gamma=sigma, C=1.0, epsilon=0.1, tol=1e-3, cache_size=40).fit(Xs, (y - yc) / ys)
    np.testing.assert_allclose(model.predict(X[:5]), ref.predict(Xs[:5]) * ys + yc)


def test_sigest_matches_kernlab_formula():
    """R pkg: kernlab/R/sigest.R:L58–64: floor(0.5 m) pairs with replacement, zero
    distances dropped, sigma = mean(1/q0.9, 1/q0.1) with type-7 quantiles."""
    rng_data = np.random.default_rng(4)
    X = rng_data.normal(size=(11, 3))
    rng = np.random.default_rng(99)
    idx1, idx2 = rng.integers(0, 11, 5), rng.integers(0, 11, 5)
    d = np.sum((X[idx1] - X[idx2]) ** 2, axis=1)
    d = d[d != 0]
    expected = np.mean(1 / np.quantile(d, [0.9, 0.1]))
    assert sigest_sigma(X, np.random.default_rng(99)) == pytest.approx(expected)


# --------------------------------------------------------------------- SL.lm --

def test_sl_lm_drops_exactly_collinear_column_like_r():
    """R pkg: SuperLearner/R/SL.lm.R:L44 uses lm(); lm.wfit (tol = 1e-7,
    r-source/src/library/stats/R/lm.R:L169) gives an aliased column an NA coefficient
    and predict.lm uses only the non-aliased columns (lm.R:L733–739). When new data
    break the collinearity, R's predictions differ from a minimum-norm solution."""
    rng = np.random.default_rng(5)
    x1, x2 = rng.normal(size=50), rng.normal(size=50)
    X = np.column_stack([x1, x2, x1 + x2, np.full(50, 2.0)])  # col 2 = x1 + x2, col 3 constant
    y = 1 + 3 * x1 - 2 * x2 + rng.normal(scale=0.1, size=50)
    model = RLinearModel().fit(X, y)
    np.testing.assert_array_equal(model.kept_columns_, [0, 1])

    X_new = rng.normal(size=(5, 4))  # breaks both collinearities
    ols = LinearRegression().fit(X[:, :2], y)
    np.testing.assert_allclose(model.predict(X_new), ols.predict(X_new[:, :2]))
    min_norm = LinearRegression().fit(X, y)
    assert not np.allclose(model.predict(X_new), min_norm.predict(X_new))


def test_sl_lm_with_no_columns_fits_intercept_only():
    """lm(Y ~ ., data = <no columns>) is the intercept-only model (R pkg:
    SuperLearner/R/SL.lm.R:L44); used when screening selects no column."""
    y = np.array([1.0, 2.0, 6.0])
    model = RLinearModel().fit(np.empty((3, 0)), y)
    np.testing.assert_allclose(model.predict(np.empty((2, 0))), [3.0, 3.0])


# --------------------------------------------------------- ranger, xgboost -----

def test_ranger_settings():
    """R pkg: SuperLearner/R/SL.ranger.R:L59–66 (2.0-40): num.trees 500,
    mtry floor(sqrt(p)), min.node.size 5 (no split at n <= 5,
    ranger/src/TreeRegression.cpp:L106, so min_samples_split = 6), replace = TRUE,
    sample.fraction = 1, num.threads = 1."""
    rf = _make_ranger(seed=3, n_features=10)
    assert rf.n_estimators == 500 and rf.max_features == 3
    assert rf.min_samples_split == 6 and rf.min_samples_leaf == 1
    assert rf.bootstrap is True and rf.max_samples is None and rf.n_jobs == 1
    assert _make_ranger(seed=3, n_features=1).max_features == 1


def test_xgboost_settings_and_base_score():
    """R pkg: SuperLearner/R/SL.xgboost.R:L43–46, L102, L106 (pre-3.0 branch) and
    llm_paper/R/functions.R:L492–494 (tree_method = "hist"); base_score 0.5 as in R
    xgboost 1.7.x (xgboost v1.7.6 include/xgboost/objective.h:L33)."""
    xgb = _make_xgboost_hist(seed=3, n_features=4)
    params = xgb.get_params()
    assert params["n_estimators"] == 1000 and params["max_depth"] == 4
    assert params["min_child_weight"] == 10 and params["learning_rate"] == 0.1
    assert params["tree_method"] == "hist" and params["n_jobs"] == 1
    assert params["base_score"] == 0.5 and params["objective"] == "reg:squarederror"


# ------------------------------------------------------------ screen.glmnet --

def test_screening_fewer_than_two_columns_raises_screen_failure():
    """R pkg: glmnet/R/glmnet.R:L393 (5.0) stops when x has fewer than 2 columns;
    SuperLearner then keeps all columns (SuperLearner.R:L121–124, tested in
    test_superlearner_mechanics.py)."""
    rng = np.random.default_rng(6)
    with pytest.raises(ScreenFailure, match="2 or more columns"):
        screen_glmnet(rng.normal(size=(30, 1)), rng.normal(size=30), seed=1)


def test_screening_constant_y_raises_screen_failure():
    """R pkg: glmnet/R/elnet.R:L23: 'y is constant; gaussian glmnet fails'."""
    with pytest.raises(ScreenFailure, match="y is constant"):
        screen_glmnet(np.random.default_rng(7).normal(size=(20, 3)), np.ones(20), seed=1)


def test_lambda_grid_matches_glmnet_definition():
    """lambda_max = max|Xs'(y - ybar)|/n with Xs standardised by the population SD
    (glmnetpp standardize.hpp:L73–76, base.hpp:L262–272, gaussian.hpp:L448); 100 values
    in total down to ratio * lambda_max, ratio 1e-4 if n >= p else 1e-2 (glmnet.R:L384)."""
    rng = np.random.default_rng(8)
    X, y = rng.normal(size=(40, 5)) * [1, 2, 3, 4, 5], rng.normal(size=40)
    Xs = (X - X.mean(0)) / X.std(0, ddof=0)
    lam_max = np.max(np.abs(Xs.T @ (y - y.mean()))) / 40
    grid = lambda_grid(X, y)
    assert len(grid) == 100
    assert grid[0] == pytest.approx(lam_max) and grid[-1] == pytest.approx(1e-4 * lam_max)
    wide = lambda_grid(rng.normal(size=(10, 20)), rng.normal(size=10))
    assert wide[-1] / wide[0] == pytest.approx(1e-2)


def test_glmnet_early_stop_rule():
    """glmnetpp elnet_path/base.hpp:L307–310 and gaussian_base.hpp:L132–134: after at
    least mnlam = 5 points, stop at the first point with relative R^2 gain < 1e-5 or
    R^2 > 0.999 (glmnet.control.R:L153–157); that point is kept."""
    rsq = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.500001, 0.6])
    assert screeners._early_stop_length(rsq) == 7
    assert screeners._early_stop_length(np.array([0, .2, .5, .9, .9995, .9999])) == 5
    assert screeners._early_stop_length(np.array([0.0, 0.0, 0.0, 0.0, 0.0])) == 5  # gain inf
    assert screeners._early_stop_length(np.linspace(0, 0.5, 20)) == 20


def test_screen_invariant_to_rescaling_a_column():
    """screen.glmnet standardises X (glmnet standardize = TRUE), so multiplying one
    column by 1000 leaves the screened set unchanged."""
    rng = np.random.default_rng(9)
    X = rng.normal(size=(80, 8))
    y = 2 * X[:, 0] - X[:, 3] + 0.5 * X[:, 5] + rng.normal(size=80)
    mask = screen_glmnet(X, y, seed=11)
    X2 = X.copy()
    X2[:, 3] *= 1000
    np.testing.assert_array_equal(screen_glmnet(X2, y, seed=11), mask)
    assert mask[[0, 3, 5]].all()


def test_screen_fallback_selects_at_least_two_when_path_allows():
    """R pkg: SuperLearner/R/screen.glmnet.R:L9–13: fewer than minscreen = 2 variables at
    lambda.min -> the first lambda along the path with >= 2 non-zero coefficients
    (which.max(sumCoef >= minscreen))."""
    rng = np.random.default_rng(10)
    X, y = rng.normal(size=(60, 6)), rng.normal(size=60)  # pure noise: lambda.min selects < 2
    grid = lambda_grid(X, y)
    beta, _, rsq = screeners._path(X, y, grid)
    beta = beta[:, : screeners._early_stop_length(rsq)]
    first = int(np.argmax((beta != 0).sum(axis=0) >= 2))
    mask = screen_glmnet(X, y, seed=3)
    assert mask.sum() >= 2
    np.testing.assert_array_equal(mask, beta[:, first] != 0)


def test_screen_fallback_with_no_eligible_lambda_selects_first_lambda():
    """When no lambda reaches 2 non-zero coefficients, which.max(all FALSE) is 1: the
    largest lambda, whose coefficients are all zero, so no column is selected
    (R pkg: SuperLearner/R/screen.glmnet.R:L12–13). A constant column is excluded by
    glmnet, so with two columns at most one can enter."""
    rng = np.random.default_rng(12)
    X = np.column_stack([rng.normal(size=40), np.full(40, 1.0)])
    y = X[:, 0] + rng.normal(size=40)
    assert not screen_glmnet(X, y, seed=1).any()


# --------------------------------------------------------------------- seeds --

def test_derive_seed_is_deterministic_and_distinct():
    """Every random component gets its own seed from (base, outer, inner, name)."""
    a = derive_seed(1, 0, 0, "SL.ranger_screen.glmnet")
    assert a == derive_seed(1, 0, 0, "SL.ranger_screen.glmnet")
    others = {derive_seed(1, 1, 0, "SL.ranger_screen.glmnet"),
              derive_seed(1, 0, 1, "SL.ranger_screen.glmnet"),
              derive_seed(1, 0, FULL_FIT, "SL.ranger_screen.glmnet"),
              derive_seed(1, 0, 0, "SL.nnet_screen.glmnet"),
              derive_seed(2, 0, 0, "SL.ranger_screen.glmnet")}
    assert a not in others and len(others) == 5
