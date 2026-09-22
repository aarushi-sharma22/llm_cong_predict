"""Oracle tests: the native port against R itself, through rpy2 (brief Task 2.2).

Every test skips, with the reason, when rpy2, R or a required R package is missing
(they need: R, `pip install -e '.[oracle]'`, and the R packages named in each
test; versions in docs/REFERENCE_SOURCES.md). Data are synthetic. The measured gaps
are recorded in docs/PORTING_NOTES.md (C9, F2) and docs/VALIDATION_CHECKLIST.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from llm_cong_predict.rbridge import r_unavailable_reason


def _needs(*packages):
    reason = r_unavailable_reason(tuple(packages))
    return pytest.mark.skipif(reason is not None, reason=reason or "")


def _factor_data(seed: int, n: int = 300) -> pd.DataFrame:
    from llm_cong_predict.cleaning.factors import FACTOR_DEFINITIONS

    rng = np.random.default_rng(seed)
    g = rng.normal(size=n)
    cols = {}
    for spec in FACTOR_DEFINITIONS.values():
        for v in spec["vars"]:
            if spec["cor"] == "cor":
                cols[v] = 50 + 10 * (0.8 * g + rng.normal(scale=0.6, size=n))
            else:  # ordinal items for the polychoric factors
                cols[v] = np.clip(np.round(2 + 0.7 * g + rng.normal(size=n)), 1, 4)
    df = pd.DataFrame(cols)
    df.insert(0, "ncdsid", [f"SYN{i:06d}" for i in range(1, n + 1)])
    miss = rng.choice(n, 15, replace=False)
    df.loc[miss[:8], "s2_co_verbal_ability"] = np.nan
    df.loc[miss[8:], "s2_co_reading_ability"] = np.nan
    return df


@_needs("psych")
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_pearson_factor_matches_psych_fa(seed):
    """R: llm_paper/R/functions.R:L297–306 via psych::fa(x, 1, cor = "cor")$scores
    (psych 2.6.5). The native scores must have the identical NA pattern (no imputation,
    factor.scores.R:L135) and correlate with psych's at >= 0.9999; measured at Task 2.2:
    correlation >= 0.99999999997, max |difference| <= 2.2e-5 (loadings APPROX, AP8)."""
    from llm_cong_predict.cleaning.factors import create_factors, create_factors_r

    df = _factor_data(seed)
    native = create_factors(df, backend="native").scores["s2_co_factor_ability"].to_numpy()
    r = create_factors_r(df)["s2_co_factor_ability"].to_numpy()
    np.testing.assert_array_equal(np.isnan(native), np.isnan(r))
    ok = ~np.isnan(native)
    assert np.corrcoef(native[ok], r[ok])[0, 1] >= 0.9999
    assert np.nanmax(np.abs(native - r)) < 1e-3


@_needs("psych")
def test_r_bridge_computes_all_four_factors():
    """factor_backend "r": all four factors from psych::fa, including the three
    polychoric ones the native path cannot compute (brief Task 2.2)."""
    from llm_cong_predict.cleaning.factors import FACTOR_DEFINITIONS, create_factors

    res = create_factors(_factor_data(0), backend="r")
    assert res.computed == list(FACTOR_DEFINITIONS) and res.deferred == []
    for name, spec in FACTOR_DEFINITIONS.items():
        scores = res.scores[name]
        if spec["cor"] == "poly":
            assert scores.notna().all()
        assert scores.std() > 0


_R_SCREEN = """
function(x, y, foldid, lambda) {
  y <- as.numeric(y); foldid <- as.integer(foldid); lambda <- as.numeric(lambda)
  fitCV <- glmnet::cv.glmnet(x = x, y = y, lambda = lambda, type.measure = 'deviance',
                             nfolds = max(foldid), foldid = foldid, family = 'gaussian', alpha = 1)
  w <- (as.numeric(coef(fitCV$glmnet.fit, s = fitCV$lambda.min))[-1] != 0)
  if (sum(w) < 2) {
    sumCoef <- apply(as.matrix(fitCV$glmnet.fit$beta), 2, function(b) sum(b != 0))
    w <- (as.matrix(fitCV$glmnet.fit$beta)[, which.max(sumCoef >= 2)] != 0)
  }
  w
}"""


@_needs("glmnet")
def test_screen_glmnet_matches_r_on_shared_folds_and_grid():
    """R pkg: SuperLearner/R/screen.glmnet.R:L1–16 replicated in R with an explicit foldid
    and lambda sequence (cv.glmnet, glmnet 5.0); the Python screener gets the same folds
    and grid and must select the same columns (measured at Task 2.2: 12/12)."""
    import rpy2.robjects as ro

    from llm_cong_predict.models import screeners
    from llm_cong_predict.rbridge import converter

    with converter():
        r_screen = ro.r(_R_SCREEN)
        r_path = ro.r("function(x, y) glmnet::glmnet(x, as.numeric(y), alpha = 1, nlambda = 100)$lambda")
    for seed in range(12):
        rng = np.random.default_rng(seed)
        n, p = [(80, 6), (60, 20), (30, 50), (100, 3)][seed % 4]
        X = rng.normal(size=(n, p)) * rng.uniform(0.5, 3, size=p)
        beta = np.zeros(p)
        beta[:3] = [1.0, -0.6, 0.3] if seed % 3 else [0.0, 0.0, 0.0]
        y = X @ beta + rng.normal(size=n)
        foldid = rng.permutation(np.resize(np.arange(10), n))
        with converter():
            lam = np.asarray(r_path(X, y), dtype=float)
            r_mask = np.asarray(r_screen(X, y, foldid + 1.0, lam), dtype=bool)
        py_mask = screeners.screen_glmnet(X, y, seed=0, foldid=foldid, lambdas=lam)
        np.testing.assert_array_equal(py_mask, r_mask, err_msg=f"seed {seed}")


_R_SCREEN_DEFAULT = """
function(x, y, foldid) {
  y <- as.numeric(y); foldid <- as.integer(foldid)
  fitCV <- glmnet::cv.glmnet(x = x, y = y, lambda = NULL, type.measure = 'deviance', nfolds = max(foldid),
                             foldid = foldid, family = 'gaussian', alpha = 1, nlambda = 100)
  w <- (as.numeric(coef(fitCV$glmnet.fit, s = fitCV$lambda.min))[-1] != 0)
  if (sum(w) < 2) {
    sumCoef <- apply(as.matrix(fitCV$glmnet.fit$beta), 2, function(b) sum(b != 0))
    w <- (as.matrix(fitCV$glmnet.fit$beta)[, which.max(sumCoef >= 2)] != 0)
  }
  list(w = w, lambda = fitCV$glmnet.fit$lambda)
}"""


@_needs("glmnet")
def test_screen_glmnet_default_path_matches_r_on_shared_folds():
    """The whole screener as used in the pipeline (each side builds its own default
    grid and stops its own path early), with shared CV folds: identical selections on
    40 synthetic datasets (measured at Task 2.2: 40/40). The grids agree to rounding on
    their common part; the early stop may differ by one point (AP7: measured in 5 of
    40 datasets, when the relative R^2 gain is within ~1e-6 of glmnet's 1e-5 threshold,
    because the two coordinate-descent solutions differ in R^2 by up to 4.8e-4)."""
    import rpy2.robjects as ro

    from llm_cong_predict.models import screeners
    from llm_cong_predict.rbridge import converter

    with converter():
        r_screen = ro.r(_R_SCREEN_DEFAULT)
    for seed in range(40):
        rng = np.random.default_rng(seed)
        n, p = [(80, 6), (60, 20), (30, 50), (100, 3)][seed % 4]
        X = rng.normal(size=(n, p)) * rng.uniform(0.5, 3, size=p)
        beta = np.zeros(p)
        beta[:3] = [1.0, -0.6, 0.3] if seed % 3 else [0.0, 0.0, 0.0]
        y = X @ beta + rng.normal(size=n)
        foldid = rng.permutation(np.resize(np.arange(10), n))
        with converter():
            res = r_screen(X, y, foldid + 1.0)
            r_mask = np.asarray(ro.r["[["](res, "w"), dtype=bool)
            r_lam = np.asarray(ro.r["[["](res, "lambda"), dtype=float)
        py_mask = screeners.screen_glmnet(X, y, seed=0, foldid=foldid)
        np.testing.assert_array_equal(py_mask, r_mask, err_msg=f"seed {seed}")

        grid = screeners.lambda_grid(X, y)
        _, _, rsq = screeners._path(X, y, grid)
        length = screeners._early_stop_length(rsq)
        assert abs(length - len(r_lam)) <= 1, f"seed {seed}: {length} vs {len(r_lam)}"
        common = min(length, len(r_lam))
        np.testing.assert_allclose(grid[:common], r_lam[:common], rtol=1e-12)


@_needs("SuperLearner", "nnls")
def test_sl_mean_and_sl_lm_match_r_superlearner_on_identical_folds():
    """R: llm_paper/R/functions.R:L526–548 (get_lm_cv_model's library) run by R's
    CV.SuperLearner (SuperLearner 2.0-42) and by the native engine on identical outer
    AND inner folds: library predictions, weights, SL predictions and CV risks agree to
    1e-10 (measured at Task 2.2: max 1.5e-14)."""
    from llm_cong_predict.models.base_learners import lm_library
    from llm_cong_predict.models.folds import make_folds, train_indices
    from llm_cong_predict.models.native_superlearner import fit_cv_superlearner
    from llm_cong_predict.models.r_superlearner import fit_r_cv_superlearner

    rng = np.random.default_rng(0)
    n = 120
    X = rng.normal(size=(n, 5))
    X[:, 4] = X[:, 0] + X[:, 1]  # exactly collinear: R drops it (SL.lm aliasing)
    y = X[:, :3] @ np.array([1.5, -1.0, 0.5]) + rng.normal(size=n)
    folds = make_folds(n, 10, seed=1)
    inner = [make_folds(len(train_indices(n, f)), 5, seed=100 + k) for k, f in enumerate(folds)]
    native = fit_cv_superlearner(X, y, lm_library(), folds=folds, inner_folds=inner)
    r = fit_r_cv_superlearner(X, y, which="lm", folds=folds, inner_folds=inner)
    assert r.library_names == native.library_names == ["SL.mean_All", "SL.lm_All"]
    for attr in ("library_predict", "coef", "sl_predict", "cv_risk", "discrete_sl_predict"):
        np.testing.assert_allclose(getattr(native, attr), getattr(r, attr), rtol=0, atol=1e-10,
                                   err_msg=attr)
