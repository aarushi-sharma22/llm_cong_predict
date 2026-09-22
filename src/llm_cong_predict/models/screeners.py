"""``screen.glmnet``: LASSO variable screening, following glmnet's own algorithm.

R pkg: SuperLearner/R/screen.glmnet.R:L1–16 (2.0-40):

    fitCV <- glmnet::cv.glmnet(x = X, y = Y, lambda = NULL, type.measure = 'deviance',
                               nfolds = 10, family = "gaussian", alpha = 1, nlambda = 100)
    whichVariable <- coef(fitCV$glmnet.fit, s = fitCV$lambda.min)[-1] != 0
    if (sum(whichVariable) < 2) {              # minscreen = 2
        sumCoef <- colSums(fitCV$glmnet.fit$beta != 0)
        whichVariable <- fitCV$glmnet.fit$beta[, which.max(sumCoef >= 2)] != 0
    }

What is reproduced (glmnet 5.0 sources, see docs/REFERENCE_SOURCES.md):
  * X is standardised with the population SD (denominator n) and y is centred;
    the penalty applies to the standardised coefficients
    (glmnetpp elnet_driver/standardize.hpp:L35, L73–76). The gaussian lasso objective
    ``(1/2n)||y - X b||^2 + lambda ||b||_1`` is the same as scikit-learn's Lasso, so the
    path is fitted with ``sklearn.linear_model.lasso_path`` on the same grid.
  * lambda grid: lambda_max = max_j |x_j' (y - ybar)| / n on standardised X (glmnetpp
    elnet_path/base.hpp:L262–272, rescaled to the y scale at elnet_driver/gaussian.hpp:
    L448; first value fixed by glmnet/R/fix.lam.R, glmnet.R:L581), then 100 values in
    total, geometric down to ratio * lambda_max with ratio 1e-4 when n >= p and 1e-2
    otherwise (glmnet/R/glmnet.R:L384; base.hpp:L209–211).
  * the full-data path stops early, as glmnet's does, once at least 5 lambdas are done
    and the relative gain in R^2 is below 1e-5 or R^2 exceeds 0.999
    (elnet_path/base.hpp:L307–310, gaussian_base.hpp:L132–134,
    glmnet/R/glmnet.control.R:L153–157); the fold fits reuse that (truncated) grid.
  * CV error per lambda is the fold-size-weighted mean of fold MSEs, i.e. the pooled
    MSE (glmnet/R/cvstats.R:L6, cvcompute.R); ties go to the LARGEST lambda
    (getOptcv.glmnet.R:L5–8).
  * fallback: the first lambda along the path with at least ``minscreen`` non-zero
    coefficients; if there is none, ``which.max`` of an all-FALSE vector is 1, the
    largest lambda, which usually selects no column at all.
  * glmnet stops with an error when X has fewer than 2 columns (glmnet.R:L393) or y
    is constant (glmnet/R/elnet.R:L23); SuperLearner then keeps all columns
    (SuperLearner.R:L121–124). This module raises :class:`ScreenFailure` in both cases.
  * constant columns of X get coefficient 0 (glmnet excludes them).

# APPROX: coordinate-descent convergence differs (glmnet thresh = 1e-7 on its own
# scale; sklearn uses a duality-gap tolerance), and glmnet's CV folds are random in R
# (cv.glmnet.R:L256–257); here they come from KFold(10, shuffle=True, random_state).
# VALIDATION_CHECKLIST V4 / the Task 2.2 oracle measures the gap with shared folds.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import lasso_path
from sklearn.model_selection import KFold

MINSCREEN = 2  # R pkg: SuperLearner/R/screen.glmnet.R:L1 minscreen = 2
NFOLDS = 10  # R pkg: SuperLearner/R/screen.glmnet.R:L1 nfolds = 10
NLAMBDA = 100  # R pkg: SuperLearner/R/screen.glmnet.R:L1 nlambda = 100
MNLAM = 5  # R pkg: glmnet/R/glmnet.control.R:L156 mnlam = 5
FDEV = 1e-5  # R pkg: glmnet/R/glmnet.control.R:L153 fdev = 1e-5
DEVMAX = 0.999  # R pkg: glmnet/R/glmnet.control.R:L157 devmax = 0.999
_LASSO_TOL = 1e-7  # APPROX: glmnet thresh = 1e-7 (glmnet.R:L384); sklearn's tol is a duality gap
_LASSO_MAX_ITER = 100_000  # glmnet maxit = 1e5 (glmnet.R:L384)


class ScreenFailure(RuntimeError):
    """glmnet would stop with an error; SuperLearner then keeps all columns."""


def _standardise(X: np.ndarray, y: np.ndarray):
    """glmnet's internal standardisation with an intercept (weights all 1)."""
    xm = X.mean(axis=0)
    xs = np.sqrt(np.mean((X - xm) ** 2, axis=0))  # denominator n (standardize.hpp:L73–76)
    active = xs > 0  # constant columns are excluded by glmnet (coefficient 0)
    Xs = np.zeros_like(X)
    Xs[:, active] = (X[:, active] - xm[active]) / xs[active]
    ym = y.mean()
    return Xs, y - ym, xm, xs, ym, active


def lambda_grid(X: np.ndarray, y: np.ndarray, nlambda: int = NLAMBDA) -> np.ndarray:
    """glmnet's default lambda sequence for gaussian, alpha = 1 (before early stopping)."""
    n, p = X.shape
    Xs, yc, *_ = _standardise(X, y)
    lambda_max = np.max(np.abs(Xs.T @ yc)) / n
    ratio = 1e-2 if n < p else 1e-4  # R pkg: glmnet/R/glmnet.R:L384 lambda.min.ratio
    return lambda_max * ratio ** (np.arange(nlambda) / (nlambda - 1))


def _path(X: np.ndarray, y: np.ndarray, lambdas: np.ndarray):
    """Lasso coefficients (on the ORIGINAL x scale) and intercepts along ``lambdas``."""
    Xs, yc, xm, xs, ym, active = _standardise(X, y)
    beta_std = np.zeros((X.shape[1], len(lambdas)))
    if active.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            _, coefs, _ = lasso_path(Xs[:, active], yc, alphas=lambdas,
                                     tol=_LASSO_TOL, max_iter=_LASSO_MAX_ITER)
        beta_std[active] = coefs
    beta = np.zeros_like(beta_std)
    beta[active] = beta_std[active] / xs[active, None]
    intercept = ym - xm @ beta
    rss = np.sum((yc[:, None] - Xs @ beta_std) ** 2, axis=0)
    rsq = 1.0 - rss / np.sum(yc**2)
    return beta, intercept, rsq


def _early_stop_length(rsq: np.ndarray) -> int:
    """Number of path points glmnet keeps (elnet_path/base.hpp:L307–310): after at
    least MNLAM points, stop at the first point whose relative R^2 gain is < FDEV or
    whose R^2 exceeds DEVMAX; that point itself is kept."""
    for m in range(len(rsq)):
        if m + 1 < MNLAM:
            continue
        gain = np.inf if rsq[m] == 0 else (rsq[m] - rsq[m - 1]) / rsq[m]
        if gain < FDEV or rsq[m] > DEVMAX:
            return m + 1
    return len(rsq)


def screen_glmnet(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    min_screen: int = MINSCREEN,
    nfolds: int = NFOLDS,
    nlambda: int = NLAMBDA,
    foldid: np.ndarray | None = None,
    lambdas: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean mask of the columns of ``X`` that ``screen.glmnet`` keeps.

    ``foldid`` (values 0..nfolds-1) and ``lambdas`` override the random CV folds and
    the default grid; they exist for the R parity test (Task 2.2), which passes the
    same folds and grid to R's ``cv.glmnet``. With ``lambdas`` given, no early stopping
    is applied (glmnet does not stop early on a user-supplied grid, base.hpp:L307).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    if p < 2:
        raise ScreenFailure("x should be a matrix with 2 or more columns (glmnet.R:L393)")
    if np.all(y == y[0]):
        raise ScreenFailure("y is constant; gaussian glmnet fails at standardization step (elnet.R:L23)")

    if lambdas is None:
        grid = lambda_grid(X, y, nlambda)
        beta, _, rsq = _path(X, y, grid)
        keep_n = _early_stop_length(rsq)
        grid, beta = grid[:keep_n], beta[:, :keep_n]
    else:
        grid = np.asarray(lambdas, dtype=float)
        beta, _, _ = _path(X, y, grid)

    if foldid is None:
        foldid = np.empty(n, dtype=int)
        for k, (_, test) in enumerate(KFold(nfolds, shuffle=True, random_state=seed).split(X)):
            foldid[test] = k
    foldid = np.asarray(foldid)

    sq_err = np.empty((n, len(grid)))
    for k in np.unique(foldid):
        test = foldid == k
        b, b0, _ = _path(X[~test], y[~test], grid)
        sq_err[test] = (y[test, None] - (b0 + X[test] @ b)) ** 2
    cvm = sq_err.mean(axis=0)  # pooled MSE = fold-size-weighted mean (cvstats.R:L6)
    # lambda.min = max(lambda[cvm <= min(cvm)]) (getOptcv.glmnet.R:L5–7); the grid is
    # decreasing, so the largest tied lambda is the first index.
    idx_min = int(np.flatnonzero(cvm <= cvm.min())[0])
    which = beta[:, idx_min] != 0
    if which.sum() < min_screen:
        sum_coef = (beta != 0).sum(axis=0)
        reach = sum_coef >= min_screen
        new_cut = int(np.argmax(reach))  # which.max: first TRUE, or 1 (index 0) if none
        which = beta[:, new_cut] != 0
    return which
