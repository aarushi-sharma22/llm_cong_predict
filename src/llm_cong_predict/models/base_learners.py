"""Base learners for the Super Learner library.

Each learner mirrors one R SuperLearner wrapper used by the original model functions.
The library and its column names match ``get_general_superlearner_cv_model``
(R: llm_paper/R/functions.R:L514–519) and ``get_lm_cv_model`` (L543):

    SL.mean_All, SL.ranger_screen.glmnet, SL.nnet_screen.glmnet,
    SL.xgboost.hist_screen.glmnet, SL.ksvm_screen.glmnet, SL.lm_screen.glmnet

Every setting below cites the wrapper line (``# R pkg: SuperLearner/R/...``, version
2.0-40, see docs/REFERENCE_SOURCES.md) or the underlying package's default. Where the
Python implementation can only approximate the R one, the line is marked
``# APPROX: <reason>`` and listed in docs/VALIDATION_CHECKLIST.md (V4, V7).

Learner failures: a learner that cannot be fitted raises; the Super Learner engine
catches any exception exactly as R's ``try()`` does (R pkg:
SuperLearner/R/SuperLearner.R:L143–146) and gives the learner NA predictions.
:class:`LearnerFailure` is raised deliberately where the R package would stop with an
error (for example nnet's weight limit).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from ..isolation import refuse_if_loaded


class LearnerFailure(RuntimeError):
    """A learner cannot be fitted where the R wrapper would stop with an error."""


@dataclass(frozen=True)
class LearnerSpec:
    """A library entry: a name, a factory ``make(seed, n_features)`` producing a fresh
    estimator for the (screened) number of columns, and the screener applied before
    fitting (``"screen.glmnet"``, or ``None`` for no screening, named ``_All`` in R)."""

    name: str
    make: Callable[[int, int], BaseEstimator]
    screener: str | None

    def library_name(self) -> str:
        # R pkg: SuperLearner/R/CV.SuperLearner.R:L65 —
        # paste(predAlgorithm, screenAlgorithm, sep = "_"), "All" when unscreened.
        suffix = self.screener if self.screener is not None else "All"
        return f"{self.name}_{suffix}"


# --- SL.mean ------------------------------------------------------------------

def _make_mean(seed: int, n_features: int) -> BaseEstimator:
    # R pkg: SuperLearner/R/SL.mean.R:L3 (2.0-40) — weighted.mean(Y, obsWeights); the
    # original passes no weights, so obsWeights = 1 (SuperLearner.R:L101–103).
    return DummyRegressor(strategy="mean")


# --- SL.lm --------------------------------------------------------------------

class RLinearModel(RegressorMixin, BaseEstimator):
    """OLS with an intercept that drops aliased columns the way R's ``lm`` does.

    R pkg: SuperLearner/R/SL.lm.R:L44 (2.0-40) calls ``lm(Y ~ ., data = X, weights =
    obsWeights)``, i.e. ``lm.wfit`` with pivoting tolerance ``tol = 1e-7``
    (R: r-source/src/library/stats/R/lm.R:L169). R's QR (LINPACK dqrdc2, limited
    pivoting) moves a column to the end, and gives it an NA coefficient, when the part
    of it orthogonal to the columns before it is tiny relative to its norm;
    ``predict.lm`` then uses only the first ``rank`` pivoted columns (lm.R:L733–739).

    Implemented as a sequential rule: with the intercept first and the columns in
    their original order, column j is kept iff ``|R_jj| >= tol * ||x_j||`` in an
    unpivoted Householder QR, where ``|R_jj|`` is the norm of the part of x_j
    orthogonal to all earlier columns (equivalently, to the earlier KEPT columns).
    # APPROX: dqrdc2 compares downdated column norms in its own order of operations;
    # the rule matches R's choice of dropped columns up to rounding at the threshold.
    scikit-learn's LinearRegression is not used because its minimum-norm lstsq keeps
    aliased columns, which changes predictions whenever new data break a collinearity
    that held in the training data.
    """

    def __init__(self, tol: float = 1e-7):
        self.tol = tol

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        design = np.column_stack([np.ones(n), X])
        norms = np.linalg.norm(design, axis=0)
        r_diag = np.abs(np.diag(np.linalg.qr(design, mode="r")))
        k = min(n, p + 1)
        keep = np.zeros(p + 1, dtype=bool)
        keep[:k] = r_diag[:k] >= self.tol * np.where(norms[:k] > 0, norms[:k], 1.0)
        keep[:k] &= norms[:k] > 0
        if not keep[0]:  # pragma: no cover - intercept column has norm sqrt(n) > 0
            raise LearnerFailure("intercept column aliased")
        coef, *_ = np.linalg.lstsq(design[:, keep], y, rcond=None)
        self.kept_columns_ = np.flatnonzero(keep[1:])  # indices into X
        self.intercept_ = float(coef[0])
        self.coef_ = coef[1:]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X[:, self.kept_columns_] @ self.coef_


def _make_lm(seed: int, n_features: int) -> BaseEstimator:
    return RLinearModel(tol=1e-7)  # R: r-source/src/library/stats/R/lm.R:L169 (tol = 1e-7)


# --- SL.ranger ----------------------------------------------------------------

def _make_ranger(seed: int, n_features: int) -> BaseEstimator:
    return RandomForestRegressor(
        n_estimators=500,  # R pkg: SuperLearner/R/SL.ranger.R:L59 (2.0-40) num.trees = 500
        # R pkg: SuperLearner/R/SL.ranger.R:L60 mtry = floor(sqrt(ncol(X))), ncol after screening
        max_features=max(1, int(np.floor(np.sqrt(n_features)))),
        # R pkg: SuperLearner/R/SL.ranger.R:L63 min.node.size = 5 (gaussian). ranger makes a
        # node terminal when num_samples_node <= min.node.size
        # (R pkg: ranger/src/TreeRegression.cpp:L106, 0.18.0), so a split needs n >= 6;
        # sklearn splits when n >= min_samples_split (sklearn/tree/_tree.pyx:L231).
        # APPROX: sklearn bootstraps with sample weights and counts DISTINCT rows per node
        # (sklearn/ensemble/_forest.py:L156); ranger counts draws including duplicates.
        min_samples_split=6,
        min_samples_leaf=1,  # R pkg: ranger/R/ranger.R:L117 min.bucket default 1 (regression)
        bootstrap=True,  # R pkg: SuperLearner/R/SL.ranger.R:L64 replace = TRUE
        max_samples=None,  # R pkg: SuperLearner/R/SL.ranger.R:L65 sample.fraction = 1 when replace
        criterion="squared_error",  # ranger regression splitrule "variance" (default)
        random_state=seed,
        n_jobs=1,  # R pkg: SuperLearner/R/SL.ranger.R:L66 num.threads = 1
    )


# --- SL.nnet ------------------------------------------------------------------

NNET_SIZE = 2  # R pkg: SuperLearner/R/SL.nnet.R:L5 (2.0-40) size = 2
NNET_MAXNWTS = 1000  # R pkg: nnet/R/nnet.R:L79 (7.3-21) MaxNWts = 1000


def nnet_weight_count(n_features: int, size: int = NNET_SIZE, n_out: int = 1) -> int:
    """Number of weights nnet allocates: every hidden and output unit gets a bias
    (R pkg: nnet/R/nnet.R:L242–268, add.net/norm.net), so size*(p+1) + n_out*(size+1),
    which is 2p + 5 for size 2 and one output."""
    return size * (n_features + 1) + n_out * (size + 1)


class NnetLike(RegressorMixin, BaseEstimator):
    """``SL.nnet``: one hidden layer of 2 logistic units, linear output, no decay.

    R pkg: SuperLearner/R/SL.nnet.R:L8 (2.0-40): nnet(size = 2, linout = TRUE,
    trace = FALSE, maxit = 500). maxit = 500 is fixed by the wrapper and overrides
    nnet's default of 100 (nnet/R/nnet.R:L78). nnet stops with "too many weights" when
    the weight count exceeds MaxNWts = 1000 (nnet/R/nnet.R:L105–106); that happens for
    2p + 5 > 1000, i.e. p >= 498 columns after screening, and is reproduced here.
    # APPROX: nnet draws start weights from U(-0.7, 0.7) (rang, nnet.R:L117) and fits by
    # BFGS with abstol 1e-4 / reltol 1e-8; MLPRegressor uses Glorot initialisation and
    # L-BFGS with its own stopping rule, so fits differ. Inputs are not scaled, as in nnet.
    """

    def __init__(self, seed: int = 1):
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        n_wts = nnet_weight_count(X.shape[1])
        if n_wts > NNET_MAXNWTS:
            raise LearnerFailure(
                f"too many ({n_wts}) weights: nnet stops when 2p+5 > MaxNWts = 1000 "
                "(R pkg: nnet/R/nnet.R:L105-106)"
            )
        self.model_ = MLPRegressor(
            hidden_layer_sizes=(NNET_SIZE,),
            activation="logistic",  # nnet hidden units are logistic
            solver="lbfgs",
            alpha=0.0,  # R pkg: nnet/R/nnet.R:L78 decay = 0
            max_iter=500,  # R pkg: SuperLearner/R/SL.nnet.R:L8 maxit = 500
            random_state=self.seed,
        )
        with warnings.catch_warnings():  # nnet with trace = FALSE is silent at maxit
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model_.fit(X, np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float))


def _make_nnet(seed: int, n_features: int) -> BaseEstimator:
    return NnetLike(seed=seed)


# --- SL.ksvm ------------------------------------------------------------------

def sigest_sigma(X: np.ndarray, rng: np.random.Generator, frac: float = 0.5) -> float:
    """kernlab's automatic RBF width: ``mean(sigest(x, scaled = FALSE)[c(1, 3)])``.

    R pkg: kernlab/R/ksvm.R:L153–156 and kernlab/R/sigest.R:L58–64 (0.9-33): draw
    floor(frac * m) row pairs with replacement, squared distances, drop zeros, then
    1/quantile(d, c(0.9, 0.5, 0.1)); sigma is the mean of the first and third.
    R's default quantile type 7 equals numpy's default ("linear").
    # APPROX: R's sample() random numbers cannot be reproduced; the pairs differ.
    """
    m = X.shape[0]
    k = int(np.floor(frac * m))
    idx1 = rng.integers(0, m, size=k)
    idx2 = rng.integers(0, m, size=k)
    dist = np.sum((X[idx1] - X[idx2]) ** 2, axis=1)
    dist = dist[dist != 0]
    if dist.size == 0:
        raise LearnerFailure("sigest: all sampled distances are zero (R returns NA)")
    q = np.quantile(dist, [0.9, 0.5, 0.1])
    srange = 1.0 / q
    return float(np.mean(srange[[0, 2]]))


class KsvmLike(RegressorMixin, BaseEstimator):
    """``SL.ksvm``: eps-regression with an RBF kernel, kernlab's scaling and sigma rule.

    R pkg: SuperLearner/R/SL.ksvm.R:L89–129 (2.0-40): ksvm(X, Y, scaled = TRUE,
    type = NULL, kernel = "rbfdot", kpar = "automatic", C = 1, nu = 0.2,
    epsilon = 0.1). The wrapper declares cache = 40, tol = 0.001, shrinking = TRUE but
    does not pass them; kernlab's own defaults are the same values
    (R pkg: kernlab/R/ksvm.R:L61–63). type = NULL with numeric y is "eps-svr"
    (ksvm.R:L106).

    Scaling (kernlab/R/ksvm.R:L127–148): if every column has non-zero variance, X is
    standardised (mean, sd with denominator n-1, R's scale()) and so is y, and
    predictions are converted back (ksvm.R:L2645–2647, L2810–2811). If ANY column is
    constant, nothing is scaled, neither X nor y (kernlab warns). "Constant" is tested
    as all values identical, which is when R's var() is exactly 0.
    # APPROX: sklearn's SVR (libsvm) and kernlab's eps-svr are different solver
    # implementations; sigma's random pair draw differs (see sigest_sigma).
    """

    def __init__(self, seed: int = 1):
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        constant = np.ptp(X, axis=0) == 0
        self.scaled_ = bool(X.shape[1] > 0 and not constant.any())
        if self.scaled_:
            self.x_center_ = X.mean(axis=0)
            self.x_scale_ = X.std(axis=0, ddof=1)
            X = (X - self.x_center_) / self.x_scale_
            self.y_center_ = float(y.mean())
            self.y_scale_ = float(y.std(ddof=1))
            y = (y - self.y_center_) / self.y_scale_
        else:
            self.x_center_ = self.x_scale_ = None
            self.y_center_, self.y_scale_ = 0.0, 1.0
        self.sigma_ = sigest_sigma(X, np.random.default_rng(self.seed))
        self.model_ = SVR(
            kernel="rbf",
            gamma=self.sigma_,  # kernlab rbfdot: exp(-sigma * ||x - x'||^2)
            C=1.0,  # R pkg: SuperLearner/R/SL.ksvm.R:L94
            epsilon=0.1,  # R pkg: SuperLearner/R/SL.ksvm.R:L96
            tol=1e-3,  # R pkg: kernlab/R/ksvm.R:L62
            cache_size=40,  # R pkg: kernlab/R/ksvm.R:L61 (MB in both)
            shrinking=True,  # R pkg: kernlab/R/ksvm.R:L63
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.scaled_:
            X = (X - self.x_center_) / self.x_scale_
        return self.model_.predict(X) * self.y_scale_ + self.y_center_


def _make_ksvm(seed: int, n_features: int) -> BaseEstimator:
    return KsvmLike(seed=seed)


# --- SL.xgboost.hist ----------------------------------------------------------

def _make_xgboost_hist(seed: int, n_features: int) -> BaseEstimator:
    # R: llm_paper/R/functions.R:L492–494 — SL.xgboost(..., params = list(tree_method = "hist")).
    # The paper-era code path is SL.xgboost's branch for xgboost < 3.0
    # (R pkg: SuperLearner/R/SL.xgboost.R:L106, 2.0-40); the branch for > 3.0 (L53–65,
    # added 2025-12-14) drops `params`, so tree_method would be ignored there.
    refuse_if_loaded("torch", "Fitting SL.xgboost.hist")  # isolation.py, docs/ORCHESTRATION.md
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",  # R pkg: SuperLearner/R/SL.xgboost.R:L102
        n_estimators=1000,  # R pkg: SuperLearner/R/SL.xgboost.R:L43 ntrees = 1000
        max_depth=4,  # R pkg: SuperLearner/R/SL.xgboost.R:L44 max_depth = 4
        min_child_weight=10,  # R pkg: SuperLearner/R/SL.xgboost.R:L44 minobspernode = 10
        learning_rate=0.1,  # R pkg: SuperLearner/R/SL.xgboost.R:L44 shrinkage = 0.1
        tree_method="hist",  # R: llm_paper/R/functions.R:L493
        n_jobs=1,  # R pkg: SuperLearner/R/SL.xgboost.R:L46 nthread = 1
        # R xgboost 1.7.x: reg:squarederror keeps the constant base score 0.5
        # (R pkg: xgboost v1.7.6 include/xgboost/objective.h:L33; RegLossObj has no
        # InitEstimation override). xgboost >= 2.0 (installed Python: 3.3.0) estimates it
        # from the labels instead (xgboost NEWS.md:L50–52), so it is set explicitly.
        # APPROX: the paper's R xgboost version is unknown; 1.7.x is assumed.
        base_score=0.5,
        random_state=seed,
        verbosity=0,
    )


# --- library assembly -----------------------------------------------------------

def superlearner_library() -> list[LearnerSpec]:
    """The 6-learner library of ``get_general_superlearner_cv_model``, in the same
    order as the original (R: llm_paper/R/functions.R:L514–519)."""
    return [
        LearnerSpec("SL.mean", _make_mean, None),
        LearnerSpec("SL.ranger", _make_ranger, "screen.glmnet"),
        LearnerSpec("SL.nnet", _make_nnet, "screen.glmnet"),
        LearnerSpec("SL.xgboost.hist", _make_xgboost_hist, "screen.glmnet"),
        LearnerSpec("SL.ksvm", _make_ksvm, "screen.glmnet"),
        LearnerSpec("SL.lm", _make_lm, "screen.glmnet"),
    ]


def lm_library() -> list[LearnerSpec]:
    """The 2-learner library of ``get_lm_cv_model`` (R: llm_paper/R/functions.R:L543):
    mean + unscreened lm, so the lm column is named ``SL.lm_All``, which is what
    ``get_cv_lm_metrics`` filters on (L732)."""
    return [
        LearnerSpec("SL.mean", _make_mean, None),
        LearnerSpec("SL.lm", _make_lm, None),
    ]
