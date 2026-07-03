"""Native (scikit-learn / xgboost) Super Learner with nested cross-validation.

Reproduces the STRUCTURE of R's ``CV.SuperLearner`` as called in the original
model functions:

  * Outer V=10 CV. For each outer fold, the fold is held out and a Super Learner
    is fit on the rest.
  * That Super Learner runs an inner V=5 CV to obtain out-of-fold predictions from
    each base learner, then the NNLS meta-learner (method.NNLS) computes convex
    ensemble weights from those predictions.
  * Each base learner is refit on the full outer-training set and predicts the
    held-out outer fold; the ensemble prediction is the weighted combination.
  * Per-observation we record: the ensemble prediction (SL.predict), each base
    learner's prediction (library.predict), and the single best learner's
    prediction (discreteSL.predict); per outer fold we record the weights (coef).

The output is a :class:`CVSuperLearnerFit`, the SAME contract consumed by the
already-verified metric layer (``metrics/cv_metrics.py``).

WHAT IS AND ISN'T ESTABLISHED HERE
----------------------------------
Testable without R (see tests/test_models.py): the nested-CV bookkeeping (no
train/test leakage, correct shapes, correct library-name ordering), the NNLS
weighting behaviour, and sane end-to-end behaviour on data with a known signal.

NOT established without R: whether the numbers match R's SuperLearner or the
paper. The base learners are [approx] (see base_learners.py). Numerical fidelity
is a hypothesis until the rpy2 oracle (VALIDATION_CHECKLIST V4) is run on the
user's machine.
"""

from __future__ import annotations

import numpy as np

from ..metrics.cv_metrics import CVSuperLearnerFit
from .base_learners import LearnerSpec
from .folds import make_folds, train_indices
from .meta import nnls_meta
from .screeners import screen_glmnet


def _fit_predict_learner(
    spec: LearnerSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_predict: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Screen (if the spec asks), fit a fresh estimator on the (screened) training
    data, and predict ``X_predict``. Screening is refit on every training subset,
    matching SuperLearner."""
    if spec.screener == "screen.glmnet":
        mask = screen_glmnet(X_train, y_train, seed=seed)
    elif spec.screener is None:
        mask = np.ones(X_train.shape[1], dtype=bool)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown screener {spec.screener!r}")

    model = spec.make(seed)
    model.fit(X_train[:, mask], y_train)
    return np.asarray(model.predict(X_predict[:, mask]), dtype=float)


def fit_cv_superlearner(
    X: np.ndarray,
    y: np.ndarray,
    library: list[LearnerSpec],
    outcome_var: str | None = None,
    outer_v: int = 10,
    inner_v: int = 5,
    seed: int = 1,
) -> CVSuperLearnerFit:
    """Fit the nested-CV Super Learner and return a :class:`CVSuperLearnerFit`.

    Parameters
    ----------
    X, y:
        Design matrix ``(n, p)`` and target ``(n,)``. Callers are responsible for
        having already dropped rows with missing values (the R side does
        ``na.omit()`` first).
    library:
        The learner library (see base_learners.superlearner_library / lm_library).
    outer_v, inner_v:
        Outer and inner fold counts (10 and 5 in the original).
    seed:
        Seed for fold generation and for learner reproducibility (original: 1).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    n_learners = len(library)
    library_names = [spec.library_name() for spec in library]

    outer_folds = make_folds(n, outer_v, seed=seed)

    sl_predict = np.full(n, np.nan)
    library_predict = np.full((n, n_learners), np.nan)
    discrete_sl_predict = np.full(n, np.nan)
    coef = np.zeros((len(outer_folds), n_learners))

    for k, test_idx in enumerate(outer_folds):
        tr_idx = train_indices(n, test_idx)
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te = X[test_idx]

        # --- inner CV: out-of-fold predictions Z on the outer-training set ---
        n_tr = len(tr_idx)
        inner_folds = make_folds(n_tr, inner_v, seed=seed)
        Z = np.full((n_tr, n_learners), np.nan)
        for inner_test in inner_folds:
            inner_train = train_indices(n_tr, inner_test)
            for li, spec in enumerate(library):
                Z[inner_test, li] = _fit_predict_learner(
                    spec, X_tr[inner_train], y_tr[inner_train], X_tr[inner_test], seed=seed
                )

        # --- meta weights from inner OOF predictions ---
        meta = nnls_meta(Z, y_tr)
        coef[k] = meta.coef

        # --- refit each learner on full outer-training, predict outer test ---
        preds_te = np.empty((len(test_idx), n_learners))
        for li, spec in enumerate(library):
            preds_te[:, li] = _fit_predict_learner(spec, X_tr, y_tr, X_te, seed=seed)

        library_predict[test_idx] = preds_te
        sl_predict[test_idx] = preds_te @ meta.coef
        discrete_sl_predict[test_idx] = preds_te[:, meta.best]

    return CVSuperLearnerFit(
        Y=y,
        sl_predict=sl_predict,
        library_predict=library_predict,
        library_names=library_names,
        folds=outer_folds,
        coef=coef,
        discrete_sl_predict=discrete_sl_predict,
        outcome_var=outcome_var,
        method="method.NNLS",
    )
