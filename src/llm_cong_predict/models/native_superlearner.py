"""Native (scikit-learn / xgboost) Super Learner with nested cross-validation.

Port of the mechanics of R's ``CV.SuperLearner`` and ``SuperLearner`` as called by
``get_general_superlearner_cv_model`` / ``get_lm_cv_model``
(R: llm_paper/R/functions.R:L496–548). Citations are to SuperLearner 2.0-40
(docs/REFERENCE_SOURCES.md); the cited files are unchanged since 2017–2020.

Per outer fold (R pkg: SuperLearner/R/CV.SuperLearner.R:L80–91):
  * a Super Learner is fitted on the outer-training rows and predicts the held-out fold.

Inside each Super Learner (R pkg: SuperLearner/R/SuperLearner.R):
  * inner V-fold CV (L109–172). In every training split each screening algorithm runs
    ONCE and its column mask is shared by every learner paired with it (L119–131, L143).
    A screener that errors is replaced by "all columns" (L121–124).
  * a learner that errors (R's ``try``) leaves NA predictions for that fold (L143–152).
    Any learner with an NA anywhere in its column of Z has the WHOLE column set to 0
    before the weights are computed (L180–183); if every column is then 0 the fit stops
    (L184–186).
  * weights: ``method.NNLS`` on Z (L189–193, models/meta.py).
  * refit on the whole training set with screening re-run once per screener on it
    (L222–223). A learner that errors there gets NA predictions (L249–252). If the
    failed learners' total weight is positive, their Z columns are set to 0 and the
    weights are recomputed; if it is already 0 nothing changes (L274–292).
  * ensemble prediction uses only learners with non-zero weight (L295, method.R:L66–75).
  * cvRisk is set to NA for learners that failed IN CV only (L303–305). A learner that
    failed only in the refit keeps a CV risk (computed from its zeroed Z column when the
    weights were recomputed), exactly as in R.
  * discrete SL = ``which.min(cvRisk)``, ignoring NA (CV.SuperLearner.R:L89).

Randomness (models/seeds.py): outer folds come from ``make_folds(n, V, seed)``; inner
folds, screeners and learners each get a seed derived from (seed, outer fold, inner
fold, name), so results are identical across repeated runs and across ``n_jobs``.
R's own random numbers cannot be reproduced (PORTING_NOTES C).
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from ..metrics.cv_metrics import CVSuperLearnerFit
from .base_learners import LearnerSpec
from .folds import make_folds, train_indices
from .meta import compute_coef, compute_pred
from .screeners import screen_glmnet
from .seeds import FULL_FIT, NO_FOLD, derive_seed

SCREENERS = {"screen.glmnet": screen_glmnet}


class AllLearnersFailed(RuntimeError):
    """R: stop("All algorithms dropped from library") (SuperLearner.R:L184–186, L280–282)."""


def _screen_masks(library, X, y, seed, outer, inner, failures) -> dict:
    """Run each distinct screener once on this training set (SuperLearner.R:L119–131)."""
    masks: dict = {None: np.ones(X.shape[1], dtype=bool)}
    for name in dict.fromkeys(s.screener for s in library if s.screener is not None):
        try:
            mask = np.asarray(SCREENERS[name](X, y, seed=derive_seed(seed, outer, inner, name)),
                              dtype=bool)
        except Exception as exc:  # R: try(); failed screener -> All (L121–124)
            failures.append({"outer": outer, "inner": inner, "learner": name, "stage": "screen",
                             "error": f"{type(exc).__name__}: {exc}"})
            mask = np.ones(X.shape[1], dtype=bool)
        masks[name] = mask
    return masks


def _fit_predict(spec: LearnerSpec, X_tr, y_tr, X_new, mask, seed):
    """Fit one learner on the screened columns and predict; ``(pred, None)`` or
    ``(None, error message)`` when it fails (R: try(), SuperLearner.R:L143–146)."""
    try:
        model = spec.make(seed, int(mask.sum()))
        model.fit(X_tr[:, mask], y_tr)
        pred = np.asarray(model.predict(X_new[:, mask]), dtype=float).reshape(-1)
        if pred.shape[0] != X_new.shape[0]:
            raise ValueError(f"prediction has {pred.shape[0]} rows, expected {X_new.shape[0]}")
        return pred, None
    except Exception as exc:  # R's try() catches every error
        return None, f"{type(exc).__name__}: {exc}"


def _superlearner(X, y, X_new, library, inner_folds, seed, outer) -> dict:
    """One ``SuperLearner()`` call: inner CV, weights, refit, prediction on ``X_new``."""
    n, k = len(y), len(library)
    names = [s.library_name() for s in library]
    failures: list[dict] = []

    # --- inner cross-validation -> Z (SuperLearner.R:L109–172) ---
    Z = np.full((n, k), np.nan)
    for j, test in enumerate(inner_folds):
        train = train_indices(n, test)
        masks = _screen_masks(library, X[train], y[train], seed, outer, j, failures)
        for li, spec in enumerate(library):
            pred, err = _fit_predict(spec, X[train], y[train], X[test], masks[spec.screener],
                                     derive_seed(seed, outer, j, names[li]))
            if err is not None:
                failures.append({"outer": outer, "inner": j, "learner": names[li],
                                 "stage": "cv", "error": err})
            else:
                Z[test, li] = pred

    errors_cv = np.isnan(Z).any(axis=0)  # L180
    Z[:, errors_cv] = 0.0  # L181–183
    if np.all(Z == 0):
        raise AllLearnersFailed("All algorithms dropped from library")
    coef, cv_risk = compute_coef(Z, y)

    # --- refit on the whole training set (L222–271) ---
    masks = _screen_masks(library, X, y, seed, outer, FULL_FIT, failures)
    pred_y = np.full((X_new.shape[0], k), np.nan)
    for li, spec in enumerate(library):
        pred, err = _fit_predict(spec, X, y, X_new, masks[spec.screener],
                                 derive_seed(seed, outer, FULL_FIT, names[li]))
        if err is not None:
            failures.append({"outer": outer, "inner": FULL_FIT, "learner": names[li],
                             "stage": "full", "error": err})
        else:
            pred_y[:, li] = pred

    errors_full = np.isnan(pred_y).any(axis=0)  # L274
    if errors_full.any() and coef[errors_full].sum() > 0:  # L275–289
        Z[:, errors_full] = 0.0
        if np.all(Z == 0):
            raise AllLearnersFailed("All algorithms dropped from library")
        coef, cv_risk = compute_coef(Z, y)

    sl_pred = compute_pred(pred_y, coef)  # L295
    cv_risk = cv_risk.copy()
    cv_risk[errors_cv] = np.nan  # L303–305: CV failures only
    best = int(np.nanargmin(cv_risk))  # CV.SuperLearner.R:L89 which.min ignores NA
    return {"sl": sl_pred, "lib": pred_y, "discrete": pred_y[:, best], "coef": coef,
            "cv_risk": cv_risk, "failures": failures}


def _outer_fold(k, test_idx, X, y, library, inner_v, seed, inner_folds_k):
    tr = train_indices(len(y), test_idx)
    if inner_folds_k is None:
        # inner folds: shuffled, not stratified (control.R:L15, CVFolds.R:L23)
        inner_folds_k = make_folds(len(tr), inner_v, seed=derive_seed(seed, k, NO_FOLD, "inner-folds"))
    return _superlearner(X[tr], y[tr], X[test_idx], library, inner_folds_k, seed, outer=k)


def fit_cv_superlearner(
    X: np.ndarray,
    y: np.ndarray,
    library: list[LearnerSpec],
    outcome_var: str | None = None,
    outer_v: int = 10,
    inner_v: int = 5,
    seed: int = 1,
    n_jobs: int = 1,
    folds: list[np.ndarray] | None = None,
    inner_folds: list[list[np.ndarray]] | None = None,
) -> CVSuperLearnerFit:
    """Fit the nested-CV Super Learner and return a :class:`CVSuperLearnerFit`.

    Parameters
    ----------
    X, y:
        Design matrix ``(n, p)`` and outcome ``(n,)`` without missing values (R stops
        on missing data, SuperLearner.R:L70–72; callers do ``na.omit`` first).
    library:
        ``base_learners.superlearner_library()`` or ``lm_library()``.
    outer_v, inner_v:
        10 and 5 in the original (R: llm_paper/R/functions.R:L512, L541).
    seed:
        Base seed for every random component (models/seeds.py).
    n_jobs:
        Outer folds run in parallel with joblib when > 1. Output does not depend on it.
    folds, inner_folds:
        Optional fixed outer folds (0-based index arrays) and, per outer fold, fixed
        inner folds (0-based indices into that fold's training rows), for the R oracle
        comparison (Task 2.2).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("missing data is currently not supported (SuperLearner.R:L70-72)")
    n, k = len(y), len(library)

    outer_folds = folds if folds is not None else make_folds(n, outer_v, seed=seed)
    args = [(i, f, X, y, library, inner_v, seed, None if inner_folds is None else inner_folds[i])
            for i, f in enumerate(outer_folds)]
    if n_jobs == 1:
        results = [_outer_fold(*a) for a in args]
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(_outer_fold)(*a) for a in args)

    sl_predict = np.full(n, np.nan)
    library_predict = np.full((n, k), np.nan)
    discrete = np.full(n, np.nan)
    coef = np.zeros((len(outer_folds), k))
    cv_risk = np.full((len(outer_folds), k), np.nan)
    failures: list[dict] = []
    for i, (test_idx, res) in enumerate(zip(outer_folds, results)):
        sl_predict[test_idx] = res["sl"]
        library_predict[test_idx] = res["lib"]
        discrete[test_idx] = res["discrete"]
        coef[i] = res["coef"]
        cv_risk[i] = res["cv_risk"]
        failures.extend(res["failures"])

    return CVSuperLearnerFit(
        Y=y,
        sl_predict=sl_predict,
        library_predict=library_predict,
        library_names=[s.library_name() for s in library],
        folds=list(outer_folds),
        coef=coef,
        discrete_sl_predict=discrete,
        outcome_var=outcome_var,
        method="method.NNLS",
        cv_risk=cv_risk,
        failures=failures,
    )
