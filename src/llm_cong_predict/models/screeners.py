"""Variable screening (approximation of SuperLearner's ``screen.glmnet``).

SuperLearner's ``screen.glmnet`` fits a LASSO (``cv.glmnet``, alpha=1) and keeps
the variables with non-zero coefficients at the CV-selected lambda, with a
minimum-count fallback (``minscreen``) so at least a few variables survive.

This is an APPROXIMATION: ``glmnet`` and scikit-learn's ``LassoCV`` use different
coordinate-descent implementations, lambda paths, and standardisation defaults,
so the exact selected set can differ. The screening decision therefore feeds into
the [approx] status of the learners that use it, and is part of what the rpy2
oracle (V4) checks.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LassoCV


def screen_glmnet(X: np.ndarray, y: np.ndarray, seed: int, min_screen: int = 2) -> np.ndarray:
    """Return a boolean mask of columns of ``X`` to keep.

    Selects features with non-zero LASSO coefficients; if fewer than
    ``min_screen`` survive, falls back to the ``min_screen`` features with the
    largest absolute coefficient. If ``X`` has at most ``min_screen`` columns,
    keeps all of them.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    p = X.shape[1]
    if p <= min_screen:
        return np.ones(p, dtype=bool)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # LassoCV can warn on tiny/degenerate folds
        model = LassoCV(cv=5, random_state=seed, n_jobs=1, max_iter=10000)
        model.fit(X, y)

    coef = model.coef_
    mask = coef != 0
    if mask.sum() >= min_screen:
        return mask

    # Fallback: keep the strongest `min_screen` coefficients (by |coef|).
    keep = np.argsort(np.abs(coef))[::-1][:min_screen]
    mask = np.zeros(p, dtype=bool)
    mask[keep] = True
    return mask
