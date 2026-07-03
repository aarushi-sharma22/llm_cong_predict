"""The Super Learner meta-learner (``method.NNLS``).

Faithful port of SuperLearner's default ``method.NNLS`` combiner. Given the
matrix ``Z`` of inner cross-validated predictions from each base learner and the
observed ``Y``, it finds non-negative weights that minimise the squared error of
the convex combination, then normalises them to sum to 1.

Reference (SuperLearner ``method.NNLS()$computeCoef``):
    cvRisk <- apply(Z, 2, function(x) mean(w * (x - Y)^2))
    fit    <- nnls(sqrt(w) * Z, sqrt(w) * Y)
    coef   <- coef(fit); coef[is.na(coef)] <- 0
    if (sum(coef) > 0) coef <- coef / sum(coef)

This module is fully verifiable without R: the optimisation is deterministic and
its behaviour on constructed inputs is known (e.g. a perfect learner column
receives all the weight). See tests/test_models.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls


@dataclass
class MetaFit:
    coef: np.ndarray      # normalised non-negative weights, shape (n_learners,)
    cv_risk: np.ndarray   # per-learner CV risk (MSE), shape (n_learners,)
    best: int             # index of the single lowest-risk learner (discrete SL)


def nnls_meta(Z: np.ndarray, Y: np.ndarray, obs_weights: np.ndarray | None = None) -> MetaFit:
    """Compute Super Learner weights from inner-CV predictions ``Z`` and target ``Y``.

    Parameters
    ----------
    Z:
        Inner cross-validated predictions, shape ``(n_obs, n_learners)``.
    Y:
        Observed outcome, shape ``(n_obs,)``.
    obs_weights:
        Optional observation weights; defaults to 1 (as in the original usage).

    Returns
    -------
    MetaFit
        Normalised weights, per-learner CV risk, and the discrete-SL index.
    """
    Z = np.asarray(Z, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, n_learners = Z.shape
    w = np.ones(n) if obs_weights is None else np.asarray(obs_weights, dtype=float)

    # Per-learner cross-validated risk (weighted MSE against Y).
    cv_risk = np.array([np.mean(w * (Z[:, j] - Y) ** 2) for j in range(n_learners)])

    # Non-negative least squares on sqrt(weight)-scaled inputs, matching R.
    sw = np.sqrt(w)
    coef, _ = nnls(sw[:, None] * Z, sw * Y)
    coef = np.nan_to_num(coef, nan=0.0)
    total = coef.sum()
    if total > 0:
        coef = coef / total
    # else: all-zero weights (degenerate); left as-is, matching R's warning path.

    best = int(np.argmin(cv_risk))
    return MetaFit(coef=coef, cv_risk=cv_risk, best=best)
