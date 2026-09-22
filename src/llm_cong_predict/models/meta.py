"""The Super Learner meta-learner (``method.NNLS``).

Port of SuperLearner's default ``method.NNLS`` (R pkg: SuperLearner/R/method.R:L39–78,
2.0-40; the original passes no ``method=``, R: llm_paper/R/functions.R:L511–520):

    computeCoef:
      cvRisk <- apply(Z, 2, function(x) mean(w * (x - Y)^2))
      fit    <- nnls(sqrt(w) * Z, sqrt(w) * Y)
      coef   <- coef(fit); coef[is.na(coef)] <- 0
      if (sum(coef) > 0) coef <- coef / sum(coef) else warning("All algorithms have zero weight")
    computePred:
      if (sum(coef != 0) == 0) predictions are all 0 (with a warning)
      else crossprod(t(predY[, coef != 0]), coef[coef != 0])

Restricting the prediction to learners with non-zero weight is what stops an NA
prediction of a failed learner (weight 0) from spreading into the ensemble.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls


@dataclass
class MetaFit:
    coef: np.ndarray      # normalised non-negative weights, shape (n_learners,)
    cv_risk: np.ndarray   # per-learner CV risk (MSE), shape (n_learners,)
    best: int             # index of the single lowest-risk learner (discrete SL)


def compute_coef(Z: np.ndarray, Y: np.ndarray, obs_weights: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """``method.NNLS()$computeCoef`` (R pkg: SuperLearner/R/method.R:L42–64).

    Returns ``(coef, cv_risk)``. ``Z`` must not contain NaN: SuperLearner zeroes the
    column of every learner that failed before calling this (SuperLearner.R:L180–183).
    """
    Z = np.asarray(Z, dtype=float)
    Y = np.asarray(Y, dtype=float)
    w = np.ones(len(Y)) if obs_weights is None else np.asarray(obs_weights, dtype=float)
    cv_risk = np.array([np.mean(w * (Z[:, j] - Y) ** 2) for j in range(Z.shape[1])])  # L44
    sw = np.sqrt(w)
    coef, _ = nnls(sw[:, None] * Z, sw * Y)  # L48
    coef = np.nan_to_num(coef, nan=0.0)  # L54
    total = coef.sum()
    if total > 0:  # L56–61
        coef = coef / total
    else:
        warnings.warn("All algorithms have zero weight", RuntimeWarning, stacklevel=2)
    return coef, cv_risk


def compute_pred(pred_y: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """``method.NNLS()$computePred`` (R pkg: SuperLearner/R/method.R:L66–75)."""
    pred_y = np.asarray(pred_y, dtype=float)
    nonzero = coef != 0
    if not nonzero.any():
        warnings.warn("All metalearner coefficients are zero, predictions will all be equal to 0",
                      RuntimeWarning, stacklevel=2)
        return np.zeros(pred_y.shape[0])
    return pred_y[:, nonzero] @ coef[nonzero]


def nnls_meta(Z: np.ndarray, Y: np.ndarray, obs_weights: np.ndarray | None = None) -> MetaFit:
    """Weights, CV risks and the discrete-SL index from inner-CV predictions ``Z``.

    The discrete SL is ``which.min(cvRisk)``, which ignores NA
    (R pkg: SuperLearner/R/CV.SuperLearner.R:L89).
    """
    coef, cv_risk = compute_coef(Z, Y, obs_weights)
    return MetaFit(coef=coef, cv_risk=cv_risk, best=int(np.nanargmin(cv_risk)))
