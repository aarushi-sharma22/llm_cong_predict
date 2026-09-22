"""Cross-validated performance metrics.

Faithful port of the hand-rolled metric extractors in the original
``R/functions.R``:
  * get_cv_predictive_r2   -> cv_predictive_r2
  * get_cv_lm_r2           -> cv_lm_r2
  * get_cv_rmse            -> cv_rmse
  * get_cv_mad             -> cv_mad
  * get_cv_superlearner_metrics -> superlearner_metrics
  * get_cv_lm_metrics      -> lm_metrics

The original functions read fields out of an R ``CV.SuperLearner`` object. To keep
the metric layer independent of *which* backend produced the fit (native sklearn
or the rpy2 oracle), both backends populate the identical :class:`CVSuperLearnerFit`
contract defined below, and the metric functions consume only that.

Fidelity notes (see docs/PORTING_NOTES.md):
  * The original computes fold-wise risks and then reports mean/min/max across
    folds. Reproduced exactly.
  * Both scorers take the MSE columns (and ``se_mse``) from the UNCLAMPED
    predictions, because the R calls ``summary(cv_fit)`` before clamping outlying
    SL predictions; R^2, MAD and RMSE of the ensemble use the clamped predictions.
    The clamp uses an n-1 (sample) standard deviation to match R's ``sd``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# The meta-learner methods under which the original computed NNLS-style risks.
_NNLS_METHODS = {"method.NNLS", "method.NNLS2", "method.CC_LS"}

# Library-learner row names as they appear in the R SuperLearner summary/coef.
MEAN_LEARNER = "SL.mean_All"
LM_LEARNER = "SL.lm_All"


@dataclass
class CVSuperLearnerFit:
    """Backend-agnostic mirror of the fields the R ``CV.SuperLearner`` object
    exposed and that the metric functions consume.

    Parameters
    ----------
    Y:
        Observed outcome, shape ``(n_obs,)``.
    sl_predict:
        Cross-validated Super Learner predictions, shape ``(n_obs,)``.
    library_predict:
        Cross-validated predictions for each library learner, shape
        ``(n_obs, n_learners)``; column order matches ``library_names``.
    library_names:
        Learner names, e.g. ``["SL.mean_All", "SL.lm_screen.glmnet", ...]``.
    folds:
        List of outer-fold index arrays (row indices into ``Y``).
    coef:
        Per-fold learner weights, shape ``(n_folds, n_learners)``. Optional;
        used by the appendix weight tables, not by the core metrics.
    discrete_sl_predict:
        Discrete Super Learner predictions. Optional; computed by the original
        but unused in the returned metrics.
    outcome_var:
        Name of the outcome variable (the original carried this as the 2nd
        element of the (fit, var) list).
    method:
        Meta-learner method name; defaults to ``method.NNLS`` as in the original.
    """

    Y: np.ndarray
    sl_predict: np.ndarray
    library_predict: np.ndarray
    library_names: list[str]
    folds: list[np.ndarray]
    coef: Optional[np.ndarray] = None
    discrete_sl_predict: Optional[np.ndarray] = None
    outcome_var: Optional[str] = None
    method: str = "method.NNLS"

    @property
    def V(self) -> int:  # noqa: N802  (mirror R's `V`)
        return len(self.folds)

    def _learner_col(self, name: str) -> int:
        try:
            return self.library_names.index(name)
        except ValueError as exc:  # pragma: no cover - defensive
            raise KeyError(
                f"Learner {name!r} not found in library_names={self.library_names}"
            ) from exc


def _obs_weights(fit: CVSuperLearnerFit, obs_weights: Optional[np.ndarray]) -> np.ndarray:
    if obs_weights is None:
        return np.ones_like(fit.Y, dtype=float)
    return np.asarray(obs_weights, dtype=float)


def cv_predictive_r2(fit: CVSuperLearnerFit, obs_weights: Optional[np.ndarray] = None) -> dict:
    """Fold-wise predictive R^2 of the Super Learner vs the mean model.

    Port of ``get_cv_predictive_r2``: per fold, ``1 - Risk.SL / Risk.mean`` where
    ``Risk.SL`` is the SL MSE and ``Risk.mean`` is the ``SL.mean_All`` MSE.
    """
    if fit.method not in _NNLS_METHODS:
        raise NotImplementedError(f"method {fit.method!r} not supported (original only handled NNLS-family)")
    w = _obs_weights(fit, obs_weights)
    mean_col = fit._learner_col(MEAN_LEARNER)
    r2 = np.empty(fit.V)
    for i, idx in enumerate(fit.folds):
        risk_sl = np.mean(w[idx] * (fit.Y[idx] - fit.sl_predict[idx]) ** 2)
        risk_mean = np.mean(w[idx] * (fit.Y[idx] - fit.library_predict[idx, mean_col]) ** 2)
        r2[i] = 1.0 - risk_sl / risk_mean
    return {"mean_r2": float(r2.mean()), "min_r2": float(r2.min()), "max_r2": float(r2.max())}


def cv_lm_r2(fit: CVSuperLearnerFit, obs_weights: Optional[np.ndarray] = None) -> dict:
    """Fold-wise predictive R^2 of the linear model learner vs the mean model.

    Port of ``get_cv_lm_r2``: uses the ``SL.lm_All`` library column in place of
    the Super Learner predictions.
    """
    if fit.method not in _NNLS_METHODS:
        raise NotImplementedError(f"method {fit.method!r} not supported")
    w = _obs_weights(fit, obs_weights)
    mean_col = fit._learner_col(MEAN_LEARNER)
    lm_col = fit._learner_col(LM_LEARNER)
    r2 = np.empty(fit.V)
    for i, idx in enumerate(fit.folds):
        risk_lm = np.mean(w[idx] * (fit.Y[idx] - fit.library_predict[idx, lm_col]) ** 2)
        risk_mean = np.mean(w[idx] * (fit.Y[idx] - fit.library_predict[idx, mean_col]) ** 2)
        r2[i] = 1.0 - risk_lm / risk_mean
    return {"mean_r2": float(r2.mean()), "min_r2": float(r2.min()), "max_r2": float(r2.max())}


def cv_rmse(fit: CVSuperLearnerFit, obs_weights: Optional[np.ndarray] = None) -> dict:
    """Fold-wise RMSE of the Super Learner. Port of ``get_cv_rmse``."""
    if fit.method not in _NNLS_METHODS:
        raise NotImplementedError(f"method {fit.method!r} not supported")
    w = _obs_weights(fit, obs_weights)
    rmse = np.empty(fit.V)
    for i, idx in enumerate(fit.folds):
        rmse[i] = np.sqrt(np.mean(w[idx] * (fit.Y[idx] - fit.sl_predict[idx]) ** 2))
    return {"mean_rmse": float(rmse.mean()), "min_rmse": float(rmse.min()), "max_rmse": float(rmse.max())}


def cv_mad(fit: CVSuperLearnerFit, obs_weights: Optional[np.ndarray] = None) -> dict:
    """Fold-wise mean absolute deviation of the Super Learner. Port of ``get_cv_mad``.

    Note: the original applies the observation weight *inside* the absolute value,
    ``mean(abs(w * (Y - SL.predict)))``. Reproduced exactly (identical when w==1).
    """
    if fit.method not in _NNLS_METHODS:
        raise NotImplementedError(f"method {fit.method!r} not supported")
    w = _obs_weights(fit, obs_weights)
    mad = np.empty(fit.V)
    for i, idx in enumerate(fit.folds):
        mad[i] = np.mean(np.abs(w[idx] * (fit.Y[idx] - fit.sl_predict[idx])))
    return {"mean_mad": float(mad.mean()), "min_mad": float(mad.min()), "max_mad": float(mad.max())}


def _summary_risk_row(Y: np.ndarray, pred: np.ndarray, folds: list[np.ndarray],
                      w: np.ndarray) -> dict:
    """One row of ``summary.CV.SuperLearner(...)$Table`` for the NNLS family.

    R pkg: SuperLearner/R/summary.CV.SuperLearner.R:L39–43, L66 (2.0-40):
      * per fold, ``mean(w * (Y - pred)^2)``; Ave/Min/Max over the V folds (Ave is the
        plain, unweighted mean of the fold risks);
      * ``se = (1/sqrt(n)) * sd(w * (Y - pred)^2)`` over all observations, with R's
        ``sd`` (denominator n - 1) and ``n = length(SL.predict)`` (L19).
    Returned as ``mean_mse``/``se_mse``/``min_mse``/``max_mse``. R names the SE column
    ``se`` (PORTING_NOTES B4); the brief names it ``se_mse``.
    """
    risk = np.array([np.mean(w[idx] * (Y[idx] - pred[idx]) ** 2) for idx in folds])
    se = (1.0 / np.sqrt(len(pred))) * np.std(w * (Y - pred) ** 2, ddof=1)
    return {"mean_mse": float(risk.mean()), "se_mse": float(se),
            "min_mse": float(risk.min()), "max_mse": float(risk.max())}


def cv_mse(fit: CVSuperLearnerFit, obs_weights: Optional[np.ndarray] = None) -> dict:
    """Fold-wise MSE (mean/min/max) and SE of the Super Learner predictions.

    Port of the "Super Learner" row of ``summary(cv_fit)$Table``
    (R: llm_paper/R/functions.R:L704–705), computed from ``fit.sl_predict`` as given.
    Callers that follow ``get_cv_superlearner_metrics`` must pass the UNCLAMPED fit,
    because the R takes the summary before the outlier clamp on L707.
    """
    return _summary_risk_row(fit.Y, fit.sl_predict, fit.folds, _obs_weights(fit, obs_weights))


def cv_learner_mse(fit: CVSuperLearnerFit, learner: str,
                   obs_weights: Optional[np.ndarray] = None) -> dict:
    """Fold-wise MSE (mean/min/max) and SE of one library learner's predictions.

    Port of the ``<learner>`` row of ``summary(cv_fit)$Table``, e.g. ``SL.lm_All`` in
    ``get_cv_lm_metrics`` (R: llm_paper/R/functions.R:L731–732). Library predictions
    are never clamped in the R.
    """
    col = fit.library_predict[:, fit._learner_col(learner)]
    return _summary_risk_row(fit.Y, col, fit.folds, _obs_weights(fit, obs_weights))


def _winsorise_sl_predict(sl_predict: np.ndarray) -> np.ndarray:
    """Replicate the R wrapper's outlier clamp on SL predictions.

    Original line:
        cv_fit$SL.predict[abs(cv_fit$SL.predict) >
            10*sd(abs(cv_fit$SL.predict)) + mean(cv_fit$SL.predict)] <-
                mean(cv_fit$SL.predict)

    Faithfully reproduced, including the (slightly unusual) mixing of the SD of the
    absolute values with the mean of the raw values in the threshold. R's ``sd``
    uses the n-1 denominator, so ddof=1 is required to match.
    """
    sl = np.asarray(sl_predict, dtype=float).copy()
    m = float(np.mean(sl))
    threshold = 10.0 * float(np.std(np.abs(sl), ddof=1)) + m
    sl[np.abs(sl) > threshold] = m
    return sl


def superlearner_metrics(fit: CVSuperLearnerFit) -> dict:
    """Port of ``get_cv_superlearner_metrics`` (R: llm_paper/R/functions.R:L696–723).

    Order of operations, as in the R:
      1. MSE mean/min/max and ``se_mse`` from ``summary(cv_fit)`` on the UNCLAMPED
         Super Learner predictions (L704–705);
      2. the outlier clamp on ``SL.predict`` (L707, :func:`_winsorise_sl_predict`);
      3. R^2, MAD and RMSE from the CLAMPED predictions (L709–713).
    The R row also carries ``Algorithm = "Super Learner"``; it is not reproduced
    (PORTING_NOTES B4).
    """
    fit_w = _replace_sl_predict(fit, _winsorise_sl_predict(fit.sl_predict))
    row: dict = {}
    row.update(cv_mse(fit))  # unclamped (L704–705 run before the clamp on L707)
    row.update(cv_predictive_r2(fit_w))
    row.update(cv_mad(fit_w))
    row.update(cv_rmse(fit_w))
    row["var"] = fit.outcome_var
    row["n"] = int(len(fit.Y))
    return row


def lm_metrics(fit: CVSuperLearnerFit) -> dict:
    """Port of ``get_cv_lm_metrics`` (R: llm_paper/R/functions.R:L725–749).

      * MSE mean/min/max and ``se_mse`` from the ``SL.lm_All`` row of
        ``summary(cv_fit)`` (L731–732): the linear model alone, unclamped;
      * R^2 from ``SL.lm_All`` against ``SL.mean_All`` (``get_cv_lm_r2``, L736);
      * MAD and RMSE from the CLAMPED ensemble predictions (L734, L738–740).
    The R names the sample-size column ``length(cv_fit$Y)`` (L748) because
    ``mutate`` got an unnamed argument; the port keeps ``n`` (PORTING_NOTES B4).
    """
    fit_w = _replace_sl_predict(fit, _winsorise_sl_predict(fit.sl_predict))
    row: dict = {}
    row.update(cv_learner_mse(fit, LM_LEARNER))  # L731–732: the SL.lm_All row, unclamped
    row.update(cv_lm_r2(fit_w))
    row.update(cv_mad(fit_w))
    row.update(cv_rmse(fit_w))
    row["var"] = fit.outcome_var
    row["n"] = int(len(fit.Y))
    return row


def _replace_sl_predict(fit: CVSuperLearnerFit, new_sl: np.ndarray) -> CVSuperLearnerFit:
    """Return a shallow copy of ``fit`` with ``sl_predict`` replaced."""
    return CVSuperLearnerFit(
        Y=fit.Y,
        sl_predict=new_sl,
        library_predict=fit.library_predict,
        library_names=fit.library_names,
        folds=fit.folds,
        coef=fit.coef,
        discrete_sl_predict=fit.discrete_sl_predict,
        outcome_var=fit.outcome_var,
        method=fit.method,
    )
