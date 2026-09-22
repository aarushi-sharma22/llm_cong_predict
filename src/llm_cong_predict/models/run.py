"""Model runner: ports of ``get_general_superlearner_cv_model`` and ``get_lm_cv_model``.

R (llm_paper/R/functions.R:L496–524 and L526–548):

    data <- dplyr::select(data, outcome_var, predictors) %>% na.omit()
    y_train <- dplyr::pull(data, outcome_var)
    x_train <- dplyr::select(data, predictors)
    colnames(x_train) <- make.names(colnames(x_train))
    cv_sl <- CV.SuperLearner(Y = y_train, X = x_train, family = gaussian(),
                             cvControl = list(V = 10), innerCvControl = list(list(V = 5)),
                             SL.library = <6 learners | mean + lm>, ...)
    return(list(fit = cv_sl, var = outcome_var))

``make.names`` only changes column names; the port keeps the original names.
Per-person predictions are participant-level data and are written only under
``$LCP_DATA_ROOT/fits/`` (:func:`save_predictions`).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..metrics.cv_metrics import CVSuperLearnerFit
from .base_learners import lm_library, superlearner_library
from .native_superlearner import default_folds, fit_cv_superlearner


class NonNumericPredictorError(TypeError):
    """A predictor (or the outcome) is not numeric after data preparation."""


def _is_plain_numeric(dtype) -> bool:
    return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)


def fit_model(
    outcome_var: str,
    predictors,
    data: pd.DataFrame,
    method: str,
    backend: str = "native",
    seed: int = 1,
    n_jobs: int = 1,
    *,
    outer_v: int | None = None,
    inner_v: int | None = None,
) -> tuple[CVSuperLearnerFit, str]:
    """Fit one model as the R model functions do; returns ``(fit, outcome_var)``.

    * select the outcome and the predictors (a missing column raises, as
      ``dplyr::select`` with names does); a name listed twice is selected once;
    * drop rows with a missing value among them (``na.omit``);
    * refuse, with :class:`NonNumericPredictorError`, any predictor that is not
      integer or float (booleans, categories and text included): after the recorded
      data preparation (models/data_prep.py) every predictor should be numeric, so
      nothing is coerced silently. A non-numeric outcome is refused too (R:
      SuperLearner.R:L73–75 "the outcome Y must be a numeric vector");
    * fit with ``method`` "superlearner" (6 learners, functions.R:L514–519) or "lm"
      (mean + lm, L543) on the ``backend`` "native" or "r", both given the same outer
      and inner folds (``native_superlearner.default_folds``).

    ``outer_v``/``inner_v`` default to 10 and 5 (functions.R:L512, L541); only the
    smoke configuration changes them. The fit carries the ``ncdsid`` of its
    rows in ``fit.ids`` when ``data`` has an ``ncdsid`` column.
    """
    cfg = config.SUPERLEARNER if method == "superlearner" else config.LM
    if method not in ("superlearner", "lm"):
        raise ValueError(f"method must be 'superlearner' or 'lm', got {method!r}")
    outer_v = cfg.outer_folds if outer_v is None else outer_v
    inner_v = cfg.inner_folds if inner_v is None else inner_v

    predictors = list(dict.fromkeys(predictors))
    columns = list(dict.fromkeys([outcome_var, *predictors]))
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise KeyError(f"fit_model: column(s) {missing} do not exist in the sample "
                       "(dplyr::select stops here)")
    bad = {c: str(data[c].dtype) for c in predictors if not _is_plain_numeric(data[c].dtype)}
    if bad:
        raise NonNumericPredictorError(
            f"fit_model: non-numeric predictor(s) {bad}. After the documented data preparation "
            "every predictor should be numeric; nothing is coerced here (PORTING_NOTES L1).")
    if not _is_plain_numeric(data[outcome_var].dtype):
        raise NonNumericPredictorError(
            f"fit_model: the outcome {outcome_var!r} has dtype {data[outcome_var].dtype}; the R "
            "stops: 'the outcome Y must be a numeric vector' (SuperLearner.R:L73-75)")

    complete = data.loc[:, columns].notna().all(axis=1).to_numpy()  # na.omit
    if not complete.any():
        raise ValueError(f"fit_model: no complete rows for outcome {outcome_var!r}")
    y = data.loc[complete, outcome_var].to_numpy(dtype=float)
    X = data.loc[complete, predictors].to_numpy(dtype=float)
    ids = data.loc[complete, "ncdsid"].to_numpy() if "ncdsid" in data.columns else None

    folds, inner = default_folds(len(y), outer_v, inner_v, seed)
    if backend == "native":
        library = superlearner_library() if method == "superlearner" else lm_library()
        fit = fit_cv_superlearner(X, y, library, outcome_var=outcome_var, outer_v=outer_v,
                                  inner_v=inner_v, seed=seed, n_jobs=n_jobs, folds=folds,
                                  inner_folds=inner)
    elif backend == "r":
        from .r_superlearner import fit_r_cv_superlearner

        fit = fit_r_cv_superlearner(X, y, which=method, outcome_var=outcome_var, seed=seed,
                                    folds=folds, inner_folds=inner)
    else:
        raise ValueError(f"backend must be 'native' or 'r', got {backend!r}")
    fit.ids = ids
    return fit, outcome_var


_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.]+$")


def save_predictions(fit: CVSuperLearnerFit, target: str, prefix: str = "", label: str = "") -> Path:
    """Write the per-person cross-validated predictions of ``fit`` to
    ``$LCP_DATA_ROOT/fits/<prefix><target>__<outcome>.csv`` and return the path.

    Participant-level output: there is deliberately no way to choose another location
    Columns: ``ncdsid`` (if known), ``fold`` (1-based outer fold),
    ``Y``, ``SL.predict`` and one column per library learner. A non-empty ``label``
    (the smoke configuration's, config.SMOKE_RUN) is added as a first column
    ``run_label``, and ``prefix`` starts the file name, so such a file cannot be taken
    for a real result.
    """
    for part in (target, str(fit.outcome_var)) + ((prefix,) if prefix else ()):
        if not _SAFE_NAME.match(part):
            raise ValueError(f"unsafe name for a file: {part!r}")
    fold = np.empty(len(fit.Y), dtype=int)
    for k, idx in enumerate(fit.folds, start=1):
        fold[idx] = k
    table = pd.DataFrame({"fold": fold, "Y": fit.Y, "SL.predict": fit.sl_predict})
    for j, name in enumerate(fit.library_names):
        table[name] = fit.library_predict[:, j]
    if fit.ids is not None:
        table.insert(0, "ncdsid", fit.ids)
    if label:
        table.insert(0, "run_label", label)
    path = config.fits_dir() / f"{prefix}{target}__{fit.outcome_var}.csv"
    table.to_csv(path, index=False)
    return path
