"""R oracle backend: R's ``CV.SuperLearner`` through rpy2, on folds fixed from Python.

Purpose: the numerical reference for the native backend (VALIDATION_CHECKLIST V4). It
calls R's ``CV.SuperLearner`` with the library of the original model functions and
returns the same :class:`CVSuperLearnerFit` contract as the native backend, so the two
can be compared learner by learner and observation by observation.

Folds: ``cvControl = list(V, validRows = <outer folds>)`` fixes the outer folds
(R pkg: SuperLearner/R/control.R:L15–29, CVFolds.R:L13–15). To make the inner CV
identical as well, ``innerCvControl`` is a list of V control lists, each with its own
``validRows`` (CV.SuperLearner.R:L22–39; SuperLearner.R:L91). Indices are 1-based
in R; inner indices refer to the rows of that outer fold's training set, in their
original order, exactly as ``models.folds.train_indices`` orders them.

Requires R with SuperLearner and nnls (and ranger, nnet, kernlab, glmnet for the full
library), rpy2 via ``pip install -e '.[oracle]'``. R's xgboost is deliberately not
installed (docs/REFERENCE_SOURCES.md); with the full library, R's ``SL.xgboost.hist``
then fails inside ``try()`` and gets weight 0, which ``fit.failures`` does not report
(it only exists on the native side). Use it only where that is understood.
"""

from __future__ import annotations

import numpy as np

from ..metrics.cv_metrics import CVSuperLearnerFit
from ..rbridge import converter, require_r
from .folds import make_folds, train_indices

# R: llm_paper/R/functions.R:L514–519
_R_SUPERLEARNER_LIBRARY = """list("SL.mean",
     c("SL.ranger", "screen.glmnet"),
     c("SL.nnet", "screen.glmnet"),
     c("SL.xgboost.hist", "screen.glmnet"),
     c("SL.ksvm", "screen.glmnet"),
     c("SL.lm", "screen.glmnet"))"""
_R_LM_LIBRARY = 'list("SL.mean", c("SL.lm"))'  # R: llm_paper/R/functions.R:L543

_SUPERLEARNER_NAMES = [
    "SL.mean_All", "SL.ranger_screen.glmnet", "SL.nnet_screen.glmnet",
    "SL.xgboost.hist_screen.glmnet", "SL.ksvm_screen.glmnet", "SL.lm_screen.glmnet",
]
_LM_NAMES = ["SL.mean_All", "SL.lm_All"]


def _r_int_list(folds) -> str:
    """R code for a list of 1-based integer vectors."""
    return "list(" + ", ".join("c(" + ", ".join(str(int(i) + 1) for i in f) + ")" for f in folds) + ")"


def fit_r_cv_superlearner(
    X: np.ndarray,
    y: np.ndarray,
    *,
    which: str = "superlearner",
    outcome_var: str | None = None,
    outer_v: int = 10,
    inner_v: int = 5,
    seed: int = 1,
    folds: list[np.ndarray] | None = None,
    inner_folds: list[list[np.ndarray]] | None = None,
) -> CVSuperLearnerFit:
    """Fit R's ``CV.SuperLearner`` and return a :class:`CVSuperLearnerFit`.

    ``which``: ``"superlearner"`` (6-learner library) or ``"lm"`` (mean + lm).
    ``folds``: outer folds (0-based index arrays); default ``make_folds(n, outer_v, seed)``.
    ``inner_folds``: per outer fold, the inner folds (0-based indices into that fold's
    training rows); if None, R draws them after ``set.seed(seed)``.
    """
    require_r(("SuperLearner", "nnls"))
    import rpy2.robjects as ro

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if folds is None:
        folds = make_folds(n, outer_v, seed=seed)
    names = _SUPERLEARNER_NAMES if which == "superlearner" else _LM_NAMES
    library_r = _R_SUPERLEARNER_LIBRARY if which == "superlearner" else _R_LM_LIBRARY

    if inner_folds is None:
        inner_r = f"list(list(V = {int(inner_v)}))"
    else:
        for k, fold in enumerate(folds):  # inner indices must address the training rows
            n_tr = len(train_indices(n, fold))
            assert np.array_equal(np.sort(np.concatenate(inner_folds[k])), np.arange(n_tr))
        inner_r = "list(" + ", ".join(
            f"list(V = {len(f)}, validRows = {_r_int_list(f)})" for f in inner_folds) + ")"

    with converter():
        ro.globalenv["lcp_X"] = ro.r["as.data.frame"](X)
        ro.globalenv["lcp_y"] = ro.FloatVector(y)
        ro.r(f"set.seed({int(seed)})")
        ro.r(f"""
            suppressPackageStartupMessages(library(SuperLearner))
            SL.xgboost.hist <- function(...) SL.xgboost(..., params = list(tree_method = "hist"))
            lcp_fit <- suppressWarnings(CV.SuperLearner(
                Y = lcp_y, X = lcp_X, family = gaussian(),
                cvControl = list(V = {len(folds)}, validRows = {_r_int_list(folds)}),
                innerCvControl = {inner_r},
                SL.library = {library_r},
                saveAll = TRUE, verbose = FALSE))
            NULL
        """)
        out = {
            "SL.predict": ro.r("as.numeric(lcp_fit$SL.predict)"),
            "discreteSL.predict": ro.r("as.numeric(lcp_fit$discreteSL.predict)"),
            "library.predict": ro.r("unname(lcp_fit$library.predict)"),
            "library.names": ro.r("colnames(lcp_fit$library.predict)"),
            "coef": ro.r("unname(lcp_fit$coef)"),
            "cvRisk": ro.r("unname(t(sapply(lcp_fit$AllSL, function(s) s$cvRisk)))"),
        }
        ro.r("rm(lcp_X, lcp_y, lcp_fit)")

    r_names = [str(v) for v in out["library.names"]]
    if r_names != names:
        raise RuntimeError(f"R library names {r_names} differ from the expected {names}")
    return CVSuperLearnerFit(
        Y=y,
        sl_predict=np.asarray(out["SL.predict"], dtype=float),
        library_predict=np.asarray(out["library.predict"], dtype=float),
        library_names=names,
        folds=list(folds),
        coef=np.asarray(out["coef"], dtype=float),
        discrete_sl_predict=np.asarray(out["discreteSL.predict"], dtype=float),
        outcome_var=outcome_var,
        method="method.NNLS",
        cv_risk=np.asarray(out["cvRisk"], dtype=float).reshape(len(folds), len(names)),
    )
