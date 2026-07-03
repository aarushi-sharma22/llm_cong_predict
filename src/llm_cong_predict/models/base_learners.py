"""Base learners for the Super Learner library.

Each learner mirrors one of the R SuperLearner wrappers used in the original
model functions. The library and its column-name ordering match
``get_general_superlearner_cv_model`` (and the ``name_mapping`` in the original
``create_data.R``):

    SL.mean_All, SL.ranger_screen.glmnet, SL.nnet_screen.glmnet,
    SL.xgboost.hist_screen.glmnet, SL.ksvm_screen.glmnet, SL.lm_screen.glmnet

HONESTY ABOUT DEFAULTS (see docs/PORTING_NOTES.md, section C):
The paper relied on each wrapper's *default* hyperparameters. Those defaults
differ from scikit-learn's, and matching them exactly is what makes the native
backend numerically comparable to R. Below, each learner records the R default we
are targeting and a confidence tag:

  [match]     equivalent method; high confidence (SL.mean, SL.lm).
  [approx]    same family, but the underlying implementation differs enough that
              exact agreement is not expected (SL.nnet/nnet, SL.ksvm/kernlab,
              SL.ranger/ranger, SL.xgboost). Values marked ``# TO_VERIFY`` are our
              best current reading of the R default and MUST be confirmed against
              the SuperLearner wrapper source and, ultimately, the rpy2 oracle.

None of the [approx] learners can be claimed faithful until VALIDATION_CHECKLIST
V4 passes. This file sets defensible values and flags every uncertain one rather
than inventing precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


@dataclass(frozen=True)
class LearnerSpec:
    """A library entry: a name, a factory producing a fresh estimator, and the
    screener applied before fitting (``"screen.glmnet"`` or ``None`` for no
    screening, which the R side names ``_All``)."""

    name: str
    make: Callable[[int], BaseEstimator]
    screener: str | None

    def library_name(self) -> str:
        # SuperLearner names a (learner, screener) pair "<learner>_<screener>",
        # and uses "_All" when there is no screener.
        suffix = self.screener if self.screener is not None else "All"
        return f"{self.name}_{suffix}"


# --- individual learner factories -------------------------------------------

def _make_mean(seed: int) -> BaseEstimator:
    # SL.mean: predict the mean of Y, ignoring X. [match]
    return DummyRegressor(strategy="mean")


def _make_lm(seed: int) -> BaseEstimator:
    # SL.lm: ordinary least squares. [match] (both OLS)
    return LinearRegression()


def _make_ranger(seed: int) -> BaseEstimator:
    # SL.ranger -> R ranger. [approx]
    # ranger default num.trees = 500. mtry and min.node.size for *regression* are
    # NOT asserted here (ranger's regression defaults are easy to get wrong); left
    # at sklearn defaults and flagged.
    return RandomForestRegressor(
        n_estimators=500,          # ranger num.trees default
        # mtry:            # TO_VERIFY vs ranger regression default (floor(sqrt(p))?/p/3?)
        # min_node_size:   # TO_VERIFY vs ranger default (5 for regression?)
        random_state=seed,
        n_jobs=1,
    )


def _make_nnet(seed: int) -> BaseEstimator:
    # SL.nnet -> R nnet. [approx]
    # SL.nnet default size = 2 (two hidden units). nnet uses a single logistic
    # hidden layer trained with a BFGS-type optimiser; MLPRegressor is a different
    # implementation, so agreement is not expected even with matched size.
    return MLPRegressor(
        hidden_layer_sizes=(2,),   # SL.nnet size = 2
        activation="logistic",     # nnet hidden units are logistic
        solver="lbfgs",            # closest to nnet's optimiser
        max_iter=100,              # nnet default maxit = 100   # TO_VERIFY
        random_state=seed,
    )


def _make_ksvm(seed: int) -> BaseEstimator:
    # SL.ksvm -> kernlab ksvm (RBF). [approx]
    # kernlab estimates the RBF sigma via a median heuristic (sigest); sklearn's
    # gamma='scale' is a different rule, so results diverge. C=1 matches kernlab's
    # default cost.
    return SVR(
        kernel="rbf",
        C=1.0,                     # kernlab default C = 1
        gamma="scale",             # TO_VERIFY vs kernlab sigest median heuristic
    )


def _make_xgboost_hist(seed: int) -> BaseEstimator:
    # SL.xgboost.hist -> SL.xgboost with tree_method="hist". [approx]
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=1000,         # SL.xgboost default ntrees = 1000   # TO_VERIFY
        max_depth=4,               # SL.xgboost default max_depth = 4   # TO_VERIFY
        learning_rate=0.1,         # SL.xgboost default shrinkage = 0.1 # TO_VERIFY
        min_child_weight=10,       # approx SL.xgboost minobspernode=10 # TO_VERIFY
        tree_method="hist",        # the ".hist" override (this part is certain)
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


# --- library assembly --------------------------------------------------------

def superlearner_library() -> list[LearnerSpec]:
    """The 6-learner library of ``get_general_superlearner_cv_model``, in the same
    order as the original (so ``library_name()`` values match downstream code)."""
    return [
        LearnerSpec("SL.mean", _make_mean, None),
        LearnerSpec("SL.ranger", _make_ranger, "screen.glmnet"),
        LearnerSpec("SL.nnet", _make_nnet, "screen.glmnet"),
        LearnerSpec("SL.xgboost.hist", _make_xgboost_hist, "screen.glmnet"),
        LearnerSpec("SL.ksvm", _make_ksvm, "screen.glmnet"),
        LearnerSpec("SL.lm", _make_lm, "screen.glmnet"),
    ]


def lm_library() -> list[LearnerSpec]:
    """The 2-learner library of ``get_lm_cv_model``: mean + unscreened lm.

    Note the lm here has NO screener, so its library name is ``SL.lm_All`` -- which
    is exactly what ``get_cv_lm_metrics`` filters on in the original.
    """
    return [
        LearnerSpec("SL.mean", _make_mean, None),
        LearnerSpec("SL.lm", _make_lm, None),
    ]
