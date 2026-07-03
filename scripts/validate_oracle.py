#!/usr/bin/env python
"""Oracle validation: measure native-vs-R agreement on IDENTICAL folds.

Run this ON A MACHINE WITH R (the dev sandbox has none). It is the concrete check
behind VALIDATION_CHECKLIST V4. It does not assume the backends agree -- it
measures the gap and reports it as numbers, so any divergence is quantified rather
than hand-waved.

Prerequisites:
    pip install -e '.[oracle]'
    # and R with: SuperLearner, ranger, nnet, xgboost, kernlab, glmnet

Usage:
    python scripts/validate_oracle.py                 # synthetic data, lm library
    python scripts/validate_oracle.py --library full  # full 6-learner library (slow)

What it reports, per library-learner column and for the ensemble:
    * max |native - R| prediction difference across observations
    * correlation between native and R predictions
    * the fold-wise R^2/RMSE/MAD from each backend, side by side

Interpretation: SL.mean and SL.lm should agree very closely (they are [match]
learners). The tree/net/svm learners are [approx] and will differ; this script
tells you by how much, which is exactly the information needed to decide whether
the native backend is an acceptable default or whether rpy2 should be the default
for the affected outcomes.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _synthetic(n: int = 160, p: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.array([2.0, -1.5, 1.0] + [0.0] * (p - 3))
    y = X @ beta + rng.normal(scale=0.5, size=n)
    return X, y


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", choices=["lm", "full"], default="lm",
                    help="which library to compare (default: lm, fast)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="max abs prediction diff to consider a [match] learner OK")
    args = ap.parse_args()

    from llm_cong_predict.models.folds import make_folds
    from llm_cong_predict.models.native_superlearner import fit_cv_superlearner
    from llm_cong_predict.models.base_learners import lm_library, superlearner_library
    from llm_cong_predict.metrics.cv_metrics import superlearner_metrics

    try:
        from llm_cong_predict.models.r_superlearner import fit_r_cv_superlearner
    except ImportError as exc:
        print(f"[oracle unavailable] {exc}")
        return 2

    X, y = _synthetic(seed=args.seed)
    n = len(y)
    folds = make_folds(n, 10, seed=args.seed)  # SAME folds for both backends

    if args.library == "lm":
        native = fit_cv_superlearner(X, y, lm_library(), outcome_var="y", seed=args.seed)
        r_which = "lm"
    else:
        native = fit_cv_superlearner(X, y, superlearner_library(), outcome_var="y", seed=args.seed)
        r_which = "superlearner"

    print("Fitting R oracle (this calls CV.SuperLearner)...")
    r_fit = fit_r_cv_superlearner(X, y, which=r_which, outcome_var="y", seed=args.seed, folds=folds)

    assert native.library_names == r_fit.library_names, (
        f"library name mismatch: {native.library_names} vs {r_fit.library_names}"
    )

    print("\n=== per-learner prediction agreement (native vs R) ===")
    print(f"{'learner':<34}{'max|diff|':>12}{'corr':>10}")
    all_ok = True
    for j, name in enumerate(native.library_names):
        a = native.library_predict[:, j]
        b = r_fit.library_predict[:, j]
        max_diff = float(np.max(np.abs(a - b)))
        corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
        flag = ""
        if name in ("SL.mean_All", "SL.lm_All", "SL.lm_screen.glmnet"):
            ok = max_diff <= args.tol
            all_ok &= ok
            flag = "  <- [match] " + ("OK" if ok else "OUT OF TOLERANCE")
        print(f"{name:<34}{max_diff:>12.3e}{corr:>10.4f}{flag}")

    ens_diff = float(np.max(np.abs(native.sl_predict - r_fit.sl_predict)))
    print(f"\nensemble SL.predict max|diff|: {ens_diff:.3e}")

    print("\n=== fold-wise metrics, side by side ===")
    mn, mr = superlearner_metrics(native), superlearner_metrics(r_fit)
    for key in ("mean_r2", "mean_rmse", "mean_mad", "mean_mse"):
        print(f"  {key:<10}  native={mn[key]:+.5f}   R={mr[key]:+.5f}   diff={mn[key]-mr[key]:+.2e}")

    print("\nSummary:", "match-learners within tolerance."
          if all_ok else "SOME MATCH-LEARNERS OUT OF TOLERANCE -- investigate before trusting native.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
