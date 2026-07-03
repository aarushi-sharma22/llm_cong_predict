"""Factor-score construction (``psych::fa``).

Port of ``create_factors`` from ``R/functions.R``. Builds four single-factor scores
from grounded, hard-coded variable lists (the lists are explicit in the R):

  * ``s2_co_factor_ability``              (4 vars, Pearson correlation)   cor = "cor"
  * ``s3_co_factor_scholastic_motivation`` (4 vars, polychoric)          cor = "poly"
  * ``s3_te_factor_internalizing``        (5 vars, polychoric)           cor = "poly"
  * ``s3_te_factor_externalizing``        (11 vars, polychoric)          cor = "poly"

!!! NUMERICAL PARITY IS NOT ESTABLISHED HERE — see VALIDATION_CHECKLIST V3 !!!

Two reasons this is the least-certain port in the project and is explicitly flagged:

1. ``psych::fa`` defaults must be matched, not assumed. The R relies on ``psych``
   defaults: factoring method ``fm = "minres"`` (OLS/minimum-residual) and
   regression-based (Thurstone) factor scores. We set ``method="minres"`` in
   ``factor_analyzer`` to match the factoring method. Score computation differs
   between packages and is part of what V3 checks.

2. Polychoric correlation must be computed by hand (no standard dependency provides
   a psych-matching one). Three of the four
   factors use ``cor="poly"``. To reproduce them we would compute a polychoric
   correlation matrix separately and pass it via ``is_corr_matrix=True``. Polychoric
   implementations differ across packages, so this is the most likely source of
   divergence. Rather than silently substitute a Pearson correlation (which would
   change the numbers while looking fine), the polychoric path is implemented behind
   an explicit flag and, until V3 validates it against R's ``psych::fa``, the honest
   status is: mechanism ported, numbers UNVERIFIED.

Factor scores are identified only up to sign and scale, so the V3 oracle compares
via ABSOLUTE correlation (a sign flip is a match, not a failure). If parity cannot
be reached, the documented fallback is an rpy2 path calling ``psych::fa`` directly
for this step (same tradeoff as the SuperLearner oracle).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Grounded factor definitions (verbatim from the R). type is the psych `cor` arg:
# "cor" = Pearson, "poly" = polychoric.
FACTOR_DEFINITIONS: dict[str, dict] = {
    "s2_co_factor_ability": {
        "vars": ["s2_co_verbal_ability", "s2_co_nonverbal_ability",
                 "s2_co_reading_ability", "s2_co_mathematics_ability"],
        "cor": "cor",
    },
    "s3_co_factor_scholastic_motivation": {
        "vars": ["s3_co_school_waste_of_time", "s3_co_homework_a_bore",
                 "s3_co_never_take_work_seriously", "s3_co_not_like_schol"],
        "cor": "poly",
    },
    "s3_te_factor_internalizing": {
        "vars": ["s3_te_worried", "s3_te_solitary", "s3_te_miserable",
                 "s3_te_fearful", "s3_te_cries_in_school"],
        "cor": "poly",
    },
    "s3_te_factor_externalizing": {
        "vars": ["s3_te_restlessness", "s3_te_squirmy", "s3_te_destructive",
                 "s3_te_fight_others", "s3_te_irritable", "s3_te_disobedient",
                 "s3_te_cannot_settle", "s3_te_lying", "s3_te_steals",
                 "s3_te_resentful", "s3_te_bully"],
        "cor": "poly",
    },
}


def _single_factor_score(data: pd.DataFrame, cor: str) -> np.ndarray:
    """Fit a 1-factor model and return regression-based scores for ``data``.

    Targets ``psych::fa(data, 1, cor = <cor>)$score``. Implemented directly in
    numpy/scipy rather than via ``factor_analyzer`` because ``factor_analyzer``
    0.5.1 (the latest release) is incompatible with the installed scikit-learn (it
    calls the removed ``force_all_finite`` argument). Doing the factor math here has
    the side benefit of being auditable for a replication, but it is still
    PARITY-UNVERIFIED against ``psych`` (V3): psych's minres implementation and score
    computation differ in detail.
    """
    X = data.to_numpy(dtype=float)

    if cor == "cor":
        R = _pearson_corr(X)
    elif cor == "poly":
        R = _polychoric_corr(X)  # raises until V3 (see below)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown correlation type {cor!r}")

    loadings = _minres_one_factor(R)
    return _regression_scores(X, loadings, R).ravel()


def _pearson_corr(X: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix, pairwise-complete (``use="pairwise"`` style)."""
    df = pd.DataFrame(X)
    return df.corr(method="pearson").to_numpy()


def _minres_one_factor(R: np.ndarray, max_iter: int = 500, tol: float = 1e-8) -> np.ndarray:
    """Minimum-residual (OLS) single-factor loadings.

    minres finds loadings ``L`` (a vector, for one factor) minimising the sum of
    squared off-diagonal residuals of ``R - L L'``. Standard iterative scheme:
    repeatedly put the current communalities on the diagonal and take the leading
    eigenpair. This mirrors ``psych``'s default ``fm = "minres"`` in spirit; exact
    agreement is a V3 question.
    """
    p = R.shape[0]
    Rc = R.copy()
    comm = np.full(p, 0.5)  # initial communalities
    prev = None
    for _ in range(max_iter):
        np.fill_diagonal(Rc, comm)
        vals, vecs = np.linalg.eigh(Rc)
        lead = np.argmax(vals)
        lam = max(vals[lead], 0.0)
        loadings = vecs[:, lead] * np.sqrt(lam)
        comm = loadings ** 2
        if prev is not None and np.max(np.abs(comm - prev)) < tol:
            break
        prev = comm.copy()
    # sign convention: make the sum of loadings non-negative (psych orients too;
    # factor sign is arbitrary and V3 compares via |correlation|).
    if loadings.sum() < 0:
        loadings = -loadings
    return loadings


def _polychoric_corr(X: np.ndarray) -> np.ndarray:
    """Polychoric correlation matrix.

    NOT YET IMPLEMENTED TO PARITY (V3). A faithful polychoric estimator matching R's
    ``polycor``/``psych`` is nontrivial. Rather than silently substitute Pearson
    (which would change the numbers while looking fine), this raises until V3
    provides either a validated estimator or an rpy2 ``psych::fa`` fallback.
    """
    raise NotImplementedError(
        "Polychoric correlation is not yet implemented to parity with psych. "
        "The three 'poly' factors (scholastic_motivation, internalizing, "
        "externalizing) are blocked on VALIDATION_CHECKLIST V3: either a validated "
        "polychoric estimator or an rpy2 psych::fa fallback. The Pearson factor "
        "(s2_co_factor_ability) is computed; the poly factors are not, by design, "
        "rather than guessed. See docs/PORTING_NOTES.md."
    )


def _regression_scores(X: np.ndarray, loadings: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Regression (Thurstone) factor scores: standardise X, then X_std @ inv(R) @ loadings."""
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0, ddof=0)
    Xs = (X - mean) / std
    Xs = np.nan_to_num(Xs, nan=0.0)  # mean-impute standardised missings for scoring
    weights = np.linalg.pinv(corr) @ loadings
    return Xs @ weights


@dataclass
class FactorResult:
    scores: pd.DataFrame                      # ncdsid + one column per computed factor
    computed: list[str] = field(default_factory=list)   # factors actually produced
    deferred: list[str] = field(default_factory=list)   # factors blocked (poly, V3)


def create_factors(ncds_cleaned: pd.DataFrame, include_polychoric: bool = False) -> FactorResult:
    """Port of ``create_factors(ncds_cleaned)``.

    R:
        for each factor def: ncds_cleaned %>% select(vars) %>% psych::fa(1, cor=type)
        then $score; column-bind all four; prepend ncdsid.

    Because the three polychoric factors are not yet validated (V3), this returns a
    ``FactorResult`` that separates what was actually computed (the Pearson ability
    factor) from what is deferred. Set ``include_polychoric=True`` only once V3 has a
    validated estimator/fallback; until then it will raise for the poly factors,
    deliberately, rather than emit unverified numbers.
    """
    out = pd.DataFrame({"ncdsid": ncds_cleaned["ncdsid"].values})
    computed: list[str] = []
    deferred: list[str] = []

    for name, spec in FACTOR_DEFINITIONS.items():
        if spec["cor"] == "poly" and not include_polychoric:
            deferred.append(name)
            continue
        sub = ncds_cleaned.loc[:, spec["vars"]]
        out[name] = _single_factor_score(sub, spec["cor"])
        computed.append(name)

    return FactorResult(scores=out, computed=computed, deferred=deferred)
