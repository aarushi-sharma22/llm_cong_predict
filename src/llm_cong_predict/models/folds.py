"""Deterministic cross-validation fold assignment.

Why this exists as its own module: the whole point of the rpy2 oracle (see
docs/VALIDATION_CHECKLIST.md, V4) is to compare the native backend against R's
SuperLearner *on identical folds*. R's internal fold RNG
(``clusterSetRNGStream``) cannot be reproduced in Python, so if each backend
picked its own folds, any numerical comparison would be confounded by RNG
differences and would tell us nothing about whether the ensemble math agrees.

So folds are generated *here*, once, and fed to BOTH backends. This module does
not attempt to replicate R's default fold RNG -- it deliberately owns fold
assignment so both sides see the same partition.
"""

from __future__ import annotations

import numpy as np


def make_folds(n: int, v: int, seed: int) -> list[np.ndarray]:
    """Partition ``range(n)`` into ``v`` test folds (a shuffled, near-equal split).

    Parameters
    ----------
    n:
        Number of observations.
    v:
        Number of folds.
    seed:
        RNG seed controlling the shuffle. The original used seed 1
        (``clusterSetRNGStream(cluster, 1)``); we reuse that value by default at
        the call sites, but the fold *layout* here is our own, shared contract.

    Returns
    -------
    list of np.ndarray
        ``v`` arrays of row indices; together they partition ``range(n)`` with no
        overlap and no omission.
    """
    if v < 2:
        raise ValueError(f"need at least 2 folds, got v={v}")
    if n < v:
        raise ValueError(f"cannot make {v} folds from {n} observations")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    # np.array_split gives near-equal sizes when n is not divisible by v.
    return [np.sort(fold) for fold in np.array_split(perm, v)]


def train_indices(n: int, test_fold: np.ndarray) -> np.ndarray:
    """Complement of a test fold: all indices in ``range(n)`` not in ``test_fold``."""
    mask = np.ones(n, dtype=bool)
    mask[test_fold] = False
    return np.nonzero(mask)[0]
