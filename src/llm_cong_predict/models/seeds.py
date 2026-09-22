"""Deterministic seed derivation for every random component of the Super Learner.

R's seeds cannot be reproduced: ``CV.SuperLearner`` draws folds in the master
process and the learners run on workers seeded by ``clusterSetRNGStream``
(R: llm_paper/R/functions.R:L506–509). The port therefore owns its randomness and
makes it independent of execution order and of ``n_jobs``: each random component gets
a seed derived only from the base seed, the outer fold, the inner fold and a
component name (a learner's library name, a screener name, or a folds label).
"""

from __future__ import annotations

import zlib

import numpy as np

FULL_FIT = -1  # "inner fold" index used for the refit on the whole training set
NO_FOLD = -2  # used where a component is not tied to a fold (e.g. outer-fold layout)


def derive_seed(base: int, outer: int, inner: int, name: str) -> int:
    """A 32-bit seed that depends only on ``(base, outer, inner, name)``.

    Uses ``numpy.random.SeedSequence`` with a spawn key built from the fold indices
    and a CRC-32 of the name (stable across processes and Python versions, unlike
    ``hash()``).
    """
    key = (outer - NO_FOLD, inner - NO_FOLD, zlib.crc32(name.encode("utf-8")))
    return int(np.random.SeedSequence(int(base), spawn_key=key).generate_state(1)[0])
