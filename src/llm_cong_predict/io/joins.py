"""dplyr-style joins without ``by``: join on every column the two frames share.

R pkg: dplyr/R/join.R:L624 and dplyr/R/join-by.R:L376–379 (1.2.1): when ``by`` is
not given, ``join_by_common`` uses ``intersect(names(x), names(y))`` and dplyr prints
"Joining with `by = join_by(...)`". With no shared column dplyr stops with an error.
Missing key values match each other (dplyr's default ``na_matches = "na"``), as in
pandas. Row order follows ``x``; several matches in ``y`` repeat the ``x`` row.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def natural_join(x: pd.DataFrame, y: pd.DataFrame, how: str, step: str = "") -> tuple[pd.DataFrame, list[str]]:
    """``dplyr::<how>_join(x, y)`` without ``by``; returns ``(joined, keys)``.

    The keys are logged, because a column that happens to exist in both frames (for
    example a metric two tools both report) silently becomes a join key.
    """
    keys = [c for c in x.columns if c in y.columns]
    if not keys:
        raise ValueError(f"{step}: `by` must be supplied when `x` and `y` have no common variables "
                         "(dplyr join without by)")
    logger.info("natural %s join%s by %s", how, f" ({step})" if step else "", keys)
    return x.merge(y, on=keys, how=how), keys
