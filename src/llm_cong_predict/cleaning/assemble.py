"""Assembly and complete-case subsetting.

Faithful ports of ``get_complete_ncds``, ``find_full_overlap`` and
``find_essay_teacher_genetics_overlap`` from ``R/functions.R``.

VALIDATION STATUS: ``find_full_overlap`` and ``get_complete_ncds`` are generic
join/subset logic and tested for mechanics on synthetic frames.
``find_essay_teacher_genetics_overlap`` depends on cleaned/derived columns, so its
end-to-end behaviour is validated only once ``clean_ncds`` and real data exist.
"""

from __future__ import annotations

from functools import reduce

import pandas as pd

from ..io.labels import as_factor


def get_complete_ncds(
    ncds_cleaned: pd.DataFrame,
    ncds_factors: pd.DataFrame,
    ncds_aspirations: pd.DataFrame,
    ncds_essay: pd.DataFrame,
    ncds_gene: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Port of ``get_complete_ncds``.

    R:
        ncds_cleaned %>%
          left_join(ncds_factors) %>%
          left_join(ncds_aspirations) %>%
          left_join(ncds_essay)

    Successive left joins onto the cleaned frame. All joins are on the shared key
    ``ncdsid`` (the only common column across these frames).

    NOTE on the gene argument (PORTING_NOTES A3): the R *definition* takes 4 args
    but ``_targets.R`` *calls* it with a 5th (``gene_data``), which R silently drops.
    Here the gene frame is an explicit optional argument; when provided it is joined
    on ``ncdsid`` too, when ``None`` the behaviour matches the 4-arg R exactly.
    """
    frames = [ncds_cleaned, ncds_factors, ncds_aspirations, ncds_essay]
    if ncds_gene is not None:
        frames.append(ncds_gene)
    return reduce(lambda left, right: left.merge(right, on="ncdsid", how="left"), frames)


def find_full_overlap(ncds_complete: pd.DataFrame, varlist: list[str]) -> pd.DataFrame:
    """Port of ``find_full_overlap(ncds_complete, varlist)``.

    R:
        ncds_complete %>% select(ncdsid, varlist) %>% na.omit()

    Keep ``ncdsid`` plus the requested variables, then drop any row with a missing
    value among them (complete-case subset).
    """
    cols = ["ncdsid"] + [c for c in varlist if c != "ncdsid"]
    return ncds_complete.loc[:, cols].dropna()


# The raw teacher-evaluation N-codes used by the extended-teacher overlap. These
# are grounded: they appear explicitly in the R and correspond to the age-11
# teacher ratings (n876-n885).
_EXTENDED_TEACHER_CODES = ["n876", "n877", "n878", "n879", "n880", "n881", "n882", "n883", "n884", "n885"]

_ABILITY_NONMISSING = [
    "s2_co_factor_ability", "s2_co_verbal_ability", "s2_co_nonverbal_ability",
    "s2_co_reading_ability", "s2_co_mathematics_ability",
]


def find_essay_teacher_genetics_overlap(
    ncds_complete_genetics: pd.DataFrame,
    ncds_essays: pd.DataFrame,
    ncds_1_to_9: pd.DataFrame,
) -> pd.DataFrame:
    """Port of ``find_essay_teacher_genetics_overlap``.

    R:
        extended_teacher_evaluations <- ncds_1_to_9 %>%
          select(ncdsid, n876:n885) %>%
          mutate_if(is.numeric, as_factor) %>%
          mutate_all(~ ifelse(. == "Dont know", NA, .)) %>%
          na.omit()

        ncds_complete_genetics %>%
          filter(!is.na(<the five ability vars>)) %>%
          inner_join(ncds_essays) %>%
          inner_join(extended_teacher_evaluations)

    Faithful translation. The raw teacher codes are converted to their labels, the
    "Dont know" label is set to missing, and complete cases are kept; the result is
    inner-joined (on ``ncdsid``) with essays and the ability-complete genetics frame.
    """
    ext = ncds_1_to_9.loc[:, ["ncdsid"] + _EXTENDED_TEACHER_CODES].copy()
    for code in _EXTENDED_TEACHER_CODES:
        labelled = pd.Series(as_factor(ncds_1_to_9, code).astype("string"), index=ncds_1_to_9.index)
        labelled = labelled.where(labelled != "Dont know")
        ext[code] = labelled.values
    ext = ext.dropna()

    base = ncds_complete_genetics.dropna(subset=_ABILITY_NONMISSING)
    return base.merge(ncds_essays, on="ncdsid", how="inner").merge(ext, on="ncdsid", how="inner")
