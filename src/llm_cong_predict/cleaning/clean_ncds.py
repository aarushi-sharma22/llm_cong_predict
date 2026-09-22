"""``clean_ncds``: recode and rename the combined NCDS data (brief F2).

Port of ``clean_ncds(ncds_complete, mapping_df)`` (R: llm_paper/R/functions.R:L64–242).
The published function cannot run: the public ``variables.xlsx`` has no header row
(PORTING_NOTES A1), ``select(-s2_co_total_ability)`` names a column the table never
creates (L202), and the final join refers to four undefined blocks (L224–226). This
module is a RECONSTRUCTION of the evident intent (PORTING_NOTES A2); every other step
follows the R line cited next to it.

Representation of R types in pandas:
  * an R factor built by ``factor(x)`` / ``factor(x, levels, ordered = TRUE)`` is a
    ``pd.Categorical`` whose categories are the R levels in order, so ``as.numeric()``
    of it (the pipeline's data preparation) is the level position = ``codes + 1``;
  * an R logical with NA is pandas' nullable ``boolean`` dtype.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..io.labels import VALUE_LABELS_KEY, observed_rank_codes, r_number_string, to_character

logger = logging.getLogger(__name__)

# R: llm_paper/R/functions.R:L81–90, verbatim, including the duplicated entries and the
# leading space in " Cant say,inappl".
MISSING_LABELS = (
    "Dont know", "Inapplicable", "Not known",
    "Not answered", "Do not know, DNA",
    "Do not know", "Refused", "Not applicable",
    "Self completion qnaire not completed",
    "Item not applicable", "Misrouted - incomplete interview",
    "N/a: proxy/block not entered",
    "No usual pay", " Cant say,inappl",
    "Cant say", "Inapplicable", "Self completion qnaire not completed",
    "Unclassifiable", "Too vague",
    "Imprecise",
)

# R: llm_paper/R/functions.R:L106–114 — closed ranges, first match wins, else NA.
NSSEC_RANGES = (
    (1.0, 2.0, 1), (3.1, 3.4, 2), (4.1, 6.0, 3), (7.1, 7.4, 4),
    (8.1, 9.2, 5), (10.0, 11.2, 6), (12.1, 12.7, 7), (13.1, 13.5, 8),
)

# R: llm_paper/R/functions.R:L116–125 — "< bound" tested in order, first match wins.
EDU_LEVELS = ("No Qualifications", "Lower Secondary", "Upper Secondary", "Degree")
EDU_BOUNDS = ((4, "No Qualifications"), (6, "Lower Secondary"), (8, "Upper Secondary"), (11, "Degree"))

# R: llm_paper/R/functions.R:L167–189, in this order.
TEACHER_COLUMNS = (
    "s2_te_general_knowledge", "s2_te_number_work", "s2_te_use_of_books", "s2_te_oral_ability",
    "s2_te_poor_hand_control", "s2_te_squirmy", "s2_te_poor_coordination",
    "s2_te_hardly_ever_still", "s2_te_poor_speech", "s2_te_inconsequential_behavior",
    "s2_te_miscellneous_symptoms", "s2_te_anxiety_adults", "s2_te_anxiety_children",
    "s2_te_hostility_children", "s2_te_writing_off_adults", "s2_te_misc",
    "s2_te_hostility_adults", "s2_te_restlessness", "s2_te_unforthcomingness",
    "s2_te_depression", "s2_te_withdrawal",
)


class MissingColumnError(KeyError):
    """A column the R selects by name is absent; dplyr stops with an error here."""


def _require(data: pd.DataFrame, columns, where: str) -> None:
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise MissingColumnError(f"clean_ncds: column(s) {missing} are absent; the R selects "
                                 f"them by name and stops here ({where})")


def variables_table(mapping_df: pd.DataFrame) -> pd.DataFrame:
    """R: llm_paper/R/functions.R:L75–78 — ``full_name = paste0("s", sweep, "_",
    substr(respondent, 1, 2), "_", new_varname)`` and ``variable = tolower(variable)``.
    ``paste0`` prints the double sweep as R does (0, not 0.0)."""
    out = mapping_df.copy()
    out["full_name"] = ("s" + out["sweep"].map(r_number_string) + "_"
                        + out["respondent"].astype(str).str[:2] + "_" + out["new_varname"].astype(str))
    out["variable"] = out["variable"].astype(str).str.lower()
    return out


def blank_missing_labels(ncds_complete: pd.DataFrame) -> pd.DataFrame:
    """R: llm_paper/R/functions.R:L80–90 — ``mutate_all(function(x)
    ifelse(sjlabelled::to_character(x) %in% <MISSING_LABELS>, NA, x))``.

    Only values whose LABEL is in the list become NA; unlabelled values never match
    (``to_character`` gives NA for them, io/labels.py). ``ifelse`` returns a plain vector,
    so every value label is gone afterwards and is never re-applied."""
    out = ncds_complete.copy()
    for col in out.columns:
        hit = to_character(ncds_complete, col).isin(MISSING_LABELS).fillna(False).to_numpy(dtype=bool)
        if hit.any():
            out[col] = out[col].mask(hit)
    out.attrs = {k: v for k, v in ncds_complete.attrs.items() if k != VALUE_LABELS_KEY}
    return out


def select_and_rename(data: pd.DataFrame, variables: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """R: llm_paper/R/functions.R:L91–95 — ``select(ncdsid, one_of(variables$variable))``
    then rename each column from ``variable`` to ``full_name`` (first match).

    ``one_of`` keeps the table's order and drops absent names with a warning
    (tidyselect/R/helpers.R:L124–127); the absent codes are returned and logged."""
    codes = list(variables["variable"])
    absent = [c for c in codes if c not in data.columns]
    if absent:
        logger.warning("clean_ncds: %d code(s) of variables.xlsx are absent from the data: %s",
                       len(absent), absent)
    present = [c for c in dict.fromkeys(codes) if c in data.columns and c != "ncdsid"]
    first_full = dict(zip(variables["variable"][::-1], variables["full_name"][::-1]))  # first match
    out = data.loc[:, ["ncdsid"] + present].rename(columns={c: first_full[c] for c in present})
    return out, absent


def blank_year_month_codes(data: pd.DataFrame) -> pd.DataFrame:
    """R: llm_paper/R/functions.R:L97–98 — 9998 and 9999 become NA in columns whose name
    ends in "year" or "month" (``ends_with`` ignores case). No such column exists in
    the public table; the step is kept."""
    out = data.copy()
    for col in out.columns:
        if col.lower().endswith(("year", "month")):
            out[col] = out[col].mask(out[col].isin([9998, 9999]))
    return out


def recode_nssec(x: pd.Series) -> pd.Series:
    """R: llm_paper/R/functions.R:L104–115 — closed ranges, gaps and everything else NA."""
    v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    out = np.full(v.shape, np.nan)
    for lo, hi, code in NSSEC_RANGES:
        hit = np.isnan(out) & (v >= lo) & (v <= hi)
        out[hit] = code
    return pd.Series(out, index=x.index)


def recode_parent_education(age_left: pd.Series) -> pd.Categorical:
    """R: llm_paper/R/functions.R:L116–129 — ``as.numeric(x) < 4`` No Qualifications,
    ``< 6`` Lower Secondary, ``< 8`` Upper Secondary, ``< 11`` Degree, else NA; first
    match wins; an ordered factor with the levels in that order."""
    v = pd.to_numeric(age_left, errors="coerce").to_numpy(dtype=float)
    labels = np.full(v.shape, None, dtype=object)
    for bound, name in EDU_BOUNDS:
        hit = (labels == None) & (v < bound)  # noqa: E711 — element-wise "not yet assigned"
        labels[hit] = name
    return pd.Categorical(labels, categories=list(EDU_LEVELS), ordered=True)


def combine_parent_education(mother: pd.Categorical, father: pd.Categorical) -> pd.Categorical:
    """R: llm_paper/R/functions.R:L130–145.

    ``pmin``/``pmax`` of the two ordered factors (NA if either is NA,
    r-source/src/library/base/R/pmax.R:L101), then:
      NoQual/NoQual 1, NoQual/Lower 2, Lower/Lower 3, (<Upper)/Upper 4, Upper/Upper 5,
      (<Degree)/Degree 6, Degree/Degree 7, else NA;
    then ``factor(s3_pa_edu)``: levels are the sorted distinct OBSERVED values, so
    ``as.numeric()`` later gives the level position, not the value (brief F2)."""
    m = np.asarray(mother.codes, dtype=float)
    f = np.asarray(father.codes, dtype=float)
    m[m < 0] = np.nan
    f[f < 0] = np.nan
    lo, hi = np.fmin(m, f), np.fmax(m, f)  # NaN handled below: NA if either is NA
    lo[np.isnan(m) | np.isnan(f)] = np.nan
    hi[np.isnan(m) | np.isnan(f)] = np.nan
    nq, ls, us, dg = 0, 1, 2, 3
    rules = (
        ((lo == nq) & (hi == nq), 1), ((lo == nq) & (hi == ls), 2), ((lo == ls) & (hi == ls), 3),
        ((lo < us) & (hi == us), 4), ((lo == us) & (hi == us), 5), ((lo < dg) & (hi == dg), 6),
        ((lo == dg) & (hi == dg), 7),
    )
    value = np.full(m.shape, np.nan)
    for cond, code in rules:  # case_when: first TRUE wins
        value[np.isnan(value) & cond] = code
    observed = np.unique(value[~np.isnan(value)])
    return pd.Categorical(value, categories=observed)


def _block(data: pd.DataFrame, variables: pd.DataFrame, types) -> pd.DataFrame:
    """``select(ncdsid, one_of(filter(variables, type %in% types)$full_name))``."""
    names = [n for n in variables.loc[variables["type"].isin(types), "full_name"] if n in data.columns]
    return data.loc[:, ["ncdsid"] + names]


def clean_ncds(ncds_complete: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """Port of ``clean_ncds`` (R: llm_paper/R/functions.R:L64–242; brief F2).

    Returns one row per ``ncdsid`` with the columns of sex, birthweight, height,
    teacher, parents, personality, behavior, ability, motivation and highest_edu, in
    that order (L228–239). ``attrs`` carries ``clean_ncds_absent_codes`` (codes of the
    table missing from the data) and ``clean_ncds_collisions`` (for each of the nine
    teacher columns that also appear in the ability or behavior block, whether the
    rank-coded teacher version and the raw version are identical).
    """
    variables = variables_table(mapping_df)  # L75–78
    data = blank_missing_labels(ncds_complete)  # L80–90
    data, absent = select_and_rename(data, variables)  # L91–95
    data = blank_year_month_codes(data)  # L97–98
    if data["ncdsid"].duplicated().any():
        raise ValueError("clean_ncds: ncdsid is not unique in the combined data")

    # ---- parents, L102–145
    parents = _block(data, variables, ["parental class"]).copy()  # L103
    for col in [c for c in parents.columns if "nssec" in c.lower()]:  # contains() ignores case
        parents[col] = recode_nssec(parents[col])  # L104–115
    _require(data, ["s3_pa_age_edu_left_mother", "s3_pa_age_edu_left_father"], "functions.R:L116-125")
    parents["s3_pa_mother_edu"] = recode_parent_education(parents["s3_pa_age_edu_left_mother"])
    parents["s3_pa_father_edu"] = recode_parent_education(parents["s3_pa_age_edu_left_father"])
    parents["s3_pa_edu"] = combine_parent_education(
        parents["s3_pa_mother_edu"].array, parents["s3_pa_father_edu"].array)

    # ---- sex, L150–152: ifelse(s0_co_sex == 1, TRUE, FALSE), NA stays NA
    _require(data, ["s0_co_sex"], "functions.R:L151")
    sex = data.loc[:, ["ncdsid"]].copy()
    sex["s0_co_male"] = pd.array(np.where(data["s0_co_sex"].isna(), None, data["s0_co_sex"] == 1),
                                 dtype="boolean")

    # ---- height, L156–157; birthweight, L161–162
    _require(data, ["s3_co_height"], "functions.R:L157")
    height = data.loc[:, ["ncdsid", "s3_co_height"]]
    _require(data, ["s0_mo_birthweight"], "functions.R:L162")
    birthweight = data.loc[:, ["ncdsid", "s0_mo_birthweight"]]

    # ---- teacher, L166–191: labels are gone, so haven::as_factor is factor(x)
    # (forcats/R/as_factor.R:L52–54) and ifelse(x == "Dont know", NA, x) returns the
    # integer codes: the rank among the column's sorted distinct observed values.
    _require(data, TEACHER_COLUMNS, "functions.R:L167-189")
    teacher = data.loc[:, ["ncdsid", *TEACHER_COLUMNS]].copy()
    for col in TEACHER_COLUMNS:
        teacher[col] = observed_rank_codes(teacher[col])

    behavior = _block(data, variables, ["behavior"])  # L195–196
    ability = _block(data, variables, ["ability"])  # L200–201
    # L202 select(-s2_co_total_ability): that name is never created by the table, so
    # the R stops here; reconstruction: drop it only when present (PORTING_NOTES A2).
    ability = ability.drop(columns=[c for c in ["s2_co_total_ability"] if c in ability.columns])
    personality = _block(data, variables, ["personality"])  # L206–207
    motivation = _block(data, variables, ["academic_motivation"])  # L211–212
    highest_edu = _block(data, variables, ["education"])  # L216–217

    # ---- L224–240: plyr::join_all(type = "full") keeps the FIRST occurrence of a column
    # present in several blocks (io/readers.py::_plyr_full_join); the join list starts
    # with teacher, so its rank-coded columns win. The undefined blocks bsag,
    # aspirations, parenting and camsis are left out (reconstruction, PORTING_NOTES A2).
    # All blocks have the same rows, so the join is row-aligned.
    join_order = [teacher, parents, height, birthweight, behavior, ability, personality,
                  motivation, highest_edu, sex]
    first_source: dict[str, pd.Series] = {}
    for block in join_order:
        for col in block.columns:
            first_source.setdefault(col, block[col])
    select_order = [sex, birthweight, height, teacher, parents, personality, behavior, ability,
                    motivation, highest_edu]
    columns = list(dict.fromkeys(["ncdsid"] + [c for b in select_order for c in b.columns]))
    out = pd.DataFrame({c: first_source[c] for c in columns}, index=data.index).reset_index(drop=True)

    collisions = {}
    for col in TEACHER_COLUMNS:
        for other_name, other in (("ability", ability), ("behavior", behavior)):
            if col in other.columns:
                a, b = teacher[col].to_numpy(dtype=float), other[col].to_numpy(dtype=float)
                same = bool(np.array_equal(a, b, equal_nan=True))
                collisions[col] = {"other_block": other_name, "identical": same,
                                   "cells_differing": int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())}
    for col, rec in collisions.items():
        logger.info("clean_ncds: %s appears in teacher and %s; the teacher (rank-coded) version is "
                    "kept; identical to the raw version: %s", col, rec["other_block"], rec["identical"])
    out.attrs = {"clean_ncds_absent_codes": absent, "clean_ncds_collisions": collisions}
    return out
