"""The mapping table at the top of ``R/create_data.R`` and the learner-name table.

R: llm_paper/R/create_data.R:L11–44 (``mapping``) and L504–517 (``name_mapping``).
Both are copied verbatim, including the two things that matter downstream:

  * ``category_name`` has no ``TRUE ~`` fallback (L31–34), so the four ``confounder``
    rows get NA. A ``filter(category == ...)`` in dplyr drops NA rows, and
    ``filter(category != ...)`` drops them too — :func:`dplyr_filter_equals` and
    :func:`dplyr_filter_not_equals` reproduce that, which pandas does not do by itself;
  * the age suffix comes from the variable's sweep prefix (L35–43), e.g.
    ``s2_co_reading_ability`` -> "Reading Ability (Age 11)".

Variables that are not in the table (the Big Five outcomes, for example) get NA for
``name`` and ``category``, as the R's ``left_join`` gives them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# R: llm_paper/R/create_data.R:L13–28 (variable, category, name), verbatim.
MAPPING_ROWS: tuple[tuple[str, str, str], ...] = (
    ("s2_co_factor_ability", "cognitive_abilities", "General Factor of Cognitive Ability"),
    ("s2_co_reading_ability", "cognitive_abilities", "Reading Ability"),
    ("s2_co_verbal_ability", "cognitive_abilities", "Verbal Ability"),
    ("s2_co_nonverbal_ability", "cognitive_abilities", "Nonverbal Ability"),
    ("s2_co_mathematics_ability", "cognitive_abilities", "Mathematical Ability"),
    ("s3_co_reading_ability", "cognitive_abilities", "Reading Ability"),
    ("s3_co_mathematics_ability", "cognitive_abilities", "Mathematical Ability"),
    ("s3_co_factor_scholastic_motivation", "motivation_aspiration", "Scholastic Motivation"),
    ("s2_co_aspiration_camsis", "motivation_aspiration", "Occupational Aspirations"),
    ("s5_co_highest_edu", "life_outcomes", "Highest Education"),
    ("s3_te_factor_externalizing", "personality", "Externalizing Behavior"),
    ("s3_te_factor_internalizing", "personality", "Internalizing Behavior"),
    ("s3_pa_edu", "confounder", "Parental SES"),
    ("s3_co_height", "confounder", "Height"),
    ("s0_co_male", "confounder", "Sex"),
    ("s0_mo_birthweight", "confounder", "Birthweight"),
)

# R: llm_paper/R/create_data.R:L31–34 — case_when with no TRUE fallback, so "confounder" is NA.
CATEGORY_NAMES: dict[str, str] = {
    "cognitive_abilities": "Cognitive Abilities",
    "motivation_aspiration": "Non-cognitive Traits",
    "personality": "Non-cognitive Traits",
    "life_outcomes": "Life Outcomes",
}

# R: llm_paper/R/create_data.R:L35–43 — the sweep prefix gives the age in the label.
AGE_BY_SWEEP: dict[str, str] = {
    "s2": "(Age 11)", "s3": "(Age 16)", "s4": "(Age 23)", "s5": "(Age 33)",
    "s6": "(Age 42)", "s7": "(Age 46)", "s8": "(Age 50)", "s9": "(Age 55)",
}

# R: llm_paper/R/create_data.R:L504–517 — library name -> the name used in appendix D3.
LEARNER_NAMES: dict[str, str] = {
    "SL.mean_All": "Mean",
    "SL.ranger_screen.glmnet": "Random_Forest",
    "SL.nnet_screen.glmnet": "Neural_Network",
    "SL.xgboost.hist_screen.glmnet": "XGBoost",
    "SL.ksvm_screen.glmnet": "SVM",
    "SL.lm_screen.glmnet": "Linear_Model",
}


def mapping_table() -> pd.DataFrame:
    """``mapping`` (R: llm_paper/R/create_data.R:L11–44): variable, category, name
    (with the age suffix) and category_name (NA for the confounders)."""
    frame = pd.DataFrame(MAPPING_ROWS, columns=["variable", "category", "name"])
    frame["category_name"] = frame["category"].map(CATEGORY_NAMES)  # NA where no case matches
    suffix = frame["variable"].str[:2].map(AGE_BY_SWEEP)
    frame["name"] = np.where(suffix.notna(), frame["name"] + " " + suffix.fillna(""), frame["name"])
    return frame


def with_labels(frame: pd.DataFrame, var_column: str = "var") -> pd.DataFrame:
    """``left_join(select(mapping, var = variable, name, category = category_name))``
    (R: llm_paper/R/create_data.R:L66 and the same line in every table)."""
    labels = mapping_table().loc[:, ["variable", "name", "category_name"]].rename(
        columns={"variable": var_column, "category_name": "category"})
    return frame.merge(labels, on=var_column, how="left")


def dplyr_filter_equals(frame: pd.DataFrame, column: str, value) -> pd.DataFrame:
    """``dplyr::filter(column == value)``: rows where the comparison is TRUE. A missing
    value compares to NA, which dplyr drops and pandas would keep."""
    return frame[frame[column].notna() & (frame[column] == value)]


def dplyr_filter_not_equals(frame: pd.DataFrame, column: str, value) -> pd.DataFrame:
    """``dplyr::filter(column != value)``: rows where the comparison is TRUE, so rows
    with a missing value are dropped here too."""
    return frame[frame[column].notna() & (frame[column] != value)]
