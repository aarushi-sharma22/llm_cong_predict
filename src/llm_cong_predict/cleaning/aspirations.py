"""Occupational-aspiration -> CAMSIS scoring.

Faithful port of ``create_aspirations`` from ``R/functions.R``. This one is fully
grounded: it uses raw NCDS codes that are explicit in the R (``n2771`` aspiration,
``n622`` sex) and the real CAMSIS columns (``co1970``/``mcamsis``/``fcamsis``/
``stdempst``), plus the hand-crafted occupation mapping that ships in the repo.

VALIDATION STATUS: logic is a faithful translation and tested for mechanics on
synthetic frames. End-to-end numerical validation waits on the real NCDS data
(and on ``clean_ncds``, since the aspiration frame is joined into the cleaned data
downstream). See docs/VALIDATION_CHECKLIST.md (V5).
"""

from __future__ import annotations

import pandas as pd

from ..io.labels import as_factor, get_value_labels


def create_aspirations(
    ncds_complete: pd.DataFrame,
    camsis: pd.DataFrame,
    occupation_aspiration_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Port of ``create_aspirations(ncds_complete, camsis, occupation_aspiration_mapping)``.

    R logic reproduced step for step:

      aspiration_camsis <- camsis %>%
        select(occupation_1970 = co1970, camsis_male = mcamsis,
               camsis_female = fcamsis, stdempst) %>%
        filter(stdempst == 0) %>%                       # employment status unknown
        mutate(occupation_1970 = as_factor(occupation_1970)) %>%
        select(-stdempst) %>% unique() %>%
        full_join(occupation_aspiration_mapping) %>% na.omit()

      ncds_complete %>%
        select(ncdsid, aspiration_n2771 = n2771, sex = n622) %>%
        mutate(aspiration_n2771 = as.character(as_factor(aspiration_n2771)),
               sex = as.character(as_factor(sex))) %>%
        left_join(aspiration_camsis) %>%
        mutate(camsis = ifelse(sex == 1, camsis_male, camsis_female)) %>%
        select(ncdsid,
               s2_co_aspiration_camsis_male = camsis_male,
               s2_co_aspiration_camsis_female = camsis_female,
               s2_co_aspiration_camsis = camsis)

    Notes on faithful translation:
      * ``co1970`` is converted to its text label (``haven::as_factor``) so it can be
        merged with the mapping's ``occupation_1970`` label column.
      * the merge with the mapping is on the shared column ``occupation_1970``
        (a ``full_join`` in R; dropped to matched rows by the subsequent ``na.omit``).
      * the sex-specific CAMSIS score: the R compares the *character* ``sex`` to the
        literal ``1``. That comparison is reproduced exactly, including its quirk
        (see the flag below).
    """
    # --- build the occupation -> CAMSIS lookup ------------------------------
    camsis_lut = camsis.loc[
        :, ["co1970", "mcamsis", "fcamsis", "stdempst"]
    ].rename(
        columns={"co1970": "occupation_1970", "mcamsis": "camsis_male", "fcamsis": "camsis_female"}
    )
    # employment status unknown (stdempst == 0)
    camsis_lut = camsis_lut[camsis_lut["stdempst"] == 0].copy()
    # co1970 -> its factor label (needs the value labels carried by read_camsis)
    camsis_lut["occupation_1970"] = as_factor(camsis, "co1970")[camsis_lut.index].astype("string")
    camsis_lut = camsis_lut.drop(columns=["stdempst"]).drop_duplicates()

    merged_lut = camsis_lut.merge(
        occupation_aspiration_mapping, on="occupation_1970", how="outer"
    ).dropna()

    # --- score each respondent's aspiration ---------------------------------
    asp = ncds_complete.loc[:, ["ncdsid", "n2771", "n622"]].rename(
        columns={"n2771": "aspiration_n2771", "n622": "sex"}
    )
    # as.character(as_factor(...)) for both the aspiration code and sex
    asp["aspiration_n2771"] = as_factor(ncds_complete, "n2771").astype("string")
    asp["sex"] = as_factor(ncds_complete, "n622").astype("string")

    scored = asp.merge(merged_lut, on="aspiration_n2771", how="left")

    # sex-specific joint score. FAITHFUL to the R: it tests the *character* sex
    # against the literal 1 (`ifelse(sex == 1, ...)`). See flag E-cleaning-1.
    male = scored["sex"] == "1"
    scored["camsis"] = scored["camsis_female"]
    scored.loc[male, "camsis"] = scored.loc[male, "camsis_male"]

    return scored.loc[:, ["ncdsid", "camsis_male", "camsis_female", "camsis"]].rename(
        columns={
            "camsis_male": "s2_co_aspiration_camsis_male",
            "camsis_female": "s2_co_aspiration_camsis_female",
            "camsis": "s2_co_aspiration_camsis",
        }
    )
