"""Synthetic data fixtures for testing reader MECHANICS.

These generate structurally-valid dummy files (Stata .dta with value labels, essay
text files) that match the SHAPE of the real NCDS/CAMSIS inputs. They are NOT real
data and encode no real values or real variable meanings — they exist purely so the
parsing/reshaping code has something to parse. Numerical validation against the real
access-restricted data is deferred (docs/VALIDATION_CHECKLIST.md).
"""

from __future__ import annotations

import pandas as pd
import pyreadstat


def make_synthetic_ncds_dta(path: str) -> dict:
    """Write a small NCDS-shaped .dta with value labels and missing codes.

    Deliberately uses an UPPER-CASE column name (``N622``) to exercise the
    lower-casing in ``read_ncds``, and includes ``-1`` codes to exercise the
    ``set_na(-99:-1)`` recode. Returns the value-label dict used, for assertions.
    """
    df = pd.DataFrame(
        {
            "ncdsid": ["0001", "0002", "0003", "0004"],
            "N622": [1, 2, 1, -1],          # "sex", with one missing code (-1)
            "n876": [3, 4, -1, 5],          # a teacher rating, one missing
            "n2771": [10, 47, 20, -99],     # an aspiration code, one missing (-99)
        }
    )
    value_labels = {
        "N622": {1: "Male", 2: "Female"},
        "n876": {1: "Very poor", 2: "Poor", 3: "Average", 4: "Good", 5: "Very good"},
    }
    pyreadstat.write_dta(df, path, variable_value_labels=value_labels)
    return value_labels


def make_synthetic_camsis_dta(path: str) -> dict:
    """Write a small CAMSIS-shaped .dta matching the real column names."""
    df = pd.DataFrame(
        {
            "co1970": [1.0, 1.0, 2.0, 3.0],
            "stdempst": [0.0, 2.0, 0.0, 0.0],
            "mcamsis": [66.7, 69.9, 40.1, 55.2],
            "fcamsis": [62.2, 77.0, 44.3, 58.0],
        }
    )
    value_labels = {"co1970": {1: "Fishermen", 2: "Farmers", 3: "Agricultural workers"}}
    pyreadstat.write_dta(df, path, variable_value_labels=value_labels)
    return value_labels


def make_synthetic_essays(folder: str) -> list[dict]:
    """Write a few essay text files in the ``ID:/----/Words:`` format.

    Returns the expected parsed rows for assertions.
    """
    import os

    os.makedirs(folder, exist_ok=True)
    essays = [
        {"ncdsid": "0001", "text": "When i grow up i want to be a teacher.", "words": "8"},
        {"ncdsid": "0002", "text": "I will work on a farm with animals.", "words": "9"},
    ]
    dashes = "-" * 22
    for i, e in enumerate(essays, start=1):
        content = f"ID: {e['ncdsid']}\n{dashes}\n{e['text']}  Words: {e['words']}"
        with open(os.path.join(folder, f"essay_{i}.txt"), "w", encoding="utf-8") as fh:
            fh.write(content)
    return essays
