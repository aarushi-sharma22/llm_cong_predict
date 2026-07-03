"""Stata value-label handling — the ``haven`` / ``sjlabelled`` semantics.

In the R code, ``haven::read_dta`` returns *labelled* columns: numeric codes with
an attached ``value -> label`` map. Downstream the R uses:
  * ``haven::as_factor(x)``        -> replace numeric codes with their text labels;
  * ``sjlabelled::to_character(x)``-> the label text as a plain string;
  * ``sjlabelled::set_na(x, -99:-1)`` -> recode a range of codes to missing.

``pyreadstat`` exposes the same information via its ``meta`` object
(``variable_value_labels`` = ``{column: {value: label}}``). The readers in
``io/readers.py`` carry that dict on the DataFrame using ``df.attrs`` so the
cleaning layer can apply ``as_factor`` / ``to_character`` at exactly the points the
R does.

NOTE on ``df.attrs``: pandas does not always propagate ``attrs`` across operations,
so the readers re-attach after transforms, and the cleaning layer should read the
labels early (right after loading) rather than deep in a chain.
"""

from __future__ import annotations

import pandas as pd

VALUE_LABELS_KEY = "value_labels"
COLUMN_LABELS_KEY = "column_labels"


def get_value_labels(df: pd.DataFrame) -> dict[str, dict]:
    """Return the ``{column: {value: label}}`` map carried on ``df`` (or ``{}``)."""
    return df.attrs.get(VALUE_LABELS_KEY, {})


def attach_labels(
    df: pd.DataFrame,
    value_labels: dict[str, dict] | None = None,
    column_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Attach value/column label maps to ``df.attrs`` (in place) and return ``df``."""
    if value_labels is not None:
        df.attrs[VALUE_LABELS_KEY] = value_labels
    if column_labels is not None:
        df.attrs[COLUMN_LABELS_KEY] = column_labels
    return df


def set_na_range(df: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """Recode integer codes in the inclusive range ``[lo, hi]`` to missing.

    Port of ``sjlabelled::set_na(na = lo:hi)`` as used in ``read_ncds`` with
    ``-99:-1``. Applies element-wise across all columns; string columns and values
    outside the range are unaffected. Value labels are preserved on ``attrs``.
    """
    codes = list(range(lo, hi + 1))
    out = df.mask(df.isin(codes))
    out.attrs = dict(df.attrs)  # mask() drops attrs; restore
    return out


def as_factor(df: pd.DataFrame, col: str) -> pd.Categorical:
    """Port of ``haven::as_factor(df[[col]])``: numeric codes -> label categories.

    Values with no label map to ``NaN`` (as ``as_factor`` does when labels are
    partial). If the column has no attached labels, returns it as a plain category.
    """
    labels = get_value_labels(df).get(col, {})
    series = df[col]
    if not labels:
        return pd.Categorical(series)
    return pd.Categorical(series.map(labels))


def to_character(df: pd.DataFrame, col: str) -> pd.Series:
    """Port of ``sjlabelled::to_character(df[[col]])``: label text as a string.

    Used in ``clean_ncds`` to test membership against a set of "Dont know" /
    "Inapplicable" strings. Labelled values become their label; unlabelled values
    fall back to their string form (they simply won't match the missing-string set).
    """
    labels = get_value_labels(df).get(col, {})
    series = df[col]
    if not labels:
        return series.astype("string")
    mapped = series.map(labels)
    # Fall back to the raw value's string form where no label exists.
    fallback = series.astype("string")
    return mapped.astype("string").fillna(fallback)


def merge_value_labels(frames: list[pd.DataFrame]) -> dict[str, dict]:
    """Union the value-label maps carried on several frames (later frames win on
    key collisions). Used by ``combine_ncds`` so labels survive the join."""
    merged: dict[str, dict] = {}
    for f in frames:
        merged.update(get_value_labels(f))
    return merged
