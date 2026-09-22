"""Stata value-label handling — the ``haven`` / ``sjlabelled`` / ``forcats`` semantics.

In the R code, ``haven::read_dta`` returns *labelled* columns: numeric codes with an
attached ``value -> label`` map. Downstream the R uses:
  * ``haven::as_factor(x)`` on a labelled column -> a factor whose levels are every
    label plus every observed unlabelled value, sorted by the underlying value;
  * ``haven::as_factor(x)`` on a plain numeric column (labels gone), which dispatches
    to ``forcats::as_factor.numeric`` = ``factor(x)``: levels are the sorted distinct
    observed values;
  * ``ifelse(x == "Dont know", NA, x)`` on such a factor, which returns the factor's
    INTEGER CODES (the position in the level set), not the labels;
  * ``sjlabelled::to_character(x)`` -> the label text as a plain string;
  * ``sjlabelled::set_na(x, na = -99:-1)`` -> recode a range of codes to missing and
    drop their value labels.

``pyreadstat`` exposes the labels via ``meta.variable_value_labels``
(``{column: {value: label}}``). The readers in ``io/readers.py`` carry that dict on
``df.attrs['value_labels']``. pandas does not always propagate ``df.attrs`` across
operations, so read labels early in a chain.
"""

from __future__ import annotations

import re

import numpy as np
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
    """Port of ``sjlabelled::set_na(na = lo:hi)`` as used in ``read_ncds`` (-99:-1).

    R pkg: sjlabelled/R/set_na.R (1.2.0):
      * values in ``lo:hi`` become NA (L258–263);
      * the value labels of those values are removed (L268–272), so they do not
        become factor levels later (haven::as_factor);
      * a column that is entirely missing is returned untouched, labels included (L177).
    String columns and values outside the range are unaffected.
    """
    codes = list(range(lo, hi + 1))
    out = df.mask(df.isin(codes))
    attrs = dict(df.attrs)
    labels = dict(attrs.get(VALUE_LABELS_KEY, {}))
    for col, lab in labels.items():
        if col in df.columns and df[col].notna().any():
            labels[col] = {v: t for v, t in lab.items() if not (lo <= v <= hi)}
    if VALUE_LABELS_KEY in attrs:
        attrs[VALUE_LABELS_KEY] = labels
    out.attrs = attrs  # mask() drops attrs; restore
    return out


def r_number_string(value: float) -> str:
    """``as.character()`` of a double, as R prints it: at most 15 significant digits,
    fixed notation unless scientific notation is strictly shorter (R's default
    ``scipen = 0``), e.g. 1 -> "1", 20.5 -> "20.5", 1e5 -> "1e+05".
    # APPROX: follows R's width rule for the common cases (integers, short decimals);
    # R's exact digit selection for long decimals was not re-derived.
    """
    v = float(value)
    if v.is_integer() and abs(v) < 1e15:
        fixed = str(int(v))
    else:
        fixed = format(v, ".15g")
        if "e" in fixed:
            fixed = np.format_float_positional(v, precision=15, unique=True, trim="-")
    mantissa, _, exp = format(v, ".15e").partition("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    sci = f"{mantissa}e{int(exp):+03d}"
    return sci if len(sci) < len(fixed) else fixed


_R_NUMBER = re.compile(r"^[+-]?(?:(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|Inf|inf|NaN)$")
_R_HEX = re.compile(r"^[+-]?0[xX][0-9a-fA-F]+$")


def r_as_numeric(values) -> pd.Series:
    """``as.numeric()`` of character values, as R converts them: leading and trailing
    whitespace ignored; decimal, scientific, hexadecimal (``0x1A`` -> 26), ``Inf`` and
    ``NaN`` accepted; anything else NA (R warns). Checked against R 4.6.1 on the cases
    in tests/test_io.py. ``pd.to_numeric`` differs on hexadecimal and would accept
    ``1_000``-style forms through Python's ``float``.
    # APPROX: R's full grammar (e.g. "infinity", hexadecimal fractions) is not covered.
    """
    s = pd.Series(values, dtype="object")

    def one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        t = str(v).strip()
        if _R_NUMBER.match(t):
            return float(t)
        if _R_HEX.match(t):
            return float(int(t, 16))
        return np.nan

    return s.map(one).astype(float)


def haven_levels(values: pd.Series, labels: dict) -> list[str]:
    """Levels of ``haven::as_factor(x, levels = "default")`` for a labelled column.

    R pkg: haven/R/as_factor.R:L74–78 (2.5.5): observed values (labelled ones named by
    their label, unlabelled ones by ``as.character(value)``) plus every label, sorted
    by the underlying value, then unique names. Labels that never occur are levels.
    """
    observed = [float(v) for v in pd.unique(values.dropna())]
    named = [(v, labels[v] if v in labels else r_number_string(v)) for v in observed]
    named += [(float(v), str(t)) for v, t in labels.items()]
    named.sort(key=lambda pair: pair[0])  # stable: equal values keep their order
    return list(dict.fromkeys(name for _, name in named))


def as_factor(df: pd.DataFrame, col: str) -> pd.Categorical:
    """Port of ``haven::as_factor(df[[col]])``.

    Labelled column (labels on ``df.attrs``): ``as_factor.haven_labelled`` with the
    default ``levels = "default"`` (R pkg: haven/R/as_factor.R:L62–84). Values without
    a label KEEP their value (as text), they are not set to missing.
    Unlabelled numeric column: ``forcats::as_factor.numeric`` = ``factor(x)``
    (R pkg: forcats/R/as_factor.R:L52–54): levels are the sorted distinct values.
    """
    series = df[col]
    labels = get_value_labels(df).get(col, {})
    if not labels:
        levels = [r_number_string(v) for v in np.sort(pd.unique(series.dropna()).astype(float))]
        text = series.map(lambda v: r_number_string(v) if pd.notna(v) else np.nan)
        return pd.Categorical(text, categories=list(dict.fromkeys(levels)))
    text = series.map(lambda v: (labels[v] if v in labels else r_number_string(v)) if pd.notna(v) else np.nan)
    return pd.Categorical(text, categories=haven_levels(series, labels))


def factor_codes(factor: pd.Categorical) -> np.ndarray:
    """What ``ifelse(test, NA, x)`` returns for a factor ``x``: its 1-based integer codes
    (R: base ifelse assigns the factor into a logical vector, keeping the codes;
    r-source/src/library/base/R/ifelse.R:L46–55). Missing -> NaN."""
    codes = np.asarray(factor.codes, dtype=float) + 1.0
    codes[codes == 0] = np.nan
    return codes


def labelled_factor_codes(df: pd.DataFrame, col: str, missing_labels: tuple[str, ...] = ()) -> np.ndarray:
    """``ifelse(haven::as_factor(x) %in% missing_labels, NA, haven::as_factor(x))``:
    integer codes = position in :func:`haven_levels` (labelled column) or in the sorted
    observed values (unlabelled column), with the given labels set to NaN."""
    fac = as_factor(df, col)
    codes = factor_codes(fac)
    if missing_labels:
        codes[np.isin(np.asarray(fac.astype(object)), list(missing_labels))] = np.nan
    return codes


def observed_rank_codes(values: pd.Series) -> np.ndarray:
    """Integer codes of ``factor(x)`` for a plain numeric vector: the rank of each value
    among the sorted distinct observed values, starting at 1 (R pkg:
    forcats/R/as_factor.R:L52–54; r-source/src/library/base/R/factor.R:L26–38).
    Used where labels are already gone (clean_ncds' teacher block, brief F2)."""
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    distinct = np.unique(v[~np.isnan(v)])
    out = np.full(v.shape, np.nan)
    ok = ~np.isnan(v)
    out[ok] = np.searchsorted(distinct, v[ok]) + 1.0
    return out


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
    """Union of the value-label maps carried on several frames. For a column present in
    more than one frame the FIRST frame's labels are kept, as in ``plyr::join_all``,
    whose ``rbind.fill`` takes each column's attributes from its first occurrence
    (R pkg: plyr/R/rbind-fill.r:L70–71, L80; 1.8.9)."""
    merged: dict[str, dict] = {}
    for f in frames:
        for col, lab in get_value_labels(f).items():
            merged.setdefault(col, lab)
    return merged
