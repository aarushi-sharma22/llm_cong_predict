"""Data preparation applied to a sample inside a model target.

The R prepares the sample inside four kinds of model target; the steps are recorded in
``pipeline/model_spec.py::DATA_PREP`` and applied here:

  * ``as_numeric`` (R: llm_paper/_targets.R:L208, L258, L368, L374): R's
    ``as.numeric()`` — a factor gives its level position (``codes + 1``), a
    logical gives 0/1, a number stays itself, text is parsed as R parses it;
  * ``inner_join`` (L227, L230): ``dplyr::inner_join(table, by = c("ncdsid" = "id"))``;
  * ``drop_columns_starting_with`` (L230): ``select(-starts_with(prefix))``, where
    ``starts_with`` ignores case (tidyselect/R/helpers-pattern.R).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.labels import r_as_numeric


def r_as_numeric_column(col: pd.Series) -> pd.Series:
    """R's ``as.numeric()`` of one column."""
    if isinstance(col.dtype, pd.CategoricalDtype):
        codes = col.cat.codes.to_numpy(dtype=float) + 1.0  # level position
        codes[codes == 0] = np.nan  # code -1 = NA
        return pd.Series(codes, index=col.index, name=col.name)
    if pd.api.types.is_bool_dtype(col.dtype):
        return col.astype("Float64").astype(float)  # TRUE 1, FALSE 0, NA NaN
    if pd.api.types.is_numeric_dtype(col.dtype):
        return col.astype(float)
    return pd.Series(r_as_numeric(col).to_numpy(), index=col.index, name=col.name)


def apply_data_prep(sample: pd.DataFrame, steps, tables: dict[str, pd.DataFrame] | None = None,
                    variable_lists: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Apply the recorded data-preparation ``steps`` to ``sample`` in order.

    ``tables`` supplies joined frames by target name; ``variable_lists`` supplies the
    columns named by a variable-list target (``columns_from``).
    """
    out = sample
    for step in steps:
        op = step["op"]
        if op == "as_numeric":
            cols = step.get("columns") or list((variable_lists or {})[step["columns_from"]])
            out = out.copy()
            for c in cols:
                out[c] = r_as_numeric_column(out[c])
        elif op == "inner_join":
            ((left, right),) = step["by"].items()
            table = (tables or {})[step["table"]]
            out = out.merge(table, left_on=left, right_on=right, how="inner")
            if right != left and right in out.columns:
                out = out.drop(columns=[right])  # dplyr keeps only x's key column
        elif op == "drop_columns_starting_with":
            prefix = step["prefix"].lower()
            out = out.loc[:, [c for c in out.columns if not str(c).lower().startswith(prefix)]]
        else:  # pragma: no cover - the spec only records these three operations
            raise ValueError(f"unknown data-preparation step {op!r}")
    return out
