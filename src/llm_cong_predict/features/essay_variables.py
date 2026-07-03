"""Assemble essay-derived variables into one feature frame.

Faithful port of ``create_essay_variables`` from ``R/functions.R``.

R:
    salat_metrics %>%
      left_join(readability_metrics) %>%
      left_join(spelling_errors) %>%
      left_join(embeddings, by = c("ncdsid" = "id")) %>%
      select(-filename) %>%
      mutate_at(vars(-ncdsid), as.numeric) %>%
      select_if(function(x) is.character(x) ||
                (all(!is.na(x) & max(x) != Inf & min(x) != -Inf & var(x) > 0)))

NAMING NOTE (PORTING_NOTES A4): the R parameter is called ``roberta_embeddings`` but
the pipeline (`_targets.R`) passes ``gpt_embeddings`` into it. The embedding source
is therefore made an explicit argument named ``embeddings`` here.

DATA-DEPENDENT FILTER (flagged): the final ``select_if`` keeps a column only if it
is character, or has no NAs, finite range, and non-zero variance. WHICH columns
survive depends on the actual data, so the essay feature-matrix width cannot be
fixed a priori (matches the R). Tested for mechanics on synthetic frames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_essay_variables(
    salat_metrics: pd.DataFrame,
    readability_metrics: pd.DataFrame,
    spelling_errors: pd.DataFrame,
    embeddings: pd.DataFrame,
) -> pd.DataFrame:
    """Join the four essay-feature sources and drop degenerate columns.

    Joins follow the R: natural joins (shared columns) for salat/readability/
    spelling, and an explicit ``ncdsid = id`` join for the embeddings. Then every
    non-``ncdsid`` column is coerced numeric, ``filename`` is dropped, and columns
    that are all-NA / non-finite / zero-variance are removed.
    """
    out = salat_metrics
    out = out.merge(readability_metrics, on=_shared(out, readability_metrics), how="left")
    out = out.merge(spelling_errors, on=_shared(out, spelling_errors), how="left")
    out = out.merge(embeddings.rename(columns={"id": "ncdsid"}), on="ncdsid", how="left")

    if "filename" in out.columns:
        out = out.drop(columns=["filename"])

    # coerce all-but-ncdsid to numeric (mutate_at(vars(-ncdsid), as.numeric))
    value_cols = [c for c in out.columns if c != "ncdsid"]
    for c in value_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # keep ncdsid + columns with no NA, finite range, and positive variance
    keep = ["ncdsid"]
    for c in value_cols:
        col = out[c]
        if col.notna().all() and np.isfinite(col.to_numpy()).all() and col.var(ddof=1) > 0:
            keep.append(c)
    return out.loc[:, keep]


def _shared(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    """Columns common to both frames — the keys dplyr uses for a natural join."""
    shared = [c for c in a.columns if c in b.columns]
    return shared if shared else ["ncdsid"]
