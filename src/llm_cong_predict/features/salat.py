"""External-tool feature INGESTION: SALAT metrics and spelling errors.

Faithful ports of ``get_salat_metrics`` and ``get_spelling_error_metrics`` from
``R/functions.R``. These read CSVs produced by tools OUTSIDE this codebase and
reshape them — they do NOT generate the metrics:

  * SALAT metrics (lexical diversity/sophistication/sentiment) come from the
    desktop tools at https://www.linguisticanalysistools.org/ (TAALED / TAALES /
    SEANCE), one CSV per tool per essay batch.
  * Spelling/grammar errors come from the LanguageTool CLI, as a single CSV of
    per-essay rule issues.

DECISION (user, this project): we port INGESTION ONLY and flag it. We do NOT
reimplement these metrics with Python NLP libraries, because a different tool would
produce different numbers while appearing to work — a silent divergence from the
paper. Generation therefore remains external; these functions consume its output.

VALIDATION: mechanics are tested on synthetic CSVs. Real values require the real
essays run through the external tools (or the author's shared derived CSVs).
"""

from __future__ import annotations

import pandas as pd

# The spelling rule-issue categories the R fills-to-zero and sums (verbatim order).
_SPELLING_FILL_ZERO = [
    "grammar", "misspelling", "typographical", "locale-violation", "duplication",
    "style", "whitespace", "uncategorized", "inconsistency",
]
_SPELLING_OTHER = ["locale-violation", "whitespace", "uncategorized", "inconsistency"]


def _read_and_concat(paths: list[str]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def get_salat_metrics(
    ncds_essays: pd.DataFrame,
    taaled: list[str],
    taales: list[str],
    seance: list[str],
) -> pd.DataFrame:
    """Port of ``get_salat_metrics``.

    R reads three CSVs each for TAALED / TAALES / SEANCE, row-binds within each
    tool, renames TAALES' ``Filename`` -> ``filename``, then left-joins all three
    onto the essay frame (keyed on ``filename`` = the essay ``doc_id``) and drops
    ``filename``.

    Signature simplified: the R took nine positional CSV paths
    (``taaled_1..3``, ``taales_1..3``, ``seance_1..3``); here they are three lists.
    Join keys follow dplyr's natural-join (shared columns) — ``filename`` after the
    rename.
    """
    taaled_df = _read_and_concat(taaled)
    taales_df = _read_and_concat(taales).rename(columns={"Filename": "filename"})
    seance_df = _read_and_concat(seance)

    base = ncds_essays.rename(columns={"doc_id": "filename"}).loc[:, ["filename", "ncdsid"]]
    out = (
        base.merge(taaled_df, on=_common(base, taaled_df), how="left")
        .merge(taales_df, on="filename", how="left")
        .merge(seance_df, on=_common_after(seance_df), how="left")
    )
    return out.drop(columns=[c for c in ["filename"] if c in out.columns])


def _common(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    """Shared columns for a dplyr natural join (falls back to 'filename')."""
    shared = [c for c in a.columns if c in b.columns]
    return shared if shared else ["filename"]


def _common_after(seance_df: pd.DataFrame) -> str:
    # SEANCE CSVs key on filename too (the R relies on a shared 'filename' column).
    return "filename"


def get_spelling_error_metrics(ncds_essays: pd.DataFrame, path: str) -> pd.DataFrame:
    """Port of ``get_spelling_error_metrics``.

    R:
        spelling <- read_csv(path)
        counts <- spelling %>% group_by(ncdsid, rule_issue_type) %>% count()
        essays %>% select(ncdsid, words) %>%
          left_join(counts, by="ncdsid") %>%
          mutate(error_per_words = n / as.numeric(words)) %>%
          select(ncdsid, rule_issue_type, error_per_words) %>%
          pivot_wider(names_from=rule_issue_type, values_from=error_per_words) %>%
          mutate_at(<the nine categories>, ~ ifelse(is.na(.), 0, .)) %>%
          mutate(total = rowSums(<nine>), other = rowSums(<four>)) %>%
          select(-"NA")

    Reproduced faithfully: per-essay error counts by rule type, normalised by word
    count, pivoted wide, missing categories filled with 0, plus ``total`` and
    ``other`` row-sums, and the literal ``NA`` column (from missing rule types)
    dropped.
    """
    spelling = pd.read_csv(path)
    counts = (
        spelling.groupby(["ncdsid", "rule_issue_type"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    # Align the join-key dtype (pandas will not merge str vs int keys; R was
    # type-tolerant here). Coerce both sides' ncdsid to string.
    essays = ncds_essays.loc[:, ["ncdsid", "words"]].copy()
    essays["ncdsid"] = essays["ncdsid"].astype(str)
    counts["ncdsid"] = counts["ncdsid"].astype(str)

    merged = essays.merge(counts, on="ncdsid", how="left")
    merged["error_per_words"] = merged["n"] / pd.to_numeric(merged["words"], errors="coerce")
    wide = merged.loc[:, ["ncdsid", "rule_issue_type", "error_per_words"]].pivot_table(
        index="ncdsid", columns="rule_issue_type", values="error_per_words", aggfunc="first"
    ).reset_index()
    wide.columns.name = None

    for cat in _SPELLING_FILL_ZERO:
        if cat in wide.columns:
            wide[cat] = wide[cat].fillna(0)
        else:
            wide[cat] = 0.0  # category absent entirely -> all zero (R's one_of tolerates absence)

    wide["total"] = wide[_SPELLING_FILL_ZERO].sum(axis=1)
    wide["other"] = wide[_SPELLING_OTHER].sum(axis=1)

    # drop the literal "NA" column (essays whose rule_issue_type was missing)
    for na_col in ("NA", "nan", float("nan")):
        if na_col in wide.columns:
            wide = wide.drop(columns=[na_col])
    return wide
