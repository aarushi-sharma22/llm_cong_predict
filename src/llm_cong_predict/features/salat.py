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

from ..io.joins import natural_join
from ..io.labels import r_as_numeric

# The spelling rule-issue categories the R fills-to-zero and sums (verbatim order).
_SPELLING_FILL_ZERO = [
    "grammar", "misspelling", "typographical", "locale-violation", "duplication",
    "style", "whitespace", "uncategorized", "inconsistency",
]
_SPELLING_OTHER = ["locale-violation", "whitespace", "uncategorized", "inconsistency"]


class SpellingMetricsError(ValueError):
    """Raised where the R ``get_spelling_error_metrics`` stops with an error
    (owner decision C8: reproduce R's errors)."""


def _read_and_concat(paths: list[str]) -> pd.DataFrame:
    # dplyr::bind_rows: stack, aligning columns by name (R: llm_paper/R/functions.R:L424–435)
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def get_salat_metrics(
    ncds_essays: pd.DataFrame,
    taaled: list[str],
    taales: list[str],
    seance: list[str],
) -> pd.DataFrame:
    """Port of ``get_salat_metrics`` (R: llm_paper/R/functions.R:L419–444).

    R reads three CSVs each for TAALED / TAALES / SEANCE, row-binds within each tool,
    renames TAALES' ``Filename`` -> ``filename`` (L431), then (L437–443):

        ncds_essays %>% select(filename = doc_id, ncdsid) %>%
          left_join(taaled) %>% left_join(taales) %>% left_join(seance) %>%
          select(-filename)

    Each ``left_join`` has no ``by``, so dplyr joins on EVERY column the two sides
    share at that step (io/joins.py). If two tools report a column with the same name
    (a shared metric), that column becomes a join key for the later tool, and essays
    whose values differ get NA in all of that tool's columns. The keys used at each
    step are logged and stored in ``attrs["salat_join_keys"]``.

    Signature simplified: the R took nine positional CSV paths; here, three lists.
    """
    taaled_df = _read_and_concat(taaled)
    taales_df = _read_and_concat(taales).rename(columns={"Filename": "filename"})
    seance_df = _read_and_concat(seance)

    out = ncds_essays.rename(columns={"doc_id": "filename"}).loc[:, ["filename", "ncdsid"]]
    keys = {}
    for step, tool in (("taaled", taaled_df), ("taales", taales_df), ("seance", seance_df)):
        out, keys[step] = natural_join(out, tool, "left", step=f"get_salat_metrics: {step}")
    out = out.drop(columns=["filename"])
    out.attrs["salat_join_keys"] = keys
    return out


def get_spelling_error_metrics(ncds_essays: pd.DataFrame, path: str) -> pd.DataFrame:
    """Port of ``get_spelling_error_metrics`` (R: llm_paper/R/functions.R:L390–417).

    R:
        counts <- read_csv(path) %>% group_by(ncdsid, rule_issue_type) %>% count()   # L394–396
        ncds_essays %>% select(ncdsid, words) %>%
          left_join(counts, by = "ncdsid") %>%                                      # L399–400
          mutate(error_per_words = n / as.numeric(words)) %>%                       # L402
          select(ncdsid, rule_issue_type, error_per_words) %>%
          pivot_wider(names_from = rule_issue_type, values_from = error_per_words) %>%  # L404
          mutate_at(vars(one_of(<nine categories>)), ~ ifelse(is.na(.), 0, .)) %>%  # L405–408
          mutate(total = rowSums(select(., <nine>)), other = rowSums(select(., <four>))) %>%  # L410–414
          select(-"NA")                                                             # L415

    Reproduced:
      * an essay with no error row is KEPT: the left join gives it rule type NA, so
        ``pivot_wider`` makes a column literally named "NA" and the essay's categories
        are filled with 0 (brief F8: the old port dropped these essays);
      * rows in essay order; columns in order of first appearance, where each essay's
        rule types come sorted (``count()`` returns groups sorted, NA last);
      * the two cases where the R stops with an error raise
        :class:`SpellingMetricsError` (owner decision C8): a category of the nine never
        occurring (``select(., grammar, ...)`` on L410–414 names a missing column), and
        no essay without errors (``select(-"NA")`` on L415 names a missing column).
    ``ncdsid`` is compared as text on both sides (pandas refuses mixed key types).
    """
    spelling = pd.read_csv(path)
    counts = (
        spelling.groupby(["ncdsid", "rule_issue_type"], dropna=False, sort=True)
        .size()
        .reset_index(name="n")
    )
    essays = ncds_essays.loc[:, ["ncdsid", "words"]].copy()
    essays["ncdsid"] = essays["ncdsid"].astype(str)
    counts["ncdsid"] = counts["ncdsid"].astype(str)

    long = essays.merge(counts, on="ncdsid", how="left")
    long["error_per_words"] = long["n"] / r_as_numeric(long["words"]).to_numpy()  # as.numeric(words), L402
    # pivot_wider names the column of a missing rule type "NA"
    long["rule_issue_type"] = long["rule_issue_type"].astype(object).where(long["rule_issue_type"].notna(), "NA")
    if long.duplicated(["ncdsid", "rule_issue_type"]).any():
        raise SpellingMetricsError(
            "duplicate (ncdsid, rule_issue_type) pairs after the join: an ncdsid occurs more "
            "than once in the essays; pivot_wider would build list-columns here")
    rows = list(dict.fromkeys(long["ncdsid"]))
    cols = list(dict.fromkeys(long["rule_issue_type"]))
    wide = (long.pivot(index="ncdsid", columns="rule_issue_type", values="error_per_words")
            .reindex(index=rows, columns=cols).reset_index())
    wide.columns.name = None

    for cat in _SPELLING_FILL_ZERO:  # one_of(): categories that exist
        if cat in wide.columns:
            wide[cat] = wide[cat].fillna(0)

    missing = [c for c in _SPELLING_FILL_ZERO if c not in wide.columns]
    if missing:
        raise SpellingMetricsError(
            f"spelling categories {missing} never occur. The R stops here: "
            "rowSums(dplyr::select(., grammar, misspelling, ...)) names a column that does not "
            "exist (R: llm_paper/R/functions.R:L410-414).")
    wide["total"] = wide[_SPELLING_FILL_ZERO].sum(axis=1)
    wide["other"] = wide[_SPELLING_OTHER].sum(axis=1)

    if "NA" not in wide.columns:
        raise SpellingMetricsError(
            "every essay has at least one spelling error, so pivot_wider created no 'NA' column. "
            "The R stops here: dplyr::select(-\"NA\") names a column that does not exist "
            "(R: llm_paper/R/functions.R:L415).")
    return wide.drop(columns=["NA"])
