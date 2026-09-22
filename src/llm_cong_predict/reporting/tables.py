"""Port of the output tables of ``R/create_data.R``.

Every file the R writes is listed, with its inputs and its state, in
docs/reference/create_data_outputs.md. Each function below builds one of them from

  * the metric rows the runner produces (``pipeline/execute.py``: one row per scored
    fit, with ``target`` and ``var``), which stand in for the R's ``tar_read(<target>)``
    followed by ``bind_rows()``;
  * the mapping table at the top of ``create_data.R`` (``reporting/mapping.py``);
  * for appendix D3, the fits themselves (the per-fold learner weights);
  * for D5, D7 and D8, the combined NCDS data; for D4, the essays.

Four of the R's outputs cannot run as written (fig_4, D2, D9, D11) and are
reconstructed here, each marked in its docstring and in PORTING_NOTES N2. D6 is not
reconstructed: it needs a code the variable table does not contain, so it raises, as
the R stops (N2). D4 and D7 are participant-level in the R, so instead of
those two tables the port builds aggregate summaries (N3).

Labels are NOT wrapped: the R wraps them for plotting with ``stringr::str_wrap``
(PORTING_NOTES N4), and the two filters that referenced wrapped strings are applied to
the underlying variable instead, which selects the same rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.labels import as_factor, r_as_numeric
from .mapping import LEARNER_NAMES, dplyr_filter_equals, dplyr_filter_not_equals, with_labels

# --- the R's `type` labels, verbatim -------------------------------------------
# R: llm_paper/R/create_data.R:L48, L51, L54, L57
TYPE_ESSAY_FULL = "Prediction based on ~250 Word Essay"
TYPE_GENES_FULL = "Prediction based on Combination of various Polygenic Scores"
TYPE_TEACHER_FULL = "Prediction based on Teacher Evaluation"
TYPE_TGE = "Polygenic Scores, ~250 Word Essay & Teacher Evaluation"
# R: llm_paper/R/create_data.R:L324 — this one has a line break inside the label
TYPE_GENES_BFI = "Prediction based on Combination\nof various Polygenic Scores"

# R: llm_paper/R/create_data.R:L74–94 — feature set -> label, in the R's bind order
FEATURE_SET_TYPES: tuple[tuple[str, str], ...] = (
    ("essay", "~250 Word Essay"),
    ("teacher", "Teacher Evaluation"),
    ("gene", "Polygenic Scores"),
    ("essay_genes", "Polygenic Scores & ~250 Word Essay"),
    ("essay_teacher", "~250 Word Essay & Teacher Evaluation"),
    ("teacher_genes", "Polygenic Scores & Teacher Evaluation"),
    ("teacher_genes_essay", TYPE_TGE),
)
# R: llm_paper/R/create_data.R:L100–121
TYPE_COG = "Cognitive Abilities"
TYPE_NONCOG = "Non-cognitive Traits"
TYPE_BIRTHWEIGHT = "Birthweight"
TYPE_HEIGHT = "Height"
TYPE_PEDU = "Parental Education"
TYPE_SOCIOLOGICAL = "Sociological Baseline Model"
# R: llm_paper/R/create_data.R:L163–170
TYPE_TEXT_LENGTH = "Number of Words"
TYPE_SALAT = "Linguistic Metrics of Lexical Diversity, Sophistication and Sentiment"
TYPE_READABILITY = "Simple Indices of Readability"
TYPE_SPELLING = "Grammatical and Typographical Errors"
TYPE_ALL_EXCEPT_GPT = "All Textual Information (except Embeddings)"
TYPE_GPT = "GPT 3.5-based Embeddings"
TYPE_ROBERTA = "RoBERTa-based Embeddings"
TYPE_GPT4 = "GPT 4-based Embeddings"
TYPE_ALL_TEXT = "All Textual Information"
# R: llm_paper/R/create_data.R:L341, L400, L415, L476, L488
SAMPLE_COMPLETE = "Complete Information on all Variables"
SAMPLE_MAXIMUM = "Maximum Observations"

# The port's own bookkeeping columns; the R's tables do not have them. `sample_note`
# and `run_label` are kept, so a warning attached to the numbers travels with them.
_DROP_COLUMNS = ("r_target", "sample", "run_config")

# The general cognitive factor, dropped by two filters that the R writes against the
# wrapped label ("General Factor\nof Cognitive\nAbility (Age\n11)", L69, L209, L239).
_GENERAL_FACTOR = "s2_co_factor_ability"
_HIGHEST_EDU = "s5_co_highest_edu"

# The twelve BSAG totals of appendix D7 (R: llm_paper/R/create_data.R:L275–286)
BSAG_ITEMS: dict[str, str] = {
    "n1001": "Inconsequential Behavior", "n1005": "Nervous Symptoms",
    "n983": "Anxiety for Acceptance by Adults", "n992": "Anxiety for Acceptance by Children",
    "n995": "Hostility towards Children", "n989": "Writing off of Adults and Adult Standards",
    "n986": "Hostility towards Adults", "n1004": "Miscellaneous Symptoms",
    "n998": "Restlessness", "n974": "Unforthcomingness", "n980": "Depression",
    "n977": "Withdrawal",
}
# R: llm_paper/R/create_data.R:L246–249
TEACHER_RATING_ITEMS: dict[str, str] = {
    "n876": "General Knowledge", "n877": "Number Work",
    "n878": "Use of Books", "n879": "Oral Ability",
}
# R: llm_paper/R/create_data.R:L259–264 — n885 is NOT in variables.xlsx (see appendix_d6)
BEHAVIOUR_ITEMS: dict[str, str] = {
    "n880": "Poor Hand Control", "n881": "Squirmy, Fidgety",
    "n882": "Poor Physical Coordination", "n883": "Hardly Ever Still",
    "n884": "Speech Difficulties", "n885": "Imperfect Grasp of English",
}


class MissingMetricsError(KeyError):
    """A table needs the metric rows of a target the run did not produce."""


class MissingColumnError(KeyError):
    """A table needs a column the data does not have; the R's ``select`` stops here."""


def _rows(metrics: pd.DataFrame, target: str, type_label: str) -> pd.DataFrame:
    """The metric rows of one model target, labelled with the R's ``type``.

    Stands for ``tar_read(<target>) %>% bind_rows() %>% mutate(type = ...)``.
    """
    out = metrics[metrics["target"] == target]
    if out.empty:
        raise MissingMetricsError(
            f"no metric rows for {target!r}: the run did not produce them (see the run log's "
            "not_run, e.g. gene data absent or a factor the backend did not compute)")
    out = out.drop(columns=[c for c in _DROP_COLUMNS if c in out.columns]).copy()
    out["type"] = type_label
    return out


def _stack(metrics: pd.DataFrame, pairs) -> pd.DataFrame:
    """``bind_rows`` over ``(target, type label)`` pairs, in the R's order."""
    return pd.concat([_rows(metrics, t, label) for t, label in pairs], ignore_index=True)


# ----------------------------------------------------------------- figures ----

def fig_2(metrics: pd.DataFrame) -> pd.DataFrame:
    """``plot_1_data`` -> fig_2_data.csv (R: llm_paper/R/create_data.R:L201–210).

    The essay, polygenic-score and teacher models on the maximum sample, without the
    life outcome and without the general cognitive factor.
    """
    out = _stack(metrics, (("essay_superlearner", TYPE_ESSAY_FULL),
                           ("gene_superlearner", TYPE_GENES_FULL),
                           ("teacher_superlearner", TYPE_TEACHER_FULL)))
    out = with_labels(out)
    out = dplyr_filter_not_equals(out, "category", "Life Outcomes")
    return dplyr_filter_not_equals(out, "var", _GENERAL_FACTOR).reset_index(drop=True)


def fig_3(metrics: pd.DataFrame) -> pd.DataFrame:
    """``plot_2_data`` -> fig_3_data.csv (R: llm_paper/R/create_data.R:L213–235).

    The seven feature sets on the mmg sample (educational attainment) and on the
    mmg-cog sample (the general cognitive factor).
    """
    pairs = [(f"{fs}_superlearner_mmg", label) for fs, label in FEATURE_SET_TYPES]
    pairs += [(f"{fs}_superlearner_cog_mmg", label) for fs, label in FEATURE_SET_TYPES]
    return with_labels(_stack(metrics, pairs)).reset_index(drop=True)


def fig_4(metrics: pd.DataFrame) -> pd.DataFrame:
    """``plot_3_data`` -> fig_4_data.csv (R: llm_paper/R/create_data.R:L124–136).

    RECONSTRUCTION (PORTING_NOTES N2): L98 calls
    ``get_cv_superlearner_metrics(cog_superlearner_social_lm)`` on a target that was
    never read with ``tar_read``, so the R stops. The evident intent is that target's
    metric rows, which is what is used here.

    Two things the R does that are easy to misread, and are reproduced: the
    "teacher + genes + essay" row is the **mmg-sample** target, because L113 overwrites
    the full-sample variable with ``teacher_genes_essay_metrics`` (defined at L92); and
    ``pedu_full_metrics`` is computed at L110 but never added to this table.
    """
    out = _stack(metrics, (("cog_lm_social_lm", TYPE_COG),
                           ("noncog_lm_social_lm", TYPE_NONCOG),
                           ("birthweight_superlearner_social", TYPE_BIRTHWEIGHT),
                           ("height_superlearner_social", TYPE_HEIGHT),
                           ("sociological_lm_social_lm", TYPE_SOCIOLOGICAL),
                           ("teacher_genes_essay_superlearner_mmg", TYPE_TGE)))
    out = with_labels(out)
    out = dplyr_filter_equals(out, "category", "Life Outcomes").copy()
    out["naming"] = "Educational Attainment"  # L136
    return out.reset_index(drop=True)


def fig_5(metrics: pd.DataFrame) -> pd.DataFrame:
    """``plot_4_data`` -> fig_5_data.csv (R: llm_paper/R/create_data.R:L184–194, L238–239).

    Each textual feature block, with its performance relative to the word-count
    baseline (``text_length``). The R drops two outcomes by their wrapped labels
    (L239); the port drops the same two rows by variable (N4).
    """
    out = _stack(metrics, (("salat_metrics_superlearner_text", TYPE_SALAT),
                           ("readability_metrics_superlearner_text", TYPE_READABILITY),
                           ("spelling_errors_superlearner_text", TYPE_SPELLING),
                           ("spelling_salat_readability_superlearner_text", TYPE_ALL_EXCEPT_GPT),
                           ("gpt_embeddings_superlearner_text", TYPE_GPT),
                           ("essay_superlearner", TYPE_ALL_TEXT)))
    out = with_labels(out)
    baseline = _rows(metrics, "text_length_lm", TYPE_TEXT_LENGTH).loc[:, ["var", "mean_r2"]]
    out = out.merge(baseline.rename(columns={"mean_r2": "lm_performance"}), on="var", how="left")
    out["relative_performance"] = out["mean_r2"] / out["lm_performance"]  # L194
    out = out[~out["var"].isin([_HIGHEST_EDU, _GENERAL_FACTOR])]  # L239
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- appendix ----

def appendix_d1(metrics: pd.DataFrame) -> pd.DataFrame:
    """``appendix_12_data`` -> appendix_D1_data.csv (R: llm_paper/R/create_data.R:L497–501)."""
    return with_labels(_rows(metrics, "gene_superlearner", TYPE_GENES_FULL)).reset_index(drop=True)


def appendix_d2(metrics: pd.DataFrame) -> pd.DataFrame:
    """``appendix_11_data`` -> appendix_D2_data.csv (R: llm_paper/R/create_data.R:L173–179).

    The three embedding models. FAITHFUL TO WHAT RUNS (PORTING_NOTES N2): the R's pipe
    ends at L176, so the ``name`` wrapping and the step that would relabel ``type`` as
    "RoBERTa" / "GPT 3.5" / "GPT 4" (L178–179) are a separate statement that errors and
    never reaches the table. The long labels are therefore kept.
    """
    out = _stack(metrics, (("roberta_embeddings_superlearner_text", TYPE_ROBERTA),
                           ("gpt_embeddings_superlearner_text", TYPE_GPT),
                           ("gpt4_embeddings_superlearner_text", TYPE_GPT4)))
    return with_labels(out).reset_index(drop=True)


def appendix_d3(fits: dict) -> pd.DataFrame:
    """``appendix_10_data`` -> appendix_D3_data.csv (R: llm_paper/R/create_data.R:L519–580).

    Per outcome and learner, the mean and standard deviation of the learner's weight
    over the outer folds (``x$fit$coef``; R's ``sd``, denominator n-1). ``fits`` maps a
    model target to its ``[(fit, outcome), ...]``, i.e. ``RunResult.values``.
    """
    rows = []
    for target, type_label in (("essay_superlearner", TYPE_ESSAY_FULL),
                               ("gene_superlearner", TYPE_GENES_FULL),
                               ("teacher_superlearner", TYPE_TEACHER_FULL),
                               ("teacher_genes_essay_superlearner", TYPE_TGE)):
        pairs = fits.get(target)
        if not pairs:
            raise MissingMetricsError(f"no fits for {target!r}: appendix D3 needs the per-fold "
                                      "learner weights of that target")
        for fit, outcome in pairs:
            coef = np.asarray(fit.coef, dtype=float)
            rows.append(pd.DataFrame({
                "type": type_label,
                "outcome": outcome,
                "model": fit.library_names,
                "mean": coef.mean(axis=0),
                "sd": coef.std(axis=0, ddof=1),
            }))
    out = pd.concat(rows, ignore_index=True)
    out = with_labels(out, var_column="outcome")
    out["model_name"] = out["model"].map(LEARNER_NAMES)  # L577
    return out.loc[:, ["type", "name", "model_name", "mean", "sd"]].rename(
        columns={"model_name": "model", "mean": "mean_weight"})  # L578


def summary_d4_essays(essays: pd.DataFrame) -> pd.DataFrame:
    """Replaces appendix_D4_data.csv, which is PARTICIPANT-LEVEL in the R
    (``appendix_1_data``, R: llm_paper/R/create_data.R:L241–242, is the full essay text
    with ``ncdsid``). The port never builds that table outside
    ``$LCP_DATA_ROOT``; this is an aggregate summary of the word counts instead
    (PORTING_NOTES N3). ``words`` is converted as the R converts it, with
    ``as.numeric`` (L242).
    """
    words = r_as_numeric(essays["words"])
    return pd.DataFrame([{
        "n_essays": int(len(essays)),
        "n_word_counts_missing": int(words.isna().sum()),
        "mean_words": float(words.mean()), "sd_words": float(words.std(ddof=1)),
        "min_words": float(words.min()), "p25_words": float(words.quantile(0.25)),
        "median_words": float(words.median()), "p75_words": float(words.quantile(0.75)),
        "max_words": float(words.max()),
    }])


def _response_proportions(ncds_1_to_9: pd.DataFrame, items: dict[str, str], r_lines: str) -> pd.DataFrame:
    """``select(...) %>% mutate_if(is.numeric, as_factor) %>% pivot_longer %>%
    group_by(name) %>% count(value) %>% mutate(n = n/sum(n)) %>% na.omit()``.

    The denominator includes the missing responses, because ``count`` counts NA as its
    own group and ``na.omit`` only runs afterwards; the NA row itself is dropped.
    """
    missing = [code for code in items if code not in ncds_1_to_9.columns]
    if missing:
        raise MissingColumnError(
            f"column(s) {missing} are not in the combined data, so the R's select stops "
            f"(R: llm_paper/R/create_data.R:{r_lines})")
    frames = []
    for code, label in items.items():
        factor = as_factor(ncds_1_to_9, code)
        counts = pd.Series(factor).value_counts(dropna=False)
        total = float(counts.sum())
        observed = [level for level in factor.categories if level in counts.index]
        frames.append(pd.DataFrame({"name": label, "value": observed,
                                    "n": [counts[level] / total for level in observed]}))
    return pd.concat(frames, ignore_index=True)


def appendix_d5(ncds_1_to_9: pd.DataFrame) -> pd.DataFrame:
    """``appendix_2_data`` -> appendix_D5_data.csv (R: llm_paper/R/create_data.R:L244–255):
    the distribution of the four teacher ability ratings."""
    return _response_proportions(ncds_1_to_9, TEACHER_RATING_ITEMS, "L244-255")


def appendix_d6(ncds_1_to_9: pd.DataFrame) -> pd.DataFrame:
    """``appendix_3_data`` -> appendix_D6_data.csv (R: llm_paper/R/create_data.R:L257–270):
    the distribution of the six sweep-2 behaviour items.

    THIS CANNOT BE BUILT (PORTING_NOTES N2). One of the six codes, ``n885``
    ("Imperfect Grasp of English"), is not in ``variables.xlsx``, so ``read_ncds`` never
    loads it and the R's ``select`` stops. It is not reconstructed from the five codes
    that do exist, because that would silently change the table; it raises instead.
    """
    return _response_proportions(ncds_1_to_9, BEHAVIOUR_ITEMS, "L257-270")


def summary_d7_bsag(ncds_1_to_9: pd.DataFrame) -> pd.DataFrame:
    """Replaces appendix_D7_data.csv, which is PARTICIPANT-LEVEL in the R
    (``appendix_4_data``, R: llm_paper/R/create_data.R:L273–288, is every person's BSAG
    values with ``ncdsid``). This is a per-item aggregate summary instead
    (PORTING_NOTES N3).
    """
    missing = [code for code in BSAG_ITEMS if code not in ncds_1_to_9.columns]
    if missing:
        raise MissingColumnError(f"BSAG column(s) {missing} are not in the combined data")
    rows = []
    for code, label in BSAG_ITEMS.items():
        values = pd.to_numeric(ncds_1_to_9[code], errors="coerce")
        rows.append({"name": label, "n": int(values.notna().sum()),
                     "n_missing": int(values.isna().sum()),
                     "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                     "min": float(values.min()), "p25": float(values.quantile(0.25)),
                     "median": float(values.median()), "p75": float(values.quantile(0.75)),
                     "max": float(values.max())})
    return pd.DataFrame(rows)


def appendix_d8(ncds_1_to_9: pd.DataFrame) -> pd.DataFrame:
    """``appendix_5_data`` -> appendix_D8_data.csv (R: llm_paper/R/create_data.R:L290–297):
    how many cohort members aspired to each job.

    COUNTS PER JOB, so small cells are possible; the export guard decides
    whether the table may leave ``$LCP_DATA_ROOT``. The R's
    ``filter(!aspired_job %in% c(47, 20.5))`` compares a factor with numbers, so it
    removes a job only if its LABEL is "47" or "20.5" — reproduced, quirk included.
    """
    if "n2771" not in ncds_1_to_9.columns:
        raise MissingColumnError("column 'n2771' (aspired job) is not in the combined data")
    jobs = pd.Series(as_factor(ncds_1_to_9, "n2771")).dropna().astype(object)
    jobs = jobs[~jobs.isin(["47", "20.5"])]
    counts = jobs.value_counts(sort=False)
    order = [level for level in as_factor(ncds_1_to_9, "n2771").categories if level in counts.index]
    out = pd.DataFrame({"aspired_job": order, "n": [int(counts[level]) for level in order]})
    return out.sort_values("n", ascending=False, kind="stable").reset_index(drop=True)


def appendix_d9(metrics: pd.DataFrame) -> pd.DataFrame:
    """``appendix_6_data`` -> appendix_D9_data.csv (R: llm_paper/R/create_data.R:L300–316).

    RECONSTRUCTION (PORTING_NOTES N2): the R binds ``essay_full_metrics_lm``,
    ``genes_full_metrics_lm`` and ``teacher_full_metrics_lm``, which are never defined
    anywhere in the repository, and its ``bind_rows`` has a trailing comma (L306), so
    it stops. The names and the ``pivot_wider(names_from = "method")`` that follows say
    what they were meant to be: the same three feature sets fitted by
    ``get_lm_cv_model`` (``essay_lm``, ``gene_lm``, ``teacher_lm``), with the same
    ``type`` labels so that the two methods line up.
    """
    superlearner = _stack(metrics, (("essay_superlearner", TYPE_ESSAY_FULL),
                                    ("gene_superlearner", TYPE_GENES_FULL),
                                    ("teacher_superlearner", TYPE_TEACHER_FULL)))
    superlearner["method"] = "SuperLearner"
    linear = _stack(metrics, (("essay_lm_lm", TYPE_ESSAY_FULL),
                              ("gene_lm_lm", TYPE_GENES_FULL),
                              ("teacher_lm_lm", TYPE_TEACHER_FULL)))
    linear["method"] = "Linear Model"
    out = pd.concat([superlearner, linear], ignore_index=True).loc[:, ["mean_r2", "var", "type", "method"]]
    out = dplyr_filter_not_equals(out, "var", _GENERAL_FACTOR)  # L309
    wide = out.pivot(index=["var", "type"], columns="method", values="mean_r2").reset_index()
    wide.columns.name = None
    wide["diff"] = wide["SuperLearner"] - np.maximum(wide["Linear Model"], 0.0)  # L311
    wide = with_labels(wide)
    return dplyr_filter_not_equals(wide, "category", "Life Outcomes").reset_index(drop=True)  # L316


def appendix_d10(metrics: pd.DataFrame) -> pd.DataFrame:
    """``appendix_7_data`` -> appendix_D10_data.csv (R: llm_paper/R/create_data.R:L319–342):
    the Big Five outcomes. The R computes the teacher+genes+essay BFI metrics at L328
    and does not put them in the table. ``name`` is overwritten at L342 with
    ``"Big 5: " + str_to_title(<var without s8_co_>) + "\\n (Age 50)"``; that line break
    is in the R's own string, not from wrapping.
    """
    out = _stack(metrics, (("essay_superlearner_bfi", TYPE_ESSAY_FULL),
                           ("gene_superlearner_bfi", TYPE_GENES_BFI),
                           ("teacher_superlearner_bfi", TYPE_TEACHER_FULL)))
    out = with_labels(out)
    out["sample"] = SAMPLE_COMPLETE  # L341
    out["name"] = "Big 5: " + out["var"].str.replace("s8_co_", "", regex=False).str.title() + "\n (Age 50)"
    return out.reset_index(drop=True)


def appendix_d11(metrics: pd.DataFrame) -> pd.DataFrame:
    """``appendix_8_data`` -> appendix_D11_data.csv (R: llm_paper/R/create_data.R:L344–420).

    RECONSTRUCTION (PORTING_NOTES N2). Two lines stop the R: L359 defines
    ``teacher_genes_essay_overlap_metrics`` from itself, before it exists (it is built
    at L462 from ``teacher_genes_essay_superlearner_overlap_metrics``, which is what is
    used here), and L367 repeats the missing ``tar_read`` of L98.

    Faithful to a third oddity: the "Maximum Observations" half has no
    teacher+genes+essay row. L382 takes ``teacher_genes_essay_metrics``, which by then
    (L328) holds the **Big Five** metrics, whose outcomes are not in the mapping table,
    so the ``category == "Life Outcomes"`` filter removes every one of those rows.
    """
    overlap = _stack(metrics, (("cog_lm_social_lm_overlap", TYPE_COG),
                               ("noncog_lm_social_lm_overlap", TYPE_NONCOG),
                               ("pedu_superlearner_social_overlap", TYPE_PEDU),
                               ("birthweight_superlearner_social_overlap", TYPE_BIRTHWEIGHT),
                               ("height_superlearner_social_overlap", TYPE_HEIGHT),
                               ("teacher_genes_essay_superlearner_overlap", TYPE_TGE)))
    overlap["sample"] = SAMPLE_COMPLETE  # L415
    full = _stack(metrics, (("cog_lm_social_lm", TYPE_COG),
                            ("noncog_lm_social_lm", TYPE_NONCOG),
                            ("pedu_superlearner_social", TYPE_PEDU),
                            ("birthweight_superlearner_social", TYPE_BIRTHWEIGHT),
                            ("height_superlearner_social", TYPE_HEIGHT)))
    full["sample"] = SAMPLE_MAXIMUM  # L400
    out = with_labels(pd.concat([overlap, full], ignore_index=True))
    return dplyr_filter_equals(out, "category", "Life Outcomes").reset_index(drop=True)


def appendix_d12(metrics: pd.DataFrame) -> pd.DataFrame:
    """``appendix_9_data`` -> appendix_D12_data.csv (R: llm_paper/R/create_data.R:L422–495):
    the seven feature sets on the mmg sample and on the full-overlap sample, for the
    life outcome."""
    mmg = _stack(metrics, [(f"{fs}_superlearner_mmg", label) for fs, label in FEATURE_SET_TYPES])
    mmg["sample"] = SAMPLE_MAXIMUM  # L476
    overlap = _stack(metrics, [(f"{fs}_superlearner_overlap", label) for fs, label in FEATURE_SET_TYPES])
    overlap["sample"] = SAMPLE_COMPLETE  # L488
    out = with_labels(pd.concat([mmg, overlap], ignore_index=True))
    return dplyr_filter_equals(out, "category", "Life Outcomes").reset_index(drop=True)


# Output file name -> the function that builds it. D4 and D7 are the aggregate
# summaries that replace the R's participant-level tables (N3).
METRIC_TABLES = {
    "fig_2_data": fig_2, "fig_3_data": fig_3, "fig_4_data": fig_4, "fig_5_data": fig_5,
    "appendix_D1_data": appendix_d1, "appendix_D2_data": appendix_d2,
    "appendix_D9_data": appendix_d9, "appendix_D10_data": appendix_d10,
    "appendix_D11_data": appendix_d11, "appendix_D12_data": appendix_d12,
}
NCDS_TABLES = {
    "appendix_D5_data": appendix_d5, "appendix_D6_data": appendix_d6,
    "appendix_D8_data": appendix_d8, "appendix_D7_summary": summary_d7_bsag,
}
