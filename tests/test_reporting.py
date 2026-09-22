"""The reporting tables: the port of ``R/create_data.R`` (brief Task 2.6).

Synthetic metric rows and synthetic NCDS frames only. Every output file of the R, with
its inputs and its state, is in docs/reference/create_data_outputs.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from llm_cong_predict.io.labels import attach_labels
from llm_cong_predict.metrics.cv_metrics import CVSuperLearnerFit
from llm_cong_predict.reporting import build_tables, write_tables
from llm_cong_predict.reporting.mapping import mapping_table, with_labels
from llm_cong_predict.reporting.tables import (
    FEATURE_SET_TYPES,
    MissingColumnError,
    appendix_d1,
    appendix_d2,
    appendix_d3,
    appendix_d5,
    appendix_d6,
    appendix_d8,
    appendix_d9,
    appendix_d10,
    appendix_d11,
    appendix_d12,
    fig_2,
    fig_3,
    fig_4,
    fig_5,
    summary_d4_essays,
    summary_d7_bsag,
)
from fixtures.data_root import data_root

ALL_OUTCOMES = ["s2_co_factor_ability", "s2_co_verbal_ability", "s2_co_nonverbal_ability",
                "s2_co_reading_ability", "s2_co_mathematics_ability", "s3_co_reading_ability",
                "s3_co_mathematics_ability", "s2_co_aspiration_camsis",
                "s3_co_factor_scholastic_motivation", "s3_te_factor_externalizing",
                "s3_te_factor_internalizing", "s5_co_highest_edu"]
SOCIAL = ["s5_co_highest_edu"]
BFI = ["s8_co_extraversion", "s8_co_agreeableness", "s8_co_conscientiousness",
       "s8_co_neuroticism", "s8_co_openness"]


def _metric_rows(target: str, outcomes, r2_start: float = 0.1) -> pd.DataFrame:
    """Metric rows as pipeline/execute.py writes them, with made-up numbers."""
    return pd.DataFrame([{
        "target": target, "r_target": "", "var": var, "n": 100 + i,
        "mean_mse": 1.0 + i, "se_mse": 0.1, "min_mse": 0.9, "max_mse": 1.2,
        "mean_r2": round(r2_start + 0.01 * i, 4), "min_r2": 0.0, "max_r2": 0.5,
        "mean_mad": 0.5, "min_mad": 0.4, "max_mad": 0.6,
        "mean_rmse": 1.0, "min_rmse": 0.9, "max_rmse": 1.1,
        "sample": "ncds_complete", "sample_note": None, "run_config": "smoke",
        "run_label": "SMOKE RUN on synthetic data: not results",
    } for i, var in enumerate(outcomes)])


@pytest.fixture(scope="module")
def metrics() -> pd.DataFrame:
    """One metric row per scored fit, for every target the reporting tables read."""
    frames = [
        _metric_rows("essay_superlearner", ALL_OUTCOMES, 0.20),
        _metric_rows("gene_superlearner", ALL_OUTCOMES, 0.05),
        _metric_rows("teacher_superlearner", ALL_OUTCOMES, 0.30),
        _metric_rows("teacher_genes_essay_superlearner", ALL_OUTCOMES, 0.35),
        _metric_rows("essay_lm_lm", ALL_OUTCOMES, -0.10),  # a negative R², for D9's pmax
        _metric_rows("gene_lm_lm", ALL_OUTCOMES, 0.02),
        _metric_rows("teacher_lm_lm", ALL_OUTCOMES, 0.25),
        _metric_rows("cog_lm_social_lm", SOCIAL, 0.40),
        _metric_rows("noncog_lm_social_lm", SOCIAL, 0.15),
        _metric_rows("sociological_lm_social_lm", SOCIAL, 0.22),
        _metric_rows("birthweight_superlearner_social", SOCIAL, 0.01),
        _metric_rows("height_superlearner_social", SOCIAL, 0.02),
        _metric_rows("pedu_superlearner_social", SOCIAL, 0.18),
        _metric_rows("cog_lm_social_lm_overlap", SOCIAL, 0.41),
        _metric_rows("noncog_lm_social_lm_overlap", SOCIAL, 0.16),
        _metric_rows("pedu_superlearner_social_overlap", SOCIAL, 0.19),
        _metric_rows("birthweight_superlearner_social_overlap", SOCIAL, 0.02),
        _metric_rows("height_superlearner_social_overlap", SOCIAL, 0.03),
        _metric_rows("text_length_lm", ALL_OUTCOMES, 0.10),
    ]
    for feature_set, _label in FEATURE_SET_TYPES:
        frames.append(_metric_rows(f"{feature_set}_superlearner_mmg", SOCIAL, 0.30))
        frames.append(_metric_rows(f"{feature_set}_superlearner_cog_mmg", ["s2_co_factor_ability"], 0.45))
        frames.append(_metric_rows(f"{feature_set}_superlearner_overlap", ALL_OUTCOMES, 0.28))
    for feature_set in ("essay", "gene", "teacher"):
        frames.append(_metric_rows(f"{feature_set}_superlearner_bfi", BFI, 0.05))
    for name in ("salat_metrics", "readability_metrics", "spelling_errors",
                 "spelling_salat_readability", "gpt_embeddings", "roberta_embeddings",
                 "gpt4_embeddings"):
        frames.append(_metric_rows(f"{name}_superlearner_text", ALL_OUTCOMES, 0.12))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------- the mapping ----

def test_mapping_table_is_the_one_at_the_top_of_create_data():
    """R: llm_paper/R/create_data.R:L11–44 — the age suffix comes from the sweep
    prefix, and the four confounders have no category_name (no TRUE fallback)."""
    table = mapping_table().set_index("variable")
    assert table.loc["s2_co_reading_ability", "name"] == "Reading Ability (Age 11)"
    assert table.loc["s3_co_reading_ability", "name"] == "Reading Ability (Age 16)"
    assert table.loc["s5_co_highest_edu", "name"] == "Highest Education (Age 33)"
    assert table.loc["s3_te_factor_externalizing", "category_name"] == "Non-cognitive Traits"
    assert pd.isna(table.loc["s3_pa_edu", "category_name"])  # "confounder" matches no case
    assert len(table) == 16


def test_labels_of_a_variable_outside_the_mapping_are_missing():
    """The Big Five outcomes are not in the mapping, so the R's left_join gives NA —
    which is why the category filters drop them (appendix D11)."""
    labelled = with_labels(pd.DataFrame({"var": ["s8_co_openness", "s5_co_highest_edu"]}))
    assert pd.isna(labelled.loc[0, "name"]) and pd.isna(labelled.loc[0, "category"])
    assert labelled.loc[1, "category"] == "Life Outcomes"


# --------------------------------------------------------------- the figures ----

def test_fig_2_keeps_three_models_without_the_life_outcome_or_the_general_factor(metrics):
    """R: llm_paper/R/create_data.R:L201–210."""
    table = fig_2(metrics)
    assert set(table["target"]) == {"essay_superlearner", "gene_superlearner", "teacher_superlearner"}
    assert "s5_co_highest_edu" not in set(table["var"])  # category != "Life Outcomes"
    assert "s2_co_factor_ability" not in set(table["var"])  # the wrapped-name filter (L209)
    assert len(table) == 3 * 10
    assert set(table["type"]) == {"Prediction based on ~250 Word Essay",
                                  "Prediction based on Combination of various Polygenic Scores",
                                  "Prediction based on Teacher Evaluation"}


def test_fig_3_has_the_seven_feature_sets_on_both_mmg_samples(metrics):
    """R: llm_paper/R/create_data.R:L213–235."""
    table = fig_3(metrics)
    assert len(table) == 14
    assert set(table["var"]) == {"s5_co_highest_edu", "s2_co_factor_ability"}
    assert table["type"].nunique() == 7


def test_fig_4_takes_the_mmg_teacher_model_and_leaves_pedu_out(metrics):
    """R: llm_paper/R/create_data.R:L98–136. Reconstruction of the missing tar_read at
    L98; the teacher+genes+essay row is the mmg-sample target, because L113 overwrites
    the full-sample variable (L92); pedu_full_metrics (L110) is computed and unused."""
    table = fig_4(metrics)
    assert "teacher_genes_essay_superlearner_mmg" in set(table["target"])
    assert "teacher_genes_essay_superlearner" not in set(table["target"])
    assert "pedu_superlearner_social" not in set(table["target"])
    assert set(table["var"]) == {"s5_co_highest_edu"}
    assert (table["naming"] == "Educational Attainment").all()
    assert len(table) == 6


def test_fig_5_divides_by_the_word_count_baseline(metrics):
    """R: llm_paper/R/create_data.R:L184–194, L238–239."""
    table = fig_5(metrics)
    assert not {"s5_co_highest_edu", "s2_co_factor_ability"} & set(table["var"])
    baseline = metrics[(metrics["target"] == "text_length_lm")].set_index("var")["mean_r2"]
    row = table.iloc[0]
    assert row["lm_performance"] == baseline[row["var"]]
    assert row["relative_performance"] == pytest.approx(row["mean_r2"] / baseline[row["var"]])
    assert len(table) == 6 * 10


# -------------------------------------------------------------- the appendix ----

def test_appendix_d1_and_d2_are_the_metric_rows_with_labels(metrics):
    """R: llm_paper/R/create_data.R:L497–501 (D1) and L173–179 (D2). D2 keeps the long
    type labels, because the R's pipe ends before the relabelling step."""
    assert len(appendix_d1(metrics)) == 12
    d2 = appendix_d2(metrics)
    assert set(d2["type"]) == {"RoBERTa-based Embeddings", "GPT 3.5-based Embeddings",
                               "GPT 4-based Embeddings"}
    assert len(d2) == 36


def test_appendix_d3_averages_the_learner_weights_over_the_folds():
    """R: llm_paper/R/create_data.R:L519–580: per outcome and learner, colMeans and sd
    of the per-fold weight matrix, with the R's learner names."""
    names = ["SL.mean_All", "SL.ranger_screen.glmnet", "SL.nnet_screen.glmnet",
             "SL.xgboost.hist_screen.glmnet", "SL.ksvm_screen.glmnet", "SL.lm_screen.glmnet"]
    coef = np.array([[0.1, 0.2, 0.0, 0.3, 0.0, 0.4], [0.3, 0.0, 0.1, 0.2, 0.0, 0.4]])
    fit = CVSuperLearnerFit(Y=np.zeros(4), sl_predict=np.zeros(4),
                            library_predict=np.zeros((4, 6)), library_names=names,
                            folds=[np.array([0, 1]), np.array([2, 3])], coef=coef,
                            outcome_var="s5_co_highest_edu")
    fits = {t: [(fit, "s5_co_highest_edu")] for t in
            ("essay_superlearner", "gene_superlearner", "teacher_superlearner",
             "teacher_genes_essay_superlearner")}
    table = appendix_d3(fits)
    assert list(table.columns) == ["type", "name", "model", "mean_weight", "sd"]
    assert set(table["model"]) == {"Mean", "Random_Forest", "Neural_Network", "XGBoost",
                                   "SVM", "Linear_Model"}
    mean_row = table[(table["model"] == "Mean") & (table["type"].str.contains("Essay"))].iloc[0]
    assert mean_row["mean_weight"] == pytest.approx(0.2)
    assert mean_row["sd"] == pytest.approx(np.std([0.1, 0.3], ddof=1))
    assert (table["name"] == "Highest Education (Age 33)").all()


def test_appendix_d9_reconstructs_the_linear_model_half_and_clamps_it_at_zero(metrics):
    """R: llm_paper/R/create_data.R:L300–316. The three *_lm objects the R binds are
    never defined and its bind_rows has a trailing comma, so it stops; the port uses
    the get_lm_cv_model targets, as the names say. diff = SuperLearner - pmax(LM, 0)."""
    table = appendix_d9(metrics)
    assert {"SuperLearner", "Linear Model", "diff"} <= set(table.columns)
    assert "s2_co_factor_ability" not in set(table["var"])
    assert "s5_co_highest_edu" not in set(table["var"])  # category != Life Outcomes
    essay = table[table["type"].str.contains("Essay")].iloc[0]
    assert essay["Linear Model"] < 0  # the fixture's lm R² is negative here
    assert essay["diff"] == pytest.approx(essay["SuperLearner"])  # pmax(., 0) clamps it away
    assert len(table) == 3 * 10


def test_appendix_d10_renames_the_big_five_outcomes(metrics):
    """R: llm_paper/R/create_data.R:L333–342 — name is overwritten with the R's own
    string, line break included, and the teacher+genes+essay BFI rows are left out."""
    table = appendix_d10(metrics)
    assert len(table) == 15
    assert set(table["name"]) == {f"Big 5: {trait}\n (Age 50)" for trait in
                                  ("Extraversion", "Agreeableness", "Conscientiousness",
                                   "Neuroticism", "Openness")}
    assert (table["sample"] == "Complete Information on all Variables").all()
    assert "teacher_genes_essay_superlearner_bfi" not in set(table["target"])


def test_appendix_d11_has_no_teacher_model_in_the_maximum_sample_half(metrics):
    """R: llm_paper/R/create_data.R:L344–420. Reconstruction of two broken lines (L359,
    L367). The "Maximum Observations" half has five rows, not six: at L382 the R takes
    teacher_genes_essay_metrics, which by then holds the Big Five metrics (L328), and
    those outcomes are not in the mapping, so the Life Outcomes filter removes them."""
    table = appendix_d11(metrics)
    maximum = table[table["sample"] == "Maximum Observations"]
    complete = table[table["sample"] == "Complete Information on all Variables"]
    assert len(maximum) == 5 and len(complete) == 6
    assert "teacher_genes_essay_superlearner_overlap" in set(complete["target"])
    assert not any("teacher_genes_essay" in t for t in maximum["target"])
    assert set(table["var"]) == {"s5_co_highest_edu"}


def test_appendix_d12_pairs_the_two_samples(metrics):
    """R: llm_paper/R/create_data.R:L422–495."""
    table = appendix_d12(metrics)
    assert len(table) == 14
    assert set(table["sample"]) == {"Maximum Observations", "Complete Information on all Variables"}
    assert set(table["var"]) == {"s5_co_highest_edu"}


# ------------------------------------------ tables built from the NCDS data ----

def _labelled_ncds() -> pd.DataFrame:
    frame = pd.DataFrame({
        "ncdsid": [f"SYN{i:06d}" for i in range(1, 7)],
        "n876": [1.0, 1.0, 2.0, 3.0, np.nan, 2.0],
        "n877": [2.0, 2.0, 2.0, 1.0, 3.0, 3.0],
        "n878": [1.0, 2.0, 3.0, 3.0, 3.0, 3.0],
        "n879": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        "n880": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "n2771": [1.0, 1.0, 2.0, 3.0, np.nan, 2.0],
        **{code: np.arange(6, dtype=float) for code in
           ("n1001", "n1005", "n983", "n992", "n995", "n989", "n986", "n1004",
            "n998", "n974", "n980", "n977")},
    })
    labels = {code: {1: "Above average", 2: "Average", 3: "Below average"}
              for code in ("n876", "n877", "n878", "n879", "n880")}
    labels["n2771"] = {1: "Teacher", 2: "Engineer", 3: "47"}
    return attach_labels(frame, value_labels=labels)


def test_appendix_d5_proportions_count_the_missing_responses_in_the_denominator():
    """R: llm_paper/R/create_data.R:L244–255: count() counts NA as its own group, the
    proportion divides by that total, and na.omit drops the NA row afterwards."""
    table = appendix_d5(_labelled_ncds())
    assert set(table["name"]) == {"General Knowledge", "Number Work", "Use of Books", "Oral Ability"}
    knowledge = table[table["name"] == "General Knowledge"].set_index("value")["n"]
    assert knowledge["Above average"] == pytest.approx(2 / 6)  # 6 rows, one of them missing
    assert knowledge.sum() == pytest.approx(5 / 6)
    assert "Below average" in set(table["value"]) and table["n"].notna().all()


def test_appendix_d6_builds_under_the_corrected_variant():
    """With config.INCLUDE_N885 on, read_ncds loads n885 and the table can be built
    (owner decision at Checkpoint D). Every output of such a run is marked with
    config.N885_NOTE, the way a sample built without gene data is marked."""
    from llm_cong_predict import config

    frame = _labelled_ncds().copy()
    labels = dict(frame.attrs["value_labels"])
    for code in ("n881", "n882", "n883", "n884", "n885"):  # the rest of the behaviour items
        frame[code] = [1.0, 2.0, 3.0, 1.0, 2.0, np.nan]
        labels[code] = {1: "Does not apply", 2: "Applies somewhat", 3: "Certainly applies"}
    frame.attrs["value_labels"] = labels

    table = appendix_d6(frame)
    assert set(table["name"]) == {"Poor Hand Control", "Squirmy, Fidgety",
                                  "Poor Physical Coordination", "Hardly Ever Still",
                                  "Speech Difficulties", "Imperfect Grasp of English"}
    built = build_tables(metrics=_metric_rows("gene_superlearner", ALL_OUTCOMES),
                         ncds_1_to_9=frame, note=config.N885_NOTE)
    assert (built["appendix_D6_data"]["variables_note"] == config.N885_NOTE).all()
    assert (built["appendix_D1_data"]["variables_note"] == config.N885_NOTE).all()


def test_appendix_d6_cannot_be_built_because_n885_is_not_in_the_variable_table():
    """R: llm_paper/R/create_data.R:L257–270 selects n880–n885, but n885 is not in
    data/variables.xlsx, so read_ncds never loads it and the R's select stops. The port
    does not rebuild the table from the five codes that do exist."""
    with pytest.raises(MissingColumnError, match="n885"):
        appendix_d6(_labelled_ncds())


def test_appendix_d8_counts_jobs_and_keeps_the_r_filter_quirk():
    """R: llm_paper/R/create_data.R:L290–297: !aspired_job %in% c(47, 20.5) compares a
    factor with numbers, so it removes a job only if its LABEL is "47" or "20.5"."""
    table = appendix_d8(_labelled_ncds())
    assert list(table.columns) == ["aspired_job", "n"]
    assert table["n"].tolist() == sorted(table["n"], reverse=True)
    assert dict(zip(table["aspired_job"], table["n"])) == {"Teacher": 2, "Engineer": 2}
    assert "47" not in set(table["aspired_job"])  # the label "47" is dropped by the quirk


def test_d4_and_d7_summaries_hold_no_participant_data():
    """Brief F9: appendix D4 (essay text) and D7 (every person's BSAG values) are
    participant-level in the R. The port builds aggregate summaries instead
    (PORTING_NOTES N3)."""
    essays = pd.DataFrame({"doc_id": ["e1", "e2"], "ncdsid": ["SYN000001", "SYN000002"],
                           "text": ["synthetic essay one", "synthetic essay two"],
                           "words": ["120", "80"]})
    d4 = summary_d4_essays(essays)
    assert len(d4) == 1 and "ncdsid" not in d4.columns and "text" not in d4.columns
    assert d4.loc[0, "n_essays"] == 2 and d4.loc[0, "mean_words"] == pytest.approx(100.0)

    d7 = summary_d7_bsag(_labelled_ncds())
    assert len(d7) == 12 and "ncdsid" not in d7.columns
    assert set(d7.columns) == {"name", "n", "n_missing", "mean", "sd", "min", "p25",
                               "median", "p75", "max"}
    assert d7.loc[0, "mean"] == pytest.approx(2.5)


# ------------------------------------------------------ building and writing ----

def test_a_table_whose_inputs_are_missing_is_reported_not_faked(metrics):
    """A run without gene data produces no gene metric rows, so the tables that need
    them cannot be built. They are listed with the reason instead of being built from
    fewer rows."""
    without_genes = metrics[~metrics["target"].str.startswith("gene_")]
    built = build_tables(without_genes)
    # fig_2 and fig_3 both show a polygenic-score model, so neither can be built
    assert "gene_superlearner" in built.skipped["fig_2_data"]
    assert "gene_superlearner_mmg" in built.skipped["fig_3_data"]
    # fig_4 needs no gene-only target, so it is still built
    assert "fig_4_data" in built.tables
    assert "appendix_D3_data" in built.skipped  # no fits were passed


def test_write_tables_writes_only_under_the_data_root(metrics, tmp_path):
    """Brief Section 2.3 / Task 2.6: nothing is written outside $LCP_DATA_ROOT until it
    passes the export guard (Task 2.7)."""
    built = build_tables(metrics)
    with data_root(tmp_path):
        written = write_tables(built, prefix="SMOKE_")
    assert written and all(tmp_path.resolve() in p.resolve().parents for p in written)
    assert {p.name for p in written} >= {"SMOKE_fig_2_data.csv", "SMOKE_appendix_D12_data.csv"}
    assert (tmp_path / "reporting").is_dir()
    assert len(pd.read_csv(written[0])) == len(built["fig_2_data"])


def test_the_built_tables_go_through_the_export_guard(metrics, monkeypatch, tmp_path):
    """Task 2.7: a reporting table leaves $LCP_DATA_ROOT only through the guard. The
    metric tables pass; appendix D8 holds counts per job, so small cells stop it."""
    from llm_cong_predict import config
    from llm_cong_predict.export import export_tables

    monkeypatch.setattr(config, "MINIMUM_CELL_SIZE", 10)
    built = build_tables(metrics, ncds_1_to_9=_labelled_ncds())
    written, refused = export_tables(built, tmp_path, n_people=[100, 111])
    assert {p.name for p in written} >= {"fig_2_data.csv", "appendix_D12_data.csv"}
    assert "minimum cell size" in refused["appendix_D8_data"]  # two people per job here
    assert not (tmp_path / "appendix_D8_data.csv").exists()


def test_every_table_of_create_data_is_accounted_for(metrics):
    """The sixteen files the R writes (docs/reference/create_data_outputs.md): ten from
    metrics, one from the fits, three from the NCDS data, and D4/D7 replaced by
    aggregate summaries. D6 is the one that cannot be built."""
    built = build_tables(metrics, fits=None, ncds_1_to_9=_labelled_ncds(),
                         essays=pd.DataFrame({"doc_id": ["e1"], "ncdsid": ["SYN000001"],
                                              "text": ["x"], "words": ["10"]}))
    expected = {"fig_2_data", "fig_3_data", "fig_4_data", "fig_5_data", "appendix_D1_data",
                "appendix_D2_data", "appendix_D5_data", "appendix_D8_data", "appendix_D9_data",
                "appendix_D10_data", "appendix_D11_data", "appendix_D12_data",
                "appendix_D4_summary", "appendix_D7_summary"}
    assert expected <= set(built.tables)
    assert "n885" in built.skipped["appendix_D6_data"]
    assert set(built.skipped) == {"appendix_D3_data", "appendix_D6_data"}
