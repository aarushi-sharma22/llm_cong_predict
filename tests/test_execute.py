"""The synthetic input set and the end-to-end run (brief Task 2.4).

Every test here runs on data written by tests/fixtures/synthetic_ncds.py into a
temporary ``$LCP_DATA_ROOT``: obviously synthetic IDs (SYN000001...), deterministic
from a seed, with planted relations. No real data is involved.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
import pytest

from fixtures.synthetic_ncds import SPELLING_CATEGORIES, write_synthetic_inputs
from llm_cong_predict import config
from llm_cong_predict.cleaning.clean_ncds import MISSING_LABELS
from llm_cong_predict.io.readers import read_datalist, read_essays
from llm_cong_predict.pipeline import execute
from llm_cong_predict.pipeline.build import build_pipeline
from llm_cong_predict.pipeline.execute import SAMPLE_NOTE_NO_GENE_DATA, SmokeRunRefused, run_pipeline
from llm_cong_predict.rbridge import r_unavailable_reason

SMOKE_N = 120  # small n, as the smoke configuration allows (brief Task 2.4)
SEED = 7


@contextlib.contextmanager
def data_root(path):
    """Point $LCP_DATA_ROOT at ``path`` for the duration of the block."""
    before = os.environ.get(config.LCP_DATA_ROOT_ENV)
    os.environ[config.LCP_DATA_ROOT_ENV] = str(path)
    try:
        yield Path(path)
    finally:
        if before is None:
            os.environ.pop(config.LCP_DATA_ROOT_ENV, None)
        else:
            os.environ[config.LCP_DATA_ROOT_ENV] = before


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    """A complete synthetic input set, polygenic scores included."""
    root = tmp_path_factory.mktemp("synthetic_with_genes")
    with data_root(root):
        write_synthetic_inputs(root, seed=SEED, n=SMOKE_N, gene_data=True)
    return root


@pytest.fixture(scope="session")
def synthetic_root_no_genes(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synthetic_no_genes")
    with data_root(root):
        write_synthetic_inputs(root, seed=SEED, n=SMOKE_N, gene_data=False)
    return root


def _smoke(**changes) -> config.RunConfig:
    import dataclasses

    return dataclasses.replace(config.SMOKE_RUN, **changes)


# ------------------------------------------------------- the synthetic set ----

def test_synthetic_inputs_are_deterministic_given_the_seed(tmp_path):
    """Brief Section 2.3: synthetic data is generated deterministically from a seed."""
    roots = []
    for name in ("a", "b"):
        root = tmp_path / name
        root.mkdir()
        with data_root(root):
            write_synthetic_inputs(root, seed=3, n=30)
        roots.append(root)
    a, b = roots
    files = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    for rel in files:
        if rel.suffix == ".dta":  # .dta headers carry a timestamp: compare the contents
            fa, ma = pyreadstat.read_dta(str(a / rel))
            fb, mb = pyreadstat.read_dta(str(b / rel))
            pd.testing.assert_frame_equal(fa, fb)
            assert ma.variable_value_labels == mb.variable_value_labels
        else:
            assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel


def test_every_code_of_the_variable_table_is_in_exactly_one_ncds_file(synthetic_root):
    """The set holds every code of data/variables.xlsx (brief Task 2.4), each in one
    file, so that combine_ncds' full join has no column collision."""
    codes = {c.lower() for c in read_datalist(str(config.VARIABLES_XLSX))["variable"]}
    seen: dict[str, str] = {}
    for key in ("ncds_1_2_3", "ncds_4", "ncds_5", "ncds_6", "ncds_7", "ncds_8", "ncds_9",
                "ncds_occ_2", "ncds_occ_5", "ncds_occ_6", "ncds_occ_7", "ncds_occ_8"):
        frame, _ = pyreadstat.read_dta(str(synthetic_root / config.RESTRICTED_INPUTS[key]))
        for column in frame.columns:
            name = column.lower()
            if name in codes:
                assert name not in seen, f"{name} is in both {seen.get(name)} and {key}"
                seen[name] = key
    assert set(seen) == codes


def test_value_labels_include_missing_strings_negative_codes_and_the_aspiration_labels(synthetic_root):
    """Brief Task 2.4: realistic value-label sets that include the missing-label
    strings, negative missing codes, and n2771 labels from the public mapping file."""
    _, meta = pyreadstat.read_dta(str(synthetic_root / config.RESTRICTED_INPUTS["ncds_1_2_3"]))
    labels = meta.variable_value_labels
    used = {text for mapping in labels.values() for text in mapping.values()}
    assert len(used & set(MISSING_LABELS)) >= 8
    assert " Cant say,inappl" in used  # the entry with the leading space (brief F2)
    assert any(code < 0 for mapping in labels.values() for code in mapping)
    mapping_file = pd.read_excel(config.OCCUPATION_ASPIRATION_XLSX)
    # the codes keep the mixed case of variables.xlsx ("N2771"), which read_ncds lower-cases
    aspiration = {k.lower(): v for k, v in labels.items()}["n2771"]
    for code, text in zip(mapping_file["aspiration_code"], mapping_file["aspiration_n2771"]):
        assert aspiration[int(code)] == text


def test_essays_salat_and_spelling_are_in_the_format_the_readers_parse(synthetic_root):
    """The essay folder parses with no malformed file, the shared SALAT metric becomes a
    join key (brief F8), and the spelling CSV has essays with no error and all nine
    categories (so get_spelling_error_metrics does not stop, PORTING_NOTES G3/C8)."""
    from llm_cong_predict.features.salat import get_salat_metrics, get_spelling_error_metrics

    with data_root(synthetic_root):
        essays = read_essays(str(config.restricted_path("essays")))
        assert essays.attrs["read_essays_malformed"] == {"files_with_extra_pieces": 0,
                                                         "files_with_missing_pieces": 0}
        assert essays["ncdsid"].str.match(r"^SYN\d{6}$").all()
        assert essays["words"].astype(int).gt(0).all()

        salat = get_salat_metrics(essays, **{tool: [str(config.restricted_path(f"{tool}_{b}"))
                                                    for b in (1, 2, 3)]
                                             for tool in ("taaled", "taales", "seance")})
        assert salat.attrs["salat_join_keys"]["taales"] == ["filename", "nwords"]
        assert salat.notna().all().all()

        spelling = pd.read_csv(config.restricted_path("spelling_mistakes"))
        assert set(spelling["rule_issue_type"]) == set(SPELLING_CATEGORIES)
        errors = get_spelling_error_metrics(essays, str(config.restricted_path("spelling_mistakes")))
        assert (errors["total"] == 0).any()  # essays with no error are kept (brief F8)
        assert len(errors) == len(essays)


# ------------------------------------------------------------- the bindings ----

def test_every_target_is_bound_to_the_function_it_calls():
    """Brief Task 2.4: each pipeline target is bound to the function it calls."""
    from llm_cong_predict.cleaning.clean_ncds import clean_ncds
    from llm_cong_predict.features.readability import ingest_readability_metrics
    from llm_cong_predict.io.readers import read_ncds
    from llm_cong_predict.metrics.cv_metrics import lm_metrics, superlearner_metrics
    from llm_cong_predict.models.run import fit_model

    pipe = build_pipeline()
    specs = execute.model_specs_by_name()
    for name in pipe.names():
        assert execute.binding_for(name, specs).function is not None, name
    assert execute.binding_for("ncds_4", specs).function is read_ncds
    assert execute.binding_for("ncds_1_to_9_cleaned", specs).function is clean_ncds
    assert execute.binding_for("readability_metrics", specs).function is ingest_readability_metrics
    assert execute.binding_for("essay_superlearner_overlap", specs).function is fit_model
    assert execute.binding_for("essay_superlearner_overlap_metrics", specs).function is superlearner_metrics
    assert execute.binding_for("essay_lm_lm_metrics", specs).function is lm_metrics  # the lm scorer (F7)


# ------------------------------------------------ refusing a real smoke run ----

def test_smoke_configuration_is_refused_without_the_synthetic_marker(tmp_path):
    """Brief Task 2.4: the smoke configuration is impossible to use for a real run."""
    with data_root(tmp_path):
        with pytest.raises(SmokeRunRefused, match=config.SYNTHETIC_MARKER_FILE):
            run_pipeline(config.SMOKE_RUN, targets=["mapping_df"])


def test_smoke_configuration_is_refused_when_an_id_is_not_synthetic(tmp_path):
    """Even with the marker copied in, a non-synthetic ID stops the run."""
    with data_root(tmp_path):
        write_synthetic_inputs(tmp_path, seed=1, n=20)
        essay = sorted((tmp_path / config.RESTRICTED_INPUTS["essays"]).glob("*.txt"))[0]
        essay.write_text(essay.read_text().replace("ID: SYN", "ID: NOT"), encoding="utf-8")
        with pytest.raises(SmokeRunRefused, match="not of the form SYN000001"):
            run_pipeline(config.SMOKE_RUN, targets=["ncds_essays"])


# ------------------------------------- gene data absent: skips and the note ----

@pytest.fixture(scope="module")
def no_gene_result(synthetic_root_no_genes):
    """One smoke run without gene data: a gene target, a target on a gene-defined
    sample, and a target on ncds_complete."""
    with data_root(synthetic_root_no_genes):
        return run_pipeline(_smoke(factor_backend="native"),
                            targets=["teacher_superlearner_mmg_metrics", "teacher_lm_lm_metrics",
                                     "gene_superlearner_mmg_metrics"])


def test_gene_targets_are_skipped_and_gene_defined_samples_are_marked(no_gene_result):
    """Owner decision at Checkpoint C: targets whose predictors need gene_variables are
    skipped (PORTING_NOTES L2), and every metric row from a sample the R defines with
    gene_variables carries sample_note, so the warning travels with the numbers."""
    result = no_gene_result
    notes = result.metrics.groupby("target")["sample_note"].first()
    assert notes["teacher_superlearner_mmg"] == SAMPLE_NOTE_NO_GENE_DATA
    assert pd.isna(notes["teacher_lm_lm"])  # ncds_complete is not defined with gene_variables
    skipped = [e for e in result.not_run if e["target"] == "gene_superlearner_mmg"]
    assert len(skipped) == 1 and "gene data absent" in skipped[0]["reason"]
    assert "gene_superlearner_mmg" not in set(result.metrics["target"])
    log = json.loads(result.log_path.read_text())
    assert log["gene_data_available"] is False
    assert log["skipped_targets"] == ["gene_superlearner_mmg"]


def test_every_smoke_output_is_labelled(no_gene_result):
    """Brief Task 2.4: the smoke configuration labels every output it produces."""
    result = no_gene_result
    assert result.metrics_path.name == "SMOKE_metrics.csv"
    assert result.log_path.name == "SMOKE_run_log.json"
    assert (result.metrics["run_label"] == config.SMOKE_RUN.label).all()
    assert (result.metrics["run_config"] == "smoke").all()
    log = json.loads(result.log_path.read_text())
    assert log["run_config"] == "smoke" and log["run_label"] == config.SMOKE_RUN.label
    assert log["data_source"] == "synthetic"
    predictions = result.prediction_paths
    assert predictions and all(p.name.startswith("SMOKE_") for p in predictions)
    first = pd.read_csv(predictions[0])
    assert (first["run_label"] == config.SMOKE_RUN.label).all()
    assert first.columns[1] == "ncdsid" and "SL.predict" in first.columns


def test_predictions_and_metrics_are_written_only_under_the_data_root(no_gene_result, synthetic_root_no_genes):
    """Participant-level output stays under $LCP_DATA_ROOT (brief Section 2.3)."""
    root = synthetic_root_no_genes.resolve()
    for path in [no_gene_result.metrics_path, no_gene_result.log_path, *no_gene_result.prediction_paths]:
        assert root in path.resolve().parents


# ------------------------------------- the native factor backend: not run ----

def test_native_factor_backend_reports_the_polychoric_outcomes_as_not_run(synthetic_root):
    """Owner decision at Checkpoint C: with factor_backend="native" the three
    polychoric factor outcomes are reported as not run, with a reason, and no metric
    row exists for them (PORTING_NOTES F1)."""
    polychoric = ["s3_co_factor_scholastic_motivation", "s3_te_factor_externalizing",
                  "s3_te_factor_internalizing"]
    with data_root(synthetic_root):
        result = run_pipeline(_smoke(factor_backend="native", save_predictions=False),
                              targets=["teacher_lm_lm_metrics"])
    reported = {e["outcome"]: e["reason"] for e in result.not_run if e["target"] == "teacher_lm_lm"}
    assert sorted(reported) == sorted(polychoric)
    assert all("polychoric" in reason and "native" in reason for reason in reported.values())
    assert not set(result.metrics["var"]) & set(polychoric)
    assert len(result.metrics) == 9  # the other nine outcomes of all_outcomes did run
    log = json.loads(result.log_path.read_text())
    assert sorted(log["unavailable_columns"]) == sorted(polychoric)


def test_results_do_not_depend_on_n_jobs(synthetic_root):
    """The fits are spread over worker processes; the numbers must not change."""
    targets = ["cog_lm_social_lm_metrics", "text_length_lm_metrics"]
    with data_root(synthetic_root):
        one = run_pipeline(_smoke(n_jobs=1, save_predictions=False), targets=targets)
        many = run_pipeline(_smoke(n_jobs=3, save_predictions=False), targets=targets)
    columns = [c for c in one.metrics.columns if c not in ("sample_note",)]
    pd.testing.assert_frame_equal(one.metrics[columns], many.metrics[columns])


# ------------------------------------------------------------ end to end ----

@pytest.mark.slow
@pytest.mark.skipif(r_unavailable_reason(("psych",)) is not None,
                    reason=(r_unavailable_reason(("psych",)) or "") +
                           " — the end-to-end run needs the 'r' factor backend for the three "
                           "polychoric factors (PORTING_NOTES F1)")
def test_smoke_run_scores_every_scored_target(synthetic_root):
    """Brief Task 2.4: the smoke run on synthetic data produces a metric row for every
    scored target. 63 targets are scored and 409 of the 416 fits are scored; the seven
    *_mmg_lm targets are scored nowhere in the R (owner decision C6)."""
    specs = execute.model_specs_by_name()
    scored = {name: s for name, s in specs.items() if s.scorer != "none"}
    with data_root(synthetic_root):
        result = run_pipeline(_smoke(n_jobs=min(8, os.cpu_count() or 1)))
    metrics = result.metrics
    assert result.not_run == []
    assert set(metrics["target"]) == set(scored)
    assert len(metrics) == sum(s.n_fits() for s in scored.values()) == 409
    for name, spec in scored.items():
        rows = metrics[metrics["target"] == name]
        assert len(rows) == spec.n_fits(), name
        assert rows["n"].gt(0).all()
    assert metrics[["mean_mse", "mean_r2", "mean_rmse", "mean_mad", "se_mse"]].notna().all().all()
    assert metrics["sample_note"].isna().all()  # gene data is present here
    # the planted relation: the teacher's ratings predict the cognitive outcomes
    teacher = metrics[(metrics["target"] == "teacher_superlearner") &
                      (metrics["var"] == "s2_co_verbal_ability")]
    assert float(teacher["mean_r2"].iloc[0]) > 0.2
    assert pd.read_csv(result.metrics_path).shape[0] == len(metrics)


@pytest.mark.slow
def test_paper_configuration_runs_on_one_target(synthetic_root):
    """Brief Task 2.4: one slow test runs the full paper configuration (10 outer folds,
    5 inner folds, the 6-learner library) on one small synthetic target. Its outputs
    say they come from synthetic data."""
    import dataclasses

    paper = dataclasses.replace(config.PAPER_RUN, factor_backend="native", n_jobs=min(5, os.cpu_count() or 1))
    with data_root(synthetic_root):
        result = run_pipeline(paper, targets=["teacher_superlearner_mmg_metrics"])
    (fit, var), = result.values["teacher_superlearner_mmg"]
    assert var == "s5_co_highest_edu"
    assert len(fit.folds) == config.SUPERLEARNER.outer_folds == 10
    assert fit.library_names == ["SL.mean_All", "SL.ranger_screen.glmnet", "SL.nnet_screen.glmnet",
                                 "SL.xgboost.hist_screen.glmnet", "SL.ksvm_screen.glmnet",
                                 "SL.lm_screen.glmnet"]
    assert np.isfinite(fit.sl_predict).all()
    row = result.metrics.iloc[0]
    assert row["run_config"] == "paper" and "SYNTHETIC" in row["run_label"]
    assert result.metrics_path.name == "SYNTHETIC_metrics.csv"
    assert float(row["mean_r2"]) > 0.1  # planted: ability drives both the ratings and the outcome
