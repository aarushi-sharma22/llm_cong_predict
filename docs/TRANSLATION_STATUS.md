# Translation status

Every R object in the original repository (`tobiaswolfram/llm_paper`, commit
`b0cfe4c`) and its Python counterpart. Each item has exactly
one status:

| Status | Meaning |
|---|---|
| **PORTED** | Python exists, follows the cited R lines, and has tests of its behaviour on synthetic data. Numbers on real data are not yet checked (see the VALIDATION_CHECKLIST item). |
| **PORTED-APPROX** | As PORTED, but part of it can only approximate the R implementation. Each approximation is marked `# APPROX` and listed in the VALIDATION_CHECKLIST APPROX register. |
| **INGESTION** | The R computes the values with an external tool; the port reads that tool's output instead of generating it. |
| **BOUNDARY** | Not reimplemented; the Python raises and points to the ingestion path. |
| **BLOCKED** | Cannot be produced, for a reason in the R itself; the port raises with that reason rather than guess. |
| **REPLACED** | Participant-level in the R; the port builds an aggregate summary instead. |

Nothing is left unported: no entry below is "not started".

## `R/functions.R` (798 lines, 30 top-level functions; counted with grep)

| R function (line) | Python | Status | Notes |
|---|---|---|---|
| `get_gpt_embeddings` (2) | `features/embeddings.py::gpt_embeddings` | PORTED | Reads a saved embeddings file instead of the R's contradictory RDS reshaper (PORTING_NOTES G1). |
| `get_gpt4_embeddings` (12) | `features/embeddings.py::gpt_embeddings` | PORTED | Identical to the above in the R (A5). |
| `read_datalist` (22) | `io/readers.py::read_datalist` | PORTED | Reads the file with assigned column names: 63 rows (reconstruction, A1). |
| `read_essays` (26) | `io/readers.py::read_essays` | PORTED | Reads like `readtext`, splits like `tidyr::separate`, including malformed files (E6). |
| `read_gene_data` (34) | `io/readers.py::read_gene_data` | PORTED | The R body is empty; without a path the Python raises (E1). It reads the optional polygenic score file, in the port's placeholder format (M5). |
| `read_camsis` (39) | `io/readers.py::read_camsis` | PORTED | Runs on the shipped file. |
| `read_occupation_aspiration_mapping` (43) | `io/readers.py::read_occupation_aspiration_mapping` | PORTED | Runs on the shipped file. |
| `read_ncds` (47) | `io/readers.py::read_ncds` | PORTED | Missing-code labels dropped as `set_na` does (F6). |
| `combine_ncds` (57) | `io/readers.py::combine_ncds` | PORTED | plyr collision semantics (E3). |
| `clean_ncds` (64) | `cleaning/clean_ncds.py::clean_ncds` | PORTED | Reconstruction of a function that cannot run as published (A1, A2, F7). V1. |
| `create_aspirations` (244) | `cleaning/aspirations.py::create_aspirations` | PORTED | Sex-comparison quirk reproduced (F3). V5. |
| `create_factors` (272) | `cleaning/factors.py::create_factors` | PORTED | Default backend `"r"`: all four factors from `psych::fa` through rpy2 (F1). Native backend: Pearson factor only (APPROX, AP8). V3. |
| `get_complete_ncds` (309) | `cleaning/assemble.py::get_complete_ncds` | PORTED | Natural joins with the keys checked; the R call passes an unused 5th argument, which is an error in R; reconstruction (A3). |
| `create_essay_variables` (317) | `features/essay_variables.py::create_essay_variables` | PORTED | Output width depends on the data (G5). V2. |
| `find_essay_teacher_genetics_overlap` (330) | `cleaning/assemble.py::find_essay_teacher_genetics_overlap` | PORTED | haven integer codes (F5). Never called by the R pipeline. |
| `find_full_overlap` (348) | `cleaning/assemble.py::find_full_overlap` | PORTED | |
| `tokenize_essays` (356) | `features/readability.py::tokenize_essays`, `r/readability.R` | BOUNDARY | TreeTagger; the Python function raises. Ported to `r/readability.R`, which has never been run (G4). The pipeline target is EXTERNAL. |
| `calculate_readability_metrics` (369) | `features/readability.py::ingest_readability_metrics`, `r/readability.R` | INGESTION | koRpus; the Python generator raises. `r/readability.R` (never run, G4) writes the CSV the pipeline reads (`config.RESTRICTED_INPUTS["readability_metrics"]`). V9. |
| `get_spelling_error_metrics` (390) | `features/salat.py::get_spelling_error_metrics` | INGESTION | LanguageTool output; the R's error cases are reproduced (G3). |
| `get_salat_metrics` (419) | `features/salat.py::get_salat_metrics` | INGESTION | SALAT tools' output; natural joins (G3). |
| `get_roberta_embeddings` (446) | `features/embeddings.py::roberta_embeddings`, `features/roberta_step.py` | PORTED | Batched; a separate step in its own process (G2, ORCHESTRATION). Not run on real weights here. |
| `SL.xgboost.hist` (492) | `models/base_learners.py::_make_xgboost_hist` | PORTED-APPROX | C8, AP6. |
| `get_general_superlearner_cv_model` (496) | `models/run.py::fit_model(method="superlearner")` | PORTED-APPROX | Selection, `na.omit`, `(fit, var)` faithful (L1); engine mechanics faithful and matched to R for mean + lm (C1); learners and screener APPROX (C2–C9). V4. |
| `get_lm_cv_model` (526) | `models/run.py::fit_model(method="lm")` | PORTED | Matches R's `CV.SuperLearner` to 1e-14 on identical folds (C1, C4). |
| `get_cv_predictive_r2` (550) | `metrics/cv_metrics.py::cv_predictive_r2` | PORTED | B1. |
| `get_cv_rmse` (602) | `metrics/cv_metrics.py::cv_rmse` | PORTED | |
| `get_cv_mad` (649) | `metrics/cv_metrics.py::cv_mad` | PORTED | |
| `get_cv_superlearner_metrics` (696) | `metrics/cv_metrics.py::superlearner_metrics` | PORTED | MSE and `se_mse` before the clamp (B4). |
| `get_cv_lm_metrics` (725) | `metrics/cv_metrics.py::lm_metrics` | PORTED | MSE from `SL.lm_All` (B4). |
| `get_cv_lm_r2` (751) | `metrics/cv_metrics.py::cv_lm_r2` | PORTED | |

## `_targets.R` (504 lines)

| Part | Python | Status | Notes |
|---|---|---|---|
| Data, essay and cleaning targets (L41–174) | `pipeline/build.py`, `pipeline/execute.py` | PORTED | The graph validates and runs. No stub roots remain: `tokenized_essays` is EXTERNAL (r/readability.R), readability is ingested, gene data is optional (M1, M4, M5). |
| Variable lists (L120–159) | `pipeline/variable_lists.py` | PORTED | Checked against the inventory; the data-dependent ones follow the R, its two dropped-column quirks included (M3). |
| 70 model targets, 416 fits (L178–385) | `pipeline/model_spec.py` | PORTED (structure) | Each target records method, outcome, predictors, sample, data preparation and scorer; `R_TO_PYTHON_TARGET` maps the names (I1, I2). |
| 53 metric targets (L387–494) | `pipeline/model_spec.py` | PORTED (structure) | 63 metric targets in Python: the 53, plus the 10 targets scored only in `create_data.R`; the 7 mmg_lm targets have none (I1). |
| Execution | `pipeline/execute.py` | PORTED | Bindings for every target, dependency order, model fits over worker processes, metric rows, run log (M1). Exercised end to end on synthetic data (M6). |

Computed with `python run.py`: the graph has 186 targets, none blocked; 185 run given
their inputs and one (`tokenized_essays`) is produced by another program.

## `R/create_data.R` (601 lines)

Figure and appendix tables (`fig_2..5_data.csv`, `appendix_D1..D12_data.csv`), ported
into `src/llm_cong_predict/reporting/`. Every output, its inputs and its
state: docs/reference/create_data_outputs.md.

| Output | Python | Status | Notes |
|---|---|---|---|
| `fig_2_data`, `fig_3_data`, `fig_5_data` | `reporting/tables.py::fig_2`, `fig_3`, `fig_5` | PORTED | N1, N4. |
| `fig_4_data`, `appendix_D9_data`, `appendix_D11_data` | `reporting/tables.py::fig_4`, `appendix_d9`, `appendix_d11` | PORTED (reconstruction) | The R stops on these (undefined objects, a self-reference, a trailing comma): N2. |
| `appendix_D1_data`, `appendix_D2_data`, `appendix_D10_data`, `appendix_D12_data` | `reporting/tables.py` | PORTED | D2 keeps the long `type` labels, because the R's pipe ends before the relabelling (N2). |
| `appendix_D3_data` | `reporting/tables.py::appendix_d3` | PORTED | Per-fold learner weights from the fits. |
| `appendix_D5_data`, `appendix_D8_data` | `reporting/tables.py::appendix_d5`, `appendix_d8` | PORTED | D8 holds counts per job; the export guard decides whether it may leave the data root. |
| `appendix_D6_data` | `reporting/tables.py::appendix_d6` | BLOCKED | Needs `n885`, which is not in `variables.xlsx`, so the R's `select` stops; the port raises rather than rebuild the table from five of the six codes. A corrected variant behind `config.INCLUDE_N885` (off by default) adds the row and marks every output of the run (N2). |
| `appendix_D4_data`, `appendix_D7_data` | `reporting/tables.py::summary_d4_essays`, `summary_d7_bsag` | REPLACED | Participant-level in the R: aggregate summaries instead, written only under `$LCP_DATA_ROOT` (N3). |

## `R/get_gpt_embeddings.R` (52 lines)

`scripts/get_gpt_embeddings.py`. **PORTED**, but it refuses to run without a double
opt-in, because it sends essays to an external API (H2).

## `r/readability.R` (this port, no R counterpart file)

The standalone TreeTagger + koRpus script: `read_essays`,
`tokenize_essays` and `calculate_readability_metrics` from `R/functions.R`, with the
TreeTagger path and the data paths taken from the environment. **Never run** (G4, V9).

## No R counterpart (added by this port)

| Python | What it is |
|---|---|
| `export.py` | The export guard: the one function through which a table may be written outside `$LCP_DATA_ROOT`. The minimum cell size has no default (PORTING_NOTES H4, V10). |
| `scripts/check_no_restricted_data.py`, `scripts/hooks/pre-commit` | Refuse restricted data in a commit (H3). |
| `scripts/check_essay_format.py` | Aggregate format counts for the real essays (E6). |
| `isolation.py` | torch and xgboost never share a process (G2). |
| `pipeline/execute.py` | The runner (M1); `run.R` is its R counterpart, listed below. |

## `run.R` (7 lines)

`run.py`. **PORTED**: `python run.py` prints the plan; `python run.py --run
[--config smoke] [--targets ...]` runs the graph through `pipeline/execute.py` (M1).
