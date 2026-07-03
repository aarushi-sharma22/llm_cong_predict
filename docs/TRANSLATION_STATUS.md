# Translation Status — every R object, honestly accounted for

This is the complete surface of the original R repo and exactly what has and has
not been translated. Updated as work proceeds. No green-checkmark inflation: a row
is only "DONE" if the Python exists AND has a test that checks behaviour (not just
"it runs"). "STRUCTURAL" means ported and tested for mechanics but not numerically
validated against R. "NOT STARTED" means the R has not been translated at all.

Legend: ✅ DONE · 🟡 STRUCTURAL (needs oracle/data to validate) · ⬜ NOT STARTED

## `R/functions.R` (798 lines, 27 defs)

| R function (line) | Purpose | Python target | Status |
|---|---|---|---|
| `get_cv_predictive_r2` (550) | fold-wise SL R² | `metrics/cv_metrics.py::cv_predictive_r2` | ✅ |
| `get_cv_lm_r2` (751) | fold-wise lm R² | `metrics/cv_metrics.py::cv_lm_r2` | ✅ |
| `get_cv_rmse` (602) | fold-wise RMSE | `metrics/cv_metrics.py::cv_rmse` | ✅ |
| `get_cv_mad` (649) | fold-wise MAD | `metrics/cv_metrics.py::cv_mad` | ✅ |
| `get_cv_superlearner_metrics` (696) | assemble SL metric row | `metrics/cv_metrics.py::superlearner_metrics` | ✅ |
| `get_cv_lm_metrics` (725) | assemble lm metric row | `metrics/cv_metrics.py::lm_metrics` | ✅ |
| `SL.xgboost.hist` (492) | xgboost hist variant | `models/base_learners.py::_make_xgboost_hist` | 🟡 |
| `get_general_superlearner_cv_model` (496) | nested-CV SL fit | `models/native_superlearner.py` + `models/backend` | 🟡 |
| `get_lm_cv_model` (526) | nested-CV lm fit | `models/native_superlearner.py` (lm_library) | 🟡 |
| `read_datalist` (22) | read variables.xlsx | `io/readers.py::read_datalist` | 🟡 |
| `read_essays` (26) | parse essay text files | `io/readers.py::read_essays` | 🟡 |
| `read_gene_data` (34) | **empty stub in R** | `io/readers.py::read_gene_data` (raises, documented) | ✅ |
| `read_camsis` (39) | read CAMSIS .dta | `io/readers.py::read_camsis` | 🟡 (runs on real shipped file) |
| `read_occupation_aspiration_mapping` (43) | read mapping xlsx | `io/readers.py::read_occupation_aspiration_mapping` | 🟡 (runs on real shipped file) |
| `read_ncds` (47) | read one NCDS .dta wave | `io/readers.py::read_ncds` | 🟡 |
| `combine_ncds` (57) | full-join all waves | `io/readers.py::combine_ncds` | 🟡 |
| `clean_ncds` (64) | **the big recode** (180 lines) | `cleaning/clean_ncds.py` | ⬜ DEFERRED: needs real variables.xlsx (option b) |
| `create_aspirations` (244) | CAMSIS aspiration score | `cleaning/aspirations.py::create_aspirations` | 🟡 |
| `create_factors` (272) | psych::fa factor scores | `cleaning/factors.py::create_factors` | 🟡 Pearson factor done; 3 polychoric factors DEFERRED (V3) |
| `get_complete_ncds` (309) | join cleaned+factors+aspir+essay | `cleaning/assemble.py::get_complete_ncds` | 🟡 |
| `create_essay_variables` (317) | assemble+filter essay features | `features/essay_variables.py` | ⬜ |
| `find_essay_teacher_genetics_overlap` (330) | overlap subset | `cleaning/assemble.py::find_essay_teacher_genetics_overlap` | 🟡 |
| `find_full_overlap` (348) | complete-case subset | `cleaning/assemble.py::find_full_overlap` | 🟡 |
| `tokenize_essays` (356) | TreeTagger tokenization | `features/readability.py` (external-tool boundary) | ⬜ |
| `calculate_readability_metrics` (369) | koRpus readability | `features/readability.py` | ⬜ |
| `get_spelling_error_metrics` (390) | LanguageTool spelling CSV | `features/spelling.py` (CSV-ingestion contract) | ⬜ |
| `get_salat_metrics` (419) | SALAT tool CSVs | `features/salat.py` (CSV-ingestion contract) | ⬜ |
| `get_roberta_embeddings` (446) | RoBERTa embeddings | `features/embeddings.py::roberta` | ⬜ |
| `get_gpt_embeddings` (2) | read GPT embedding RDS | `features/embeddings.py::gpt` | ⬜ |

## `R/_targets.R` (504 lines, 177 targets)
The DAG: data-load, essay, clean, and ~110 model/metric targets (mostly
combinatorial). Python target: `pipeline/` (declarative model spec + runner per
`docs/ORCHESTRATION.md`). Status: ⬜ NOT STARTED.

## `R/create_data.R` (601 lines)
Post-pipeline figure/appendix CSV generation (`fig_2..5_data.csv`,
`appendix_D1..D12`). Has known bugs (PORTING_NOTES A6). Python target:
`scripts/make_figures.py`. Status: ⬜ NOT STARTED.

## `R/get_gpt_embeddings.R` (52 lines)
Standalone OpenAI embedding generation (ada-002 + text-embedding-3-large). Python
target: `scripts/get_gpt_embeddings.py`. Status: ⬜ NOT STARTED.

## `R/run.R` (7 lines)
`targets::tar_make()`. Python target: a `run.py` / CLI entry. Status: ⬜ NOT STARTED.

---

## Honest completion estimate
By translated source lines: ~55% of `functions.R` (metrics + models + IO + the
cleaning block except clean_ncds). clean_ncds (~180 lines) is deferred pending the
real variables.xlsx (option b). 0% of the other three code files. The empty package folders (`io/`, `cleaning/`, `features/`, `pipeline/`) are
placeholders for the ~75% not yet written.

## What "validation deferred" means per item
Anything touching restricted data (all `io/` readers, `clean_ncds`, `create_factors`,
`create_aspirations`, essay features) will be translated from the R faithfully but
CANNOT be run/validated until the real NCDS data (and ideally the author's correct
`variables.xlsx` + derived features) arrive. Those get tested for mechanics on
synthetic fixtures now; numerical validation is logged in
`docs/VALIDATION_CHECKLIST.md` and happens later. We do NOT guess values where the R
depends on data we don't have — we port the logic and mark the gap.
