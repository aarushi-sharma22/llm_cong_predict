# llm_cong_predict

Python replication of the analysis pipeline from:

> Wolfram, T. (2025). *Large Language Models Predict Cognition and Education Close
> to or Better than Genomics or Expert Assessment.* Communications Psychology.
> https://www.nature.com/articles/s44271-025-00274-x

A Python port of the original R code (https://github.com/tobiaswolfram/llm_paper,
commit `b0cfe4c`). By default it reproduces what the R evidently did, including its
quirks. Where the published R cannot have run, the port implements the evident intent
and logs a *reconstruction*. **Every deviation is logged** in
[`docs/PORTING_NOTES.md`](docs/PORTING_NOTES.md) with its R source lines and the test
that covers it. The progress of each R function is in
[`docs/TRANSLATION_STATUS.md`](docs/TRANSLATION_STATUS.md).

## Reproducibility status (read this first)

This is a code artifact, not a one-click reproduction. The original inputs are
access-restricted and are **not** included:

- **NCDS phenotypic + essay data** — via UK Data Service registration.
- **Genetic / polygenic scores** — separate NCDS Data Access Committee application.
- **Derived essay features** (embeddings, SALAT linguistic metrics, spelling) —
  confidential; shareable by the original author to UKDS-approved users, or
  regenerable locally (RoBERTa weights run on the local machine; the SALAT desktop
  tools; a local LanguageTool). Sending essays to an external API (OpenAI, hosted
  inference, the public LanguageTool API) is against this project's rules, see
  *Data safety* below.

Accordingly, the pipeline is developed and tested against **synthetic fixtures that
match the real data schemas**. Reproducing the paper's *numbers* requires the real
data and completion of [`docs/VALIDATION_CHECKLIST.md`](docs/VALIDATION_CHECKLIST.md).

## Model backends

- **native** (default): the Super Learner with nested CV in Python
  (`models/native_superlearner.py`), with each SuperLearner wrapper's settings
  reproduced and cited (`models/base_learners.py`). Installs without R. Where Python
  can only approximate an R learner, the code says `# APPROX` and
  `docs/VALIDATION_CHECKLIST.md` lists how the gap is measured.
- **oracle** (optional, `pip install -e '.[oracle]'`): calls R's `SuperLearner`
  through `rpy2` on the same folds, to measure the native backend's gap. Requires R
  and the R packages listed in [`docs/REFERENCE_SOURCES.md`](docs/REFERENCE_SOURCES.md).

R is not only for the oracle: three of the twelve outcomes are polychoric factor scores
that only `psych::fa` produces, so a full run needs R, rpy2 and psych
(`config.FACTOR_BACKEND = "r"`). With `"native"` those three outcomes, and the samples
that need them, are reported as not run.
[`docs/ORCHESTRATION.md`](docs/ORCHESTRATION.md) has the full table.

## Running the pipeline

```bash
python run.py                       # print the plan: what runs, what comes from elsewhere
python run.py --run --n-jobs 8      # the paper configuration (10 outer, 5 inner folds)
python run.py --run --config smoke  # the smoke configuration: synthetic data only
python run.py --run --targets essay_lm_lm_metrics   # one target and its dependencies
```

A run reads the restricted inputs from `$LCP_DATA_ROOT` and writes the metric rows, the
run log and the per-person predictions back under it. Three steps happen outside the
pipeline and are read as files: the RoBERTa embeddings
(`python -m llm_cong_predict.features.roberta_step`), the koRpus readability CSV
(`r/readability.R`, which needs TreeTagger and koRpus and has never been run here), and
the SALAT and LanguageTool outputs. R is needed at run time for
the three polychoric factor scores; see
[`docs/ORCHESTRATION.md`](docs/ORCHESTRATION.md) for where R is used and what happens
without it. The pipeline has so far run end to end only on synthetic data
(`tests/fixtures/synthetic_ncds.py`).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest                       # the full suite, including tests marked slow
```

Optional extras: `.[embeddings]` (torch, transformers, pyarrow: local RoBERTa
embeddings), `.[oracle]` (rpy2), `.[external-api]` (the OpenAI client, used only by a
script that refuses to run by default). On macOS, xgboost needs the OpenMP runtime
(`brew install libomp`). torch bundles a different OpenMP runtime, and using torch and
xgboost in one process crashes, so RoBERTa embeddings are generated in a separate
process (`docs/PORTING_NOTES.md` G2).

## Data safety

Restricted data (NCDS sweeps, essays, derived essay features, embeddings, polygenic
scores) and every participant-level output live **outside** this repository, in the
directory named by the environment variable `LCP_DATA_ROOT`. There is no default, and a
directory inside the repository is refused. File names are listed in one place,
`RESTRICTED_INPUTS` in `src/llm_cong_predict/config.py`.

A table leaves `LCP_DATA_ROOT` only through the export guard,
`export.py::export_table`, which refuses a table with an ID-like column, a free-text
column, one row per person, or a count below the minimum cell size. That minimum has no
default: `config.MINIMUM_CELL_SIZE` is `None`, so every export is refused until the
value has been confirmed against the UK Data Service's output rules. Passing the guard
is four mechanical checks, not a disclosure review.

`.gitignore` is an allow-list for `data/`: only `data/variables.xlsx`,
`data/occupation_aspiration_mapping.xlsx` and `data/camsis/*.dta` (public reference
files) can be tracked. Because `git add -f` bypasses `.gitignore`, a checker also runs
before every commit. Enable it once per clone:

```bash
git config core.hooksPath scripts/hooks
```

The hook runs `scripts/check_no_restricted_data.py`, which refuses staged data-like
files, anything else under `data/`, CSV/TXT files outside `tests/` and `docs/`, and any
file over 5 MB. Run it by hand on every tracked file with
`python scripts/check_no_restricted_data.py --all`.

`scripts/get_gpt_embeddings.py` sends essays to the OpenAI API. It is kept only for
provenance and refuses to run without both
`--i-confirm-the-data-licence-permits-external-processing` and
`LCP_ALLOW_EXTERNAL_API=1`. Its client is in a separate extra (`.[external-api]`), so
installing `.[embeddings]` never installs it.

## Attribution / licensing

The original repository ships no license (all rights reserved) and no citation
file. Attribution and reuse terms should be settled with the original author before
any public release of this replication.

## Layout

```
src/llm_cong_predict/
  config.py        # public file paths, LCP_DATA_ROOT, restricted file names, CV settings
  io/              # readers (.dta, .xlsx, essays), value labels, dplyr-style joins
  cleaning/        # create_aspirations, create_factors, assembly and overlap subsets
  features/        # SALAT/spelling/readability ingestion, embeddings, essay variables,
                   # roberta_step.py (RoBERTa embeddings, run as its own process)
  isolation.py     # torch and xgboost never share a process
  models/          # native Super Learner, base learners, screen.glmnet, rpy2 oracle
  metrics/         # cross-validated metric rows (get_cv_superlearner/lm_metrics)
  reporting/       # the figure and appendix tables (port of R/create_data.R)
  pipeline/        # dependency graph, variable lists, model specification, the runner
r/
  readability.R    # TreeTagger + koRpus readability, run by hand in R; NEVER RUN HERE
scripts/
  check_no_restricted_data.py   # refuses restricted data in commits (pre-commit hook)
  hooks/pre-commit
  extract_r_targets.py          # _targets.R -> docs/reference/r_targets_inventory.json
  validate_oracle.py            # native vs R SuperLearner on the same folds (needs R)
  check_essay_format.py         # aggregate format counts for the essay files (numbers only)
  get_gpt_embeddings.py         # provenance only; refuses to run by default
data/              # public reference files only: variables.xlsx, occupation mapping, camsis/*.dta
docs/              # porting notes, validation checklist, reference sources, plan, brief
tests/             # unit tests on synthetic data; nothing real
  fixtures/synthetic_ncds.py    # the complete synthetic input set for an end-to-end run
```

`reference/` (git-ignored) holds clones of the R sources the port cites; see
[`docs/REFERENCE_SOURCES.md`](docs/REFERENCE_SOURCES.md).
