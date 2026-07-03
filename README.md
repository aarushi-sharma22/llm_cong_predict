# llm_cong_predict

Python replication of the analysis pipeline from:

> Wolfram, T. (2025). *Large Language Models Predict Cognition and Education Close
> to or Better than Genomics or Expert Assessment.* Communications Psychology.
> https://www.nature.com/articles/s44271-025-00274-x

A faithful Python port of the original R code
(https://github.com/tobiaswolfram/llm_paper), restructured as a maintainable
software project. Bugs in the original are fixed and **every deviation is logged**
in [`docs/PORTING_NOTES.md`](docs/PORTING_NOTES.md).

## Reproducibility status (read this first)

This is a code artifact, not a one-click reproduction. The original inputs are
access-restricted and are **not** included:

- **NCDS phenotypic + essay data** — via UK Data Service registration.
- **Genetic / polygenic scores** — separate NCDS Data Access Committee application.
- **Derived essay features** (embeddings, SALAT linguistic metrics, spelling) —
  confidential; shareable by the original author to UKDS-approved users, or
  regenerable via external tools (OpenAI/HuggingFace, LanguageTool CLI, the SALAT
  desktop tools).

Accordingly, the pipeline is developed and tested against **synthetic fixtures that
match the real data schemas**. Reproducing the paper's *numbers* requires the real
data and completion of [`docs/VALIDATION_CHECKLIST.md`](docs/VALIDATION_CHECKLIST.md).

## Model backends

- **native** (default): scikit-learn / XGBoost stacking ensemble with nested CV.
  Installs without R, so the artifact is usable by anyone.
- **oracle** (optional, `pip install -e '.[oracle]'`): calls the original R
  `SuperLearner` via `rpy2`. Used solely to validate the native backend's numbers
  against the original on real data. Requires a working R install.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Attribution / licensing

The original repository ships no license (all rights reserved) and no citation
file. Attribution and reuse terms should be settled with the original author before
any public release of this replication.

## Layout

```
src/llm_cong_predict/
  config.py        # all paths + model params (no hardcoded machine paths)
  io/              # readers for NCDS/CAMSIS/essays + metadata (replaces haven/readxl)
  cleaning/        # clean_ncds, create_factors, create_aspirations
  features/        # embeddings (native) + SALAT/readability CSV ingestion contracts
  models/          # native + rpy2 SuperLearner backends (shared contract)
  metrics/         # cv_metrics.py  <-- ported + tested
  pipeline/        # DAG wiring (declarative model spec)
scripts/           # get_gpt_embeddings.py, make_figures.py, setup_git.sh
data/schema/       # versioned, provenance-flagged variable mapping + feature schema
tests/             # golden-value + property tests on synthetic fixtures
```
