"""Central configuration for the Python replication.

Every path and model parameter lives here so that nothing is hardcoded in the
logic modules. This directly replaces the machine-specific paths in the original
R code (e.g. ``C:/TreeTagger``, ``C:/Users/usr/anaconda3/python.exe``).

Every value meant to match the R cites its source line (``# R: llm_paper/...:L<n>``).
Restricted data is never read from or written into the repository: see
``LCP_DATA_ROOT`` below and docs/PORTING_NOTES.md section H.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- Paths: public files inside the repository --------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Public reference files shipped with the original R repository. These are the only
# data files allowed in the repository (.gitignore, scripts/check_no_restricted_data.py).
CAMSIS_FILE = DATA_DIR / "camsis" / "gb71co70.dta"  # R: llm_paper/_targets.R:L92 (the only CAMSIS file used)
VARIABLES_XLSX = DATA_DIR / "variables.xlsx"  # R: llm_paper/_targets.R:L43
OCCUPATION_ASPIRATION_XLSX = DATA_DIR / "occupation_aspiration_mapping.xlsx"  # R: llm_paper/_targets.R:L94

# --- Paths: restricted data, outside the repository ---------------------------
# Restricted inputs (NCDS sweeps, essays, derived essay features, polygenic scores)
# and participant-level outputs live ONLY under the directory named by this
# environment variable. There is deliberately no default, and a directory inside the
# repository is refused (brief Section 2.3).
LCP_DATA_ROOT_ENV = "LCP_DATA_ROOT"

# Every restricted file name, relative to $LCP_DATA_ROOT, in one place. The names
# follow the paths the original _targets.R read from its data/ folder. They are NOT
# confirmed: the real UK Data Service file names will be checked when the downloads
# arrive, and this table is the one place to change them.
RESTRICTED_INPUTS: dict[str, str] = {
    "essays": "essays",  # folder; R: llm_paper/_targets.R:L42
    "ncds_1_2_3": "ncds_1_2_3/ncds0123.dta",  # R: llm_paper/_targets.R:L45
    "ncds_4": "ncds_4/ncds4.dta",  # R: llm_paper/_targets.R:L49
    "ncds_5": "ncds_5/ncds5cmi.dta",  # R: llm_paper/_targets.R:L53
    "ncds_6": "ncds_6/ncds6.dta",  # R: llm_paper/_targets.R:L57
    "ncds_7": "ncds_7/ncds7.dta",  # R: llm_paper/_targets.R:L61
    "ncds_8": "ncds_8/ncds_2008_followup.dta",  # R: llm_paper/_targets.R:L65
    "ncds_9": "ncds_9/ncds_2013_flatfile.dta",  # R: llm_paper/_targets.R:L69
    "ncds_occ_2": "ncds_occ_coding/ncds2_occupation_coding_father.dta",  # R: llm_paper/_targets.R:L73
    "ncds_occ_5": "ncds_occ_coding/ncds5_occupation_coding_cm.dta",  # R: llm_paper/_targets.R:L77
    "ncds_occ_6": "ncds_occ_coding/ncds6_occupation_coding_cm.dta",  # R: llm_paper/_targets.R:L81
    "ncds_occ_7": "ncds_occ_coding/ncds7_occupation_coding_cm.dta",  # R: llm_paper/_targets.R:L85
    "ncds_occ_8": "ncds_occ_coding/ncds8_occupation_coding_cm.dta",  # R: llm_paper/_targets.R:L89
    "spelling_mistakes": "spelling_mistakes.csv",  # R: llm_paper/_targets.R:L100
    "taaled_1": "salat_metrics/essays_1_taaled.csv",  # R: llm_paper/_targets.R:L103
    "taaled_2": "salat_metrics/essays_2_taaled.csv",  # R: llm_paper/_targets.R:L104
    "taaled_3": "salat_metrics/essays_3_taaled.csv",  # R: llm_paper/_targets.R:L105
    "taales_1": "salat_metrics/essays_1_taales.csv",  # R: llm_paper/_targets.R:L106
    "taales_2": "salat_metrics/essays_2_taales.csv",  # R: llm_paper/_targets.R:L107
    "taales_3": "salat_metrics/essays_3_taales.csv",  # R: llm_paper/_targets.R:L108
    "seance_1": "salat_metrics/essays_1_seance.csv",  # R: llm_paper/_targets.R:L109
    "seance_2": "salat_metrics/essays_2_seance.csv",  # R: llm_paper/_targets.R:L110
    "seance_3": "salat_metrics/essays_3_seance.csv",  # R: llm_paper/_targets.R:L111
    # GPT embeddings. The R saved raw API responses as data/embeddings_gpt35_raw.rds and
    # data/embeddings_gpt4_raw.rds (R: llm_paper/R/get_gpt_embeddings.R:L29, L50). The
    # port stores them as CSV files with an ncdsid column (PORTING_NOTES G1); names and
    # format chosen by the port (CSV, so no Parquet library is needed).
    "gpt35_embeddings": "embeddings/embeddings_gpt35.csv",
    "gpt4_embeddings": "embeddings/embeddings_gpt4.csv",
    # Readability indices computed outside Python (TreeTagger + koRpus, r/readability.R,
    # Task 2.5), read by features/readability.py::ingest_readability_metrics. Name chosen
    # by the port.
    "readability_metrics": "readability_metrics.csv",
    # Polygenic scores, OPTIONAL. The R reader is an empty placeholder (R:
    # llm_paper/R/functions.R:L34–36). Placeholder format chosen by the port: CSV with
    # ncdsid first, then one numeric column per score (io/readers.py::read_gene_data).
    # The released files' format is a Phase 5/6 item. When the file is absent, the
    # gene-dependent targets are skipped (PORTING_NOTES L2).
    "gene_data": "genetics/polygenic_scores.csv",
}

# OUTPUT folders, always under $LCP_DATA_ROOT: participant-level outputs (derived,
# fits, logs) and the aggregate tables — the metric rows and the reporting tables —
# which stay there until they pass the export guard (Task 2.7).
PARTICIPANT_OUTPUT_DIRS = ("derived", "fits", "logs", "metrics", "reporting")

# Files written by separate pipeline steps into $LCP_DATA_ROOT/derived/ (names chosen by
# the port). RoBERTa embeddings are generated in their own process (isolation.py).
DERIVED_FILES: dict[str, str] = {
    "roberta_embeddings": "roberta_embeddings.csv",
}


class DataRootError(RuntimeError):
    """$LCP_DATA_ROOT is unset, missing, or points inside the repository."""


def data_root() -> Path:
    """Return the restricted-data root from ``$LCP_DATA_ROOT``, after checking it.

    Raises :class:`DataRootError` when the variable is unset or empty, when the
    directory does not exist, or when it resolves (symlinks followed) to the
    repository itself or to a path inside it.
    """
    raw = os.environ.get(LCP_DATA_ROOT_ENV, "").strip()
    if not raw:
        raise DataRootError(
            f"{LCP_DATA_ROOT_ENV} is not set. Restricted data (NCDS sweeps, essays, "
            "derived essay features, polygenic scores) is read only from the directory "
            f"named by {LCP_DATA_ROOT_ENV}, which must lie outside this repository. "
            "There is no default."
        )
    root = Path(raw).expanduser().resolve()
    if root == PROJECT_ROOT or PROJECT_ROOT in root.parents:
        raise DataRootError(
            f"{LCP_DATA_ROOT_ENV}={raw!r} resolves to {root}, which is inside the "
            f"repository ({PROJECT_ROOT}). Restricted data must live outside the "
            "repository so it can never be committed."
        )
    if not root.is_dir():
        raise DataRootError(f"{LCP_DATA_ROOT_ENV}={raw!r} does not exist or is not a directory.")
    return root


def restricted_path(key: str) -> Path:
    """Path of a restricted input, e.g. ``restricted_path("ncds_4")``."""
    if key not in RESTRICTED_INPUTS:
        raise KeyError(f"unknown restricted input {key!r}; known: {sorted(RESTRICTED_INPUTS)}")
    return data_root() / RESTRICTED_INPUTS[key]


def participant_output_dir(kind: str) -> Path:
    """Folder for participant-level outputs (``derived``, ``fits`` or ``logs``) under
    ``$LCP_DATA_ROOT``, created if missing."""
    if kind not in PARTICIPANT_OUTPUT_DIRS:
        raise ValueError(f"kind must be one of {PARTICIPANT_OUTPUT_DIRS}, got {kind!r}")
    path = data_root() / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def derived_dir() -> Path:
    return participant_output_dir("derived")


def derived_path(key: str) -> Path:
    """Path of a derived, participant-level file under ``$LCP_DATA_ROOT/derived/``."""
    if key not in DERIVED_FILES:
        raise KeyError(f"unknown derived file {key!r}; known: {sorted(DERIVED_FILES)}")
    return derived_dir() / DERIVED_FILES[key]


def fits_dir() -> Path:
    return participant_output_dir("fits")


def logs_dir() -> Path:
    return participant_output_dir("logs")


def metrics_dir() -> Path:
    return participant_output_dir("metrics")


# Written ONLY by the synthetic-data generator (tests/fixtures/synthetic_ncds.py) at the
# top of a synthetic $LCP_DATA_ROOT. The package never writes it. The runner refuses
# the smoke configuration unless it is present (pipeline/execute.py).
SYNTHETIC_MARKER_FILE = "SYNTHETIC_DATA_MARKER.json"


# --- Factor scores -------------------------------------------------------------
# "r": all four create_factors scores from R's psych::fa through rpy2, the reference
# implementation (needs R + rpy2 + psych). "native": the Pearson factor in numpy; the
# three polychoric factors raise (PORTING_NOTES F1, F2). "r" is the default because it
# is the only backend that produces all four factors as the R does.
FACTOR_BACKEND = "r"


# --- Model / CV parameters ---------------------------------------------------
@dataclass(frozen=True)
class SuperLearnerConfig:
    # R: llm_paper/R/functions.R:L512 — cvControl = list(V = 10), innerCvControl = list(list(V = 5)).
    outer_folds: int = 10
    inner_folds: int = 5
    # The base seed of the port's own random streams (models/seeds.py). 1 echoes
    # parallel::clusterSetRNGStream(cluster, 1) (R: llm_paper/R/functions.R:L509), but
    # that call seeds only the worker processes. CV.SuperLearner draws the outer folds
    # in the master process (R pkg: SuperLearner/R/CV.SuperLearner.R:L19,
    # CVFolds.R:L23), whose RNG state the published code does not set, so R's fold
    # split cannot be recovered. The port therefore owns its folds (PORTING_NOTES C2).
    seed: int = 1
    family: str = "gaussian"
    # R: llm_paper/R/functions.R:L514–519 — the SL.library; every learner except
    # SL.mean is paired with the screen.glmnet screener.
    learners: tuple[str, ...] = (
        "SL.mean",
        "SL.ranger",
        "SL.nnet",
        "SL.xgboost.hist",  # SL.xgboost with params=list(tree_method="hist")
        "SL.ksvm",
        "SL.lm",
    )
    screener: str = "screen.glmnet"  # LASSO pre-screen, applied to all but SL.mean
    # No `method =` is passed (R: llm_paper/R/functions.R:L511–520), so SuperLearner's
    # default applies (R pkg: SuperLearner/R/CV.SuperLearner.R:L3, method.NNLS).
    meta_method: str = "method.NNLS"


@dataclass(frozen=True)
class LmConfig:
    # R: llm_paper/R/functions.R:L540–543 — get_lm_cv_model: same CV control,
    # SL.library = list("SL.mean", c("SL.lm")).
    outer_folds: int = 10
    inner_folds: int = 5
    seed: int = 1
    family: str = "gaussian"
    learners: tuple[str, ...] = ("SL.mean", "SL.lm")


# Per-learner settings live in models/base_learners.py, where each one cites its
# SuperLearner wrapper line (or package default) or is marked APPROX; see
# docs/PORTING_NOTES.md section C and VALIDATION_CHECKLIST V4/V7.

SUPERLEARNER = SuperLearnerConfig()
LM = LmConfig()


# --- Run configurations (pipeline/execute.py) ----------------------------------
@dataclass(frozen=True)
class RunConfig:
    """How the runner fits the models. ``n_jobs`` only spreads the model fits over
    worker processes; the output does not depend on it."""

    name: str
    outer_folds: int
    inner_folds: int
    seed: int = SUPERLEARNER.seed
    n_jobs: int = 1
    factor_backend: str | None = None  # None: FACTOR_BACKEND
    save_predictions: bool = True
    # Written into every output of a run with this configuration (file-name prefix and
    # a run_label column); empty for the paper configuration on real data.
    label: str = ""
    file_prefix: str = ""


# The configuration of the paper: 10 outer and 5 inner folds, the 6-learner library
# (R: llm_paper/R/functions.R:L512, L541).
PAPER_RUN = RunConfig(name="paper", outer_folds=SUPERLEARNER.outer_folds, inner_folds=SUPERLEARNER.inner_folds)

# SMOKE configuration: for checking the plumbing on synthetic data only. Fewer folds
# (2 outer, 2 inner); same learners. NOT A REAL RUN: the runner refuses it unless
# $LCP_DATA_ROOT holds SYNTHETIC_MARKER_FILE and every ID is synthetic, and every
# output it writes carries the label below.
SMOKE_RUN = RunConfig(
    name="smoke", outer_folds=2, inner_folds=2,
    label="SMOKE RUN on synthetic data: not results",
    file_prefix="SMOKE_",
)
