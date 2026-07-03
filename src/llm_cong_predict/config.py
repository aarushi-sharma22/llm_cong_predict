"""Central configuration for the Python replication.

Every path and model parameter lives here so that nothing is hardcoded in the
logic modules. This directly replaces the machine-specific paths in the original
R code (e.g. ``C:/TreeTagger``, ``C:/Users/usr/anaconda3/python.exe``).

Parameters are annotated by how confident we are in them:
  * CERTAIN  -> read directly and unambiguously from the original source.
  * TO_VERIFY -> must be checked against the SuperLearner wrapper source or the
                 paper before we can claim numerical fidelity. Not guessed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SCHEMA_DIR = DATA_DIR / "schema"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Restricted inputs (absent until the user obtains UKDS/NCDS access). The paths
# mirror the layout the original _targets.R expected under data/.
NCDS_DIRS = {
    "ncds_1_2_3": DATA_DIR / "ncds_1_2_3" / "ncds0123.dta",
    "ncds_4": DATA_DIR / "ncds_4" / "ncds4.dta",
    "ncds_5": DATA_DIR / "ncds_5" / "ncds5cmi.dta",
    "ncds_6": DATA_DIR / "ncds_6" / "ncds6.dta",
    "ncds_7": DATA_DIR / "ncds_7" / "ncds7.dta",
    "ncds_8": DATA_DIR / "ncds_8" / "ncds_2008_followup.dta",
    "ncds_9": DATA_DIR / "ncds_9" / "ncds_2013_flatfile.dta",
    "ncds_occ_2": DATA_DIR / "ncds_occ_coding" / "ncds2_occupation_coding_father.dta",
    "ncds_occ_5": DATA_DIR / "ncds_occ_coding" / "ncds5_occupation_coding_cm.dta",
    "ncds_occ_6": DATA_DIR / "ncds_occ_coding" / "ncds6_occupation_coding_cm.dta",
    "ncds_occ_7": DATA_DIR / "ncds_occ_coding" / "ncds7_occupation_coding_cm.dta",
    "ncds_occ_8": DATA_DIR / "ncds_occ_coding" / "ncds8_occupation_coding_cm.dta",
}
CAMSIS_FILE = DATA_DIR / "camsis" / "gb71co70.dta"  # the ONLY camsis file the pipeline uses
VARIABLES_XLSX = DATA_DIR / "variables.xlsx"
OCCUPATION_ASPIRATION_XLSX = DATA_DIR / "occupation_aspiration_mapping.xlsx"

# Reconstructed, provenance-flagged mapping (see data/schema/). This is our
# stand-in for the correct variables.xlsx until the author provides the real one.
VARIABLE_MAPPING_YAML = SCHEMA_DIR / "ncds_variable_mapping.yaml"
ESSAY_FEATURE_SCHEMA_YAML = SCHEMA_DIR / "essay_feature_schema.yaml"


# --- Model / CV parameters ---------------------------------------------------
@dataclass(frozen=True)
class SuperLearnerConfig:
    # CERTAIN: from cvControl=list(V=10) / innerCvControl=list(list(V=5)).
    outer_folds: int = 10
    inner_folds: int = 5
    # CERTAIN: from parallel::clusterSetRNGStream(cluster, 1).
    seed: int = 1
    family: str = "gaussian"
    # CERTAIN: the SL.library list is written explicitly in
    # get_general_superlearner_cv_model(). Each learner is paired with the
    # screen.glmnet screener except SL.mean.
    learners: tuple[str, ...] = (
        "SL.mean",
        "SL.ranger",
        "SL.nnet",
        "SL.xgboost.hist",  # SL.xgboost with params=list(tree_method="hist")
        "SL.ksvm",
        "SL.lm",
    )
    screener: str = "screen.glmnet"  # LASSO pre-screen, applied to all but SL.mean
    # The meta-learner. SuperLearner's default is method.NNLS (non-negative least
    # squares on the convex combination of learners). CERTAIN: no `method=` arg is
    # passed, so the default applies.
    meta_method: str = "method.NNLS"


@dataclass(frozen=True)
class LmConfig:
    # get_lm_cv_model: SL.library = list("SL.mean", c("SL.lm")); same CV control.
    outer_folds: int = 10
    inner_folds: int = 5
    seed: int = 1
    family: str = "gaussian"
    learners: tuple[str, ...] = ("SL.mean", "SL.lm")


# TO_VERIFY: the per-learner default hyperparameters differ between the R
# SuperLearner wrappers and scikit-learn, and the paper relied on defaults. These
# must be extracted from the SL.* wrapper source (SuperLearner package) before the
# native backend can be claimed to match. We deliberately do NOT guess them here;
# they are pinned during the model-layer phase and cross-checked with the rpy2
# oracle. See docs/PORTING_NOTES.md ("Base-learner defaults").
BASE_LEARNER_DEFAULTS_STATUS = "TO_VERIFY: extract from SuperLearner wrapper source"

SUPERLEARNER = SuperLearnerConfig()
LM = LmConfig()
