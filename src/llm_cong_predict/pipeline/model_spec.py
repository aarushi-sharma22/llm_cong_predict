"""Declarative model spec.

The original ``_targets.R`` hand-wrote ~110 near-identical model + metric targets,
which is why ``create_data.R`` became 600 fragile lines. Here the model targets are
GENERATED from a compact list of families. This is the single biggest structural
improvement over the original: adding a model is one spec line, not a copy-paste.

FAITHFULNESS NOTE: the families below reproduce the original's actual combinations
of {feature set × outcome × sample × method}, and therefore the same dependency
edges. The generated target NAMES are regularised (systematic), whereas the
original used ad-hoc names (e.g. ``essay_lm`` but ``essay_superlearner_mmg_lm``).
The dependency GRAPH is faithful; the string names are tidied. This is a
reimplementation, not a byte-for-byte name match.
"""

from __future__ import annotations

from dataclasses import dataclass

from .graph import Status, Target

# Feature set -> the variable-list target(s) it depends on. Feature sets whose
# "predictors" are a single literal column in the original (birthweight, height,
# pedu, text_length) carry no variable-list dependency — only the data frame.
FEATURE_DEPS: dict[str, tuple[str, ...]] = {
    "essay": ("essay_variables",),
    "gene": ("gene_variables",),
    "teacher": ("teacher_variables",),
    "essay_genes": ("essay_variables", "gene_variables"),
    "essay_teacher": ("essay_variables", "teacher_variables"),
    "teacher_genes": ("teacher_variables", "gene_variables"),
    "teacher_genes_essay": ("teacher_variables", "gene_variables", "essay_variables"),
    "sociological": ("sociological_variables",),
    "cog": ("cog_variables",),
    "noncog": ("noncog_variables",),
    "birthweight": (),   # literal s0_mo_birthweight
    "height": (),        # literal s3_co_height
    "pedu": (),          # literal s3_pa_edu
    "text_length": (),   # literal nwords
    "salat_metrics": ("salat_metrics_variables",),
    "readability_metrics": ("readability_metrics_variables",),
    "spelling_errors": ("spelling_errors_variables",),
    "spelling_salat_readability": (
        "salat_metrics_variables", "readability_metrics_variables", "spelling_errors_variables",
    ),
    "gpt_embeddings": ("gpt_embeddings_variables",),
    "roberta_embeddings": ("roberta_embeddings_variables",),
    "gpt4_embeddings": ("gpt4_embeddings_variables",),
}

_MAIN = ["essay", "gene", "teacher", "essay_genes", "essay_teacher", "teacher_genes", "teacher_genes_essay"]


@dataclass(frozen=True)
class ModelFamily:
    feature_sets: tuple[str, ...]
    outcome: str      # an outcome-list target name, or a literal outcome variable
    sample: str       # a data-frame target name
    method: str       # "superlearner" or "lm"
    suffix: str       # naming suffix for this family


# The families, reproducing the original's model combinations.
FAMILIES: list[ModelFamily] = [
    ModelFamily(tuple(_MAIN), "all_outcomes", "ncds_complete_all_overlap", "superlearner", "_overlap"),
    ModelFamily(("cog", "noncog", "birthweight", "height", "pedu"), "social_outcomes", "ncds_complete_all_overlap", "superlearner", "_social_overlap"),
    ModelFamily(("salat_metrics", "readability_metrics", "spelling_errors", "spelling_salat_readability", "gpt_embeddings", "roberta_embeddings", "gpt4_embeddings"), "all_outcomes", "ncds_complete", "superlearner", "_text"),
    ModelFamily(tuple(_MAIN) + ("sociological",), "all_outcomes", "ncds_complete", "superlearner", ""),
    ModelFamily(("text_length",), "all_outcomes", "ncds_complete", "lm", ""),
    ModelFamily(tuple(_MAIN), "all_outcomes", "ncds_complete", "lm", "_lm"),
    ModelFamily(("essay", "gene", "teacher", "teacher_genes_essay"), "bfi_variables", "ncds_complete", "superlearner", "_bfi"),
    ModelFamily(tuple(_MAIN), "social_outcomes", "ncds_complete_mmg", "superlearner", "_mmg"),
    ModelFamily(tuple(_MAIN), "social_outcomes", "ncds_complete_mmg", "lm", "_mmg_lm"),
    ModelFamily(tuple(_MAIN), "s2_co_factor_ability", "ncds_complete_mmg_cog", "superlearner", "_cog_mmg"),
    ModelFamily(("cog", "noncog", "birthweight", "height", "pedu"), "social_outcomes", "ncds_complete", "superlearner", "_social"),
    ModelFamily(("sociological", "cog", "noncog"), "social_outcomes", "ncds_complete", "lm", "_social_lm"),
    ModelFamily(("cog", "noncog"), "social_outcomes", "ncds_complete_all_overlap", "lm", "_social_lm_overlap"),
]

# Outcomes that are literal variable names rather than outcome-list targets (so they
# are not dependencies to be resolved in the graph).
_LITERAL_OUTCOMES = {"s2_co_factor_ability"}


def expand_family(fam: ModelFamily) -> list[Target]:
    """Expand one family into model + metric targets with correct dependencies."""
    targets: list[Target] = []
    for fs in fam.feature_sets:
        model_name = f"{fs}_{fam.method}{fam.suffix}"
        deps: list[str] = list(FEATURE_DEPS.get(fs, ()))
        deps.append(fam.sample)
        if fam.outcome not in _LITERAL_OUTCOMES:
            deps.append(fam.outcome)
        # The model function (native SuperLearner / lm) is BUILT; the target is only
        # ever *blocked* through its data/feature dependencies (e.g. clean_ncds).
        targets.append(
            Target(model_name, tuple(dict.fromkeys(deps)), Status.BUILT,
                   note=f"{fam.method} on {fs} predictors, outcome={fam.outcome}, sample={fam.sample}")
        )
        # metric target consumes the model
        metric_name = f"{model_name}_metrics"
        targets.append(Target(metric_name, (model_name,), Status.BUILT, note="cv metrics"))
    return targets


def model_targets() -> list[Target]:
    """All model + metric targets, generated from FAMILIES."""
    out: list[Target] = []
    seen: set[str] = set()
    for fam in FAMILIES:
        for t in expand_family(fam):
            if t.name in seen:
                continue  # a feature set may recur across families with same name
            seen.add(t.name)
            out.append(t)
    return out
