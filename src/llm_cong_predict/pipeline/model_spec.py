"""Declarative model specification: the 70 model targets of ``_targets.R``.

The original hand-wrote every model target (R: llm_paper/_targets.R:L178–385). Here
they are generated from 13 families. Each generated target records what the R target
does: method, outcome (and so the ``pattern`` it iterates over), predictor sets,
sample, the data preparation applied to the sample inside the target, and which
scorer turns the fit into a metric row. ``R_TO_PYTHON_TARGET`` maps every R name to
its (regularised) Python name. tests/test_pipeline.py checks all of this against
``docs/reference/r_targets_inventory.json``, which scripts/extract_r_targets.py
derives from the R files.

Scorers: ``_targets.R`` uses ``get_cv_lm_metrics`` only for the seven
``*_lm`` targets (L452–464) and ``get_cv_superlearner_metrics`` for every other
metric target, including the three ``*_social_lm`` lm fits (L489–494). Targets
without a metric target in ``_targets.R`` are scored by ``R/create_data.R`` with
``get_cv_superlearner_metrics`` (text_length and the seven text components,
L163–170; the two ``*_social_lm_overlap`` targets, L344, L347). The seven
``*_superlearner_mmg_lm`` targets are scored nowhere; they get scorer "none", so they
are fitted but produce no metric row.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .graph import Status, Target
from .variable_lists import CONSTANT_LISTS


@dataclass(frozen=True)
class Ref:
    """A predictor or outcome reference: a variable-list target or a literal column."""

    kind: str  # "list" or "literal"
    value: str

    def as_dict(self) -> dict:
        return {"kind": self.kind, "value": self.value}


def _lists(*names: str) -> tuple[Ref, ...]:
    return tuple(Ref("list", n) for n in names)


def _literal(column: str) -> tuple[Ref, ...]:
    return (Ref("literal", column),)


# Feature set -> predictor references, in the R's order.
FEATURE_SETS: dict[str, tuple[Ref, ...]] = {
    "essay": _lists("essay_variables"),
    "gene": _lists("gene_variables"),
    "teacher": _lists("teacher_variables"),
    "essay_genes": _lists("essay_variables", "gene_variables"),
    "essay_teacher": _lists("essay_variables", "teacher_variables"),
    "teacher_genes": _lists("teacher_variables", "gene_variables"),
    "teacher_genes_essay": _lists("teacher_variables", "gene_variables", "essay_variables"),
    "sociological": _lists("sociological_variables"),
    "cog": _lists("cog_variables"),
    "noncog": _lists("noncog_variables"),
    "birthweight": _literal("s0_mo_birthweight"),  # R: llm_paper/_targets.R:L204, L364
    "height": _literal("s3_co_height"),  # R: llm_paper/_targets.R:L206, L366
    "pedu": _literal("s3_pa_edu"),  # R: llm_paper/_targets.R:L208, L368
    "text_length": _literal("nwords"),  # R: llm_paper/_targets.R:L261
    "salat_metrics": _lists("salat_metrics_variables"),
    "readability_metrics": _lists("readability_metrics_variables"),
    "spelling_errors": _lists("spelling_errors_variables"),
    # R: llm_paper/_targets.R:L220 — c(salat, readability, spelling)
    "spelling_salat_readability": _lists(
        "salat_metrics_variables", "readability_metrics_variables", "spelling_errors_variables"),
    "gpt_embeddings": _lists("gpt_embeddings_variables"),
    "roberta_embeddings": _lists("roberta_embeddings_variables"),
    "gpt4_embeddings": _lists("gpt4_embeddings_variables"),
}

# Data preparation applied to the sample inside the model target. It
# depends only on the feature set: the same feature set gets the same preparation in
# every family where it appears.
DATA_PREP: dict[str, tuple[dict, ...]] = {
    # R: llm_paper/_targets.R:L208, L368 — mutate(s3_pa_edu = as.numeric(s3_pa_edu)):
    # the factor's level position, not its value.
    "pedu": ({"op": "as_numeric", "columns": ["s3_pa_edu"]},),
    # R: llm_paper/_targets.R:L258, L374 — mutate_at(vars(sociological_variables), as.numeric):
    # logical -> 0/1, ordered parent-education factors -> level positions 1..4.
    "sociological": ({"op": "as_numeric", "columns_from": "sociological_variables"},),
    # R: llm_paper/_targets.R:L227 — inner_join(roberta_embeddings, by = c("ncdsid" = "id"))
    "roberta_embeddings": ({"op": "inner_join", "table": "roberta_embeddings", "by": {"ncdsid": "id"}},),
    # R: llm_paper/_targets.R:L230 — select(-starts_with("embedding")) %>% inner_join(gpt4_embeddings, ...)
    "gpt4_embeddings": (
        {"op": "drop_columns_starting_with", "prefix": "embedding"},
        {"op": "inner_join", "table": "gpt4_embeddings", "by": {"ncdsid": "id"}},
    ),
}

_MAIN = ("essay", "gene", "teacher", "essay_genes", "essay_teacher", "teacher_genes",
         "teacher_genes_essay")
_TEXT = ("salat_metrics", "readability_metrics", "spelling_errors", "spelling_salat_readability",
         "gpt_embeddings", "roberta_embeddings", "gpt4_embeddings")
_SOCIAL = ("cog", "noncog", "birthweight", "height", "pedu")


@dataclass(frozen=True)
class ModelFamily:
    feature_sets: tuple[str, ...]
    outcome: Ref | str  # an outcome-list target (iterated, R `pattern`) or a literal column
    sample: str         # a data-frame target name
    method: str         # "superlearner" or "lm"
    suffix: str         # naming suffix for this family
    scorer: str = "superlearner"  # "superlearner", "lm" or "none"
    predictor_overrides: dict = field(default_factory=dict)  # feature set -> refs

    def __post_init__(self):
        # A plain string names an outcome-list target if one exists, else a column.
        if isinstance(self.outcome, str):
            kind = "list" if self.outcome in CONSTANT_LISTS else "literal"
            object.__setattr__(self, "outcome", Ref(kind, self.outcome))


# The families, reproducing the original's model combinations (R line ranges given).
FAMILIES: list[ModelFamily] = [
    # L178–198, metrics L387–400
    ModelFamily(_MAIN, Ref("list", "all_outcomes"), "ncds_complete_all_overlap", "superlearner", "_overlap", "superlearner"),
    # L200–209, metrics L467–476
    ModelFamily(_SOCIAL, Ref("list", "social_outcomes"), "ncds_complete_all_overlap", "superlearner", "_social_overlap", "superlearner"),
    # L211–231, scored in R/create_data.R:L164–170
    ModelFamily(_TEXT, Ref("list", "all_outcomes"), "ncds_complete", "superlearner", "_text", "superlearner"),
    # L234–259, metrics L434–450
    ModelFamily(_MAIN + ("sociological",), Ref("list", "all_outcomes"), "ncds_complete", "superlearner", "", "superlearner"),
    # L261–263, scored in R/create_data.R:L163 with get_cv_superlearner_metrics
    ModelFamily(("text_length",), Ref("list", "all_outcomes"), "ncds_complete", "lm", "", "superlearner"),
    # L265–285, metrics L452–464 (get_cv_lm_metrics)
    ModelFamily(_MAIN, Ref("list", "all_outcomes"), "ncds_complete", "lm", "_lm", "lm"),
    # L287–298, metrics L417–424
    ModelFamily(("essay", "gene", "teacher", "teacher_genes_essay"), Ref("list", "bfi_variables"), "ncds_complete", "superlearner", "_bfi", "superlearner"),
    # L301–321, metrics L402–415
    ModelFamily(_MAIN, Ref("list", "social_outcomes"), "ncds_complete_mmg", "superlearner", "_mmg", "superlearner"),
    # L323–343: scored nowhere
    ModelFamily(_MAIN, Ref("list", "social_outcomes"), "ncds_complete_mmg", "lm", "_mmg_lm", "none"),
    # L345–358 (no pattern: literal outcome), metrics L426–432
    ModelFamily(_MAIN, Ref("literal", "s2_co_factor_ability"), "ncds_complete_mmg_cog", "superlearner", "_cog_mmg", "superlearner"),
    # L360–369, metrics L478–487
    ModelFamily(_SOCIAL, Ref("list", "social_outcomes"), "ncds_complete", "superlearner", "_social", "superlearner"),
    # L372–380, metrics L489–494 (get_cv_superlearner_metrics on lm fits).
    # cog uses the single column s2_co_factor_ability here, not cog_variables (L377).
    ModelFamily(("sociological", "cog", "noncog"), Ref("list", "social_outcomes"), "ncds_complete", "lm", "_social_lm", "superlearner",
                predictor_overrides={"cog": _literal("s2_co_factor_ability")}),
    # L382–385, scored in R/create_data.R:L344, L347
    ModelFamily(("cog", "noncog"), Ref("list", "social_outcomes"), "ncds_complete_all_overlap", "lm", "_social_lm_overlap", "superlearner",
                predictor_overrides={"cog": _literal("s2_co_factor_ability")}),
]


@dataclass(frozen=True)
class ModelTargetSpec:
    name: str
    feature_set: str
    method: str
    outcome: Ref
    predictors: tuple[Ref, ...]
    sample: str
    data_prep: tuple[dict, ...]
    scorer: str

    @property
    def pattern(self) -> str | None:
        """The outcome list the R target iterates over (``pattern = ...``), if any."""
        return self.outcome.value if self.outcome.kind == "list" else None

    def n_fits(self) -> int:
        return len(CONSTANT_LISTS[self.pattern]) if self.pattern else 1

    def dependencies(self) -> tuple[str, ...]:
        deps = [p.value for p in self.predictors if p.kind == "list"]
        deps.append(self.sample)
        if self.outcome.kind == "list":
            deps.append(self.outcome.value)
        for step in self.data_prep:
            if "table" in step:
                deps.append(step["table"])
            if "columns_from" in step:
                deps.append(step["columns_from"])
        return tuple(dict.fromkeys(deps))


def family_specs(fam: ModelFamily) -> list[ModelTargetSpec]:
    """The model targets of one family."""
    return [
        ModelTargetSpec(
            name=f"{fs}_{fam.method}{fam.suffix}",
            feature_set=fs,
            method=fam.method,
            outcome=fam.outcome,
            predictors=fam.predictor_overrides.get(fs, FEATURE_SETS[fs]),
            sample=fam.sample,
            data_prep=DATA_PREP.get(fs, ()),
            scorer=fam.scorer,
        )
        for fs in fam.feature_sets
    ]


def model_specs() -> list[ModelTargetSpec]:
    """All 70 model targets, generated from FAMILIES."""
    specs = [s for fam in FAMILIES for s in family_specs(fam)]
    names = [s.name for s in specs]
    if len(names) != len(set(names)):  # pragma: no cover - guards the spec itself
        raise ValueError("duplicate model target names in FAMILIES")
    return specs


def _graph_targets(specs: list[ModelTargetSpec]) -> list[Target]:
    out: list[Target] = []
    for s in specs:
        predictors = "+".join(p.value for p in s.predictors)
        out.append(Target(s.name, s.dependencies(), Status.BUILT,
                          note=f"{s.method} on {predictors}, outcome={s.outcome.value}, "
                               f"sample={s.sample}, scorer={s.scorer}"))
        if s.scorer != "none":
            out.append(Target(f"{s.name}_metrics", (s.name,), Status.BUILT,
                              note=f"{s.scorer} scorer"))
    return out


def expand_family(fam: ModelFamily) -> list[Target]:
    """Graph targets of one family: each model, plus its metrics target when scored."""
    return _graph_targets(family_specs(fam))


def model_targets() -> list[Target]:
    """Graph targets: one per model, plus a ``<model>_metrics`` target when it is scored."""
    return _graph_targets(model_specs())


# Every R model target -> its Python target (names regularised as
# <feature set>_<method><family suffix>).
R_TO_PYTHON_TARGET: dict[str, str] = {
    "essay_superlearner_overlap": "essay_superlearner_overlap",
    "gene_superlearner_overlap": "gene_superlearner_overlap",
    "teacher_superlearner_overlap": "teacher_superlearner_overlap",
    "essay_genes_superlearner_overlap": "essay_genes_superlearner_overlap",
    "essay_teacher_superlearner_overlap": "essay_teacher_superlearner_overlap",
    "teacher_genes_superlearner_overlap": "teacher_genes_superlearner_overlap",
    "teacher_genes_essay_superlearner_overlap": "teacher_genes_essay_superlearner_overlap",
    "cog_superlearner_social_overlap": "cog_superlearner_social_overlap",
    "noncog_superlearner_social_overlap": "noncog_superlearner_social_overlap",
    "birthweight_superlearner_social_overlap": "birthweight_superlearner_social_overlap",
    "height_superlearner_social_overlap": "height_superlearner_social_overlap",
    "pedu_superlearner_social_overlap": "pedu_superlearner_social_overlap",
    "salat_metrics_superlearner": "salat_metrics_superlearner_text",
    "readability_metrics_superlearner": "readability_metrics_superlearner_text",
    "spelling_errors_superlearner": "spelling_errors_superlearner_text",
    "spelling_salat_readability_superlearner": "spelling_salat_readability_superlearner_text",
    "gpt_embeddings_superlearner": "gpt_embeddings_superlearner_text",
    "roberta_embeddings_superlearner": "roberta_embeddings_superlearner_text",
    "gpt4_embeddings_superlearner": "gpt4_embeddings_superlearner_text",
    "essay_superlearner": "essay_superlearner",
    "gene_superlearner": "gene_superlearner",
    "teacher_superlearner": "teacher_superlearner",
    "essay_genes_superlearner": "essay_genes_superlearner",
    "essay_teacher_superlearner": "essay_teacher_superlearner",
    "teacher_genes_superlearner": "teacher_genes_superlearner",
    "teacher_genes_essay_superlearner": "teacher_genes_essay_superlearner",
    "sociological_superlearner": "sociological_superlearner",
    "text_length": "text_length_lm",
    "essay_lm": "essay_lm_lm",
    "gene_lm": "gene_lm_lm",
    "teacher_lm": "teacher_lm_lm",
    "essay_genes_lm": "essay_genes_lm_lm",
    "essay_teacher_lm": "essay_teacher_lm_lm",
    "teacher_genes_lm": "teacher_genes_lm_lm",
    "teacher_genes_essay_lm": "teacher_genes_essay_lm_lm",
    "essay_bfi_superlearner": "essay_superlearner_bfi",
    "gene_bfi_superlearner": "gene_superlearner_bfi",
    "teacher_bfi_superlearner": "teacher_superlearner_bfi",
    "teacher_genes_essay_bfi_superlearner": "teacher_genes_essay_superlearner_bfi",
    "essay_superlearner_mmg": "essay_superlearner_mmg",
    "gene_superlearner_mmg": "gene_superlearner_mmg",
    "teacher_superlearner_mmg": "teacher_superlearner_mmg",
    "essay_genes_superlearner_mmg": "essay_genes_superlearner_mmg",
    "essay_teacher_superlearner_mmg": "essay_teacher_superlearner_mmg",
    "teacher_genes_superlearner_mmg": "teacher_genes_superlearner_mmg",
    "teacher_genes_essay_superlearner_mmg": "teacher_genes_essay_superlearner_mmg",
    "essay_superlearner_mmg_lm": "essay_lm_mmg_lm",
    "gene_superlearner_mmg_lm": "gene_lm_mmg_lm",
    "teacher_superlearner_mmg_lm": "teacher_lm_mmg_lm",
    "essay_genes_superlearner_mmg_lm": "essay_genes_lm_mmg_lm",
    "essay_teacher_superlearner_mmg_lm": "essay_teacher_lm_mmg_lm",
    "teacher_genes_superlearner_mmg_lm": "teacher_genes_lm_mmg_lm",
    "teacher_genes_essay_superlearner_mmg_lm": "teacher_genes_essay_lm_mmg_lm",
    "essay_superlearner_cog_mmg": "essay_superlearner_cog_mmg",
    "gene_superlearner_cog_mmg": "gene_superlearner_cog_mmg",
    "teacher_superlearner_cog_mmg": "teacher_superlearner_cog_mmg",
    "essay_genes_superlearner_cog_mmg": "essay_genes_superlearner_cog_mmg",
    "essay_teacher_superlearner_cog_mmg": "essay_teacher_superlearner_cog_mmg",
    "teacher_genes_superlearner_cog_mmg": "teacher_genes_superlearner_cog_mmg",
    "teacher_genes_essay_superlearner_cog_mmg": "teacher_genes_essay_superlearner_cog_mmg",
    "cog_superlearner_social": "cog_superlearner_social",
    "noncog_superlearner_social": "noncog_superlearner_social",
    "birthweight_superlearner_social": "birthweight_superlearner_social",
    "height_superlearner_social": "height_superlearner_social",
    "pedu_superlearner_social": "pedu_superlearner_social",
    "sociological_superlearner_social_lm": "sociological_lm_social_lm",
    "cog_superlearner_social_lm": "cog_lm_social_lm",
    "noncog_superlearner_social_lm": "noncog_lm_social_lm",
    "cog_superlearner_social_lm_overlap": "cog_lm_social_lm_overlap",
    "noncog_superlearner_social_lm_overlap": "noncog_lm_social_lm_overlap",
}


GENE_VARIABLES = "gene_variables"


def is_gene_dependent(spec: ModelTargetSpec) -> bool:
    """True when the target's predictors include the polygenic scores."""
    return any(p.kind == "list" and p.value == GENE_VARIABLES for p in spec.predictors)


def split_gene_dependent(specs: list[ModelTargetSpec], gene_data_available: bool):
    """``(runnable, skipped)``: without gene data, targets whose predictors include
    ``gene_variables`` are skipped.

    In R, ``gene_data`` is the function ``read_gene_data`` itself, so
    ``gene_variables`` is ``NULL`` (R: llm_paper/_targets.R:L93, L132). A gene-only model
    would then have no predictor and a combined one would silently lose the gene
    block. The port skips these targets instead and lists them in the run log
    (PORTING_NOTES L2). Samples defined with ``gene_variables`` (mmg, all_overlap) are
    kept: without gene data they are defined without it, as in R.
    """
    if gene_data_available:
        return list(specs), []
    runnable = [s for s in specs if not is_gene_dependent(s)]
    skipped = [s for s in specs if is_gene_dependent(s)]
    return runnable, skipped
