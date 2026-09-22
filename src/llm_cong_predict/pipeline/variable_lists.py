"""The variable-list targets of ``_targets.R``.

Constant lists are copied verbatim; composite lists are built the same way the R
builds them; lists read from data frames (``colnames(...)[-1]``) are data-dependent
and are computed when the pipeline runs. Checked against
``docs/reference/r_targets_inventory.json`` by tests/test_pipeline.py.
"""

from __future__ import annotations

# R: llm_paper/_targets.R:L120–130
TEACHER_VARIABLES = (
    "s2_te_general_knowledge", "s2_te_number_work",
    "s2_te_use_of_books", "s2_te_oral_ability",
    "s2_te_poor_hand_control", "s2_te_squirmy",
    "s2_te_poor_coordination", "s2_te_hardly_ever_still",
    "s2_te_poor_speech", "s2_te_inconsequential_behavior",
    "s2_te_miscellneous_symptoms", "s2_te_anxiety_adults",
    "s2_te_anxiety_children", "s2_te_hostility_children",
    "s2_te_writing_off_adults", "s2_te_misc",
    "s2_te_hostility_adults", "s2_te_restlessness",
    "s2_te_unforthcomingness", "s2_te_depression",
    "s2_te_withdrawal",
)
# R: llm_paper/_targets.R:L131
SOCIOLOGICAL_VARIABLES = (
    "s0_co_male", "s3_pa_father_edu", "s3_pa_mother_edu", "s0_mo_class_mother_husband",
    "s0_mo_persons_per_room", "s2_pa_nssec_father", "s0_mo_class_mother_father",
)
# R: llm_paper/_targets.R:L139 — paste0("roberta_dim_", 1:768)
ROBERTA_EMBEDDINGS_VARIABLES = tuple(f"roberta_dim_{i}" for i in range(1, 769))
# R: llm_paper/_targets.R:L143–146
COG_VARIABLES = (
    "s2_co_factor_ability",
    "s2_co_verbal_ability", "s2_co_nonverbal_ability",
    "s2_co_reading_ability", "s2_co_mathematics_ability",
    "s3_co_reading_ability", "s3_co_mathematics_ability",
)
# R: llm_paper/_targets.R:L147–149
NONCOG_VARIABLES = (
    "s2_co_aspiration_camsis",
    "s3_co_factor_scholastic_motivation",
    "s3_te_factor_externalizing", "s3_te_factor_internalizing",
)
COG_NONCOG_VARIABLES = COG_VARIABLES + NONCOG_VARIABLES  # R: llm_paper/_targets.R:L150
# R: llm_paper/_targets.R:L151–153
BFI_VARIABLES = (
    "s8_co_extraversion",
    "s8_co_agreeableness", "s8_co_conscientiousness",
    "s8_co_neuroticism", "s8_co_openness",
)
# R: llm_paper/_targets.R:L154
CONFOUNDER_VARIABLES = ("s3_pa_edu", "s3_co_height", "s0_co_male", "s0_mo_birthweight")
SOCIAL_OUTCOMES = ("s5_co_highest_edu",)  # R: llm_paper/_targets.R:L155
ALL_OUTCOMES = COG_NONCOG_VARIABLES + SOCIAL_OUTCOMES  # R: llm_paper/_targets.R:L159

CONSTANT_LISTS: dict[str, tuple[str, ...]] = {
    "teacher_variables": TEACHER_VARIABLES,
    "sociological_variables": SOCIOLOGICAL_VARIABLES,
    "roberta_embeddings_variables": ROBERTA_EMBEDDINGS_VARIABLES,
    "cog_variables": COG_VARIABLES,
    "noncog_variables": NONCOG_VARIABLES,
    "cog_noncog_variables": COG_NONCOG_VARIABLES,
    "bfi_variables": BFI_VARIABLES,
    "confounder_variables": CONFOUNDER_VARIABLES,
    "social_outcomes": SOCIAL_OUTCOMES,
    "all_outcomes": ALL_OUTCOMES,
}

# Lists read from data frames at run time: colnames(<frame>)[-1] and variants
# (R: llm_paper/_targets.R:L132–138).
DATA_DEPENDENT_LISTS = (
    "gene_variables", "essay_variables", "salat_metrics_variables",
    "readability_metrics_variables", "spelling_errors_variables",
    "gpt_embeddings_variables", "gpt4_embeddings_variables",
)

# Composites of data-dependent lists (R: llm_paper/_targets.R:L140–142, L156–158):
# (member lists in order, extra literal columns appended).
COMPOSITE_LISTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "mmg_variables": (("gene_variables", "essay_variables", "teacher_variables"), ()),
    "mmg_edu_variables": (("gene_variables", "essay_variables", "teacher_variables"), ("s5_co_highest_edu",)),
    "mmg_cog_variables": (("gene_variables", "essay_variables", "teacher_variables"), ("s2_co_factor_ability",)),
    "all_vars": (("teacher_variables", "gene_variables", "essay_variables", "cog_noncog_variables",
                  "social_outcomes", "confounder_variables"), ()),
}
