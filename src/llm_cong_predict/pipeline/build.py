"""Assemble the full pipeline DAG — a faithful reconstruction of ``_targets.R``.

Registers the data-load, essay, and cleaning targets with their real dependency
edges (verified against ``_targets.R``), then appends the generated model + metric
targets from ``model_spec``. Each target carries a status flag:

  * BUILT — the underlying Python function is implemented (io/cleaning/features/
            models/metrics);
  * STUB  — deliberately not implemented, and the code raises:
      - ``ncds_1_to_9_cleaned``  (clean_ncds, deferred — needs real variables.xlsx);
      - ``tokenized_essays`` / ``readability_metrics`` (TreeTagger/koRpus boundary);
      - ``gene_data`` (empty in original; reader raises).

The point of this module is to make the WIRING explicit and checkable, and to let
the blocked-node analysis show precisely how much is gated on ``clean_ncds``. It
does not run — that needs the real data and clean_ncds.
"""

from __future__ import annotations

from .graph import Pipeline, Status, Target
from .model_spec import model_targets

# NCDS wave + occupation-coding load targets (all via read_ncds, depend on mapping_df).
_NCDS_WAVES = [
    "ncds_1_2_3", "ncds_4", "ncds_5", "ncds_6", "ncds_7", "ncds_9",
    "ncds_occ_2", "ncds_occ_5", "ncds_occ_6", "ncds_occ_7", "ncds_occ_8",
]
# (ncds_8 is loaded too in the original; kept for completeness)
_NCDS_WAVES.insert(5, "ncds_8")


def _load_targets() -> list[Target]:
    t: list[Target] = [
        Target("ncds_essays", (), Status.BUILT, "read_essays"),
        Target("mapping_df", (), Status.BUILT, "read_datalist (note: public variables.xlsx has wrong schema, A1)"),
        Target("camsis_data", (), Status.BUILT, "read_camsis (runs on real shipped file)"),
        Target("occupation_aspiration_mapping", (), Status.BUILT, "read_occupation_aspiration_mapping (real file)"),
        Target("gene_data", (), Status.STUB, "read_gene_data raises (access-restricted; empty in original)"),
    ]
    for w in _NCDS_WAVES:
        t.append(Target(w, ("mapping_df",), Status.BUILT, "read_ncds"))
    return t


def _essay_targets() -> list[Target]:
    return [
        Target("tokenized_essays", ("ncds_essays",), Status.STUB, "tokenize_essays: TreeTagger external boundary"),
        Target("spelling_errors", ("ncds_essays",), Status.BUILT, "get_spelling_error_metrics (ingestion; needs CSV)"),
        Target("readability_metrics", ("ncds_essays", "tokenized_essays"), Status.STUB, "calculate_readability_metrics: koRpus external boundary"),
        Target("salat_metrics", ("ncds_essays",), Status.BUILT, "get_salat_metrics (ingestion; needs CSVs)"),
        Target("roberta_embeddings", ("ncds_essays",), Status.BUILT, "roberta_embeddings (native; needs model+essays)"),
        Target("gpt_embeddings", ("ncds_essays",), Status.BUILT, "gpt_embeddings reshaper (needs saved file)"),
        Target("gpt4_embeddings", ("ncds_essays",), Status.BUILT, "gpt_embeddings reshaper (gpt4 variant)"),
        Target("essay_data", ("salat_metrics", "readability_metrics", "spelling_errors", "gpt_embeddings"), Status.BUILT, "create_essay_variables"),
    ]


def _variable_list_targets() -> list[Target]:
    # Constant variable-list targets have no data dependency; derived ones depend on
    # the frame they read column names from.
    const = [
        "teacher_variables", "sociological_variables", "cog_variables", "noncog_variables",
        "bfi_variables", "confounder_variables", "social_outcomes", "roberta_embeddings_variables",
    ]
    t = [Target(n, (), Status.BUILT, "constant variable list") for n in const]
    t += [
        Target("gene_variables", ("gene_data",), Status.BUILT, "colnames(gene_data)"),
        Target("essay_variables", ("essay_data",), Status.BUILT, "colnames(essay_data)"),
        Target("salat_metrics_variables", ("salat_metrics", "essay_data"), Status.BUILT, ""),
        Target("readability_metrics_variables", ("readability_metrics", "essay_data"), Status.BUILT, ""),
        Target("spelling_errors_variables", ("spelling_errors", "essay_data"), Status.BUILT, ""),
        Target("gpt_embeddings_variables", ("gpt_embeddings", "essay_data"), Status.BUILT, ""),
        Target("gpt4_embeddings_variables", ("gpt4_embeddings",), Status.BUILT, ""),
        # composite lists used by mmg samples
        Target("cog_noncog_variables", ("cog_variables", "noncog_variables"), Status.BUILT, ""),
        Target("all_outcomes", ("cog_noncog_variables", "social_outcomes"), Status.BUILT, ""),
        Target("mmg_edu_variables", ("gene_variables", "essay_variables", "teacher_variables"), Status.BUILT, ""),
        Target("mmg_cog_variables", ("gene_variables", "essay_variables", "teacher_variables"), Status.BUILT, ""),
        Target("all_vars", ("teacher_variables", "gene_variables", "essay_variables", "cog_noncog_variables", "social_outcomes", "confounder_variables"), Status.BUILT, ""),
    ]
    return t


def _clean_targets() -> list[Target]:
    return [
        Target("ncds_1_to_9", tuple(_NCDS_WAVES), Status.BUILT, "combine_ncds (full outer join)"),
        Target("ncds_1_to_9_cleaned", ("ncds_1_to_9", "mapping_df"), Status.STUB,
               "clean_ncds DEFERRED (option b): needs the real variables.xlsx. THE key blocker."),
        Target("aspiration_data", ("ncds_1_to_9", "camsis_data", "occupation_aspiration_mapping"), Status.BUILT, "create_aspirations"),
        Target("factor_data", ("ncds_1_to_9_cleaned",), Status.BUILT,
               "create_factors (Pearson factor built; 3 polychoric factors deferred, V3)"),
        Target("ncds_complete", ("ncds_1_to_9_cleaned", "factor_data", "aspiration_data", "essay_data", "gene_data"), Status.BUILT, "get_complete_ncds"),
        Target("ncds_complete_mmg", ("ncds_complete", "mmg_edu_variables"), Status.BUILT, "find_full_overlap"),
        Target("ncds_complete_mmg_cog", ("ncds_complete", "mmg_cog_variables"), Status.BUILT, "find_full_overlap"),
        Target("ncds_complete_all_overlap", ("ncds_complete", "all_vars"), Status.BUILT, "find_full_overlap"),
    ]


def build_pipeline() -> Pipeline:
    """Construct and structurally validate the full pipeline DAG."""
    targets: list[Target] = []
    targets += _load_targets()
    targets += _essay_targets()
    targets += _variable_list_targets()
    targets += _clean_targets()
    targets += model_targets()
    pipe = Pipeline(targets)
    pipe.validate()  # raises if any dependency is dangling or the graph has a cycle
    return pipe
