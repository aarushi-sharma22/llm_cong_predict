"""Tests for the pipeline scaffolding.

These verify the WIRING is sound: dependencies resolve, the graph is acyclic, the
model spec expands deterministically, and the blocked-node analysis correctly
identifies clean_ncds (and the other stubs) as blocking the downstream. They do NOT
test execution — the pipeline cannot run without real data + clean_ncds.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from llm_cong_predict.pipeline.build import build_pipeline
from llm_cong_predict.pipeline.graph import Pipeline, Status, Target
from llm_cong_predict.pipeline.model_spec import (
    FAMILIES,
    R_TO_PYTHON_TARGET,
    expand_family,
    model_specs,
    model_targets,
)
from llm_cong_predict.pipeline.variable_lists import CONSTANT_LISTS, DATA_DEPENDENT_LISTS


# --------------------------------------------------------- generic graph ----

def test_topo_order_orders_dependencies_first():
    p = Pipeline([
        Target("a", ()),
        Target("b", ("a",)),
        Target("c", ("a", "b")),
    ])
    order = p.topo_order()
    assert order.index("a") < order.index("b") < order.index("c")


def test_dangling_dependency_raises():
    p = Pipeline([Target("b", ("missing",))])
    with pytest.raises(ValueError, match="unknown target"):
        p.validate()


def test_cycle_is_detected():
    p = Pipeline([Target("a", ("b",)), Target("b", ("a",))])
    with pytest.raises(ValueError, match="cycle"):
        p.topo_order()


def test_blocked_is_transitive():
    p = Pipeline([
        Target("leaf", (), Status.STUB),
        Target("mid", ("leaf",), Status.BUILT),
        Target("top", ("mid",), Status.BUILT),
        Target("independent", (), Status.BUILT),
    ])
    blocked = p.blocked()
    assert set(blocked) == {"leaf", "mid", "top"}          # independent is not blocked
    assert blocked["top"] == {"leaf"}                       # root cause traced to the stub
    assert p.runnable_frontier() == ["independent"]


# --------------------------------------------------------- model spec -------

def test_model_spec_expands_deterministically():
    """Every scored model target has a paired _metrics target. The seven
    *_superlearner_mmg_lm targets (R: llm_paper/_targets.R:L323–343) are scored in
    neither _targets.R nor create_data.R, so they have none (owner decision C6)."""
    targets = model_targets()
    names = [t.name for t in targets]
    assert len(names) == len(set(names))          # no duplicate target names
    specs = model_specs()
    for s in specs:
        assert s.name in names
        assert (f"{s.name}_metrics" in names) == (s.scorer != "none")
    metrics = [n for n in names if n.endswith("_metrics")]
    assert len(metrics) == sum(s.scorer != "none" for s in specs) == 63


def test_expand_family_dependencies():
    # a single-feature family expands to model + metric with correct deps
    fam = FAMILIES[0]  # overlap family, MAIN feature sets, all_outcomes, all_overlap
    ts = {t.name: t for t in expand_family(fam)}
    # essay model depends on essay_variables + the sample + the outcome list
    essay = ts["essay_superlearner_overlap"]
    assert "essay_variables" in essay.deps
    assert "ncds_complete_all_overlap" in essay.deps
    assert "all_outcomes" in essay.deps
    # its metric target depends on the model
    assert ts["essay_superlearner_overlap_metrics"].deps == ("essay_superlearner_overlap",)


def test_literal_outcome_not_a_dependency():
    # the cog_mmg family uses the literal outcome s2_co_factor_ability, which must
    # NOT appear as a graph dependency (it is a column name, not a target).
    from llm_cong_predict.pipeline.model_spec import ModelFamily
    fam = ModelFamily(("essay",), "s2_co_factor_ability", "ncds_complete_mmg_cog", "superlearner", "_cog_mmg")
    model = expand_family(fam)[0]
    assert "s2_co_factor_ability" not in model.deps
    assert "ncds_complete_mmg_cog" in model.deps


# --------------------------------------------------------- full pipeline ----

def test_full_pipeline_validates():
    # build_pipeline() calls validate(); reaching here means wiring is sound.
    pipe = build_pipeline()
    assert len(pipe) > 150          # data + essay + clean + ~140 model/metric targets
    pipe.topo_order()               # acyclic


def test_clean_ncds_is_built_and_the_readability_stub_blocks_the_models():
    """Task 2.1 removed clean_ncds' STUB status (brief Task 2.1). The model half is
    still blocked, now through the readability boundary (essay_data needs
    readability_metrics) until Task 2.5 / 2.4 provide it. Replaces a test that asserted
    the STUB status of clean_ncds."""
    pipe = build_pipeline()
    assert pipe.get("ncds_1_to_9_cleaned").status is Status.BUILT
    blocked = pipe.blocked()
    assert "essay_superlearner_overlap" in blocked
    assert "ncds_1_to_9_cleaned" not in blocked["essay_superlearner_overlap"]
    assert "readability_metrics" in blocked["essay_superlearner_overlap"]


def test_stub_roots_are_exactly_the_three_expected():
    """After Task 2.1 the stub roots are gene_data (restricted, empty in the R) and the
    readability boundary (tokenize_essays / calculate_readability_metrics, Task 2.5).
    Previously four, including clean_ncds."""
    pipe = build_pipeline()
    assert set(pipe.stub_roots()) == {"gene_data", "readability_metrics", "tokenized_essays"}


def test_readers_are_in_the_runnable_frontier():
    # the leaf readers are BUILT and not blocked -> they would run given real data.
    pipe = build_pipeline()
    frontier = set(pipe.runnable_frontier())
    assert "ncds_essays" in frontier
    assert "camsis_data" in frontier
    assert "ncds_1_2_3" in frontier
    # clean_ncds (Task 2.1) is now runnable; ncds_complete still waits for essay_data,
    # which needs the readability boundary
    assert "ncds_1_to_9_cleaned" in frontier
    assert "factor_data" in frontier
    assert "ncds_complete" not in frontier


# ------------------------------------------- inventory of the R model targets --

REPO = Path(__file__).resolve().parents[1]
INVENTORY = json.loads((REPO / "docs" / "reference" / "r_targets_inventory.json").read_text())


def test_every_r_model_target_maps_to_one_python_target_with_the_same_definition():
    """R: llm_paper/_targets.R:L178–385 (via docs/reference/r_targets_inventory.json):
    each R model target maps to exactly one Python target with the same method,
    outcome, predictors, sample, data preparation and scorer (brief F7; scorers also
    from R/create_data.R:L98, L163–170, L181, L344, L347)."""
    specs = {s.name: s for s in model_specs()}
    r_names = [m["r_name"] for m in INVENTORY["model_targets"]]
    assert sorted(R_TO_PYTHON_TARGET) == sorted(r_names)
    assert sorted(R_TO_PYTHON_TARGET.values()) == sorted(specs)  # a bijection onto the spec
    for m in INVENTORY["model_targets"]:
        s = specs[R_TO_PYTHON_TARGET[m["r_name"]]]
        assert s.method == m["method"], m["r_name"]
        assert s.outcome.as_dict() == m["outcome"], m["r_name"]
        assert s.pattern == m["pattern"], m["r_name"]
        assert [p.as_dict() for p in s.predictors] == m["predictors"], m["r_name"]
        assert s.sample == m["sample"], m["r_name"]
        assert list(s.data_prep) == m["data_prep"], m["r_name"]
        assert s.scorer == m["scorer"], m["r_name"]
        assert s.n_fits() == m["n_fits"], m["r_name"]


def test_cog_social_lm_uses_the_single_ability_factor():
    """R: llm_paper/_targets.R:L377, L382 — cog_superlearner_social_lm(_overlap) use
    "s2_co_factor_ability", not cog_variables (brief F7)."""
    specs = {s.name: s for s in model_specs()}
    for name in ("cog_lm_social_lm", "cog_lm_social_lm_overlap"):
        assert [p.as_dict() for p in specs[name].predictors] == [
            {"kind": "literal", "value": "s2_co_factor_ability"}]
    assert specs["cog_superlearner_social"].predictors[0].value == "cog_variables"


def test_expanding_over_outcomes_gives_416_fits():
    """70 model targets expand over their outcome lists (pattern = ...) into 416 fits
    (brief F7; computed from _targets.R: 12 all_outcomes, 1 social outcome, 5 BFI)."""
    specs = model_specs()
    assert len(specs) == 70
    assert sum(s.n_fits() for s in specs) == 416 == INVENTORY["totals"]["fits"]


def test_variable_lists_match_the_r_constants():
    """R: llm_paper/_targets.R:L120–159: constant lists copied verbatim; lists read
    from data frames are data-dependent."""
    for name, entry in INVENTORY["variable_lists"].items():
        if entry["members"] is not None:
            assert list(CONSTANT_LISTS[name]) == entry["members"], name
        elif entry["kind"] == "data-dependent":
            assert name in DATA_DEPENDENT_LISTS, name


def test_embedding_models_depend_on_their_embedding_targets():
    """R: llm_paper/_targets.R:L227, L230 — the RoBERTa and GPT-4 models join their
    embedding tables onto ncds_complete inside the target."""
    pipe = build_pipeline()
    assert "roberta_embeddings" in pipe.get("roberta_embeddings_superlearner_text").deps
    assert "gpt4_embeddings" in pipe.get("gpt4_embeddings_superlearner_text").deps


def test_extractor_reproduces_the_committed_inventory():
    """scripts/extract_r_targets.py re-derives docs/reference/r_targets_inventory.json
    from the pinned R sources."""
    if not (REPO / "reference" / "llm_paper" / "_targets.R").exists():
        pytest.skip("reference/llm_paper not cloned (see docs/REFERENCE_SOURCES.md)")
    result = subprocess.run([sys.executable, str(REPO / "scripts" / "extract_r_targets.py"), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
