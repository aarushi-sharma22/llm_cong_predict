"""Tests for the pipeline scaffolding.

These verify the WIRING is sound: dependencies resolve, the graph is acyclic, the
model spec expands deterministically, and the blocked-node analysis correctly
identifies clean_ncds (and the other stubs) as blocking the downstream. They do NOT
test execution — the pipeline cannot run without real data + clean_ncds.
"""

from __future__ import annotations

import pytest

from llm_cong_predict.pipeline.build import build_pipeline
from llm_cong_predict.pipeline.graph import Pipeline, Status, Target
from llm_cong_predict.pipeline.model_spec import FAMILIES, expand_family, model_targets


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
    targets = model_targets()
    names = [t.name for t in targets]
    assert len(names) == len(set(names))          # no duplicate target names
    # every model target has a paired _metrics target
    models = [n for n in names if not n.endswith("_metrics")]
    metrics = [n for n in names if n.endswith("_metrics")]
    assert len(models) == len(metrics)
    for m in models:
        assert f"{m}_metrics" in names


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


def test_clean_ncds_is_a_stub_and_blocks_the_model_half():
    pipe = build_pipeline()
    assert pipe.get("ncds_1_to_9_cleaned").status is Status.STUB
    blocked = pipe.blocked()
    # a representative model target is blocked, and clean_ncds is among its causes
    assert "essay_superlearner_overlap" in blocked
    assert "ncds_1_to_9_cleaned" in blocked["essay_superlearner_overlap"]


def test_stub_roots_are_exactly_the_four_expected():
    pipe = build_pipeline()
    assert set(pipe.stub_roots()) == {
        "gene_data", "ncds_1_to_9_cleaned", "readability_metrics", "tokenized_essays",
    }


def test_readers_are_in_the_runnable_frontier():
    # the leaf readers are BUILT and not blocked -> they would run given real data.
    pipe = build_pipeline()
    frontier = set(pipe.runnable_frontier())
    assert "ncds_essays" in frontier
    assert "camsis_data" in frontier
    assert "ncds_1_2_3" in frontier
    # clean_ncds and anything downstream of it must NOT be in the frontier
    assert "ncds_1_to_9_cleaned" not in frontier
    assert "ncds_complete" not in frontier
