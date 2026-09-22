"""Run the pipeline graph in this process (brief Task 2.4) — the port of ``tar_make()``.

Every target of ``pipeline/build.py`` is bound below to the function it calls, with the
``_targets.R`` line it comes from. The runner resolves the requested targets'
dependencies, executes them in topological order, fits the models and scores them, and
returns the metric rows. There is no caching and no job scheduler: those are Phase 3
(docs/ORCHESTRATION.md). The bindings contain no logic of their own beyond passing
values (the thin-adapter rule); the work happens in ``io/``, ``cleaning/``,
``features/``, ``models/`` and ``metrics/``.

Two run configurations (``config.PAPER_RUN``, ``config.SMOKE_RUN``):
  * the paper configuration is 10 outer and 5 inner folds with the 6-learner library
    (R: llm_paper/R/functions.R:L512, L541);
  * the SMOKE configuration is for checking the plumbing on synthetic data. The runner
    refuses it unless ``$LCP_DATA_ROOT`` holds the marker written only by the synthetic
    generator (``config.SYNTHETIC_MARKER_FILE``) and every ID is synthetic, and every
    output it writes is labelled.

Targets that cannot be run are reported, never silently dropped: each one gets an entry
in ``RunResult.not_run`` and in the run log, with its reason. There are two:
  * gene data absent, so the gene-dependent targets are skipped (PORTING_NOTES L2), and
    the samples built without gene variables carry ``sample_note`` in the metrics table;
  * a factor that the native backend does not compute, so the fits that need it as an
    outcome or a predictor, and the samples that need it, do not run (PORTING_NOTES F1).

Outputs, all under ``$LCP_DATA_ROOT``: the metrics table (``metrics/``), the run log
(``logs/``) and the per-person predictions (``fits/``, PORTING_NOTES L3).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd
from joblib import Parallel, delayed

from .. import config
from ..cleaning.aspirations import create_aspirations
from ..cleaning.assemble import find_full_overlap, get_complete_ncds
from ..cleaning.clean_ncds import clean_ncds
from ..cleaning.factors import create_factors
from ..features.embeddings import gpt_embeddings, read_roberta_embeddings
from ..features.essay_variables import create_essay_variables
from ..features.readability import ingest_readability_metrics
from ..features.salat import get_salat_metrics, get_spelling_error_metrics
from ..io.readers import (
    combine_ncds,
    read_camsis,
    read_datalist,
    read_essays,
    read_gene_data,
    read_ncds,
    read_occupation_aspiration_mapping,
)
from ..metrics.cv_metrics import lm_metrics, superlearner_metrics
from ..models.data_prep import apply_data_prep
from ..models.run import fit_model, save_predictions
from . import variable_lists as vl
from .build import build_pipeline
from .graph import Status
from .model_spec import R_TO_PYTHON_TARGET, ModelTargetSpec, is_gene_dependent, model_specs
from .run_log import skipped_targets_entry, write_run_log
from .variable_lists import CONSTANT_LISTS

logger = logging.getLogger(__name__)

# Owner decision at Checkpoint C: when gene data is absent, the samples that the R
# defines with gene_variables are built without them, so they are not the paper's
# samples. Every metric row from such a target says so.
SAMPLE_NOTE_NO_GENE_DATA = "built without gene data, not comparable to the paper"

# The sample targets and the variable list each one keeps (R: llm_paper/_targets.R:L170–172).
SAMPLE_VARIABLE_LIST = {
    "ncds_complete_mmg": "mmg_edu_variables",
    "ncds_complete_mmg_cog": "mmg_cog_variables",
    "ncds_complete_all_overlap": "all_vars",
}
# Those whose variable list contains gene_variables (all three, as the R defines them).
GENE_DEFINED_SAMPLES = {s for s, lst in SAMPLE_VARIABLE_LIST.items()
                        if "gene_variables" in vl.COMPOSITE_LISTS[lst][0]}

# The order combine_ncds gets the waves in (R: llm_paper/_targets.R:L160–165).
NCDS_WAVE_ORDER = ("ncds_1_2_3", "ncds_4", "ncds_5", "ncds_6", "ncds_7", "ncds_8", "ncds_9",
                   "ncds_occ_2", "ncds_occ_5", "ncds_occ_6", "ncds_occ_7", "ncds_occ_8")

SYNTHETIC_ID = re.compile(r"^SYN\d{6}$")
PYTHON_TO_R_TARGET = {v: k for k, v in R_TO_PYTHON_TARGET.items()}
_METRICS_SUFFIX = "_metrics"


class SmokeRunRefused(RuntimeError):
    """The smoke configuration was asked for on data that is not the synthetic set."""


@dataclass(frozen=True)
class NotRun:
    """A target that was deliberately not run, and why."""

    reason: str


@dataclass(frozen=True)
class Binding:
    """One target: the component function it calls, and how it is called."""

    function: Callable
    call: Callable[["_Run"], Any]
    r_line: str = ""


@dataclass
class RunResult:
    run_config: config.RunConfig
    metrics: pd.DataFrame
    not_run: list[dict]
    values: dict[str, Any]
    label: str
    data_source: str
    elapsed_seconds: float
    metrics_path: Path | None = None
    log_path: Path | None = None
    prediction_paths: list[Path] = field(default_factory=list)


# --------------------------------------------------------------- bindings ----

def _bind_ncds_wave(key: str) -> Binding:
    def call(r: "_Run"):
        # R: read_ncds(file, varlist = tolower(mapping_df$variable)) (_targets.R:L44–91)
        varlist = [str(v).lower() for v in r.value("mapping_df")["variable"]]
        return read_ncds(str(config.restricted_path(key)), varlist)

    return Binding(read_ncds, call, "llm_paper/_targets.R:L44-91")


def _bind_gene_data() -> Binding:
    def call(r: "_Run"):
        # R: tar_target(gene_data, read_gene_data) stores the function itself, so
        # gene_variables is NULL (_targets.R:L93, L132; PORTING_NOTES L2). Here the
        # optional file is read when it exists, and gene data is absent when it does not.
        path = config.restricted_path("gene_data")
        return read_gene_data(str(path)) if path.exists() else None

    return Binding(read_gene_data, call, "llm_paper/_targets.R:L93")


def _bind_salat() -> Binding:
    def call(r: "_Run"):
        # R: get_salat_metrics(ncds_essays, taaled_1..3, taales_1..3, seance_1..3) (L102–111)
        paths = {tool: [str(config.restricted_path(f"{tool}_{b}")) for b in (1, 2, 3)]
                 for tool in ("taaled", "taales", "seance")}
        return get_salat_metrics(r.value("ncds_essays"), **paths)

    return Binding(get_salat_metrics, call, "llm_paper/_targets.R:L102-111")


def _bind_factor_data() -> Binding:
    def call(r: "_Run"):
        # R: create_factors(ncds_1_to_9_cleaned) (L168). Factors the backend does not
        # compute are recorded as unavailable columns, with the reason.
        result = create_factors(r.value("ncds_1_to_9_cleaned"), backend=r.run_config.factor_backend)
        for name in result.deferred:
            r.unavailable[name] = (
                "not computed: the 'native' factor backend has no polychoric estimator "
                "matching psych::fa; use factor_backend='r' (PORTING_NOTES F1)")
        return result.scores

    return Binding(create_factors, call, "llm_paper/_targets.R:L168")


def _bind_sample(sample: str) -> Binding:
    list_name = SAMPLE_VARIABLE_LIST[sample]

    def call(r: "_Run"):
        # R: find_full_overlap(ncds_complete, <list>) (L170–172)
        varlist = r.value(list_name)
        blocked = [c for c in varlist if c in r.unavailable]
        if blocked:
            return NotRun(f"{list_name} needs column(s) {sorted(blocked)}: {r.unavailable[blocked[0]]}")
        return find_full_overlap(r.value("ncds_complete"), varlist)

    return Binding(find_full_overlap, call, "llm_paper/_targets.R:L170-172")


def _bind_constant_list(name: str) -> Binding:
    values = list(CONSTANT_LISTS[name])
    return Binding(CONSTANT_LISTS.get, lambda r, v=values: list(v), "llm_paper/_targets.R:L120-159")


def _bind_essay_list(frame: str, drop_first: int) -> Binding:
    def call(r: "_Run"):
        return vl.columns_kept_in_essay_data(r.value(frame), r.value("essay_data"), drop_first)

    return Binding(vl.columns_kept_in_essay_data, call, "llm_paper/_targets.R:L134-137")


def _bind_composite(name: str) -> Binding:
    members = vl.COMPOSITE_LISTS[name][0]

    def call(r: "_Run"):
        return vl.composite_list(name, {m: r.value(m) for m in members})

    return Binding(vl.composite_list, call, "llm_paper/_targets.R:L140-142, L156-158")


def bindings() -> dict[str, Binding]:
    """Every target of the graph -> the function it calls (R line in ``Binding.r_line``)."""
    b: dict[str, Binding] = {
        # --- data loading (R: _targets.R:L41–94) ---
        "ncds_essays": Binding(read_essays, lambda r: read_essays(str(config.restricted_path("essays"))),
                               "llm_paper/_targets.R:L41-42"),
        "mapping_df": Binding(read_datalist, lambda r: read_datalist(str(config.VARIABLES_XLSX)),
                              "llm_paper/_targets.R:L43"),
        "camsis_data": Binding(read_camsis, lambda r: read_camsis(str(config.CAMSIS_FILE)),
                               "llm_paper/_targets.R:L92"),
        "occupation_aspiration_mapping": Binding(
            read_occupation_aspiration_mapping,
            lambda r: read_occupation_aspiration_mapping(str(config.OCCUPATION_ASPIRATION_XLSX)),
            "llm_paper/_targets.R:L94"),
        "gene_data": _bind_gene_data(),
        # --- essays (R: _targets.R:L99–115) ---
        # TreeTagger tokenisation happens inside r/readability.R (Task 2.5): EXTERNAL.
        "tokenized_essays": Binding(lambda: None, lambda r: None, "llm_paper/_targets.R:L99"),
        "spelling_errors": Binding(
            get_spelling_error_metrics,
            lambda r: get_spelling_error_metrics(r.value("ncds_essays"),
                                                 str(config.restricted_path("spelling_mistakes"))),
            "llm_paper/_targets.R:L100"),
        "readability_metrics": Binding(
            ingest_readability_metrics,
            lambda r: ingest_readability_metrics(r.value("ncds_essays"),
                                                 str(config.restricted_path("readability_metrics"))),
            "llm_paper/_targets.R:L101"),
        "salat_metrics": _bind_salat(),
        "roberta_embeddings": Binding(
            read_roberta_embeddings,
            lambda r: read_roberta_embeddings(str(config.derived_path("roberta_embeddings"))),
            "llm_paper/_targets.R:L112"),
        "gpt_embeddings": Binding(
            gpt_embeddings, lambda r: gpt_embeddings(str(config.restricted_path("gpt35_embeddings"))),
            "llm_paper/_targets.R:L113"),
        "gpt4_embeddings": Binding(
            gpt_embeddings, lambda r: gpt_embeddings(str(config.restricted_path("gpt4_embeddings"))),
            "llm_paper/_targets.R:L114"),
        "essay_data": Binding(
            create_essay_variables,
            lambda r: create_essay_variables(r.value("salat_metrics"), r.value("readability_metrics"),
                                             r.value("spelling_errors"), r.value("gpt_embeddings")),
            "llm_paper/_targets.R:L115"),
        # --- variable lists read from frames (R: _targets.R:L132–138) ---
        "gene_variables": Binding(vl.gene_variables, lambda r: vl.gene_variables(r.value("gene_data")),
                                  "llm_paper/_targets.R:L132"),
        "essay_variables": Binding(vl.essay_variables, lambda r: vl.essay_variables(r.value("essay_data")),
                                   "llm_paper/_targets.R:L133"),
        "salat_metrics_variables": _bind_essay_list("salat_metrics", 1),
        "readability_metrics_variables": _bind_essay_list("readability_metrics", 2),
        "spelling_errors_variables": _bind_essay_list("spelling_errors", 1),
        "gpt_embeddings_variables": _bind_essay_list("gpt_embeddings", 1),
        "gpt4_embeddings_variables": Binding(
            vl.gpt4_embeddings_variables, lambda r: vl.gpt4_embeddings_variables(r.value("gpt4_embeddings")),
            "llm_paper/_targets.R:L138"),
        # --- cleaning (R: _targets.R:L160–172) ---
        "ncds_1_to_9": Binding(combine_ncds, lambda r: combine_ncds(*(r.value(w) for w in NCDS_WAVE_ORDER)),
                               "llm_paper/_targets.R:L160-165"),
        "ncds_1_to_9_cleaned": Binding(
            clean_ncds, lambda r: clean_ncds(r.value("ncds_1_to_9"), r.value("mapping_df")),
            "llm_paper/_targets.R:L166"),
        "aspiration_data": Binding(
            create_aspirations,
            lambda r: create_aspirations(r.value("ncds_1_to_9"), r.value("camsis_data"),
                                         r.value("occupation_aspiration_mapping")),
            "llm_paper/_targets.R:L167"),
        "factor_data": _bind_factor_data(),
        "ncds_complete": Binding(
            get_complete_ncds,
            lambda r: get_complete_ncds(r.value("ncds_1_to_9_cleaned"), r.value("factor_data"),
                                        r.value("aspiration_data"), r.value("essay_data"),
                                        r.value("gene_data")),
            "llm_paper/_targets.R:L169"),
    }
    for wave in NCDS_WAVE_ORDER:
        b[wave] = _bind_ncds_wave(wave)
    for name in CONSTANT_LISTS:
        b[name] = _bind_constant_list(name)
    for sample in SAMPLE_VARIABLE_LIST:
        b[sample] = _bind_sample(sample)
    for name in vl.COMPOSITE_LISTS:
        if name in b or name == "mmg_variables":  # mmg_variables is never used by a target
            continue
        b[name] = _bind_composite(name)
    return b


BINDINGS = bindings()
# The model and metric targets are bound generically, from the model spec.
MODEL_BINDING = Binding(fit_model, lambda r: None, "llm_paper/_targets.R:L178-385")
METRICS_BINDING = Binding(superlearner_metrics, lambda r: None, "llm_paper/_targets.R:L387-494")


def binding_for(target: str, specs: dict[str, ModelTargetSpec] | None = None) -> Binding:
    """The binding of any target of the graph, model and metric targets included."""
    specs = model_specs_by_name() if specs is None else specs
    if target in BINDINGS:
        return BINDINGS[target]
    if target in specs:
        return MODEL_BINDING
    if target.endswith(_METRICS_SUFFIX) and target[: -len(_METRICS_SUFFIX)] in specs:
        spec = specs[target[: -len(_METRICS_SUFFIX)]]
        return Binding(lm_metrics if spec.scorer == "lm" else superlearner_metrics,
                       METRICS_BINDING.call, METRICS_BINDING.r_line)
    raise KeyError(f"no binding for target {target!r}")


def model_specs_by_name() -> dict[str, ModelTargetSpec]:
    return {s.name: s for s in model_specs()}


# ------------------------------------------------------------- the runner ----

def _fit_one(outcome: str, predictors: list[str], data: pd.DataFrame, method: str, seed: int,
             outer_v: int, inner_v: int, n_jobs: int = 1):
    """One model fit, as the R model functions do it (models/run.py). Module level so
    that it can run in a joblib worker process."""
    return fit_model(outcome, predictors, data, method, seed=seed, n_jobs=n_jobs,
                     outer_v=outer_v, inner_v=inner_v)


@dataclass
class _FitTask:
    target: str
    outcome: str
    predictors: list[str]
    method: str
    data: pd.DataFrame


class _Run:
    def __init__(self, run_config: config.RunConfig, targets: Sequence[str] | None):
        self.run_config = run_config
        self.pipe = build_pipeline()
        self.specs = model_specs_by_name()
        self.values: dict[str, Any] = {}
        self.unavailable: dict[str, str] = {}
        self.not_run: list[dict] = []
        self.requested = list(self.pipe.names()) if targets is None else list(targets)
        for name in self.requested:
            if name not in self.pipe:
                raise KeyError(f"unknown target {name!r}")
        self.order = self._order()
        root = config.data_root()
        self.data_source, self.marker = self._data_source(root)
        if run_config.name == "smoke" and self.data_source != "synthetic":
            raise SmokeRunRefused(
                f"the smoke configuration runs only on synthetic data: {root}/"
                f"{config.SYNTHETIC_MARKER_FILE} is missing, so this may be real data. The marker "
                "is written only by tests/fixtures/synthetic_ncds.py.")
        self.label = run_config.label
        self.prefix = run_config.file_prefix
        if self.data_source == "synthetic" and run_config.name != "smoke":
            self.label = f"{run_config.name} configuration on SYNTHETIC data: not results"
            self.prefix = self.prefix or "SYNTHETIC_"

    # --- graph ---
    def _order(self) -> list[str]:
        needed: set[str] = set()
        stack = list(self.requested)
        while stack:
            name = stack.pop()
            if name in needed:
                continue
            needed.add(name)
            stack.extend(self.pipe.get(name).deps)
        return [n for n in self.pipe.topo_order() if n in needed]

    @staticmethod
    def _data_source(root: Path) -> tuple[str, dict]:
        marker = root / config.SYNTHETIC_MARKER_FILE
        if marker.exists():
            try:
                content = json.loads(marker.read_text())
            except json.JSONDecodeError:
                content = {}
            if content.get("synthetic") is True:
                return "synthetic", content
        return "restricted", {}

    def value(self, name: str):
        return self.values[name]

    def _skip(self, target: str, outcome: str | None, reason: str) -> None:
        self.not_run.append({"target": target, "outcome": outcome, "reason": reason})
        logger.warning("not run: %s%s — %s", target, f" [{outcome}]" if outcome else "", reason)

    def _check_synthetic_ids(self, name: str, value: Any) -> None:
        """In the smoke configuration every ID must be synthetic, so that the
        configuration cannot be used for a real run even if the marker is copied."""
        if self.run_config.name != "smoke" or not isinstance(value, pd.DataFrame):
            return
        for column in ("ncdsid", "id"):
            if column not in value.columns:
                continue
            ids = value[column].dropna().astype(str).unique()
            bad = [i for i in ids if not SYNTHETIC_ID.match(i)]
            if bad:
                raise SmokeRunRefused(
                    f"the smoke configuration runs only on synthetic data: target {name!r} has "
                    f"{len(bad)} ID(s) that are not of the form SYN000001.")

    # --- phases ---
    def build_inputs(self) -> None:
        for name in self.order:
            if name in self.specs or self._is_metrics(name):
                continue
            target = self.pipe.get(name)
            if target.status is Status.STUB:  # pragma: no cover - none at present
                raise NotImplementedError(f"target {name!r} is a stub: {target.note}")
            value = BINDINGS[name].call(self)
            if isinstance(value, NotRun):
                self._skip(name, None, value.reason)
            self._check_synthetic_ids(name, value)
            self.values[name] = value
        self.gene_available = self.values.get("gene_data") is not None

    def _is_metrics(self, name: str) -> bool:
        return name.endswith(_METRICS_SUFFIX) and name[: -len(_METRICS_SUFFIX)] in self.specs

    def _resolve(self, refs) -> list[str]:
        out: list[str] = []
        for ref in refs:
            out.extend(self.value(ref.value) if ref.kind == "list" else [ref.value])
        return list(dict.fromkeys(out))

    def _tasks_for(self, name: str) -> list[_FitTask]:
        spec = self.specs[name]
        if is_gene_dependent(spec) and not self.gene_available:
            self._skip(name, None, "gene data absent: the predictors include gene_variables "
                                   "(PORTING_NOTES L2)")
            return []
        sample = self.value(spec.sample)
        if isinstance(sample, NotRun):
            self._skip(name, None, f"sample {spec.sample} not built: {sample.reason}")
            return []
        tables = {s["table"]: self.value(s["table"]) for s in spec.data_prep if "table" in s}
        lists = {s["columns_from"]: self.value(s["columns_from"]) for s in spec.data_prep
                 if "columns_from" in s}
        data = apply_data_prep(sample, spec.data_prep, tables=tables, variable_lists=lists)
        predictors = self._resolve(spec.predictors)
        blocked = [p for p in predictors if p in self.unavailable]
        if blocked:
            self._skip(name, None, f"predictor(s) {sorted(blocked)}: {self.unavailable[blocked[0]]}")
            return []
        outcomes = list(CONSTANT_LISTS[spec.pattern]) if spec.pattern else [spec.outcome.value]
        tasks = []
        for outcome in outcomes:
            if outcome in self.unavailable:
                self._skip(name, outcome, self.unavailable[outcome])
                continue
            tasks.append(_FitTask(name, outcome, predictors, spec.method, data))
        return tasks

    def fit_models(self) -> list[Path]:
        """Fit every requested model target. The fits are independent, so they are
        spread over ``n_jobs`` worker processes; the results do not depend on that."""
        tasks: list[_FitTask] = []
        for name in self.order:
            if name in self.specs:
                tasks.extend(self._tasks_for(name))
                self.values.setdefault(name, [])
        if not tasks:
            return []
        cfg = self.run_config
        # With a single fit there is nothing to spread, so the workers go to that fit's
        # outer folds instead (fit_model's own n_jobs; the output does not depend on it).
        inner_jobs = cfg.n_jobs if len(tasks) == 1 else 1
        results = Parallel(n_jobs=cfg.n_jobs)(
            delayed(_fit_one)(t.outcome, t.predictors, t.data, t.method, cfg.seed,
                              cfg.outer_folds, cfg.inner_folds, inner_jobs) for t in tasks)
        written: list[Path] = []
        for task, (fit, var) in zip(tasks, results):
            self.values[task.target].append((fit, var))
            if cfg.save_predictions:
                written.append(save_predictions(fit, task.target, prefix=self.prefix, label=self.label))
        return written

    def score_models(self) -> pd.DataFrame:
        rows = []
        for name in self.order:
            if not self._is_metrics(name):
                continue
            model = name[: -len(_METRICS_SUFFIX)]
            spec = self.specs[model]
            note = (SAMPLE_NOTE_NO_GENE_DATA
                    if spec.sample in GENE_DEFINED_SAMPLES and not self.gene_available else None)
            for fit, _var in self.values.get(model, []):
                row = lm_metrics(fit) if spec.scorer == "lm" else superlearner_metrics(fit)
                rows.append({"target": model, "r_target": PYTHON_TO_R_TARGET.get(model, ""),
                             "var": row.pop("var"), "n": row.pop("n"), **row,
                             "sample": spec.sample, "sample_note": note,
                             "run_config": self.run_config.name, "run_label": self.label})
        return pd.DataFrame(rows)

    def write_outputs(self, metrics: pd.DataFrame, elapsed: float, n_predictions: int) -> tuple[Path, Path]:
        metrics_path = config.metrics_dir() / f"{self.prefix}metrics.csv"
        metrics.to_csv(metrics_path, index=False)
        gene_skipped = [s for name, s in self.specs.items()
                        if name in self.order and is_gene_dependent(s)] if not self.gene_available else []
        entries = {
            "run_config": self.run_config.name,
            "run_label": self.label,
            "data_source": self.data_source,
            "outer_folds": self.run_config.outer_folds,
            "inner_folds": self.run_config.inner_folds,
            "seed": self.run_config.seed,
            "n_jobs": self.run_config.n_jobs,
            "factor_backend": self.run_config.factor_backend or config.FACTOR_BACKEND,
            "gene_data_available": self.gene_available,
            "sample_note": SAMPLE_NOTE_NO_GENE_DATA if not self.gene_available else None,
            "targets_built": [n for n in self.order if n in self.values],
            "n_fits": int(sum(len(v) for k, v in self.values.items() if k in self.specs)),
            "n_metric_rows": int(len(metrics)),
            "n_prediction_files": n_predictions,
            "not_run": self.not_run,
            "unavailable_columns": dict(self.unavailable),
            "elapsed_seconds": round(elapsed, 2),
            **skipped_targets_entry(gene_skipped, "gene data absent (PORTING_NOTES L2)"),
        }
        log_path = write_run_log(entries, name=f"{self.prefix}run_log.json")
        return metrics_path, log_path


def run_pipeline(run_config: config.RunConfig = config.PAPER_RUN,
                 targets: Sequence[str] | None = None, *, write_outputs: bool = True) -> RunResult:
    """Run ``targets`` (default: the whole graph) with ``run_config``, in this process.

    Dependencies are resolved from the graph and executed in topological order. Returns
    the metric rows, the targets that were not run and why, and the target values.
    """
    started = time.time()
    run = _Run(run_config, targets)
    run.build_inputs()
    predictions = run.fit_models()
    metrics = run.score_models()
    elapsed = time.time() - started
    metrics_path = log_path = None
    if write_outputs:
        metrics_path, log_path = run.write_outputs(metrics, elapsed, len(predictions))
    return RunResult(run_config=run_config, metrics=metrics, not_run=run.not_run, values=run.values,
                     label=run.label, data_source=run.data_source, elapsed_seconds=elapsed,
                     metrics_path=metrics_path, log_path=log_path, prediction_paths=predictions)
