# Orchestration Design

The original used the R `targets` package: a make-like DAG giving (a) dependency
tracking, (b) skip-if-unchanged caching, (c) parallelism. This pipeline has ~110
model fits, several of them expensive, so caching and parallelism are genuinely
valuable — you should not re-run everything to regenerate one figure.

This project provides **both** a lightweight Python runner (default) and an optional
Snakemake wrapper. The decision to have both is safe only because of one rule:

> **The thin-adapter rule.** Orchestrators contain *no logic*. All real work lives
> in pure, individually-tested functions in the component modules
> (`io/`, `cleaning/`, `features/`, `models/`, `metrics/`). Both orchestrators only
> decide call-order over those functions. This prevents the two entry points from
> drifting apart, which is the only real risk of maintaining both.

Status: components are built first. The Python runner is added once components exist;
the Snakefile is added last. Neither orchestrator is implemented yet at the time of
writing — this note is the design they will follow.

---

## Primary: lightweight Python module graph

The design mirrors what `targets` gave us, in plain Python with no external tooling.

### 1. Targets as pure functions
Each step is a function with explicit inputs and one output, e.g.

```python
def clean_ncds(ncds_1_to_9, mapping) -> pd.DataFrame: ...
def create_factors(ncds_cleaned) -> pd.DataFrame: ...
```

### 2. A target registry + topological runner
A registry maps target name -> (function, dependency names). A runner
topologically sorts and executes, optionally caching each output to disk keyed by a
hash of its inputs + the function source, so unchanged targets are skipped on re-run
(the `targets` "skip if up to date" behaviour).

```python
@target(deps=["ncds_1_to_9", "mapping"])
def ncds_1_to_9_cleaned(ncds_1_to_9, mapping):
    return clean_ncds(ncds_1_to_9, mapping)

# runner: build(target_name) -> resolves deps, runs, caches
```

### 3. Declarative model spec (the key improvement over the original)
The original hand-wrote ~110 near-identical `tar_target(...)` model calls, which is
why `create_data.R` became 600 fragile lines. Here the model targets are **generated
from a spec** instead of copy-pasted:

```python
MODEL_SPEC = [
    ModelRun(feature_set="essay",   outcomes=ALL_OUTCOMES, sample="complete",  method="superlearner"),
    ModelRun(feature_set="gene",    outcomes=ALL_OUTCOMES, sample="complete",  method="superlearner"),
    ModelRun(feature_set="teacher", outcomes=ALL_OUTCOMES, sample="mmg",        method="superlearner"),
    ModelRun(feature_set=["essay","gene","teacher"], outcomes=SOCIAL_OUTCOMES, sample="mmg", method="lm"),
    ...
]
# expanded programmatically into (feature_set x outcome x sample x method) targets.
```

Adding a model becomes one spec line, not a copy-paste block. This is the single
biggest structural fix relative to the original.

### Why this is the default
Zero extra dependencies, installable and runnable by anyone who `pip install`s the
package — consistent with the project principle of not forcing tooling on users
(the same reason the model layer defaults to native Python rather than requiring R).

---

## Optional: Snakemake wrapper (for HPC / cluster runs)

For large runs on a scheduler (SLURM etc.), a `Snakefile` exposes the same targets
as Snakemake rules. Each rule does **not** reimplement anything — it calls the same
component functions (via a thin CLI entry point or `script:`), so the logic has a
single source of truth. Snakemake then contributes battle-tested cluster submission,
restart, and parallelism.

```
rule clean_ncds:
    input:  "cache/ncds_1_to_9.pkl", "data/schema/ncds_variable_mapping.yaml"
    output: "cache/ncds_1_to_9_cleaned.pkl"
    script: "scripts/steps/clean_ncds.py"   # calls cleaning.clean_ncds()
```

### When to use which
- **Reproducing a figure / running locally** -> Python runner. No extra install.
- **Running the full ~110-fit pipeline on a cluster** -> Snakemake wrapper.

### Honest tradeoff
Two entry points is a small ongoing maintenance tax and a drift risk. That risk is
bounded entirely by the thin-adapter rule above; if that rule is ever violated (logic
creeping into a Snakefile rule or the runner), collapse to one orchestrator instead.
