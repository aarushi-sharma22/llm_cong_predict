# Remaining work

What is not implemented yet, and the rules that hold from now on.

---

## Rule: the suite must be green on a machine with neither R nor torch

On a machine where R, rpy2, torch and transformers are all missing, `pytest` must end
with **zero failures and zero errors**. Everything that needs one of them must SKIP, and
each skip must name what is missing.

**Why.** The native path is meant to install and run without R (README, "Model
backends"), and torch is an optional extra that must never share a process with xgboost
(docs/ORCHESTRATION.md). A test that fails instead of skipping on such a machine tells
a reviewer nothing except that the branch looks broken, and it hides any real failure
next to it.

**What must carry a guard.** Any test that reaches:

| What it reaches | Guard to use |
|---|---|
| `run_pipeline` with the default factor backend (it computes the three polychoric factors through `psych::fa`, PORTING_NOTES F1) | `tests/test_execute.py::needs_r_factors`, or `pytest.skip(R_FACTORS_SKIP)` in a fixture |
| `fit_model(backend="r")`, `models/r_superlearner.py` | `pytest.mark.skipif(r_unavailable_reason(("SuperLearner", "nnls")) ...)`, as in `tests/test_run.py` |
| `create_factors(backend="r")`, `create_factors_r`, anything in `rbridge` | `tests/test_oracle.py::_needs("psych")` and friends |
| torch or transformers | `tests/test_features.py::_needs_torch` |

A run that passes `factor_backend="native"` needs no guard: it is exactly the path a
machine without R takes, and testing it there is the point.

**How to check, without uninstalling anything.** `rpy2` cannot start R when `R_HOME`
points nowhere, which is the same code path as R not being installed:

```bash
R_HOME=/nonexistent .venv/bin/python -m pytest -rfEs
```

For the torch half, run the suite in an environment where the `[embeddings]` extra is
not installed (`importlib.util.find_spec` is what the guard uses, so hiding it with
`PYTHONPATH` does not work).

**In CI, run the suite twice:** once with R and torch present, once with neither. The
second run must be green, with skips.

**How the rule was found.** On a machine without R or torch,
`tests/test_execute.py::test_results_do_not_depend_on_n_jobs` failed instead of skipping:
it called `run_pipeline` without naming a factor backend, so it took the default `"r"`.
It was the only test missing the guard; the rest of the suite skipped correctly.

---

## Not implemented

| Item | Why | Described in |
|---|---|---|
| Pinned package versions and a lockfile, **including R xgboost < 3.0** so that `SuperLearner` still passes `params = list(tree_method = "hist")` | the current SuperLearner takes a different code path for xgboost ≥ 3.0, so the full six-learner oracle cannot run against R until versions are pinned | PORTING_NOTES C8, K1; VALIDATION_CHECKLIST V4, V7 |
| Caching in the runner ("skip if up to date", keyed by inputs and function source) | the runner deliberately has no caching framework | docs/ORCHESTRATION.md |
| The Snakemake wrapper for cluster runs of all 416 fits | same | docs/ORCHESTRATION.md |
| Containers, CI, HPC job scripts | | |

The format of the released polygenic index files will replace the placeholder reader
(PORTING_NOTES M5).
