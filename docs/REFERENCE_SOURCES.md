# Reference sources

Third-party source code cloned into `reference/` (git-ignored, never committed) so that
every value in the port can be traced to a line of source. Citations in code use the
form `# R: llm_paper/R/functions.R:L<line>` or
`# R pkg: <package>/<path>:L<line> (<version or commit>)`, where the path is relative
to `reference/`.

**These are the current versions of each package, cloned on 2026-09-22. The package
versions the paper was actually run with are unknown. Behaviour may differ between the
versions below and the versions the author used.** Where a difference between versions
matters, the port says so at the point of use and records it in `docs/PORTING_NOTES.md`.

Only `llm_paper` is pinned to the exact commit behind the paper (`b0cfe4c`, the
repository's only commit).

## Sources listed in the Phase 1–2 brief (Task 0.2)

| Directory | Origin | Commit (HEAD at clone) | Commit date | `Version:` in DESCRIPTION |
|---|---|---|---|---|
| `reference/llm_paper` | https://github.com/tobiaswolfram/llm_paper | `b0cfe4ceba0c29fcc6121aa7ff49c761b8127285` (checked out `b0cfe4c`) | 2025-05-19 | n/a (not a package) |
| `reference/SuperLearner` | https://github.com/ecpolley/SuperLearner | `583dbb9b1c83bc36b3079c516fc7321e636c1da4` | 2026-08-15 | 2.0-40 (DESCRIPTION `Date: 2025-12-14`) |
| `reference/nnet` | https://github.com/cran/nnet | `d409ca312e7b103c72547b84460c5a32acd2ae93` | 2026-08-03 | 7.3-21 |
| `reference/kernlab` | https://github.com/cran/kernlab | `d8f05c9b1b8d220deb98fdf28cd33471f17d5eae` | 2024-08-14 | 0.9-33 |
| `reference/glmnet` | https://github.com/cran/glmnet | `51f61a9819a79a9e3dbe3e88861cadfd23cc1828` | 2026-05-04 | 5.0 |
| `reference/psych` | https://github.com/cran/psych | `2b17d1e78b536c0e4cc807d8a83f546e0a0da85d` | 2026-05-16 | 2.6.5 |
| `reference/plyr` | https://github.com/cran/plyr | `93ae6654628ad50aa9fc76521eca3b7f6489bccf` | 2023-10-02 | 1.8.9 |
| `reference/tibble` | https://github.com/cran/tibble | `507752ef4e41b379219e9a76f0c9cdb781d14cd4` | 2026-01-11 | 3.3.1 |
| `reference/haven` | https://github.com/cran/haven | `5cfb3a41fe8f9c56e8c1e61802a2b9101d3c50f5` | 2025-05-30 | 2.5.5 |
| `reference/sjlabelled` | https://github.com/cran/sjlabelled | `a6c368505afc1bf7d3aebbb685b32d8ba91a8068` | 2022-04-10 | 1.2.0 |
| `reference/forcats` | https://github.com/cran/forcats | `cdaf2911c4cb1c8d341461be885b2ff8a7aaffab` | 2025-09-25 | 1.0.1 |
| `reference/ranger` | https://github.com/imbs-hl/ranger | `dafa5db57972dd49c808db881e6e6cd5c4ebdc4a` | 2026-01-16 | 0.18.0 |

The `cran/*` repositories are read-only CRAN mirrors in which each commit is one CRAN
release, so `git log` inside them lists past versions. For example, `plyr`'s
`R/join.r`, `R/join-all.r` and `R/rbind-fill.r` have no code changes between 1.8.4
(2016) and 1.8.9 (2023). Only roxygen export tags changed (checked with `git diff`).

## Additional sources cloned during Task 0.8 (not in the brief's list)

These were cloned only to check facts in Section 4 of the brief that depend on their
behaviour. They are used the same way as the sources above.

| Directory | Origin | Commit | Commit date | Version | Used to check |
|---|---|---|---|---|---|
| `reference/dplyr` | https://github.com/cran/dplyr | `08b2d303ceb4d5364615c18c9f2e06ff72a6c6bb` | 2026-04-03 | 1.2.1 | natural-join keys (`join_by_common`), `select` |
| `reference/tidyselect` | https://github.com/cran/tidyselect | `f4a576e707ac2a7eeea537afd59255eb145177a1` | 2024-03-11 | 1.2.1 | `one_of` (warns on unknown names), `ends_with`/`contains` (`ignore.case = TRUE`) |
| `reference/r-source` | https://github.com/wch/r-source (sparse checkout: `src/library/base/R`, `src/library/stats/R`) | `675d7dc5689e8a287aa372498793f0fc22ee0770` | 2026-09-21 | R trunk, `VERSION` = "4.7.0 Under development (unstable)" | `ifelse`, `pmin`, `factor`, `Ops.ordered`, `lm.fit`/`lm.wfit` tolerance, `predict.lm` |
| `reference/xgboost` | https://github.com/dmlc/xgboost (sparse checkout: `doc/changes`, root files incl. `NEWS.md`; tags fetched blobless) | master `56f951e7419a6f66f4568865e1d7835bcb6dbbf1`; tag `v1.7.6` = `36eb41c960483c8b52b44082663c99e6a0de440a` (R package `Version: 1.7.6.1`); tag `v2.0.0` = `096047c547aa71af7d53a507cecdd2a1d3124651`; tag `v3.3.0` = `d5cd2b40725d55747447f66e4a24f9a2c341b0bf` | 2026-09-22 (master) | see tags | default `base_score` for `reg:squarederror` in 1.7.x vs ≥ 2.0 |

Cloned after Checkpoint B, for the `read_essays` decision:

| Directory | Origin | Commit | Version | Used to check |
|---|---|---|---|---|
| `reference/tidyr` | https://github.com/cran/tidyr | `a4454c3aca56ffb21683b3dba6a80fc5ae4b8c96` | 1.3.2 | `separate` with `extra = "warn"`, `fill = "warn"` (`R/separate.R`, `src/simplifyPieces.cpp`) |
| `reference/readtext` | https://github.com/cran/readtext | `ee73002dd5a4c31f67cb1454f3e41df69282ca7d` | 0.92.1 | how a .txt file is read (`R/get-functions.R`) |

`r-source` is the development trunk, not a released R. The base-R functions checked
(`ifelse`, `pmin`, `factor`) have been stable for many releases, but that stability was
not checked against the R version the author used.

## Installed R environment (2026-09-22, after Checkpoint B)

**These are the current CRAN versions installed on the development machine, not the
versions the paper used, which are unknown. Results may differ between versions.**

- R: `R version 4.6.1 (2026-06-24)`, platform `aarch64-apple-darwin23`, installed by the
  owner from the CRAN macOS installer; `R.home()` = `/Library/Frameworks/R.framework/Resources`.
- Installed with
  `Rscript -e 'install.packages(c("SuperLearner","psych","glmnet","ranger","nnet","kernlab"), repos="https://cloud.r-project.org")'`
  (CRAN macOS arm64 binaries). R's `xgboost` is deliberately **not** installed: version
  3.x takes a different SuperLearner code path that drops `params` (PORTING_NOTES C8).
  Pinning it is a Phase 3 item.
- Python bridge: `pip install -e '.[oracle]'` installed rpy2 3.6.8 (rpy2-rinterface
  3.6.7, rpy2-robjects 3.6.5) into `.venv` (Python 3.13.7) without errors.
  `importr()` loads all seven packages below.

Every package reported by `installed.packages()` after that installation:

| Package | Version | Priority |
|---|---|---|
| SuperLearner | 2.0-42 | |
| psych | 2.6.5 | |
| glmnet | 5.0 | |
| ranger | 0.18.0 | |
| nnet | 7.3-21 | recommended |
| kernlab | 0.9-33 | |
| nnls | 1.6 | |
| gam | 1.22-7 | |
| cvAUC | 1.1.4 | |
| GPArotation | 2026.8-2 | |
| mnormt | 2.1.2 | |
| foreach | 1.5.2 | |
| iterators | 1.0.14 | |
| shape | 1.4.6.1 | |
| Rcpp | 1.1.2 | |
| RcppEigen | 0.3.4.0.2 | |
| rlang | 1.3.0 | |
| ROCR | 1.0-12 | |
| gplots | 3.3.0 | |
| gtools | 3.9.5 | |
| caTools | 1.18.4 | |
| bitops | 1.1-0 | |
| data.table | 1.18.6.1 | |
| boot | 1.3-32 | recommended |
| class | 7.3-23 | recommended |
| cluster | 2.1.8.2 | recommended |
| codetools | 0.2-20 | recommended |
| foreign | 0.8-91 | recommended |
| KernSmooth | 2.23-26 | recommended |
| lattice | 0.22-9 | recommended |
| MASS | 7.3-65 | recommended |
| Matrix | 1.7-5 | recommended |
| mgcv | 1.9-4 | recommended |
| nlme | 3.1-169 | recommended |
| rpart | 4.1.27 | recommended |
| spatial | 7.3-18 | recommended |
| survival | 3.8-6 | recommended |
| base, compiler, datasets, graphics, grDevices, grid, methods, parallel, splines, stats, stats4, tcltk, tools, utils | 4.6.1 | base |

The CRAN versions of kernlab, glmnet, psych, ranger and nnet equal the `Version:` of
the source clones above. **SuperLearner differs:** CRAN has 2.0-42, while the GitHub
clone cited throughout (`reference/SuperLearner`, ecpolley master) says 2.0-40. To check
the difference, the CRAN mirror was cloned too:

| Directory | Origin | Commit | Version |
|---|---|---|---|
| `reference/SuperLearner-cran` | https://github.com/cran/SuperLearner | `50d79e3dc924ab8e21fad10f8396a394ac96b551` (2026-09-14) | 2.0-42 |

`diff -rq` of the two `R/` folders: `SuperLearner.R`, `CV.SuperLearner.R`, `method.R`,
`summary.CV.SuperLearner.R`, `CVFolds.R`, `control.R`, `SL.mean.R`, `SL.nnet.R`,
`SL.xgboost.R` and `screen.glmnet.R` are byte-identical. `SL.ksvm.R`, `SL.lm.R` and
`SL.ranger.R` differ only in roxygen comment lines; their code is identical, checked
with the `#'` lines removed. Citations in the code keep the line numbers of the 2.0-40
clone; in the 2.0-42 files, the wrapper code of those three is shifted by the removed
documentation lines.

## Environment facts relevant to these sources

- R 4.6.1 is installed (above). Before 2026-09-22 it was not, and every R-side claim
  up to Checkpoint B comes from reading source, not from running it.
- The Python equivalents installed in `.venv` are listed in `docs/PHASE_1_2_PLAN.md`
  (Task 0.7 section). Since then: torch 2.14.0, transformers 5.17.0 (owner item after
  Checkpoint A), rpy2 3.6.8, and Homebrew `libomp` 23.1.2 for xgboost.
