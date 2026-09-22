# Porting Notes

Every deviation from the original R code (`tobiaswolfram/llm_paper`), and every
non-obvious fidelity decision, is logged here. The stance agreed with the project
owner: **fix bugs, but document every fix**. This file is the audit trail that
distinguishes "a documented translation" from "a rewrite."

Status legend: ✅ done · 🔦 flagged, decision pending real data · ⏳ not yet ported.

---

## A. Bugs and inconsistencies in the original repo

These are defects in the *public* R repo. A naive line-by-line port would faithfully
reproduce broken behaviour, so each is called out with how we handle it.

### A1. `variables.xlsx` schema mismatch 🔦 (highest impact)
The whole `clean_ncds` recode keys on columns named `variable`, `sweep`,
`respondent`, `new_varname`, `type`, and constructs
`full_name = s{sweep}_{substr(respondent,1,2)}_{new_varname}`.
**None of those column names exist in the shipped `data/variables.xlsx`.** The file
has no header row, so R's `read_excel` silently promotes the first data row to
column names, yielding columns like `health`, `mother`, `birthweight`, `N646`. The
original readme itself notes the file contains variables "from previous work."
Conclusion: the public file is not the file the code was written against.

Handling: reconstruct the required schema as a versioned, provenance-flagged
artifact (`data/schema/ncds_variable_mapping.yaml`). The cleaning *logic* is ported
faithfully; the *mapping table* is provisional until the author's real file arrives.
See VALIDATION_CHECKLIST item V1.

### A2. Undefined objects inside `clean_ncds` 🔦
The final `plyr::join_all(list(teacher, parents, height, birthweight, bsag,
behavior, ability, aspirations, personality, motivation, parenting, highest_edu,
sex, camsis), ...)` references `bsag`, `aspirations`, `parenting`, and `camsis` —
none of which are defined anywhere in the function. In R this errors at runtime
(or silently picks up global-env leakage). Likely dead code from an earlier version.

Handling: determine from the paper's variable list whether these are real blocks
(BSAG = Bristol Social Adjustment Guides is plausibly a real construct) or dead
code, and either implement or drop with a note. Decision pending V1.

### A3. `get_complete_ncds` arity mismatch ✅
*Called* in `_targets.R` with 5 args
(`ncds_1_2_3_cleaned, factor_data, aspiration_data, essay_data, gene_data`) but
*defined* in `functions.R` with 4 (no gene argument). The gene arg is silently
dropped by R. Given `read_gene_data` is an empty `#PLACEHOLDER` stub, gene data is
absent regardless.

Handling: Python signature makes the gene input explicit and optional; when absent,
gene-dependent targets are skipped rather than silently no-op'd.

### A4. `create_essay_variables` parameter naming ✅
Called with `gpt_embeddings` as the 4th argument, but the parameter is named
`roberta_embeddings` and the body joins on it. The "gpt vs roberta" naming is
inconsistent across the essay targets.

Handling: the embedding source is passed explicitly and named for what it is; the
pipeline spec records which embedding feeds which target.

### A5. `get_gpt_embeddings` / `get_gpt4_embeddings` are identical ✅
The two functions are byte-identical; both read the same RDS object. The GPT-3.5 vs
GPT-4 distinction actually lives in `get_gpt_embeddings.R` (different `model=`
strings, different output RDS files), not in these readers.

Handling: a single parametrised reader; the model choice is a parameter, matching
where the real distinction lives.

### A6. `create_data.R` will not run top-to-bottom 🔦
Contains at least: a dangling `dplyr::mutate` after a broken pipe (the
`appendix_11_data` block starts a new statement with a `.` placeholder and no
upstream), and several objects used before assignment
(`essay_full_metrics_lm`, `teacher_genes_essay_overlap_metrics`,
`cog_superlearner_social_lm` used without `tar_read`). Mixed `tar_read(...)` vs
bare-symbol usage throughout.

Handling: rebuilt as a clean figures module (`scripts/make_figures.py`) that
reproduces the *intended* outputs (`fig_2..5_data.csv`, `appendix_D1..D12`). Each
fixed line noted inline in that module. Decision on ambiguous blocks pending the
model outputs being available.

### A7. Machine-specific / Windows-only paths ✅
`C:/TreeTagger` (TreeTagger install), `C:/Users/usr/anaconda3/python.exe`
(reticulate). Non-portable.

Handling: all paths centralised in `config.py`; no absolute machine paths anywhere.

### A8. Deprecated / fragile R idioms ✅ (informational)
`dplyr::as.tbl` (deprecated), `dplyr:::select` with three colons (line 201, reaches
into the namespace internals). Signals the code was written across several R
versions and not re-run cleanly end-to-end. No action beyond noting it.

---

## B. Fidelity decisions in already-ported code

### B1. CV metric fold-wise aggregation ✅
The original computes each risk *per outer fold* and reports mean/min/max across
folds (not a single pooled risk). Reproduced exactly in `metrics/cv_metrics.py`.
Verified against hand-computed values in `tests/test_cv_metrics.py`.

### B2. `sd` uses n-1 (ddof=1) ✅
R's `sd` uses the sample (n-1) denominator; NumPy's `std` defaults to n (ddof=0).
The winsorisation threshold uses `sd(abs(SL.predict))`, so `np.std(..., ddof=1)` is
required for parity. Encoded and tested.

### B3. Winsorisation formula quirk ✅ (documented, reproduced faithfully)
The R clamp is
`SL.predict[abs(SL.predict) > 10*sd(abs(SL.predict)) + mean(SL.predict)] <- mean(SL.predict)`.
Two properties reproduced exactly: (i) the threshold mixes the SD of the *absolute*
values with the mean of the *raw* values; (ii) because the outlier inflates its own
SD, a single large prediction is only clamped if it exceeds ~10 SDs above the mean —
so in practice the clamp fires rarely. This is faithful to the original, not a fix.
Tested in `test_winsorise_only_clamps_beyond_10_sd`.

### B4. Metric rows: MSE and SE from the unclamped predictions, and lm MSE from `SL.lm_All`
Rewritten in Phase 1, Task 1.2. The earlier version of this note claimed the port
was "numerically identical". It was not: it computed the MSE after the clamp, and the
lm row used the ensemble's MSE (brief F6).

- **Label:** faithful.
- **R source:** `llm_paper/R/functions.R:L704–705` (the "Super Learner" row of
  `summary(cv_fit)$Table`, taken **before** the clamp on `L707`);
  `L731–732` (`get_cv_lm_metrics` takes the `SL.lm_All` row); `L748` (the lm row's
  sample-size column is named `length(cv_fit$Y)`).
  `SuperLearner/R/summary.CV.SuperLearner.R:L39–43, L66` (2.0-40): per-fold risk
  `mean(w * (Y - pred)^2)`; Ave, Min and Max over folds, where Ave is the plain mean;
  `se = (1/sqrt(n)) * sd(w * (Y - pred)^2)` with `n = length(SL.predict)` (`L19`).
- **Python** (`metrics/cv_metrics.py`):
  - `superlearner_metrics` takes `mean_mse`, `min_mse`, `max_mse` and `se_mse` from
    the unclamped Super Learner predictions, and R², MAD and RMSE from the clamped
    ones.
  - `lm_metrics` takes `mean_mse`, `min_mse`, `max_mse` and `se_mse` from the
    unclamped `SL.lm_All` column (the linear model alone). R² comes from `SL.lm_All`
    against `SL.mean_All`, and MAD and RMSE from the clamped ensemble, as before.
  - SE uses `ddof=1`.
  - Naming: R calls the SE column **`se`**; the port calls it `se_mse` (owner decision
    C5). The R rows also carry an `Algorithm` column ("Super Learner" or "SL.lm_All"),
    which the port does not reproduce. The lm row's sample-size column stays `n`
    instead of R's `length(cv_fit$Y)`.
- **Why:** faithful order of operations (brief F6).
- **Test:** `tests/test_cv_metrics.py::test_superlearner_mse_unclamped_r2_mad_rmse_clamped`,
  `::test_lm_metrics_mse_is_sl_lm_all_fold_mean`, `::test_se_mse_hand_computation`.
- **Validation:** V6. Figure and appendix CSVs against the paper.

---

## C. Super Learner engine and base learners (rewritten in Phase 1, Task 1.3)

Citations are to SuperLearner 2.0-40 and the package versions in
`docs/REFERENCE_SOURCES.md`. The SuperLearner files involved have not changed since
2017–2020, except the xgboost wrapper (C8). All C entries are confirmed on real data
by VALIDATION_CHECKLIST **V4** (numerical parity with the R oracle on shared folds)
and **V7** (learner settings).

### C1. Nested-CV mechanics, failures and weights
- **Label:** faithful.
- **R source:**
  - `SuperLearner/R/SuperLearner.R`: `L119–131, L143` (each screener runs once per
    training set and its mask is shared); `L121–124, L211–213` (a failed screener
    means All); `L143–152, L180–186` (a learner error leaves NA, the whole Z column is
    set to 0, and the fit stops if all columns are 0); `L222–223` (screening re-run on
    the full training set); `L249–252, L274–292` (a refit error gives NA; weights are
    recomputed only if the failed learners' weight is positive); `L295` (prediction);
    `L303–305` (cvRisk NA for CV failures).
  - `method.R:L39–78` (NNLS weights; prediction over non-zero weights).
  - `CV.SuperLearner.R:L80–91` (outer loop; discrete SL = `which.min(cvRisk)`).
- **Python:** `models/native_superlearner.py` and `models/meta.py` follow each of
  those lines. Any exception in a learner or screener is caught, as R's `try()`
  catches any error, and recorded in `CVSuperLearnerFit.failures`.
  `CVSuperLearnerFit.cv_risk` holds the per-fold learner risks.
  - Owner decision C10: `cv_risk` is NaN only for learners that failed in the inner
    CV. A learner that fails only in the refit keeps a number, as in R. When its
    positive weight forced a recomputation, that number is computed from its zeroed Z
    column.
  - The brief's wording "cv_risk is NaN for failed learners" is therefore exact only
    for CV failures.
- **Why:** brief F4.
- **Test:** `tests/test_superlearner_mechanics.py`: `::test_failing_learner_gets_weight_zero_and_ensemble_has_no_nan`,
  `::test_refit_failure_with_positive_weight_recomputes_weights`,
  `::test_refit_failure_with_zero_weight_changes_nothing`,
  `::test_all_learners_failing_stops`,
  `::test_screening_runs_once_per_training_set_and_mask_is_shared`,
  `::test_screening_failure_keeps_all_columns`, `::test_compute_*`.
- **Validation:** V4.

### C2. Folds and random seeds are owned by the port
- **Label:** APPROX.
- **R source:** `llm_paper/R/functions.R:L506–512` (`clusterSetRNGStream(cluster, 1)`,
  `cvControl = list(V = 10)`, `innerCvControl = list(list(V = 5))`);
  `SuperLearner/R/control.R:L15` (shuffled, not stratified); `CVFolds.R:L23`
  (`split(sample(1:N), rep(1:V, length = N))`).
- **Python:**
  - Outer folds: `make_folds(n, 10, seed)`, a shuffled split with the same fold sizes
    as R's.
  - Inner folds, screener CV folds and learner seeds come from
    `models/seeds.py::derive_seed(seed, outer, inner, name)`. They depend only on the
    base seed, the outer fold, the inner fold and the learner or screener name.
  - Results are identical across repeated runs and across `n_jobs`. Outer folds
    optionally run in parallel with joblib (`n_jobs`, default 1).
  - Optional `folds` and `inner_folds` arguments fix the layout for the R oracle.
- **Why:** R's random numbers cannot be reproduced. The outer folds are drawn in the
  master process (`CV.SuperLearner.R:L19`), while `clusterSetRNGStream` seeds only the
  workers.
- **Test:** `tests/test_superlearner_mechanics.py::test_repeated_runs_are_identical`,
  `::test_n_jobs_does_not_change_output_small_library`,
  `::test_n_jobs_does_not_change_output_full_library` (slow),
  `::test_fixed_folds_are_used`; `tests/test_learners.py::test_derive_seed_is_deterministic_and_distinct`.
- **Validation:** V4 (compare on shared folds).

### C3. `SL.mean`
- **Label:** faithful.
- **R source:** `SuperLearner/R/SL.mean.R:L3` (`weighted.mean(Y, obsWeights)`; the
  weights are all 1, `SuperLearner.R:L101–103`).
- **Python:** `DummyRegressor(strategy="mean")`.
- **Why:** identical estimator.
- **Test:** covered by every engine test.
- **Validation:** V4.

### C4. `SL.lm`: R's rule for aliased columns
- **Label:** APPROX.
- **R source:** `SuperLearner/R/SL.lm.R:L44` (`lm(Y ~ ., weights = obsWeights)`);
  `r-source/src/library/stats/R/lm.R:L169` (`lm.wfit`, `tol = 1e-7`), `L733–739`
  (`predict.lm` uses only the non-aliased columns).
- **Python:** `RLinearModel`. With the intercept first, column j is kept iff
  `|R_jj| >= 1e-7 * ||x_j||` in an unpivoted QR, i.e. its part orthogonal to the
  earlier columns is not negligible. OLS is then fitted on the kept columns, and
  prediction uses only those. With no columns left it fits the intercept only, as
  `lm(Y ~ .)` does.
- **Why:** scikit-learn's `LinearRegression` keeps aliased columns (minimum-norm
  solution), which changes predictions when new data break a collinearity.
  APPROX: dqrdc2's limited-pivoting arithmetic can differ at the threshold.
- **Test:** `tests/test_learners.py::test_sl_lm_drops_exactly_collinear_column_like_r`,
  `::test_sl_lm_with_no_columns_fits_intercept_only`.
- **Validation:** V4, V7.

### C5. `SL.ranger`
- **Label:** APPROX.
- **R source:** `SuperLearner/R/SL.ranger.R:L59–66`: `num.trees = 500`,
  `mtry = floor(sqrt(ncol(X)))`, `min.node.size = 5` (gaussian), `replace = TRUE`,
  `sample.fraction = 1`, `num.threads = 1`. `ranger/src/TreeRegression.cpp:L106`
  (0.18.0; `<=` since 2014): a node with `n <= min.node.size` is not split.
  `ranger/R/ranger.R:L117`: `min.bucket` default 1.
- **Python:** `RandomForestRegressor(n_estimators=500,
  max_features=max(1, floor(sqrt(p))), min_samples_split=6, min_samples_leaf=1,
  bootstrap=True, max_samples=None, n_jobs=1)`, with p counted after screening.
- **Why:** sklearn splits when `n >= min_samples_split`
  (`sklearn/tree/_tree.pyx:L231`), so 6 reproduces ranger's threshold (owner decision
  C2; brief F5 said 5, which is off by one). APPROX: sklearn bootstraps with sample
  weights and counts distinct rows per node, while ranger counts draws including
  duplicates. The trees and their random numbers are different implementations.
- **Test:** `tests/test_learners.py::test_ranger_settings`.
- **Validation:** V4, V7.

### C6. `SL.nnet`
- **Label:** APPROX. The weight limit is faithful.
- **R source:** `SuperLearner/R/SL.nnet.R:L5, L8` (`size = 2, linout = TRUE,
  trace = FALSE, maxit = 500`); `nnet/R/nnet.R:L78–79` (`rang = 0.7`, `decay = 0`,
  `MaxNWts = 1000`, `abstol = 1e-4`, `reltol = 1e-8`; the default `maxit = 100` is
  overridden); `L105–106` ("too many weights"); `L242–268` (a bias for every unit).
- **Python:** `NnetLike`. It raises `LearnerFailure` when `2p + 5 > 1000` (p ≥ 498);
  otherwise it fits `MLPRegressor(hidden_layer_sizes=(2,), activation="logistic",
  solver="lbfgs", alpha=0.0, max_iter=500)` without input scaling.
- **Why:** faithful failure point. APPROX: start weights (nnet draws U(−0.7, 0.7)) and
  the optimiser's stopping rules differ, so the fitted networks differ.
- **Test:** `tests/test_learners.py::test_nnet_weight_count_is_2p_plus_5`,
  `::test_nnet_fails_at_498_columns_and_fits_at_497`,
  `::test_nnet_uses_maxit_500_and_no_decay`.
- **Validation:** V4, V7.

### C7. `SL.ksvm`
- **Label:** APPROX. The scaling rule and the sigma rule are faithful.
- **R source:** `SuperLearner/R/SL.ksvm.R:L89–129` (C = 1, epsilon = 0.1,
  rbfdot, kpar automatic, scaled = TRUE; the wrapper's `cache`/`tol`/`shrinking` are
  not passed, and kernlab's own defaults `kernlab/R/ksvm.R:L61–63` apply);
  `kernlab/R/ksvm.R:L106` (eps-svr), `L127–148` (scaling: all columns and y, or none
  if any column is constant), `L153–156`, `sigest.R:L58–64` (sigma),
  `L2645–2647, L2810–2811` (prediction scaling).
- **Python:** `KsvmLike`. It standardises X and y (ddof = 1) unless any column is
  constant (all values identical), estimates sigma with the `sigest` rule on the
  (possibly scaled) X, fits `SVR(kernel="rbf", gamma=sigma, C=1, epsilon=0.1,
  tol=1e-3, cache_size=40, shrinking=True)`, and unscales the predictions.
- **Why:** faithful rules. APPROX: libsvm and kernlab are different solvers, and
  `sigest` samples its row pairs with R's RNG.
- **Test:** `tests/test_learners.py::test_ksvm_skips_all_scaling_when_a_column_is_constant`,
  `::test_ksvm_scales_x_and_y_when_no_column_is_constant`,
  `::test_sigest_matches_kernlab_formula`.
- **Validation:** V4, V7.

### C8. `SL.xgboost.hist`
- **Label:** APPROX.
- **R source:** `llm_paper/R/functions.R:L492–494` (`params = list(tree_method =
  "hist")`); `SuperLearner/R/SL.xgboost.R:L43–46, L102, L106` (the branch for
  xgboost < 3.0: `reg:squarederror`, `nrounds = 1000`, `max_depth = 4`,
  `min_child_weight = 10`, `eta = 0.1`, `nthread = 1`). The base score comes from
  xgboost v1.7.6 `include/xgboost/objective.h:L33` and
  `src/objective/objective.cc:L34–38` (0.5 for squared error); the estimated
  intercept since 2.0.0 is in xgboost `NEWS.md:L50–52` and v2.0.0
  `src/objective/regression_obj.cu:L66`.
- **Python:** `XGBRegressor(n_estimators=1000, max_depth=4, min_child_weight=10,
  learning_rate=0.1, tree_method="hist", n_jobs=1, base_score=0.5,
  objective="reg:squarederror")`.
- **Why:** the installed xgboost 3.3.0 would estimate the base score from the labels,
  while R xgboost 1.7.x uses 0.5, so it is set explicitly. APPROX: the paper's R
  xgboost version is unknown (1.7.x is assumed), and histogram construction differs
  across versions.
- **Note:** SuperLearner added a branch for xgboost > 3.0 on 2025-12-14 (commit
  `abebb56`, `SL.xgboost.R:L53–65`). That branch **does not pass `params`**, so under
  R xgboost ≥ 3 the `tree_method = "hist"` override would be silently ignored. The R
  oracle therefore needs pinned package versions, which is a Phase 3 item. R's xgboost
  is deliberately not installed here yet (owner instruction).
- **Test:** `tests/test_learners.py::test_xgboost_settings_and_base_score`.
- **Validation:** V4, V7.

### C9. `screen.glmnet`
- **Label:** APPROX. The selection rule is faithful.
- **R source:**
  - `SuperLearner/R/screen.glmnet.R:L1–16`: alpha 1, minscreen 2, nfolds 10,
    nlambda 100, deviance (MSE), `lambda.min`, and the fallback
    `which.max(sumCoef >= minscreen)`.
  - `glmnet/R/glmnet.R:L384` (defaults), `L393` (≥ 2 columns), `L581` + `fix.lam.R`.
  - `elnet.R:L23` (constant y); `cvstats.R:L6`; `getOptcv.glmnet.R:L5–8`.
  - glmnetpp `elnet_driver/standardize.hpp:L35, L73–76`, `elnet_driver/gaussian.hpp:L448`,
    `elnet_path/base.hpp:L209–211, L262–272, L307–310`, `elnet_path/gaussian_base.hpp:L132–134`.
  - `glmnet.control.R:L153–157`.
- **Python:** `models/screeners.py::screen_glmnet`:
  - standardise with the population SD and centre y;
  - λmax = max|Xsᵀ(y−ȳ)|/n and 100 geometric values down to 1e-4·λmax (1e-2 if
    n < p);
  - `sklearn.linear_model.lasso_path` on that grid;
  - glmnet's early stop (≥ 5 points, relative R² gain < 1e-5 or R² > 0.999) on the
    full-data path, whose truncated grid is reused in the folds;
  - `KFold(10, shuffle=True)`, pooled CV MSE, ties to the largest λ;
  - fallback to the first λ with ≥ 2 non-zero coefficients. If there is none, it
    takes the first (largest) λ, which usually selects **no** column, exactly as
    `which.max` of an all-FALSE vector does;
  - `ScreenFailure` when p < 2 or y is constant. The engine then keeps all columns.
- **Why:** faithful selection rule. APPROX: convergence criteria differ, and R's CV
  folds are random. `foldid` and `lambdas` arguments let the Task 2.2 oracle compare
  on identical folds and grid.
- **Test:** `tests/test_learners.py::test_screening_*`, `::test_lambda_grid_matches_glmnet_definition`,
  `::test_glmnet_early_stop_rule`, `::test_screen_invariant_to_rescaling_a_column`,
  `::test_screen_fallback_*`.
- **Validation:** V4.

---

## D. Unused / carried-over files (informational)

- `data/camsis/gb71co60.dta`, `data/camsis/gb91soc2000.dta`: present in the original
  repo but **never referenced** by the pipeline (only `gb71co70.dta` is used, via
  `read_camsis`). Carried over for completeness; flagged so they aren't mistaken for
  live dependencies.
- Original repo has **no LICENSE and no citation file** → legally all-rights-reserved.
  Resolve attribution/licensing with the author before public release (see README).

---

## E. IO layer (`src/llm_cong_predict/io/`) — deviations & decisions

### E1. `read_gene_data` raises instead of returning NULL ✅ (documented)
The R body was an empty `#PLACEHOLDER` returning `NULL` silently. The Python port
raises `NotImplementedError` with a clear message, so any use of gene data fails
loudly rather than propagating a silent `None`. (Note: `_targets.R` also mis-wires
this target — `tar_target(gene_data, read_gene_data)` passes the function itself,
uncalled — which is handled at the pipeline layer, not here.)

### E2. Stata value labels carried via `df.attrs` ✅
`haven`/`sjlabelled` attach `value -> label` maps to columns; pandas has no direct
equivalent, so the readers carry `pyreadstat`'s `variable_value_labels` on
`df.attrs['value_labels']`, and `io/labels.py` provides `as_factor` /
`to_character` / `set_na_range` to apply them where the cleaning layer needs them.
Caveat: `df.attrs` is not always propagated across pandas operations, so labels are
re-attached after transforms and should be read early in the cleaning chain.

### E3. `combine_ncds` column-collision handling 🔦 (assumption, verify with data)
`plyr::join_all(type="full")`'s behaviour when two frames share a non-key column is
ambiguous. The port does a full outer join on `ncdsid` and, on any non-key overlap,
*coalesces* (prefer left, fill from right) into one column and records it in
`df.attrs['combine_ncds_collisions']`, rather than letting pandas suffix `_x`/`_y`.
The NCDS waves use distinct variable codes so this is not expected to trigger; the
coalescing rule must be checked against R once real data is available.

### E4. `read_camsis` does not lower-case column names ✅ (faithful)
Matches the R (`haven::read_dta` only). The real CAMSIS files already use lower-case
names (`co1970`/`mcamsis`/`fcamsis`) that `create_aspirations` relies on.

### E5. `read_datalist` reads whatever the file contains ✅ (faithful)
Thin `read_excel` wrapper. The schema mismatch (A1) is deliberately NOT fixed in the
reader — it is a cleaning-layer concern where the provisional schema is
reconstructed and flagged.

---

## F. Cleaning block (`src/llm_cong_predict/cleaning/`) — deviations & decisions

Scope note: `clean_ncds` itself is **deferred** (option (b), user decision): a
strictly-grounded schema (no guessing) would be only partial, so rather than build a
half-known variable renamer that would likely be redone, `clean_ncds` waits for the
author's real `variables.xlsx`. The other five cleaning functions are ported now.

### F1. `create_factors` — polychoric factors DEFERRED, not guessed 🔦 (V3)
Three of the four factors use `psych::fa(cor="poly")` (polychoric). A polychoric
estimator matching R's `polycor`/`psych` is nontrivial, and substituting a Pearson
correlation would change the numbers while appearing to work. So only the Pearson
factor (`s2_co_factor_ability`) is computed; the three polychoric factors raise
unless `include_polychoric=True`, which itself raises until V3 provides a validated
estimator or an rpy2 `psych::fa` fallback. This is the "don't guess where data/method
is missing" rule applied to a method gap.

### F2. `create_factors` implemented in numpy, not `factor_analyzer` ✅ (forced + better)
`factor_analyzer` 0.5.1 (its latest release) is INCOMPATIBLE with the installed
scikit-learn: it calls the removed `force_all_finite` argument and errors on `fit`.
So the single-factor minres extraction and regression (Thurstone) scores are
implemented directly in numpy/scipy. Side benefit: the factor math is auditable for
a replication rather than hidden in a library. Still PARITY-UNVERIFIED vs `psych`
(V3) — psych's minres and scoring differ in detail; V3 compares via |correlation|
because factor scores are identified only up to sign and scale.

### F3. `create_aspirations` sex comparison reproduced faithfully, incl. its quirk ✅
The R computes `sex = as.character(as_factor(sex))` and then
`ifelse(sex == 1, camsis_male, camsis_female)` — comparing a *character* ("Male"/
"Female" or "1"/"2" depending on the file's labels) to the numeric literal `1`. If
the sex variable carries text labels, that equality is never true and every
respondent takes the female score. This is reproduced exactly (the male branch fires
only when the character sex equals the string "1"), and flagged here because it is a
latent data-dependent quirk of the original to check against real data.

### F4. `get_complete_ncds` gene argument made explicit ✅ (see A3)
The R definition takes 4 args but is called with 5 (`gene_data`), silently dropped.
The port adds an explicit optional `ncds_gene`; `None` reproduces the 4-arg R exactly.

### F5. `find_essay_teacher_genetics_overlap` — raw teacher codes grounded ✅
Uses raw codes `n876`–`n885`, which are explicit in the R and correspond to the
age-11 teacher ratings; converts to labels, sets the "Dont know" label to missing,
keeps complete cases, then inner-joins with essays and the ability-complete frame.

---

## G. Feature layer (`src/llm_cong_predict/features/`) — deviations & decisions

### G1. GPT embeddings: RDS round-trip replaced; original reshaper is contradictory 🔦
The R reshaper `get_gpt_embeddings` does `readRDS(essays)` (argument = PATH to the
raw OpenAI-response `.rds`) but then `bind_cols(essays, .)` (argument = essays data
frame) — the same argument used two incompatible ways. `.rds` is also an R-only
binary format. The Python port therefore (a) ports the generation script
(`get_gpt_embeddings.R` → `scripts/get_gpt_embeddings.py`) to save a Python-native
Parquet/CSV with `ncdsid` + embedding columns, and (b) makes the reshaper read that
and return the reshaper's INTENDED output (`id` + embedding columns). Deviation and
the original contradiction both documented.

### G2. RoBERTa mean-pooling matches the R exactly, including padding ✅
`get_roberta_embeddings` takes `tf$reduce_mean(last_hidden_state, axis=1)` after
passing only input_ids (no attention_mask), so padding positions ARE included in the
mean. The port (PyTorch) reproduces this exactly — mean over all `max_len=250`
positions, padding included. A mask-weighted mean would give different numbers, so it
is deliberately NOT used. Framework differs (TF→PyTorch) but the `roberta-base`
weights are identical. Runnable with the real essays; not run in the sandbox.

### G3. SALAT + spelling = INGESTION ONLY (user decision) ✅ flagged
`get_salat_metrics` and `get_spelling_error_metrics` read CSVs from external tools
(SALAT desktop apps; LanguageTool CLI) and reshape them. They do NOT generate the
metrics. Reimplementing the metrics in Python would produce different numbers than
the paper, so generation stays external and these consume its output. Tested on
synthetic CSVs. One pandas-specific fix: the spelling merge coerces `ncdsid` to
string on both sides (pandas refuses to merge str-vs-int keys; R was type-tolerant).

### G4. Readability/tokenization are an EXTERNAL-TOOL BOUNDARY that raises 🔦
`tokenize_essays` (TreeTagger) and `calculate_readability_metrics` (koRpus) generate
metrics with R-specific / external tools. Rather than substitute a different Python
readability library (which would silently diverge from the paper), these RAISE with
guidance. An `ingest_readability_metrics` path is provided so pre-computed
readability (koRpus output, or the author's derived features) can enter the pipeline
the same way SALAT/spelling CSVs do. A validated native re-implementation is a
possible future checklist item, not a silent default.

### G5. `create_essay_variables` — data-dependent filter, embedding arg renamed ✅
The final `select_if` keeps only columns that are non-NA, finite, and non-constant,
so the essay feature-matrix width depends on the data (matched, flagged). The 4th
argument is named `embeddings` (the R named it `roberta_embeddings` but the pipeline
passes `gpt_embeddings`; see A4).

---

## H. Data safety (Phase 1, Task 1.1)

Entries from here on use the format of brief Section 7: ID, label, R source lines,
what the Python does, why, the test that covers it, and the VALIDATION_CHECKLIST
item, if any.

### H1. Restricted inputs are read only from `$LCP_DATA_ROOT`
- **Label:** data-safety change.
- **R source:** `llm_paper/_targets.R:L41–114` reads every restricted input from the
  project's own `data/` folder (for example `data/ncds_1_2_3/ncds0123.dta` at L45,
  `data/spelling_mistakes.csv` at L100).
- **Python:** `config.RESTRICTED_INPUTS` keeps the same relative names in one table.
  `config.restricted_path(key)` resolves them under `$LCP_DATA_ROOT`. Participant-level
  outputs go to `derived/`, `fits/` and `logs/` under the same root
  (`config.derived_dir()`, `fits_dir()`, `logs_dir()`). `config.data_root()` raises
  `DataRootError` when the variable is unset, when the directory is missing, or when it
  resolves (symlinks followed) inside the repository. There is no default. Importing
  the package never touches the data root. The file names are those of `_targets.R`
  and are not yet confirmed against the UK Data Service downloads.
- **Why:** restricted data must never be inside the repository (brief Section 2.3).
- **Test:** `tests/test_data_safety.py::test_data_root_*`,
  `::test_restricted_and_output_paths_built_under_data_root`,
  `::test_importing_config_never_touches_the_data_root`.
- **Validation:** none. When the real downloads arrive, confirm the names in
  `RESTRICTED_INPUTS`.

### H2. `scripts/get_gpt_embeddings.py` needs a double opt-in
- **Label:** data-safety change.
- **R source:** `llm_paper/R/get_gpt_embeddings.R:L19–29` and `L40–50` send every
  essay's text to the OpenAI embeddings API.
- **Python:** the script is kept for provenance. It refuses to run (exit code 3) unless
  both `--i-confirm-the-data-licence-permits-external-processing` and
  `LCP_ALLOW_EXTERNAL_API=1` are given, and prints why: the project rule is that essays
  must never be sent to external APIs. The check runs before any essay is read and
  before `openai` is imported. Essays are read from, and embeddings written to,
  `$LCP_DATA_ROOT`. No code in the package or pipeline calls the script (checked with
  `git grep`). The `openai` client moved from the `embeddings` extra to a separate
  `external-api` extra, so installing the local embedding stack never installs it.
- **Why:** brief Section 2.3; owner decision at Checkpoint A (item 2).
- **Test:** `tests/test_data_safety.py::test_gpt_script_refuses_without_double_opt_in`
  (three cases), `::test_embeddings_extra_does_not_include_openai`.
- **Validation:** none.

### H3. Git allow-list, restricted-data checker and pre-commit hook
- **Label:** data-safety change.
- **R source:** none. The original repository tracks `data/*.csv` and `data/*.rds`
  through Git LFS (`llm_paper/.gitattributes`).
- **Python:**
  - `.gitignore` ignores everything under `data/` except `data/variables.xlsx`,
    `data/occupation_aspiration_mapping.xlsx` and `data/camsis/*.dta`.
  - It ignores `/reference/`, `/outputs/`, `/results/`, `/.cache_pipeline/` and the
    data-like extensions everywhere.
  - It ignores CSV files at the repository root. That goes beyond the brief's list;
    it covers the root predictions CSV found in F8.6.
  - `scripts/check_no_restricted_data.py` checks staged files (or all tracked files
    with `--all`). It refuses: data-like extensions outside the allow-list, other files
    under `data/`, CSV/TXT outside `tests/` and `docs/` (except `requirements*.txt`),
    files over 5,000,000 bytes (staged blob size), and anything under `reference/`.
  - `scripts/hooks/pre-commit` runs the checker. It is enabled per clone with
    `git config core.hooksPath scripts/hooks`, which is documented in the README and
    enabled in the development clone.
- **Why:** `.gitignore` alone is bypassed by `git add -f`. Brief Task 1.1, F8.6.
- **Test:** `tests/test_data_safety.py::test_checker_*`,
  `::test_pre_commit_hook_blocks_commit`, `::test_gitignore_*`.
- **Validation:** none.

---

## I. Model specification (Phase 1, Task 1.4)

### I1. Every model target records its method, outcome, predictors, sample, data preparation and scorer
- **Label:** faithful.
- **R source:** `llm_paper/_targets.R:L178–385` (model targets), `L387–494` (metric
  targets); `llm_paper/R/create_data.R:L98, L163–170, L181, L344, L347` (scoring of
  targets that have no metric target in `_targets.R`).
- **Python:**
  - `scripts/extract_r_targets.py` parses both files and writes
    `docs/reference/r_targets_inventory.json` (70 model targets, 416 fits, 53 metric
    targets). The output was checked by hand against the R files.
  - `pipeline/model_spec.py` generates the same 70 targets from 13 families. Each
    `ModelTargetSpec` records method, outcome (and the `pattern` it iterates over),
    predictors, sample, data preparation and scorer.
  - `pipeline/variable_lists.py` holds the constant variable lists verbatim.
  - `R_TO_PYTHON_TARGET` maps every R name to its Python name. The names are
    regularised as `<feature set>_<method><suffix>`, for example `essay_lm` →
    `essay_lm_lm` and `salat_metrics_superlearner` → `salat_metrics_superlearner_text`.
- **Scorers (brief F7):** `lm` only for the seven `*_lm` targets (`L452–464`).
  `superlearner` for every other scored target, including the three `*_social_lm` lm
  fits (`L489–494`), `text_length` and the seven text components (create_data.R), and
  the two `*_social_lm_overlap` targets (create_data.R). **`none`** for the seven
  `*_superlearner_mmg_lm` targets (`L323–343`), which neither file scores (owner
  decision C6). They are fitted but get no metric target.
- **Data preparation (brief F7):** `pedu` → `as.numeric(s3_pa_edu)` (`L208, L368`);
  `sociological` → `as.numeric` on all seven sociological variables (`L258, L374`);
  RoBERTa → inner join of `roberta_embeddings` on `ncdsid = id` (`L227`); GPT-4 → drop
  columns starting with `embedding`, then inner join `gpt4_embeddings` (`L230`). The
  joined tables are graph dependencies of their model targets.
- **Why:** brief Task 1.4 and F7.
- **Test:** `tests/test_pipeline.py::test_every_r_model_target_maps_to_one_python_target_with_the_same_definition`,
  `::test_expanding_over_outcomes_gives_416_fits`, `::test_variable_lists_match_the_r_constants`,
  `::test_embedding_models_depend_on_their_embedding_targets`,
  `::test_extractor_reproduces_the_committed_inventory`,
  `::test_model_spec_expands_deterministically`.
- **Validation:** none (structure only).

### I2. `cog_superlearner_social_lm` and `…_overlap` use one predictor
- **Label:** faithful. This fixes a port bug.
- **R source:** `llm_paper/_targets.R:L377, L382`: the predictor is
  `"s2_co_factor_ability"`, not `cog_variables`.
- **Python:** the `_social_lm` and `_social_lm_overlap` families override the `cog`
  feature set with the literal column. The previous port used all seven
  `cog_variables`.
- **Why:** brief F7.
- **Test:** `tests/test_pipeline.py::test_cog_social_lm_uses_the_single_ability_factor`.
- **Validation:** none.
