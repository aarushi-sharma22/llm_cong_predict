# Porting Notes

Every deviation from the original R code (`tobiaswolfram/llm_paper`, commit
`b0cfe4c`), and every non-obvious fidelity decision, is logged here. This is the
audit trail that distinguishes a documented translation from a rewrite.

The stance (brief Section 2.2, replacing the earlier "fix bugs" stance):
- **Default: faithful.** The port reproduces what the R evidently did, including
  quirks that change results.
- **Reconstruction:** where the published R cannot have run (undefined objects, a
  wrong call), the port implements the evident intent and says so here.
- **Corrected variant (off by default):** a corrected behaviour exists only where the
  brief asks for one, behind a keyword argument that defaults to `False`.
- **APPROX:** where Python can only approximate the R, the gap is measured by an item
  in `docs/VALIDATION_CHECKLIST.md`.
- **Data-safety change:** changes made to keep restricted data out of the repository
  and away from external services.

Entries written or rewritten from Phase 1 on use the format of brief Section 7: ID,
label, R source lines, what the Python does, why, the test, and the VALIDATION_CHECKLIST
item. Older entries keep the status legend ✅ done · 🔦 flagged, decision pending real
data · ⏳ not yet ported. Line numbers refer to the clones listed in
`docs/REFERENCE_SOURCES.md`.

---

## A. Bugs and inconsistencies in the original repo

These are defects in the *public* R repo. A naive line-by-line port would faithfully
reproduce broken behaviour, so each is called out with how we handle it.

### A1. `variables.xlsx` is the right file with its header row missing (rewritten in Phase 1, Task 1.6)
- **Label:** reconstruction. It replaces the earlier note, which was wrong.
- **R source:** `llm_paper/_targets.R:L43` (`read_datalist("data/variables.xlsx")`);
  `llm_paper/R/functions.R:L22–24` (`read_excel(path)`), `L75–78` (uses `variable`,
  `sweep`, `respondent`, `new_varname`), `L103, L196, L201, L207, L212, L217` (uses
  `type`).
- **What is true** (brief F1, verified at Checkpoint A):
  - The file has no header row, so `read_excel` makes row 0 the header and the
    columns the R needs do not exist. The published R cannot have run on this exact
    file.
  - Read with `header=None` and assigned names by position (0 sweep, 1 type,
    2 respondent, 3 question, 4 variable label, 5 new_varname, 6 variable), the rows
    complete on columns 0, 1, 2, 5 and 6 are exactly 63, with no duplicated code.
  - All 74 names of the form `s<sweep>_<co|te|pa|mo>_...` in `_targets.R` and
    `functions.R` are either a `full_name` of the table or created by the R, with one
    exception: `s2_co_total_ability`, which the R only removes.
  - The earlier note said "the public file is not the file the code was written
    against". That is wrong, and `clean_ncds` is not blocked on a different file.
- **Python:** `io/readers.py::read_datalist` reads the file with `header=None`, names
  columns 0–6, drops the empty columns 7 onwards, keeps the rows complete on the five
  columns the R uses, and asserts 63 rows and no duplicate codes. No separate schema
  file (the planned `data/schema/ncds_variable_mapping.yaml`) is needed, and none
  exists.
- **Why:** the evident intent; the R cannot run otherwise.
- **Test:** `tests/test_clean_ncds.py::test_read_datalist_gives_63_named_rows`,
  `::test_all_74_r_names_resolve` (names from `docs/reference/r_variable_names.json`,
  written by `scripts/extract_r_targets.py`); `tests/test_io.py::test_read_datalist_real_file_returns_frame`.
- **Validation:** V1.

### A2. `clean_ncds` refers to four undefined blocks; reconstructed without them (rewritten in Phase 1, Task 1.6)
- **Label:** reconstruction.
- **R source:** `llm_paper/R/functions.R:L224–240`.
  `plyr::join_all(list(teacher, parents, height, birthweight, bsag, behavior,
  ability, aspirations, personality, motivation, parenting, highest_edu, sex, camsis))`
  refers to `bsag`, `aspirations`, `parenting` and `camsis`, which are defined nowhere,
  so R stops here. `L202` (`select(-s2_co_total_ability)`) also errors with this
  table, because that name is not in it (A1).
- **What the evident intent is** (brief F2, verified):
  - The final `select` keeps only the columns of sex, birthweight, height, teacher,
    parents, personality, behavior, ability, motivation and highest_edu, in that order
    (`L228–239`), and every block comes from the same rows.
  - So the output is one row per `ncdsid` with those columns. The undefined blocks
    would not contribute any selected column.
  - Nine teacher columns appear in two blocks: four ability ratings in `teacher` and
    `ability`, and five behaviour items in `teacher` and `behavior`. `plyr::join_all`
    keeps the first occurrence (E3), which is the teacher block's rank-coded version.
    The select order is not the reason.
- **Python:** `cleaning/clean_ncds.py::clean_ncds` (F7) builds that frame without the
  undefined blocks, drops `s2_co_total_ability` only when present, and reports for each
  collided column whether the two versions are identical
  (`attrs["clean_ncds_collisions"]`).
- **Why:** the published R cannot run.
- **Test:** `tests/test_clean_ncds.py::test_one_row_per_ncdsid_and_column_order`,
  `::test_s2_co_total_ability_absent_is_handled`,
  `::test_collision_check_reports_identical_and_different`.
- **Validation:** V1, and the collision report on real data.

### A3. `get_complete_ncds` is called with an argument it does not have (corrected in Phase 1, Task 1.6)
- **Label:** reconstruction. It corrects the earlier note.
- **R source:** `llm_paper/R/functions.R:L309` defines four arguments and no `...`.
  `llm_paper/_targets.R:L169` calls it with five (`…, essay_data, gene_data`).
- **What is true:** R does **not** drop the fifth argument silently, as the earlier
  note said. A call with an unmatched argument and no `...` stops with "unused
  argument (gene_data)". The published `ncds_complete` target therefore cannot have
  run either. Also, `tar_target(gene_data, read_gene_data)` (`_targets.R:L93`) stores
  the function itself, so `colnames(gene_data)` is `NULL` and `gene_variables` is empty
  (`L132`).
- **Python:** `cleaning/assemble.py::get_complete_ncds` takes an explicit optional
  `ncds_gene`. With `None` it performs the three left joins of the four-argument R.
  Gene-dependent targets are skipped when gene data is absent (Task 2.3).
  - The joins are dplyr natural joins, on every shared column, as the R's `left_join`
    without `by` does (Checkpoint B decision). The keys are logged, and a warning is
    issued whenever a join uses a key other than `ncdsid`.
  - After every join, `JoinCardinalityError` is raised unless there is exactly one row
    per `ncdsid`. The R makes no such check.
- **Why:** the evident intent: join the gene data when available.
- **Test:** `tests/test_cleaning.py::test_get_complete_ncds_optional_gene_join`,
  `::test_get_complete_ncds_left_joins`,
  `::test_get_complete_ncds_natural_join_warns_on_non_ncdsid_key`,
  `::test_get_complete_ncds_requires_one_row_per_ncdsid`.
- **Validation:** none.

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

Handling: Task 2.6 ports the intended outputs into `src/llm_cong_predict/reporting/`
(`fig_2..5_data.csv`, `appendix_D1..D12`), with each broken line logged there. The
planned `scripts/make_figures.py` was never written. D4 (essay text) and D7
(per-person BSAG values) are participant-level and become aggregate summaries
(brief F9).

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
- **Measured against R (Task 2.2):** on identical outer and inner folds, R's
  `CV.SuperLearner` (SuperLearner 2.0-42, R 4.6.1) and the native engine with the lm
  library (`SL.mean`, `SL.lm`) agree to at most 1.7e-14. That covers library
  predictions, weights, SL predictions, CV risks and discrete-SL predictions, with an
  exactly collinear column included. `tests/test_oracle.py::test_sl_mean_and_sl_lm_match_r_superlearner_on_identical_folds`.
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
  `::test_sl_lm_with_no_columns_fits_intercept_only`. Against R:
  `tests/test_oracle.py::test_sl_mean_and_sl_lm_match_r_superlearner_on_identical_folds`
  (the data include an exactly collinear column; agreement to 1e-14).
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
- **Measured against R (Task 2.2, glmnet 5.0, synthetic data):**
  - With R's lambda sequence and shared folds: identical selections in 12/12
    datasets.
  - With each side's own default grid and early stop (the screener as the pipeline
    uses it) and shared folds: identical selections in 40/40.
  - The grids agree to about 1e-15 relative.
  - The early stop gave a path one point longer or shorter than glmnet's in 5 of 40
    datasets. In each, the relative R² gain was within about 1e-6 of the 1e-5
    threshold, because the two coordinate-descent solutions' R² differ by up to 4.8e-4.
  - Not reproducible: R's random CV folds (the pipeline's own folds are seeded).
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

### E3. `combine_ncds` resolves column collisions as plyr does (rewritten in Phase 1, Task 1.5)
- **Label:** faithful. Owner decision C1 corrects brief F8 item 4.
- **R source:**
  - `llm_paper/R/functions.R:L57–62`:
    `plyr::join_all(by = "ncdsid", type = "full") %>% as_tibble()`.
  - `plyr/R/join-all.r:L14–23`; `plyr/R/join.r:L122–133` (`.join_all`, full: `matched <-
    cbind(x[ids$x, ], y[ids$y, y.cols])`, then `rbind.fill(matched, unmatched)`).
  - `plyr/R/rbind-fill.r:L70–71, L80` (1.8.9; code unchanged since 1.8.4). The output
    has one column per `unique(names)`, filled with `df[[var]]`, the first occurrence.
- **Python:** `io/readers.py::combine_ncds`, built on `_plyr_full_join`:
  - Rows: every row of the accumulated frame, in order, with its matches, then the
    right frame's unmatched rows.
  - A column in both frames: rows of the left frame keep the left value, even when it
    is NA (this is **not** a coalesce). Right-only rows take the right value. The
    column keeps the left frame's labels.
  - Every collision is logged (warning) and recorded in
    `attrs["combine_ncds_collisions"]` with the count of differing cells.
  - `strict=True` raises `ColumnCollisionError` instead. This option is not in the R.
- **Why:** the brief (F8) said `join_all` keeps both copies and `as_tibble` then
  rejects the duplicated names, so the Python should raise. That is not what plyr
  does. `rbind.fill` removes the duplicates, so `as_tibble` never sees any, and the
  author's run could have had silent collisions. The earlier port coalesced, which
  matched neither R nor the brief.
- **Test:** `tests/test_io.py::test_combine_ncds_collision_keeps_first_frame_like_plyr`,
  `::test_combine_ncds_right_only_rows_take_right_value_in_plyr_order`,
  `::test_combine_ncds_strict_mode_raises_on_collision`,
  `::test_combine_ncds_collided_column_keeps_first_frame_labels`.
- **Validation:** on real data, check `attrs["combine_ncds_collisions"]`. Any entry
  means two NCDS files carry the same variable code.

### E4. `read_camsis` does not lower-case column names ✅ (faithful)
Matches the R (`haven::read_dta` only). The real CAMSIS files already use lower-case
names (`co1970`/`mcamsis`/`fcamsis`) that `create_aspirations` relies on.

### E5. `read_datalist` reads `variables.xlsx` with assigned column names (Task 2.1)
Reconstruction. See A1.

### E6. `read_essays` reads like `readtext` and splits like `tidyr::separate` (Checkpoint B decision)
- **Label:** faithful.
- **R source:**
  - `llm_paper/R/functions.R:L26–31`;
  - `readtext/R/get-functions.R:L2–4` (0.92.1): `paste(readLines(con), collapse = "\n")`,
    so LF, CRLF and CR endings become `\n` and the final line terminator is dropped;
  - `tidyr/R/separate.R:L170–201` and `src/simplifyPieces.cpp` (1.3.2):
    `extra = "warn"` keeps the first two pieces, `fill = "warn"` fills the missing
    right piece with NA.
- **Python:** `io/readers.py::read_essays`, `parse_essay`, `_readtext_txt`,
  `_separate_two`. The first version split at the first separator only and used `""`
  for a missing piece.
  - When the ID separator occurs twice, the text is cut at the second one, and the
    "  Words: " part behind it is discarded too, so the word count is NA, as in R.
  - tidyr's warnings list row numbers. The port warns and logs only the NUMBER of
    files with extra and with missing pieces (owner instruction), and keeps the counts
    in `attrs["read_essays_malformed"]`.
  - `scripts/check_essay_format.py` gives the owner the same counts for the real
    essays, as aggregate numbers only.
- **Why:** owner decision at Checkpoint B.
- **Test:** `tests/test_io.py::test_read_essays_malformed_files_follow_tidyr_separate`,
  `::test_read_essays_reads_text_like_readtext`, `::test_read_essays_parses_format`,
  `::test_check_essay_format_prints_only_counts`.
- **Validation:** V2. Run `scripts/check_essay_format.py` on the real essays.


### E7. `to_character` follows sjlabelled (Task 2.1)
- **Label:** faithful.
- **R source:** `sjlabelled/R/as_character.R:L10–15, L33–38`, `as_label.R:L229–269`
  (1.2.0): `add.non.labelled = FALSE`.
- **Python:** `io/labels.py::to_character`. In a column with value labels, a labelled
  value becomes its label and an unlabelled value becomes NA. A column without labels
  gives `as.character(x)`. The earlier version returned the raw value as text for
  unlabelled values. That never changed which values `clean_ncds` blanks, but it was
  not the R's result.
- **Why:** faithful; used by `clean_ncds` step 2.
- **Test:** `tests/test_clean_ncds.py::test_missing_strings_apply_only_through_labels`.
- **Validation:** V1.
---

## F. Cleaning block (`src/llm_cong_predict/cleaning/`) — deviations & decisions

Scope note: `clean_ncds` was deferred in the first build (option (b)) on the belief
that the public `variables.xlsx` was the wrong file. That belief was wrong (A1): the
file is right and only lacks a header row. `clean_ncds` is ported in Phase 2,
Task 2.1, as a reconstruction (A1, A2). The other five cleaning functions are ported.

### F1. `create_factors` backends: R's `psych::fa` (default) or native (Task 2.2)
- **Label:** faithful with the `"r"` backend. The native backend is APPROX for the
  Pearson factor and cannot compute the polychoric factors.
- **R source:** `llm_paper/R/functions.R:L272–307`. Each factor is
  `psych::fa(1, cor = type) %>% .$score`. `$score` partially matches `scores`, the only
  element starting with "score" (checked in R with psych 2.6.5).
- **Python:** `cleaning/factors.py::create_factors(…, backend=None)`, where `None`
  means `config.FACTOR_BACKEND`, default `"r"`.
  - `"r"` computes all four factors with `psych::fa` through rpy2
    (`create_factors_r`). It is the reference implementation and the only one for the
    three polychoric factors.
  - `"native"` computes the Pearson factor in numpy (F2). The polychoric factors are
    deferred, or raise with `include_polychoric=True`. A polychoric estimator matching
    psych would be guesswork.
  - The default `"r"` is my choice, because it is the only backend that produces all
    four factors as the R does. It needs R, rpy2 and psych.
- **Why:** brief Task 2.2.
- **Test:** `tests/test_oracle.py::test_r_bridge_computes_all_four_factors`,
  `::test_pearson_factor_matches_psych_fa`;
  `tests/test_cleaning.py::test_create_factors_polychoric_raises_not_guesses`
  (native).
- **Validation:** V3.

### F2. `create_factors` implemented in numpy; Pearson-factor scoring follows `factor.scores` (scoring fixed in Phase 1, Task 1.5)
- **Label:** faithful for the scoring; APPROX for the loadings.
- **R source:** `llm_paper/R/functions.R:L297–306`; `psych/R/fa.R:L19–41` (defaults:
  minres, regression scores, `missing = FALSE`, `impute = "none"`), `L811` (no
  `rho` passed); `psych/R/factor.scores.R:L8, L21–30, L135`.
- **Python:** `cleaning/factors.py`. The one-factor minres loadings come from
  iterated eigen-decompositions on the pairwise Pearson correlation. The scores are
  `scale(x) %*% solve(r, loadings)`, falling back to a pseudo-inverse as psych does:
  columns centred on their means and divided by the SD with denominator **n − 1**,
  both over non-missing values, and **no imputation**, so a row with any missing item
  gets NaN. The earlier port mean-imputed missing items and used the n denominator
  (brief F3).
- **Why:** faithful scoring. `factor_analyzer` 0.5.1 is incompatible with the
  installed scikit-learn. APPROX: psych fits the uniquenesses by `optim`, which
  minimises the same criterion, but identical loadings are not guaranteed.
- **Test:** `tests/test_cleaning.py::test_pearson_factor_scores_na_for_incomplete_rows_and_n_minus_1_scaling`,
  `::test_create_factors_computes_pearson_defers_polychoric`.
- **Measured against R (Task 2.2, psych 2.6.5):** on 4 synthetic datasets with missing
  items, the NA pattern is identical, the correlation with `psych::fa(x, 1)$scores` is
  ≥ 0.99999999997, and the largest absolute difference is 2.2e-5, from the loadings
  (AP8). `tests/test_oracle.py::test_pearson_factor_matches_psych_fa`.
- **Validation:** V3.

### F3. `create_aspirations` sex comparison reproduced faithfully, incl. its quirk ✅
The R computes `sex = as.character(as_factor(sex))` and then
`ifelse(sex == 1, camsis_male, camsis_female)` — comparing a *character* ("Male"/
"Female" or "1"/"2" depending on the file's labels) to the numeric literal `1`. If
the sex variable carries text labels, that equality is never true and every
respondent takes the female score. This is reproduced exactly (the male branch fires
only when the character sex equals the string "1"), and flagged here because it is a
latent data-dependent quirk of the original to check against real data.

### F4. `get_complete_ncds` gene argument made explicit ✅ (see A3)
The R definition takes 4 arguments but `_targets.R` passes 5, which is an error in R
(A3, corrected). The port adds an explicit optional `ncds_gene`. With `None` it
performs the joins of the 4-argument R.

### F5. `find_essay_teacher_genetics_overlap`: haven integer codes, unlabelled values kept (rewritten in Phase 1, Task 1.5)
- **Label:** faithful. Owner decision C9 and brief F8 item 3.
- **R source:** `llm_paper/R/functions.R:L330–346`; `haven/R/as_factor.R:L62–84`
  (2.5.5, `levels = "default"`); `r-source/src/library/base/R/ifelse.R:L46–55`.
- **Python:**
  - `ncds_1_to_9` is still labelled, so each of `n876`–`n885` becomes
    `haven::as_factor` with the default levels. Those are every label plus every
    observed unlabelled value, sorted by the underlying value, **including labels
    that never occur**. Unlabelled values are kept as levels, not set to missing.
  - `ifelse(x == "Dont know", NA, x)` returns the factor's **integer codes**, i.e. the
    position in that level set (`io/labels.py::labelled_factor_codes`). The earlier
    port returned label text and dropped unlabelled values.
  - Both `inner_join` calls join on every shared column (`io/joins.py`).
  - This coding differs from clean_ncds' teacher block, where the labels are already
    gone and the code is the rank among observed values
    (`io/labels.py::observed_rank_codes`).
  - The function is defined but never called by `_targets.R` or `create_data.R`.
- **Why:** faithful.
- **Test:** `tests/test_cleaning.py::test_find_essay_teacher_genetics_overlap_keeps_unlabelled_codes_as_integer_codes`,
  `::test_find_essay_teacher_genetics_overlap_inner_joins_and_labels`;
  `tests/test_io.py::test_haven_codes_differ_from_observed_rank_codes_when_a_label_is_unobserved`,
  `::test_as_factor_keeps_unlabelled_values_like_haven_default`.
- **Validation:** none (unused by the pipeline).

### F6. `read_ncds` drops the labels of the codes it sets to missing (Phase 1, Task 1.5)
- **Label:** faithful.
- **R source:** `llm_paper/R/functions.R:L51` (`sjlabelled::set_na(na = -99:-1)`);
  `sjlabelled/R/set_na.R:L258–263` (values become NA), `L268–272` (their labels are
  removed), `L177` (an all-NA column is returned untouched).
- **Python:** `io/labels.py::set_na_range` drops those labels as well.
  `read_ncds` attaches the labels **before** recoding. The earlier port attached them
  afterwards, so the missing-code labels survived and would have become
  `haven::as_factor` levels.
- **Why:** faithful level sets for every later `as_factor`.
- **Test:** `tests/test_io.py::test_read_ncds_drops_labels_of_recoded_missing_codes`.
- **Validation:** none.

### F7. `clean_ncds` (Phase 2, Task 2.1)
- **Label:** faithful per step. The whole function is a reconstruction (A1, A2).
- **R source:** `llm_paper/R/functions.R:L64–242`. Each step is cited in
  `cleaning/clean_ncds.py`.
- **Python:** `cleaning/clean_ncds.py`:
  - `full_name` with R's printing of the sweep (L75–78).
  - The verbatim missing-label list applied through labels only, after which every
    value label is dropped (L80–90).
  - `one_of` selection in table order; absent codes are logged and kept in
    `attrs["clean_ncds_absent_codes"]` (L91–95).
  - Year/month recode (L97–98).
  - nssec closed ranges (L104–115).
  - Parent education as ordered categoricals (L116–129).
  - `s3_pa_edu` from pmin/pmax, as a categorical whose categories are the sorted
    observed values, so `as.numeric` gives the level position (L130–145).
  - Sex as nullable boolean (L150–152).
  - Height and birthweight (L156–162).
  - Teacher block as rank codes among observed values (L166–191).
  - The type blocks (L195–217).
  - Assembly with plyr first-occurrence semantics, in the select order (L224–240).
  - A column the R selects by NAME (sex, height, birthweight, the 21 teacher columns,
    the parents' age-left columns) raises `MissingColumnError` when absent, as dplyr
    stops there.
  - `ncdsid` must be unique.
- **Why:** brief F2.
- **Test:** `tests/test_clean_ncds.py` (14 tests: the ten the brief lists, plus the
  table, the parent-education bounds, sex, absent codes).
- **Validation:** V1 (absent codes, collision report on real data).

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

### G2. RoBERTa: mean over all positions, padding included; batched (batching added in Phase 1, Task 1.5)
- **Label:** faithful. Batching does not change the numbers.
- **R source:** `llm_paper/R/functions.R:L446–490`, in particular `L480–482`: a keras
  input of token ids only, no attention mask, and
  `tf$reduce_mean(roberta_model(input)[[1]], axis = 1L)`.
- **Python:** `features/embeddings.py::roberta_pool(input_ids, model)` returns the mean
  of the last hidden state over all positions, padding included. `roberta_embeddings`
  tokenises as the R does (`max_length = 250`, truncation, padding to max length) and
  runs the model `batch_size` essays at a time. Model and tokenizer can be passed in;
  by default the public `roberta-base` weights are loaded locally. PyTorch replaces
  TensorFlow.
- **Why:** a single forward pass over about 10,000 essays of 250 tokens runs out of
  memory (brief F8). Rows are independent, so batching gives the same numbers.
- **Test:** `tests/test_features.py::test_roberta_pool_batched_equals_unbatched`
  (bit-identical on a tiny randomly initialised `RobertaConfig` model),
  `::test_roberta_embeddings_batch_size_does_not_change_output`. The real
  `roberta-base` weights were not loaded here.
- **Validation:** V2 (essay feature width) on real essays.
- **Process isolation (owner decision, Checkpoint B).** torch bundles its own OpenMP
  runtime, and once torch has been imported an xgboost fit in the same process
  segfaults (observed with torch 2.14.0 and xgboost 3.3.0 on macOS). So RoBERTa
  embeddings are generated by a separate step in its own process,
  `python -m llm_cong_predict.features.roberta_step`, which writes
  `$LCP_DATA_ROOT/derived/roberta_embeddings.csv`. The pipeline reads that file
  (`read_roberta_embeddings`). `llm_cong_predict.isolation` refuses either library
  when the other is loaded, and no environment workaround is used. Values are saved as
  float64 holding the model's float32 outputs exactly, as R holds keras' float32
  predictions in doubles. Tests:
  `tests/test_features.py::test_xgboost_learner_refuses_after_torch_is_imported`,
  `::test_roberta_generation_refuses_after_xgboost_is_imported`,
  `::test_roberta_step_writes_derived_file_that_reads_back_exactly`,
  `::test_roberta_step_refuses_without_data_root`. See docs/ORCHESTRATION.md.

### G3. SALAT and spelling: ingestion only, with R's join and pivot semantics (rewritten in Phase 1, Task 1.5)
- **Label:** faithful. Owner decision C8 and brief F8 items 1–2.
- **R source:** `llm_paper/R/functions.R:L390–417` (spelling), `L419–444` (SALAT);
  `dplyr/R/join.R:L624`, `join-by.R:L376–379` (joins without `by`).
- **Python** (`features/salat.py`):
  - **Spelling:**
    - An essay without any error row is kept. The left join gives it rule type NA,
      `pivot_wider` makes an "NA" column, and its nine categories are filled with 0.
      The earlier port dropped such essays, and `create_essay_variables` then removed
      every spelling column for having NAs.
    - Rows follow essay order. Columns follow first appearance, with each essay's
      rule types sorted, as `count()` returns them.
    - The two cases where the R stops raise `SpellingMetricsError` with a message
      naming the R line: a category of the nine never occurs (`L410–414`), or no
      essay is error-free (`L415`).
    - `ncdsid` is compared as text (pandas refuses mixed key types).
    - The word count is converted as R's `as.numeric` converts it
      (`io/labels.py::r_as_numeric`: whitespace trimmed, hexadecimal accepted; checked
      against R 4.6.1; AP10).
  - **SALAT:**
    - Each `left_join` joins on every column the two sides share (`io/joins.py`), as
      dplyr does without `by`.
    - The keys used at each step are logged and kept in `attrs["salat_join_keys"]`.
      A metric reported by two tools becomes a join key, and essays whose values
      differ lose the later tool's columns.
  - Neither function generates metrics: the external tools do (SALAT desktop apps,
    LanguageTool).
- **Why:** faithful.
- **Test:** `tests/test_features.py::test_spelling_keeps_essays_without_errors_with_zeros`,
  `::test_spelling_columns_follow_first_appearance_with_sorted_types`,
  `::test_spelling_raises_like_r_when_no_essay_is_error_free`,
  `::test_spelling_raises_like_r_when_a_category_never_occurs`,
  `::test_salat_natural_join_uses_shared_metric_as_key`,
  `::test_get_spelling_error_metrics_pivots_fills_and_sums` (fixture extended so the R
  would run on it).
- **Validation:** V2. On real data, check `attrs["salat_join_keys"]`: any key other
  than `filename` means a shared metric column.

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

---

## J. Corrections to the facts of the Phase 1–2 brief (Section 4)

Found at Checkpoint A (`docs/PHASE_1_2_PLAN.md` A.4) and confirmed by the owner. The
code follows the corrected version.

### J1. F2: why the teacher version of the collided columns is kept
The conclusion is right, the reason is not. `plyr::join_all` keeps the first
occurrence of a duplicated column (`plyr/R/rbind-fill.r:L70–71, L80`). The teacher
block comes first in the join list (`functions.R:L224`), so its rank-coded columns
survive. The final `select` never sees duplicates. Also:
- `one_of` warns about unknown names rather than dropping them silently
  (`tidyselect/R/helpers.R:L124–127`), and keeps the table's order (`match_vars`).
- The `add.non.labelled = FALSE` default of `to_character` is set in
  `sjlabelled/R/as_character.R:L14`; `as_label.R:L269` is where unlabelled values
  become NA.
- `select(-s2_co_total_ability)` errors on every run with this table (A2).

### J2. F4: CV risk of learners that fail only in the refit
SuperLearner sets `cvRisk` to NA only for learners that failed in cross-validation
(`SuperLearner/R/SuperLearner.R:L303–305`). A learner that fails only in the
full-data refit keeps a number. The port copies this (owner decision C10; C1).

### J3. F5: learner mappings
- **ranger:** the sklearn value is `min_samples_split = 6`, not 5. ranger does not split
  a node with `n <= min.node.size` (`ranger/src/TreeRegression.cpp:L106`) (owner
  decision C2; C5).
- **ksvm:** `cache`, `tol` and `shrinking` come from kernlab's defaults
  (`kernlab/R/ksvm.R:L61–63`), not from the wrapper, which does not pass them. The
  values are the same (C7).
- **screen.glmnet:**
  - the grid has 100 values in total, the largest lambda included;
  - the CV error is the pooled (fold-size-weighted) MSE;
  - ties go to the largest lambda;
  - the full-data path stops early by glmnet's rule;
  - if no lambda reaches two non-zero coefficients, `which.max` picks the largest
    lambda, which usually selects no column (C9).
- **xgboost:** confirmed. `base_score` is 0.5 in R xgboost 1.7.x for squared error
  (v1.7.6 source), although that version's parameter documentation suggests otherwise
  (C8).

### J4. F8 item 4: `combine_ncds` collisions
`plyr::join_all(type = "full")` does not return duplicated names, so `as_tibble` does
not reject them and the Python must not raise. The later frame's values only fill rows
the earlier frames did not have (owner decision C1; E3).

### J5. F10: the aspiration mapping is many-to-one
13 `occupation_1970` values are shared by several aspirations, so the mapping is
many-to-one, not one-to-one. Each aspiration matches exactly one CAMSIS row, so the
join in `create_aspirations` adds no rows (computed with pandas: 58 joined rows for
58 aspirations). No code change.

### J6. PORTING_NOTES A3 (earlier version of this file)
An unmatched fifth argument is an error in R, not silently dropped (A3, rewritten).

---

## K. R bridge (Phase 2, Task 2.2)

### K1. `models/r_superlearner.py` fixed, and first run
- **Label:** faithful (it calls R itself).
- **R source:** `SuperLearner/R/CV.SuperLearner.R:L3–39` (`cvControl`,
  `innerCvControl`), `control.R:L15–29`, `CVFolds.R:L13–15` (`validRows`).
- **Python:** the module had never been run. Problems found and fixed:
  - It relied on the global `numpy2ri.activate()`; it now uses explicit
    `localconverter` blocks (`llm_cong_predict.rbridge.converter`).
  - rpy2 3.6 returns `None` for invisible R values; the value is now made visible.
  - It could not fix the inner folds, so R drew its own and parity could not be
    exact. It now passes per-outer-fold `validRows` through `innerCvControl`.
  - It did not check R's library names; it now does.
  - It returned no per-fold CV risks; it now returns them from `AllSL`.
  - Results are fetched piece by piece, because the numpy/pandas converter turns an
    R list into a `NamedList`.
- **R's xgboost is not installed** (owner instruction; Phase 3 pins it). With the full
  library, R's `SL.xgboost.hist` therefore fails inside `try()` and gets weight 0 on
  the R side. The oracle tests use only the lm library.
- **Why:** brief Task 2.2.
- **Test:** `tests/test_oracle.py::test_sl_mean_and_sl_lm_match_r_superlearner_on_identical_folds`;
  `scripts/validate_oracle.py` (lm library, identical outer and inner folds: every
  difference ≤ 1.7e-14, run at Task 2.2).
- **Validation:** V4 (the full library needs a pinned R xgboost < 3.0, Phase 3).

### K2. torch, xgboost and R in one process
R embedded through rpy2 and xgboost were run in one process, in both orders (glmnet,
ranger and psych in R, then xgboost, and the reverse). Neither crashed. The CRAN
builds of glmnet, ranger, kernlab and nnet link no OpenMP runtime, and psych is pure
R. torch and xgboost remain separated (docs/ORCHESTRATION.md).
