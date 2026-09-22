# Validation Checklist

What must be re-verified **once real data (and/or the author's files) arrive**.
Until every relevant item here is checked, "the pipeline runs and tests pass" means
only that the *logic* is sound on synthetic data — NOT that results are reproduced.
This separation is deliberate: it prevents green tests from being mistaken for
validation.

| ID | What to check | Blocks | How to check |
|----|---------------|--------|--------------|
| V1 | Real `variables.xlsx` mapping vs our reconstructed `ncds_variable_mapping.yaml` | A1, A2 | Diff column-by-column; every row currently flagged `provisional-guess` must be confirmed or corrected. Resolves the `bsag/aspirations/parenting/camsis` question (A2). |
| V2 | Which essay features survive the variance filter in `create_essay_variables` | essay feature width | On real essays, list surviving columns; the feature-matrix width is data-dependent and cannot be locked on synthetic data. |
| V3 | Factor-score parity for `create_factors` (`psych::fa` vs the native numpy path) | factor outcomes | Pearson factor: compare native scores with `psych::fa(x, 1)$scores` on identical input (correlation of the scores, identical NA pattern; Task 2.2 oracle test). Polychoric factors: computed by the R bridge (Task 2.2); the native path raises. |
| V4 | SuperLearner numeric parity: native backend vs the rpy2 R oracle | all model results | Feed **identical outer and inner folds** (`fit_cv_superlearner(folds=, inner_folds=)` and `validRows` in R) to both backends; compare per-learner predictions, weights and fold-wise R²/RMSE/MAD, and report every gap as a number. `SL.mean` and `SL.lm` must agree to rounding; each APPROX item below is measured, not assumed. Needs R + the packages listed in REFERENCE_SOURCES.md, with pinned versions (Phase 3). |
| V5 | CAMSIS aspiration join (`create_aspirations`) | aspiration outcome | Verify the sex-specific CAMSIS merge and the hand-crafted occupation mapping reproduce the expected `s2_co_aspiration_camsis`. |
| V6 | Final figure/appendix CSVs vs the paper | headline claims | Regenerate `fig_2..5_data.csv` and `appendix_D1..D12`; compare to the paper's reported figures/tables. |
| V7 | Base-learner settings match the SuperLearner wrappers | V4 | Done from source in Phase 1 (PORTING_NOTES C3–C9: every setting cites its wrapper or package line). Still to confirm: the package versions the paper used (unknown), in particular R xgboost < 3.0 so that `params = list(tree_method = "hist")` is passed (PORTING_NOTES C8). |

Tolerance note for V4: bit-exactness is impossible (R's `clusterSetRNGStream` RNG
stream is not reproducible in Python). The oracle test therefore fixes the folds
externally and compares the *ensemble math*, not the RNG. Any residual divergence is
reported, not smoothed over.

## APPROX register

Every line marked `# APPROX` in the code, with the check that will measure it. IDs
are cross-referenced from `docs/PORTING_NOTES.md`.

| ID | Where | Approximation | Measured by |
|----|-------|---------------|-------------|
| AP1 | `models/folds.py`, `models/seeds.py` | Folds and seeds are owned by the port; R's RNG streams cannot be reproduced (PORTING_NOTES C2). | V4 on shared folds |
| AP2 | `base_learners.RLinearModel` | Aliased-column rule reproduces `lm.wfit`'s choice up to rounding at the 1e-7 threshold (C4). | V4 (`SL.lm` parity) |
| AP3 | `base_learners._make_ranger` | sklearn counts distinct bootstrap rows per node, ranger counts draws; different tree implementation and RNG (C5). | V4 |
| AP4 | `base_learners.NnetLike` | MLPRegressor initialisation and L-BFGS stopping differ from nnet's U(-0.7, 0.7) start and BFGS (C6). | V4 |
| AP5 | `base_learners.KsvmLike`, `sigest_sigma` | libsvm vs kernlab solver; sigest's random row pairs differ (C7). | V4 |
| AP6 | `base_learners._make_xgboost_hist` | R xgboost 1.7.x assumed (base_score 0.5); histogram details differ across versions (C8). | V4, V7 |
| AP7 | `models/screeners.py` | Coordinate-descent convergence differs from glmnet's; CV folds are random in R (C9). | Task 2.2 `screen.glmnet` parity test with shared `foldid` and lambda grid; V4 |
| AP8 | `cleaning/factors.py::_minres_one_factor` | One-factor minres loadings by iterated eigen-decomposition; psych fits uniquenesses by `optim` (PORTING_NOTES F2). | V3 |
| AP9 | `io/labels.py::r_number_string` | R's `as.character()` of doubles reproduced for integers and short decimals; long decimals not re-derived. Affects level names of unlabelled values only. | Task 2.1 tests; V1 on real labels |
