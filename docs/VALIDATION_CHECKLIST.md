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
| V3 | Factor-score parity for `create_factors` (`psych::fa` vs `factor_analyzer`) | factor outcomes | Match rotation + scoring method + correlation type (Pearson vs polychoric per `type=`); compare scores on identical input to a tolerance. |
| V4 | SuperLearner numeric parity: native sklearn backend vs rpy2 R oracle | all model results | Feed **identical folds** to both backends; compare fold-wise R^2/RMSE/MAD within tolerance τ. Report the gap as a number. If τ not met, rpy2 becomes the documented default. |
| V5 | CAMSIS aspiration join (`create_aspirations`) | aspiration outcome | Verify the sex-specific CAMSIS merge and the hand-crafted occupation mapping reproduce the expected `s2_co_aspiration_camsis`. |
| V6 | Final figure/appendix CSVs vs the paper | headline claims | Regenerate `fig_2..5_data.csv` and `appendix_D1..D12`; compare to the paper's reported figures/tables. |
| V7 | Base-learner defaults extracted from SuperLearner source | V4 | Confirm ranger/nnet/ksvm/xgboost/lm wrapper defaults are matched (see PORTING_NOTES C). |

Tolerance note for V4: bit-exactness is impossible (R's `clusterSetRNGStream` RNG
stream is not reproducible in Python). The oracle test therefore fixes the folds
externally and compares the *ensemble math*, not the RNG. Any residual divergence is
reported, not smoothed over.
