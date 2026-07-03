# Porting Notes

Every deviation from the original R code (`tobiaswolfram/llm_paper`), and every
non-obvious fidelity decision, is logged here. 
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

### B4. MSE summary computed directly, not from an R summary object ✅
The original read mean/min/max MSE from `summary(cv_fit)$Table` ("Super Learner"
row). That summary risk equals the fold-wise SL MSE, which we compute directly
(`cv_mse`) so the metric layer needs no R summary object. Numerically identical.

---

## C. Base-learner defaults 🔦 (the main numerical-fidelity risk)

The paper relied on the **default** hyperparameters of each SuperLearner wrapper
(`SL.ranger`, `SL.nnet`, `SL.ksvm`, `SL.xgboost`, `SL.lm`) plus the `screen.glmnet`
LASSO screener. These defaults differ from scikit-learn's — e.g. `SL.nnet` uses
`size=2` hidden units whereas `MLPRegressor` defaults to 100. Getting the native
backend to match therefore requires extracting each wrapper's actual defaults from
the SuperLearner package source, **not** using sklearn defaults.

We deliberately do **not** guess these values in `config.py`. They are pinned during
the model-layer phase and cross-checked against the rpy2 R-SuperLearner oracle on
real data (VALIDATION_CHECKLIST V4). Being confidently wrong here would be worse
than deferring.

**Status update — model layer built (`src/llm_cong_predict/models/`).** The native
nested-CV Super Learner is implemented and its *structure* is tested
(`tests/test_models.py`): folds partition correctly, the NNLS meta-learner behaves
as known, there is no train/test leakage, library-column names match the original
(`SL.mean_All`, `SL.ranger_screen.glmnet`, ..., and `SL.lm_All` for the lm library),
and end-to-end output carries a real signal into the verified metric layer.

What is pinned vs flagged in `base_learners.py`:
  * `SL.mean`, `SL.lm` -> `[match]` (DummyRegressor mean / OLS). High confidence.
  * `SL.ranger` -> `[approx]`. `n_estimators=500` (ranger num.trees). `mtry` and
    `min.node.size` left at sklearn defaults and marked `TO_VERIFY`.
  * `SL.nnet` -> `[approx]`. `size=2` set; but nnet vs `MLPRegressor` differ in
    implementation, so agreement is not expected. `maxit` marked `TO_VERIFY`.
  * `SL.ksvm` -> `[approx]`. `C=1`; RBF `gamma` uses sklearn `'scale'` vs kernlab's
    `sigest` median heuristic (`TO_VERIFY`).
  * `SL.xgboost.hist` -> `[approx]`. `ntrees=1000`, `max_depth=4`, `shrinkage=0.1`,
    `min_child_weight~=10`, `tree_method="hist"`. The hist override is certain; the
    numeric defaults are my current reading and marked `TO_VERIFY`.

The `screen.glmnet` LASSO screener is approximated with `LassoCV`
(`screeners.py`); `glmnet` and sklearn differ in path/standardisation, so the
selected variable set can differ. This is part of what V4 measures.

**Bottom line:** the native backend is structurally correct and testable, but its
*numerical* agreement with R/the paper is a hypothesis until V4. The rpy2 oracle
(`r_superlearner.py`, UNTESTED in this sandbox — no R) plus
`scripts/validate_oracle.py` exist to measure the gap on identical folds and report
it as numbers. `validate_oracle.py` treats the `[match]` learners as a tolerance
gate and reports the `[approx]` learners' divergence rather than assuming it away.

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
