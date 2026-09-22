# `R/create_data.R`: every output file, its inputs and its state

Read from `reference/llm_paper/R/create_data.R` (601 lines, commit `b0cfe4c`) in full for
brief Task 2.6. For each file the R writes: what it is built from, what it does to it,
whether it is aggregate or participant-level, and whether the R can produce it.

The script runs top to bottom and **reuses variable names**, so what a line sees depends
on what ran before it: `essay_metrics`, `genes_metrics`, `teacher_metrics` and
`teacher_genes_essay_metrics` are defined three times (L74–94 mmg, L319–330 BFI,
L422–442 mmg again), `predictions` twice (L61, L404), and
`teacher_genes_essay_full_metrics` twice (L55 full sample, L113 **mmg sample**). The
column below says which definition each output actually gets.

`type` labels are copied verbatim from the R. Metric rows are the output of
`get_cv_superlearner_metrics` / `get_cv_lm_metrics` (mean_mse, se, min/max_mse,
mean/min/max r2, mad, rmse, var, n), so a "metric rows" input is aggregate by
construction.

| Output file | R variable (lines) | Inputs (R target names) | Transformations | Kind | Runs in R? |
|---|---|---|---|---|---|
| `fig_2_data.csv` | `plot_1_data` (L201–210) | `essay_superlearner`, `gene_superlearner`, `teacher_superlearner` metrics | add `type`; join the mapping; wrap `name` (15) and `type` (50, then 20); drop `category == "Life Outcomes"` and the general cognitive factor | aggregate | yes |
| `fig_3_data.csv` | `plot_2_data` (L213–235) | the seven `*_superlearner_mmg_metrics` and the seven `*_superlearner_cog_mmg_metrics` | add `type` per feature set; join the mapping; wrap `name` (20), `type` (20) | aggregate | yes |
| `fig_4_data.csv` | `plot_3_data` (L124–136) | `cog_superlearner_social_lm`, `noncog_superlearner_social_lm_metrics`, `birthweight_superlearner_social_metrics`, `height_superlearner_social_metrics`, `sociological_superlearner_social_lm_metrics`, and `teacher_genes_essay_full_metrics` — which at L113 is the **mmg-sample** target, not the full-sample one | join the mapping; wrap; keep `category == "Life Outcomes"`; `naming = "Educational Attainment"` | aggregate | **no** — L98 calls `get_cv_superlearner_metrics(cog_superlearner_social_lm)` on a target name that was never read with `tar_read`, so the object does not exist. `pedu_full_metrics` is computed (L110) and then not used here. |
| `fig_5_data.csv` | `plot_4_data` (L238–239) | `all_text` (L184–194): `salat_metrics_superlearner`, `readability_metrics_superlearner`, `spelling_errors_superlearner`, `spelling_salat_readability_superlearner`, `gpt_embeddings_superlearner`, `essay_superlearner`; plus `text_length` for the baseline | join the mapping; join `lm_performance` = `text_length`'s `mean_r2`; `relative_performance = mean_r2 / lm_performance`; drop Highest Education and the general cognitive factor | aggregate | yes |
| `appendix_D1_data.csv` | `appendix_12_data` (L497–501) | `gene_superlearner` metrics | join the mapping; wrap | aggregate | yes |
| `appendix_D2_data.csv` | `appendix_11_data` (L173–179) | `roberta_embeddings_superlearner`, `gpt_embeddings_superlearner`, `gpt4_embeddings_superlearner` metrics | join the mapping — and nothing else | aggregate | **partly** — the pipe ends at L176 and L177 starts a new statement, so the `name` wrapping and the `type` factor never reach `appendix_11_data`; the orphan statement itself errors (`.` with no data). |
| `appendix_D3_data.csv` | `appendix_10_data` (L572–580) | the **fits** of `essay_superlearner`, `gene_superlearner`, `teacher_superlearner`, `teacher_genes_essay_superlearner`: `x$fit$coef` | per outcome and learner: mean and sd of the per-fold weights; join the mapping and the learner-name table (L504–517) | aggregate | yes |
| `appendix_D4_data.csv` | `appendix_1_data` (L241–242) | `ncds_essays` | `words` as numeric — nothing else | **participant-level: the full essay text with `ncdsid`** (brief F9) | yes, and it must never leave the secure area |
| `appendix_D5_data.csv` | `appendix_2_data` (L244–255) | `ncds_1_to_9`: `n876`–`n879` | `haven::as_factor`; long; count per item and response; divide by the item's total; drop NA | aggregate (proportions) | yes |
| `appendix_D6_data.csv` | `appendix_3_data` (L257–270) | `ncds_1_to_9`: `n880`–**`n885`** | as D5 | aggregate (proportions) | **no** — `n885` is not in `variables.xlsx`, so `read_ncds` never loads it and `select` stops. The same code `n885` also appears in `find_essay_teacher_genetics_overlap`, which `_targets.R` never calls. |
| `appendix_D7_data.csv` | `appendix_4_data` (L273–288) | `ncds_1_to_9`: the twelve BSAG totals | as numeric; long | **participant-level: every person's BSAG values with `ncdsid`** (brief F9) | yes, and it must never leave the secure area |
| `appendix_D8_data.csv` | `appendix_5_data` (L290–297) | `ncds_1_to_9`: `n2771` | `haven::as_factor`; drop NA; `filter(!aspired_job %in% c(47, 20.5))`; count per job; sort | aggregate, but **counts per job, so small cells are possible** | yes. The filter compares a factor to numbers, so it drops nothing unless a label is literally "47" or "20.5". |
| `appendix_D9_data.csv` | `appendix_6_data` (L300–316) | `essay/gene/teacher_superlearner` metrics as "SuperLearner", plus `essay_full_metrics_lm`, `genes_full_metrics_lm`, `teacher_full_metrics_lm` | wide by method; `diff = SuperLearner - pmax(Linear Model, 0)`; join the mapping; drop the general cognitive factor and Life Outcomes | aggregate | **no** — the three `*_lm` objects are never defined anywhere in the repository, and `bind_rows(...)` has a trailing comma (L306). |
| `appendix_D10_data.csv` | `appendix_7_data` (L333–342) | `essay/gene/teacher_superlearner_bfi_metrics` | join the mapping; wrap; `sample = "Complete Information on all Variables"`; overwrite `name` with "Big 5: <Trait>\n (Age 50)" | aggregate | yes |
| `appendix_D11_data.csv` | `appendix_8_data` (L419–420) | `predictions` (L404: the overlap versions of cog, noncog, pedu, birthweight, height and `teacher_genes_essay_superlearner_overlap`) and `predictions_full` (L389: the same six on the maximum sample) | join the mapping; wrap; `sample` label; keep `category == "Life Outcomes"` | aggregate | **no** — L359 defines `teacher_genes_essay_overlap_metrics` from itself before it exists (it is only created at L462), and L367 repeats the `cog_superlearner_social_lm` mistake of L98. |
| `appendix_D12_data.csv` | `appendix_9_data` (L490–495) | the seven `*_superlearner_mmg_metrics` and the seven `*_superlearner_overlap_metrics` | add `type` and `sample`; join the mapping; wrap; keep `category == "Life Outcomes"` | aggregate | yes |

## What the port does with each of these

- **Aggregate tables that run in R** are ported as written, from the metric rows the
  runner produces (`pipeline/execute.py`) and the mapping table at the top of
  `create_data.R`.
- **D4 and D7 are participant-level** (brief F9). The port does not build them; it
  builds aggregate summaries instead (`summary_d4_essays`, `summary_d7_bsag`), and never
  writes the participant-level version anywhere outside `$LCP_DATA_ROOT`
  (PORTING_NOTES N3).
- **D8** is aggregate but holds counts per aspired job; small cells are possible, which
  is the export guard's business (Task 2.7).
- **The four broken outputs** (fig_4, D2, D9, D11) are reconstructed to the evident
  intent, each with the deviation recorded (PORTING_NOTES N2). **D6 is not
  reconstructed**: it needs a code (`n885`) that the variable table does not contain, so
  the port raises with that reason, as the R stops.
- **Label wrapping is not reproduced** (`stringr::str_wrap`): PORTING_NOTES N4.
