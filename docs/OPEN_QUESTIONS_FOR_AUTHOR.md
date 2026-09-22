# Questions for the author of `llm_paper`

We are building a Python port of the analysis pipeline of Wolfram (2025), working from
the published R at `github.com/tobiaswolfram/llm_paper`, commit `b0cfe4c`. The port
follows the R as it actually ran, quirks included, and records every place where it
could not.

Five places in `R/create_data.R` cannot run as written, so we had to guess what produced
the released figures and tables. Each question below gives the line numbers and what we
assumed. Line numbers are those of `R/create_data.R` at commit `b0cfe4c`.

---

### 1. `fig_4_data.csv` and `appendix_D11_data.csv`: `cog_superlearner_social_lm`

L98 (and again L367) reads:

```r
cog_full_metrics <- get_cv_superlearner_metrics(cog_superlearner_social_lm)
```

Every other block uses `tar_read(...)`; here the bare target name is passed, which in a
fresh session is an undefined object, so the script stops at this line.

**We assumed** you meant `get_cv_superlearner_metrics(tar_read(cog_superlearner_social_lm)[[1]])`,
i.e. the metric rows of that target, as at L344 for the `_overlap` version.

**Question:** is that what produced the "Cognitive Abilities" row of figure 4 and of
appendix D11?

### 2. `fig_4_data.csv`: which teacher + genes + essay model is in it?

L55 defines `teacher_genes_essay_full_metrics` from
`teacher_genes_essay_superlearner_metrics` (the full sample). L113 then **overwrites** it
with `teacher_genes_essay_metrics`, which at that point (L92) holds
`teacher_genes_essay_superlearner_mmg_metrics` — the mmg sample. `plot_3_data` (L124)
uses the overwritten value.

**We assumed** figure 4 shows the **mmg-sample** model, since that is what the script
would produce.

**Question:** is the teacher + genes + essay bar of figure 4 the mmg-sample model, or was
the full-sample model intended?

### 3. `appendix_D11_data.csv`: `teacher_genes_essay_overlap_metrics`

L359 reads:

```r
teacher_genes_essay_overlap_metrics <- teacher_genes_essay_overlap_metrics %>% ...
```

The variable is defined from itself, before it exists; it is only created at L462.

**We assumed** you meant the L462 definition, from
`teacher_genes_essay_superlearner_overlap_metrics`.

There is a second effect in the same table. The "Maximum Observations" half (L389) takes
`teacher_genes_essay_full_metrics`, which by L382 holds the **Big Five** metrics
(assigned at L328). Those outcomes are not in the `mapping` table, so the
`filter(category == "Life Outcomes")` at L401 removes them, and that half of the table
has five rows rather than six.

**Question:** should appendix D11's "Maximum Observations" half contain a
teacher + genes + essay row? Ours does not, because the script as written cannot produce
one.

### 4. `appendix_D9_data.csv`: the three linear-model objects

L300–307 binds `essay_full_metrics_lm`, `genes_full_metrics_lm` and
`teacher_full_metrics_lm`. None of them is defined anywhere in the repository, and the
`bind_rows(...)` call also ends with a trailing comma.

**We assumed** they are the `get_lm_cv_model` targets `essay_lm`, `gene_lm` and
`teacher_lm`, carrying the same `type` labels as their SuperLearner counterparts, so
that the `pivot_wider(names_from = "method")` at L310 lines the two methods up.

**Question:** is that what the "Linear Model" column of appendix D9 is?

### 5. `appendix_D2_data.csv`: the embedding labels

The pipe ends at L176, and L177 starts a new statement:

```r
appendix_11_data <- roberta_embeddings_superlearner %>% ... %>%
  dplyr::left_join(dplyr::select(mapping, var = variable, name, category = category_name))
dplyr::mutate(name = stringr::str_wrap(.$name, width = 20)) %>%
  dplyr::mutate(type = factor(type, levels = c(...), labels = c("RoBERTa", "GPT 3.5", "GPT 4"), ordered = TRUE))
```

So the table keeps the long labels ("RoBERTa-based Embeddings", ...) and never gets the
short ones, and the orphan statement errors.

**We kept the long labels**, since that is what the script produces.

**Question:** did the released `appendix_D2_data.csv` have the long labels or the short
ones?

---

## Also useful, if you have a moment

These are not blocking; they are places where the data, rather than the code, is
unclear to us.

1. **`n885`.** `appendix_3_data` (L259–264, written as `appendix_D6_data.csv`) selects
   `n880`–`n885`, and `find_essay_teacher_genetics_overlap` (`R/functions.R:L333`) names
   `n876`–`n885`. But `n885` is not in `data/variables.xlsx`, so `read_ncds` never loads
   it and both stop. Did you use a variable table that contained `n885`
   ("Imperfect Grasp of English")?
2. **`nwords`.** `text_length` (`_targets.R:L261`) predicts from `nwords`, but nothing in
   the repository creates that column. Which file does it come from — one of the SALAT
   outputs, or the readability output?
3. **Two variable lists drop a column.** `readability_metrics_variables`
   (`_targets.R:L135`) uses `[-c(1:2)]` on a selection that no longer contains
   `filename` (`create_essay_variables` drops it at `R/functions.R:L323`), so it removes
   `ncdsid` **and the first readability index**. `gpt_embeddings_variables` (L137) uses
   `[-1]` on a selection from which `one_of` has already removed the `id` column, so it
   removes **the first embedding dimension**. Were both intended, or should those lists
   have kept every metric?
4. **The derived essay features.** If the SALAT, readability, spelling and embedding
   files you used can be shared with UK Data Service-approved users, we would like to
   validate against them rather than regenerate them.
