#!/usr/bin/env Rscript
#
# Readability indices through TreeTagger + koRpus — the external-tool boundary.
#
# ============================== NOT RUN HERE ==============================
# This script has NEVER been run in this repository and cannot be: it needs R with
# koRpus, koRpus.lang.en, readtext and janitor, plus a TreeTagger installation, none
# of which are installed here, and it needs the real essays, which are restricted
# data. Its output has therefore never been checked against anything. Run it
# yourself, in the secure environment, and treat its first run as untested code.
# The Python pipeline never calls it: it only reads the CSV this script writes
# (features/readability.py::ingest_readability_metrics), exactly as it reads the
# SALAT and LanguageTool outputs. See docs/PORTING_NOTES.md (G4) and
# docs/ORCHESTRATION.md.
# ==========================================================================
#
# Port of three functions of the original R, unchanged except where marked:
#   * read_essays                  (llm_paper/R/functions.R:L26-31)
#   * tokenize_essays              (llm_paper/R/functions.R:L356-367)
#   * calculate_readability_metrics(llm_paper/R/functions.R:L369-388)
#
# Deviations from the original, both required by the brief (Task 2.5):
#   1. the TreeTagger path comes from the environment variable LCP_TREETAGGER_PATH,
#      not the hard-coded "C:/TreeTagger" of functions.R:L362;
#   2. the essays are read from $LCP_DATA_ROOT/essays and the result is written to
#      $LCP_DATA_ROOT/readability_metrics.csv, so that restricted data never enters
#      the repository. The original read "data/essays" inside the project.
#
# Data safety: this script reads participant data and writes it back under
# $LCP_DATA_ROOT only. It prints counts and nothing else — never essay text, file
# names or IDs.
#
# Usage:
#   export LCP_DATA_ROOT=/secure/path/to/data
#   export LCP_TREETAGGER_PATH=/opt/TreeTagger
#   Rscript r/readability.R
#
# Output: one row per essay; the columns are `filename`, `ncdsid` and one column per
# koRpus readability index, named by the index. The values are the index's `raw`
# value, or its `grade` value where `raw` is empty (functions.R:L376). They are
# written as text, as in the original, and the Python side turns them into numbers
# (create_essay_variables, functions.R:L324).

suppressPackageStartupMessages({
  library(magrittr)
  library(dplyr)
  library(purrr)
  library(tidyr)
  library(tibble)
  library(readr)
  library(readtext)
  library(janitor)
  library(koRpus)
  library(koRpus.lang.en)
})

data_root <- Sys.getenv("LCP_DATA_ROOT")
treetagger_path <- Sys.getenv("LCP_TREETAGGER_PATH")

if (!nzchar(data_root)) {
  stop("LCP_DATA_ROOT is not set. The essays are read from $LCP_DATA_ROOT/essays and the ",
       "output is written to $LCP_DATA_ROOT/readability_metrics.csv. There is no default, ",
       "and the directory must lie outside this repository.", call. = FALSE)
}
if (!nzchar(treetagger_path)) {
  stop("LCP_TREETAGGER_PATH is not set. It must point at the TreeTagger installation ",
       "(the original hard-coded C:/TreeTagger, R: llm_paper/R/functions.R:L362).", call. = FALSE)
}

essay_folder <- file.path(data_root, "essays")
output_path <- file.path(data_root, "readability_metrics.csv")

# --- read_essays (R: llm_paper/R/functions.R:L26-31), unchanged ---------------
read_essays <- function(folder) {
  readtext::readtext(paste0(folder)) %>%
    tidyr::separate(text, sep = "\n----------------------\n", into = c("ncdsid", "text")) %>%
    tidyr::separate(text, sep = "  Words: ", into = c("text", "words")) %>%
    dplyr::mutate(ncdsid = gsub("ID: ", "", ncdsid))
}

# --- tokenize_essays (R: llm_paper/R/functions.R:L356-367) --------------------
# Unchanged except for the TreeTagger path (deviation 1 above).
tokenize_essays <- function(ncds_essays, treetagger_path) {
  ncds_essays %>%
    dplyr::mutate(count = 1:dplyr::n()) %>%
    split(1:nrow(.)) %>%
    purrr::map(function(x) {
      tokenization <- treetag(file = x$text, treetagger = "manual", lang = "en", format = "obj",
                              TT.options = list(path = treetagger_path, preset = "en"))

      tokenization

    })
}

# --- calculate_readability_metrics (R: llm_paper/R/functions.R:L369-388) ------
calculate_readability_metrics <- function(ncds_essays, tokenized_essays) {

  readability_metrics <- tokenized_essays %>%
    purrr::map_dfr(function(x) {
      x %>%
        readability() %>%
        summary() %>%
        dplyr::mutate(metric = ifelse(raw == "", grade, raw)) %>%
        dplyr::select(index, metric) %>%
        t() %>%
        janitor::row_to_names(1) %>%
        as.data.frame() %>%
        tibble::remove_rownames()
    })

  ncds_essays %>%
    as.data.frame() %>%
    dplyr::select(filename = doc_id, ncdsid) %>%
    dplyr::bind_cols(readability_metrics)
}

essays <- read_essays(essay_folder)
cat(sprintf("readability: read %d essay file(s)\n", nrow(essays)))

tokenized <- tokenize_essays(essays, treetagger_path)
cat(sprintf("readability: tokenised %d essay(s) with TreeTagger\n", length(tokenized)))

metrics <- calculate_readability_metrics(essays, tokenized)
readr::write_csv(metrics, output_path)
cat(sprintf("readability: wrote %d row(s) with %d index column(s) to $LCP_DATA_ROOT/%s\n",
            nrow(metrics), ncol(metrics) - 2L, basename(output_path)))
