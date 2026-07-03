"""Tokenization and readability — an EXTERNAL-TOOL BOUNDARY.

Ports of ``tokenize_essays`` and ``calculate_readability_metrics`` from
``R/functions.R``. Unlike SALAT/spelling (which ingest CSVs), the R GENERATES these
in-process using tools that are specific to R / external binaries:

  * ``tokenize_essays`` uses **TreeTagger** (via koRpus ``treetag``), a specific
    external tagger installed at a fixed path in the original.
  * ``calculate_readability_metrics`` uses **koRpus** ``readability()`` on those
    TreeTagger tokens, producing a specific set of readability indices.

WHY THESE RAISE INSTEAD OF BEING REIMPLEMENTED (decision: don't guess, don't
diverge silently): a Python readability library (e.g. ``textstat``) is a DIFFERENT
tool with DIFFERENT formulae and would yield numbers that do not match the paper
while appearing to work. Rather than emit divergent values, these functions raise
with guidance. The faithful ways to produce these features are:

  1. run the essays through TreeTagger + koRpus (in R) and INGEST the resulting
     readability table (analogous to the SALAT/spelling ingestion), or
  2. obtain the author's pre-computed readability features (the derived-features
     bundle), or
  3. (future) an explicit, validated Python re-implementation checked against
     koRpus — a validation-checklist item, not a silent substitution.

See docs/PORTING_NOTES.md (section G) and docs/VALIDATION_CHECKLIST.md.
"""

from __future__ import annotations

import pandas as pd

_BOUNDARY_MSG = (
    "{fn} is an external-tool step in the original (TreeTagger + koRpus) and is not "
    "reimplemented in Python, because a different readability library would produce "
    "different numbers than the paper. Provide readability features by running the "
    "essays through TreeTagger/koRpus and ingesting the result, or supply the "
    "author's pre-computed readability CSV, or implement+validate a Python "
    "equivalent (see docs/PORTING_NOTES.md section G)."
)


def tokenize_essays(ncds_essays: pd.DataFrame):
    """Port of ``tokenize_essays`` — TreeTagger tokenization (external tool).

    The R tags each essay with koRpus ``treetag`` (TreeTagger backend). Raises,
    rather than substituting a different tokenizer whose output would not feed
    koRpus-equivalent readability. See module docstring.
    """
    raise NotImplementedError(_BOUNDARY_MSG.format(fn="tokenize_essays"))


def calculate_readability_metrics(ncds_essays: pd.DataFrame, tokenized_essays=None) -> pd.DataFrame:
    """Port of ``calculate_readability_metrics`` — koRpus readability (external tool).

    Raises, for the same reason as ``tokenize_essays``. See module docstring.
    """
    raise NotImplementedError(_BOUNDARY_MSG.format(fn="calculate_readability_metrics"))


def ingest_readability_metrics(ncds_essays: pd.DataFrame, path: str) -> pd.DataFrame:
    """INGESTION path (not in the original): read a pre-computed readability table
    and attach it to the essays by ``doc_id``/``filename``.

    Provided so the pipeline can proceed once readability has been generated
    externally (koRpus, or the author's derived features), mirroring how the R's
    SALAT/spelling metrics enter as CSVs. Kept minimal and faithful to that pattern:
    left-join the readability CSV onto ``ncdsid`` via the essay ``doc_id``.
    """
    readability = pd.read_csv(path)
    base = ncds_essays.rename(columns={"doc_id": "filename"}).loc[:, ["filename", "ncdsid"]]
    shared = [c for c in base.columns if c in readability.columns] or ["filename"]
    return base.merge(readability, on=shared, how="left")
