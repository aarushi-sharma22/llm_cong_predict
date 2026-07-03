"""Data readers — faithful ports of the ``read_*`` / ``combine_ncds`` functions
in the original ``R/functions.R``.

VALIDATION STATUS: these are translated from the R but can only be *numerically*
validated once the real, access-restricted NCDS data arrives (see
docs/VALIDATION_CHECKLIST.md). They are tested here for MECHANICS against
synthetic fixtures that match the real file structure (tests/fixtures). Two
readers can be exercised against real shipped files: ``read_camsis`` (the CAMSIS
.dta files are in the repo) and ``read_occupation_aspiration_mapping`` /
``read_datalist`` (the xlsx files are in the repo).

We do NOT guess data values anywhere: where the original depends on the real
dataset, the logic is ported and the gap is flagged, not filled in.
"""

from __future__ import annotations

import glob
import os

import pandas as pd
import pyreadstat

from .labels import (
    COLUMN_LABELS_KEY,
    VALUE_LABELS_KEY,
    attach_labels,
    merge_value_labels,
    set_na_range,
)


def _lower_map(d: dict | None) -> dict:
    return {str(k).lower(): v for k, v in (d or {}).items()}


def read_datalist(path: str) -> pd.DataFrame:
    """Port of ``read_datalist``: ``read_excel(path)``.

    Thin wrapper reading the variable-metadata workbook. IMPORTANT: as in R, the
    first row becomes the header. The public ``variables.xlsx`` does NOT contain the
    columns the cleaning layer expects (``variable``/``sweep``/``respondent``/
    ``new_varname``/``type``) — see PORTING_NOTES A1. This reader faithfully returns
    whatever the file contains; the schema reconstruction is handled in the cleaning
    layer, not here.
    """
    return pd.read_excel(path)


def read_occupation_aspiration_mapping(path: str) -> pd.DataFrame:
    """Port of ``read_occupation_aspiration_mapping``: ``read_excel(path)``.

    Reads the hand-crafted occupation->aspiration mapping. This file IS shipped in
    the repo, so this reader can be exercised against real data.
    """
    return pd.read_excel(path)


def read_camsis(path: str) -> pd.DataFrame:
    """Port of ``read_camsis``: ``haven::read_dta(path)``.

    Reads a CAMSIS occupation-scoring .dta. Kept faithful: like the R, column names
    are NOT altered here (the real files already use lower-case names such as
    ``co1970``/``mcamsis``/``fcamsis`` that ``create_aspirations`` relies on). Stata
    value labels are carried on ``df.attrs`` so ``create_aspirations`` can apply
    ``as_factor`` to ``co1970`` exactly as the R does.
    """
    df, meta = pyreadstat.read_dta(path)
    return attach_labels(
        df,
        value_labels=meta.variable_value_labels or {},
        column_labels={k: v for k, v in (meta.column_names_to_labels or {}).items() if v},
    )


def read_ncds(file: str, varlist: list[str]) -> pd.DataFrame:
    """Port of ``read_ncds(file, varlist)``.

    R:
        haven::read_dta(file) %>%
          setNames(tolower(colnames(.))) %>%
          dplyr::select(dplyr::one_of(tolower(c("ncdsid", varlist)))) %>%
          sjlabelled::set_na(na = -99:-1) %>%
          setNames(tolower(colnames(.)))

    Steps reproduced exactly:
      1. read the .dta (numeric codes, labels retained on attrs);
      2. lower-case all column names;
      3. keep only the requested columns (``ncdsid`` + ``varlist``) that exist,
         in the requested order (``dplyr::one_of`` ignores missing names);
      4. recode values in ``[-99, -1]`` to missing;
      5. (the R lower-cases again, a no-op here).

    The commented-out ``as_factor`` / ``mutate_if`` lines in the R are NOT executed,
    so columns stay numeric with labels attached — matched here.
    """
    df, meta = pyreadstat.read_dta(file)
    df.columns = [c.lower() for c in df.columns]
    value_labels = _lower_map(meta.variable_value_labels)
    column_labels = _lower_map(meta.column_names_to_labels)

    wanted = [str(c).lower() for c in (["ncdsid"] + list(varlist))]
    present = set(df.columns)
    # preserve requested order, drop duplicates and absent names (one_of semantics)
    keep, seen = [], set()
    for c in wanted:
        if c in present and c not in seen:
            keep.append(c)
            seen.add(c)
    df = df[keep]

    df = set_na_range(df, -99, -1)

    attach_labels(
        df,
        value_labels={c: value_labels[c] for c in keep if c in value_labels},
        column_labels={c: column_labels[c] for c in keep if c in column_labels and column_labels[c]},
    )
    return df


def read_gene_data(path: str | None = None):
    """Port of ``read_gene_data`` — an empty ``#PLACEHOLDER`` in the original.

    DEVIATION FROM R (documented in PORTING_NOTES A2/D): the R body was empty and
    returned ``NULL`` silently. We raise instead, so that any attempt to use gene
    data fails loudly and clearly rather than propagating a silent ``None`` into the
    pipeline. NCDS genetic data / polygenic scores are access-restricted and are not
    part of this repository.
    """
    raise NotImplementedError(
        "NCDS genetic data / polygenic scores are access-restricted and not included "
        "in this repository (the original `read_gene_data` was an empty placeholder "
        "returning NULL). To run the genetic models, obtain the data via the NCDS "
        "Data Access Committee and implement this reader; otherwise run the "
        "non-genetic subset of the pipeline. See docs/PORTING_NOTES.md."
    )


def read_essays(folder: str, encoding: str = "utf-8") -> pd.DataFrame:
    """Port of ``read_essays(folder)``.

    R:
        readtext::readtext(folder) %>%
          tidyr::separate(text, sep = "\\n----------------------\\n",
                          into = c("ncdsid", "text")) %>%
          tidyr::separate(text, sep = "  Words: ", into = c("text", "words")) %>%
          dplyr::mutate(ncdsid = gsub("ID: ", "", ncdsid))

    Each file in ``folder`` is one essay in the format::

        ID: <ncdsid>
        ----------------------
        <essay text>  Words: <count>

    Returns a frame with columns ``doc_id`` (filename), ``ncdsid``, ``text``,
    ``words`` — matching the R output. Splitting is done on the FIRST occurrence of
    each delimiter so essay bodies containing similar text do not break parsing
    (this mirrors the practical intent of the two ``separate`` calls).
    """
    id_delim = "\n----------------------\n"
    words_delim = "  Words: "

    rows = []
    for path in sorted(glob.glob(os.path.join(folder, "*"))):
        if not os.path.isfile(path):
            continue
        with open(path, encoding=encoding) as fh:
            content = fh.read()

        head, _, body = content.partition(id_delim)
        if _ == "":  # delimiter absent: whole content is the "id" part, no body
            head, body = content, ""
        text, _, words = body.partition(words_delim)
        words_val = words if _ != "" else None

        ncdsid = head.replace("ID: ", "")  # gsub is global; str.replace replaces all
        rows.append(
            {"doc_id": os.path.basename(path), "ncdsid": ncdsid, "text": text, "words": words_val}
        )

    return pd.DataFrame(rows, columns=["doc_id", "ncdsid", "text", "words"])


def combine_ncds(*frames: pd.DataFrame) -> pd.DataFrame:
    """Port of ``combine_ncds(...)``.

    R:
        list(...) %>% plyr::join_all(by = "ncdsid", type = "full") %>% as_tibble()

    Successive FULL outer join of all frames on ``ncdsid``. The R signature has an
    unused ``varlist`` argument, which we omit.

    COLLISION HANDLING (flagged, verify with real data — PORTING_NOTES): the NCDS
    waves use distinct variable codes, so non-key column overlaps are not expected.
    If a non-``ncdsid`` column appears in more than one frame, pandas' outer merge
    would suffix them ``_x``/``_y``; instead we coalesce (prefer the left, fill from
    the right) into a single column and record it. plyr's exact behaviour on such
    collisions is ambiguous; this assumption should be checked against R once real
    data is available.
    """
    dfs = [f for f in frames if f is not None]
    if not dfs:
        return pd.DataFrame()

    merged = dfs[0]
    collisions: list[str] = []
    for right in dfs[1:]:
        overlap = (set(merged.columns) & set(right.columns)) - {"ncdsid"}
        merged = pd.merge(merged, right, on="ncdsid", how="outer", suffixes=("", "__r"))
        for col in overlap:
            rcol = f"{col}__r"
            if rcol in merged.columns:
                merged[col] = merged[col].combine_first(merged[rcol])
                merged = merged.drop(columns=[rcol])
                collisions.append(col)

    merged.attrs[VALUE_LABELS_KEY] = merge_value_labels(dfs)
    if collisions:
        merged.attrs["combine_ncds_collisions"] = sorted(set(collisions))
    return merged
