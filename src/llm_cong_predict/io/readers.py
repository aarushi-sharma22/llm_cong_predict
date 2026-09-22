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
import logging
import os
import warnings

import pandas as pd
import pyreadstat

from .. import config
from .labels import (
    COLUMN_LABELS_KEY,
    VALUE_LABELS_KEY,
    attach_labels,
    merge_value_labels,
    set_na_range,
)

logger = logging.getLogger(__name__)


def _lower_map(d: dict | None) -> dict:
    return {str(k).lower(): v for k, v in (d or {}).items()}


# Column meanings of variables.xlsx, by position (brief F1; fixed by how clean_ncds uses
# them, R: llm_paper/R/functions.R:L75–78, L103, L196, L201, L207, L212, L217).
DATALIST_COLUMNS = ("sweep", "type", "respondent", "question", "label", "new_varname", "variable")
DATALIST_REQUIRED = ("sweep", "type", "respondent", "new_varname", "variable")
DATALIST_ROWS = 63  # computed from the public file at Checkpoint A (PORTING_NOTES A1)

# CORRECTED VARIANT, off by default (config.INCLUDE_N885; owner decision at Checkpoint D).
# n885 ("Imperfect Grasp of English") is selected by R/create_data.R:L264 for appendix D6
# and named in find_essay_teacher_genetics_overlap (functions.R:L333), but it is not in
# variables.xlsx, so the R stops in both places. Adding this row makes read_ncds load the
# code and gives clean_ncds' behaviour block one more column, s2_te_imperfect_english.
N885_ROW: dict = {"sweep": 2.0, "type": "behavior", "respondent": "teacher",
                  "question": pd.NA, "label": pd.NA, "new_varname": "imperfect_english",
                  "variable": "N885"}


def read_datalist(path: str, include_n885: bool | None = None) -> pd.DataFrame:
    """Port of ``read_datalist`` (R: llm_paper/R/functions.R:L22–24), a reconstruction.

    The R calls ``read_excel(path)``, which makes the file's first row the header. The
    public ``variables.xlsx`` has no header row, so the columns ``clean_ncds`` needs
    would not exist and the published R cannot run (PORTING_NOTES A1). The evident
    intent is reproduced: the file is read without a header, columns 0–6 are named
    ``sweep, type, respondent, question, label, new_varname, variable`` (brief F1),
    columns 7 onwards (empty, apart from one cell of spaces) are dropped, and the rows
    complete on the five columns the R uses are kept. The result must have exactly 63
    rows and no duplicated ``variable`` code.

    ``include_n885`` (default: ``config.INCLUDE_N885``, which is ``False``) turns on the
    corrected variant that adds :data:`N885_ROW`. The frame records which it was, in
    ``attrs["variables_include_n885"]``, so every output of a run can be marked.
    """
    include_n885 = config.INCLUDE_N885 if include_n885 is None else include_n885
    raw = pd.read_excel(path, header=None)
    df = raw.iloc[:, : len(DATALIST_COLUMNS)].copy()
    df.columns = list(DATALIST_COLUMNS)
    df = df.dropna(subset=list(DATALIST_REQUIRED)).reset_index(drop=True)
    if len(df) != DATALIST_ROWS:
        raise ValueError(f"variables.xlsx: expected {DATALIST_ROWS} complete rows, found {len(df)}")
    if include_n885:
        df = pd.concat([df, pd.DataFrame([N885_ROW])], ignore_index=True)
    dup = df["variable"].str.lower().duplicated()
    if dup.any():
        raise ValueError(f"variables.xlsx: duplicated codes {sorted(df.loc[dup, 'variable'])}")
    df.attrs["variables_include_n885"] = bool(include_n885)
    return df


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
      4. recode values in ``[-99, -1]`` to missing and drop their value labels;
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
    df = df[keep].copy()
    attach_labels(
        df,
        value_labels={c: value_labels[c] for c in keep if c in value_labels},
        column_labels={c: column_labels[c] for c in keep if c in column_labels and column_labels[c]},
    )
    # set_na(-99:-1) after the labels are attached, so that the labels of the recoded
    # values are dropped as sjlabelled does (R pkg: sjlabelled/R/set_na.R:L268–272).
    return set_na_range(df, -99, -1)


def read_gene_data(path: str | None = None) -> pd.DataFrame:
    """Port of ``read_gene_data`` — an empty ``#PLACEHOLDER`` in the original
    (R: llm_paper/R/functions.R:L34–36).

    Without a path it raises, rather than returning a silent ``None`` (PORTING_NOTES
    E1). With a path it reads the port's PLACEHOLDER format: a CSV whose first column
    is ``ncdsid`` and whose other columns are numeric scores. The R takes the scores as
    ``colnames(gene_data)[-1]`` (``_targets.R:L132``), so the ID must come first. The
    format of the released polygenic index files is a Phase 5/6 item; this reader is
    replaced then. The pipeline calls it only when the file exists
    (``config.RESTRICTED_INPUTS["gene_data"]``); otherwise gene data is absent.
    """
    if path is None:
        raise NotImplementedError(
            "NCDS genetic data / polygenic scores are access-restricted and not included "
            "in this repository (the original `read_gene_data` was an empty placeholder "
            "returning NULL). Place the scores at $LCP_DATA_ROOT/"
            "genetics/polygenic_scores.csv (config.RESTRICTED_INPUTS['gene_data']) or run "
            "without them: the gene-dependent models are then skipped. See "
            "docs/PORTING_NOTES.md (E1, L2)."
        )
    df = pd.read_csv(path, dtype={"ncdsid": str})
    if list(df.columns[:1]) != ["ncdsid"]:
        raise ValueError("gene data: the first column must be ncdsid (the R drops it with colnames()[-1])")
    bad = [c for c in df.columns[1:] if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise ValueError(f"gene data: score columns must be numeric, got {bad}")
    if df["ncdsid"].duplicated().any():
        raise ValueError("gene data: ncdsid is not unique")
    return df


ESSAY_ID_SEPARATOR = "\n----------------------\n"  # R: llm_paper/R/functions.R:L28
ESSAY_WORDS_SEPARATOR = "  Words: "  # R: llm_paper/R/functions.R:L29


def _readtext_txt(path: str, encoding: str) -> str:
    """A .txt file as ``readtext`` reads it: ``paste(readLines(con), collapse = "\\n")``
    (R pkg: readtext/R/get-functions.R:L2–4, 0.92.1). readLines accepts LF, CRLF and CR
    line endings and drops the final line terminator."""
    with open(path, encoding=encoding, newline=None) as fh:  # universal newlines -> "\n"
        content = fh.read()
    return content[:-1] if content.endswith("\n") else content


def _separate_two(value: str | None, sep: str) -> tuple[list[str | None], str | None]:
    """``tidyr::separate(into = <2 columns>, sep, extra = "warn", fill = "warn")`` on one
    value (R pkg: tidyr/R/separate.R:L170–201 and src/simplifyPieces.cpp, 1.3.2).

    The value is split at EVERY match of ``sep`` (the separators here contain no regex
    metacharacters, so a literal split equals tidyr's regex split). More than two
    pieces: the first two are kept and the rest discarded ("extra"). Fewer than two:
    the missing piece on the right is NA ("missing"). NA in, NA out, no flag.
    """
    if value is None:
        return [None, None], None
    pieces = value.split(sep)
    if len(pieces) > 2:
        return pieces[:2], "extra"
    if len(pieces) < 2:
        return [pieces[0], None], "missing"
    return pieces, None


def parse_essay(content: str) -> tuple[dict, set[str]]:
    """Parse one essay file's content as ``read_essays`` does.

    Returns the fields ``ncdsid``, ``text`` and ``words`` (strings or None) and the set
    of problems found: "extra" (a separator occurs more than once), "missing" (a
    separator is absent). Never logs anything itself.
    """
    (id_part, body), f1 = _separate_two(content, ESSAY_ID_SEPARATOR)
    (text, words), f2 = _separate_two(body, ESSAY_WORDS_SEPARATOR)
    ncdsid = None if id_part is None else id_part.replace("ID: ", "")  # gsub("ID: ", "", ncdsid)
    return {"ncdsid": ncdsid, "text": text, "words": words}, {f for f in (f1, f2) if f}


def read_essays(folder: str, encoding: str = "utf-8") -> pd.DataFrame:
    """Port of ``read_essays(folder)`` (R: llm_paper/R/functions.R:L26–31).

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

    Returns ``doc_id`` (file name), ``ncdsid``, ``text``, ``words``, as the R does.
    Malformed files are handled as ``tidyr::separate`` handles them (owner decision at
    Checkpoint B): pieces after a second separator are dropped, a missing piece is NA.
    tidyr warns and lists row numbers; the port only logs and warns with the NUMBER
    of files affected, never file names, IDs or text. The counts are also kept in
    ``attrs["read_essays_malformed"]``.
    """
    rows = []
    extra = missing = 0
    for path in sorted(glob.glob(os.path.join(folder, "*"))):
        if not os.path.isfile(path):
            continue
        fields, problems = parse_essay(_readtext_txt(path, encoding))
        extra += "extra" in problems
        missing += "missing" in problems
        rows.append({"doc_id": os.path.basename(path), **fields})

    out = pd.DataFrame(rows, columns=["doc_id", "ncdsid", "text", "words"])
    out.attrs["read_essays_malformed"] = {"files_with_extra_pieces": extra,
                                          "files_with_missing_pieces": missing}
    if extra or missing:
        message = (f"read_essays: {extra} file(s) had additional pieces (discarded) and "
                   f"{missing} file(s) had missing pieces (filled with NA), as "
                   "tidyr::separate(extra = 'warn', fill = 'warn') does")
        logger.warning(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return out


class ColumnCollisionError(ValueError):
    """Raised by ``combine_ncds(..., strict=True)`` when frames share a non-key column."""


def _plyr_full_join(x: pd.DataFrame, y: pd.DataFrame, by: str) -> tuple[pd.DataFrame, list[dict]]:
    """``plyr::join(x, y, by, type = "full", match = "all")``.

    R pkg: plyr/R/join.r:L104–133 and plyr/R/rbind-fill.r (1.8.9; code unchanged since
    1.8.4). ``.join_all`` builds ``matched = cbind(x[ids$x, ], y[ids$y, y.cols])`` (every
    x row with its matches, in x order), where a column present in both frames appears
    twice, then ``rbind.fill(matched, unmatched)`` appends the y rows with no match.
    ``rbind.fill`` keeps one column per name (``unique(names)``, rbind-fill.r:L80) and
    fills it with ``df[[var]]``, the FIRST occurrence (L70–71). So for a column in both
    frames: rows of x keep x's value (an NA stays NA; this is not a coalesce), rows only
    in y take y's value, and the column keeps x's attributes.
    """
    y_cols = [c for c in y.columns if c != by]
    shared = [c for c in y_cols if c in x.columns]
    new = [c for c in y_cols if c not in x.columns]
    matched = x.merge(y[[by] + new], on=by, how="left")  # x order; repeated for several matches
    unmatched = y[~y[by].isin(x[by])]
    out = pd.concat([matched, unmatched], ignore_index=True)

    records = []
    for col in shared:
        both = x[[by, col]].merge(y[[by, col]], on=by, suffixes=("_left", "_right"))
        l, r = both[f"{col}_left"], both[f"{col}_right"]
        differing = int((~((l == r) | (l.isna() & r.isna()))).sum())
        records.append({"column": col, "rows_in_both": int(len(both)),
                        "cells_differing": differing, "right_only_rows": int(len(unmatched))})
    return out, records


def combine_ncds(*frames: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Port of ``combine_ncds(...)``.

    R (llm_paper/R/functions.R:L57–62):
        list(...) %>% plyr::join_all(by = "ncdsid", type = "full") %>% as_tibble()

    Successive full joins on ``ncdsid`` with plyr's semantics (:func:`_plyr_full_join`):
    rows in the order plyr produces them, and a column that appears in more than one
    frame resolved as plyr resolves it (the earlier frame's value for its rows, the
    later frame's value only for rows the earlier frames did not have). plyr never
    produces duplicated names, so ``as_tibble``'s name check never fires (PORTING_NOTES
    E3). Every collision is logged and recorded in ``attrs["combine_ncds_collisions"]``
    with the number of cells where the two versions differ. ``strict=True`` raises
    :class:`ColumnCollisionError` instead (not in the R). The R signature's unused
    ``varlist`` argument is omitted.
    """
    dfs = [f for f in frames if f is not None]
    if not dfs:
        return pd.DataFrame()

    merged = dfs[0]
    collisions: list[dict] = []
    for i, right in enumerate(dfs[1:], start=1):
        shared = sorted((set(merged.columns) & set(right.columns)) - {"ncdsid"})
        if shared and strict:
            raise ColumnCollisionError(
                f"combine_ncds(strict=True): frame {i} shares column(s) {shared} with earlier "
                "frames; plyr::join_all would keep the earlier frame's values")
        merged, records = _plyr_full_join(merged, right, "ncdsid")
        for rec in records:
            rec["frame"] = i
            logger.warning("combine_ncds: column %r in frame %d collides with an earlier frame "
                           "(%d rows in both, %d cells differ); plyr keeps the earlier values",
                           rec["column"], i, rec["rows_in_both"], rec["cells_differing"])
        collisions.extend(records)

    merged.attrs[VALUE_LABELS_KEY] = merge_value_labels(dfs)
    if collisions:
        merged.attrs["combine_ncds_collisions"] = collisions
    return merged
