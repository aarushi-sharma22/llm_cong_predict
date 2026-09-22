"""The export guard: the one function every file written outside ``$LCP_DATA_ROOT`` goes through.

Restricted inputs and participant-level outputs stay under ``$LCP_DATA_ROOT`` (brief
Section 2.3). A table may leave it only through :func:`export_table`, which refuses it
when it:

  1. has an ID-like column (``ncdsid``, ``NCDSID``, ``id``);
  2. has a free-text column (a value longer than ``max_text_length``, or a name that
     says it holds text);
  3. has as many rows as there are people in the analysis, which is what a
     per-person table looks like;
  4. has a count column with any value below the minimum cell size.

The minimum cell size has NO default. It is a disclosure-control decision about the
cohort data, not a number this code may pick: until ``config.MINIMUM_CELL_SIZE`` is set
(or the value is passed in), every export is refused, with a message saying to confirm
the value against the UK Data Service's output rules.

Nothing here decides that an export is safe. Passing the guard means only that these
four mechanical checks found nothing; the person exporting is still responsible for the
disclosure rules that apply to their data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config

logger = logging.getLogger(__name__)

# 1. ID-like column names (brief Task 2.7), compared without regard to case.
ID_COLUMNS = frozenset({"ncdsid", "id"})
# 2. Column names that hold text rather than a label, and the length above which a
# value counts as free text. The longest label in the R's own tables is 69 characters
# ("Linguistic Metrics of Lexical Diversity, Sophistication and Sentiment",
# R: llm_paper/R/create_data.R:L164), so 120 leaves labels alone and catches essays.
TEXT_COLUMNS = frozenset({"text", "essay", "essays", "comment", "comments", "answer",
                          "answers", "response", "responses", "doc_id", "filename"})
MAX_TEXT_LENGTH = 120
# 4. Column names that hold a count, so their values are subject to the minimum cell
# size. A proportion is not a count: only integer columns are checked.
COUNT_COLUMNS = frozenset({"n", "count", "counts", "freq", "frequency", "n_obs"})


class ExportRefused(RuntimeError):
    """The guard refused to write this table outside $LCP_DATA_ROOT."""


class MinimumCellSizeNotSet(ExportRefused):
    """No minimum cell size has been confirmed, so nothing may be exported."""


def minimum_cell_size(value: int | None = None) -> int:
    """The confirmed minimum cell size, or raise. There is deliberately no default."""
    size = config.MINIMUM_CELL_SIZE if value is None else value
    if size is None:
        raise MinimumCellSizeNotSet(
            "no minimum cell size is set, so nothing may be exported. Confirm the value that "
            "applies to these data against the UK Data Service's output rules (statistical "
            "disclosure control for the NCDS), then set config.MINIMUM_CELL_SIZE or pass "
            "minimum_cell_size=. This code will not choose a value.")
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise MinimumCellSizeNotSet(f"the minimum cell size must be a positive integer, got {size!r}")
    return size


def _id_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if str(c).lower() in ID_COLUMNS]


def _text_columns(table: pd.DataFrame, max_text_length: int) -> list[str]:
    out = []
    for column in table.columns:
        if str(column).lower() in TEXT_COLUMNS:
            out.append(str(column))
            continue
        values = table[column]
        if values.dtype == object or pd.api.types.is_string_dtype(values):
            lengths = values.dropna().astype(str).str.len()
            if not lengths.empty and lengths.max() > max_text_length:
                out.append(str(column))
    return out


def _count_columns(table: pd.DataFrame) -> list[str]:
    return [str(c) for c in table.columns
            if str(c).lower() in COUNT_COLUMNS and pd.api.types.is_integer_dtype(table[c])]


def check_export(table: pd.DataFrame, *, n_people: int | Iterable[int],
                 name: str = "table", minimum_cell_size_value: int | None = None,
                 max_text_length: int = MAX_TEXT_LENGTH) -> int:
    """Raise :class:`ExportRefused` unless ``table`` passes the four checks.

    ``n_people`` is the number of people in the analysis — one number, or every sample
    size the run used (a table with that many rows is treated as per-person). Returns
    the minimum cell size that was applied.
    """
    size = minimum_cell_size(minimum_cell_size_value)  # first: nothing leaves without it
    counts = {int(n) for n in ([n_people] if isinstance(n_people, int) else n_people)}
    problems: list[str] = []

    identifiers = _id_columns(table)
    if identifiers:
        problems.append(f"ID-like column(s) {identifiers}: a table that can be linked to a "
                        "person does not leave $LCP_DATA_ROOT")
    text = _text_columns(table, max_text_length)
    if text:
        problems.append(f"free-text column(s) {text}: text may carry identifying detail "
                        f"(a value longer than {max_text_length} characters, or a text-like name)")
    if len(table) in counts:
        problems.append(f"the table has {len(table)} rows, the number of people in the analysis "
                        f"{sorted(counts)}: that is what a per-person table looks like")
    for column in _count_columns(table):
        small = table.loc[table[column] < size, column]
        if not small.empty:
            problems.append(f"count column {column!r} has {len(small)} value(s) below the minimum "
                            f"cell size {size} (smallest {int(small.min())})")
    if problems:
        raise ExportRefused(f"{name} may not be written outside $LCP_DATA_ROOT: "
                            + "; ".join(problems))
    logger.info("export guard: %s passed (%d rows, minimum cell size %d)", name, len(table), size)
    return size


def export_table(table: pd.DataFrame, path: str | Path, *, n_people: int | Iterable[int],
                 minimum_cell_size_value: int | None = None,
                 max_text_length: int = MAX_TEXT_LENGTH) -> Path:
    """Check ``table`` and, only if it passes, write it as CSV to ``path``.

    This is the single way a table leaves ``$LCP_DATA_ROOT``. A refused table is not
    written at all, and no partial file is left behind.
    """
    path = Path(path)
    check_export(table, n_people=n_people, name=path.name,
                 minimum_cell_size_value=minimum_cell_size_value,
                 max_text_length=max_text_length)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def export_tables(tables, directory: str | Path, *, n_people: int | Iterable[int],
                  prefix: str = "", minimum_cell_size_value: int | None = None,
                  max_text_length: int = MAX_TEXT_LENGTH) -> tuple[list[Path], dict[str, str]]:
    """Export every table of a mapping (or a :class:`reporting.ReportingTables`).

    Returns the paths written and, for each table the guard refused, its reason. One
    refusal does not stop the others: the point is to say which tables may leave and
    which may not.
    """
    items = tables.tables if hasattr(tables, "tables") else tables
    written: list[Path] = []
    refused: dict[str, str] = {}
    for name, table in items.items():
        try:
            written.append(export_table(
                table, Path(directory) / f"{prefix}{name}.csv", n_people=n_people,
                minimum_cell_size_value=minimum_cell_size_value, max_text_length=max_text_length))
        except MinimumCellSizeNotSet:
            raise  # nothing may be exported at all until it is set
        except ExportRefused as exc:
            refused[name] = str(exc)
    return written, refused
