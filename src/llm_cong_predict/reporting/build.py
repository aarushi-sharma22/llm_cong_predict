"""Build the reporting tables of ``R/create_data.R`` from one run's results.

``build_tables`` takes the metric rows the runner produces and, where a table needs
more than metrics, the fits (appendix D3) and the combined NCDS data or the essays
(D5-D8). A table whose inputs the run did not produce is not built and not faked: it
is listed in ``ReportingTables.skipped`` with the reason.

Nothing here writes outside ``$LCP_DATA_ROOT``: :func:`write_tables` puts the tables
under ``$LCP_DATA_ROOT/reporting/``. Sending any of them anywhere else goes through
the export guard (Task 2.7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .. import config
from .tables import (
    METRIC_TABLES,
    NCDS_TABLES,
    MissingColumnError,
    MissingMetricsError,
    appendix_d3,
    summary_d4_essays,
)


@dataclass
class ReportingTables:
    """The tables that could be built, and why the others could not."""

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.tables[name]

    def __contains__(self, name: str) -> bool:
        return name in self.tables


def _try(result: ReportingTables, name: str, build) -> None:
    try:
        result.tables[name] = build()
    except (MissingMetricsError, MissingColumnError) as exc:
        result.skipped[name] = str(exc).strip('"')


def build_tables(metrics: pd.DataFrame, fits: dict | None = None,
                 ncds_1_to_9: pd.DataFrame | None = None,
                 essays: pd.DataFrame | None = None, note: str | None = None) -> ReportingTables:
    """Build every output of ``create_data.R`` that this run's inputs allow.

    ``metrics`` is the runner's metric table, ``fits`` its ``values`` (target ->
    ``[(fit, outcome), ...]``, needed for appendix D3), ``ncds_1_to_9`` the combined
    NCDS data (D5, D6, D8 and the D7 summary) and ``essays`` the essay frame (the D4
    summary). ``note`` marks every table that does not already carry it — the corrected
    variant that adds n885 uses it (config.N885_NOTE, PORTING_NOTES N2).
    """
    result = ReportingTables()
    for name, build in METRIC_TABLES.items():
        _try(result, name, lambda build=build: build(metrics))
    if fits is not None:
        _try(result, "appendix_D3_data", lambda: appendix_d3(fits))
    else:
        result.skipped["appendix_D3_data"] = "needs the fits (the per-fold learner weights)"
    if ncds_1_to_9 is not None:
        for name, build in NCDS_TABLES.items():
            _try(result, name, lambda build=build: build(ncds_1_to_9))
    else:
        for name in NCDS_TABLES:
            result.skipped[name] = "needs the combined NCDS data (target ncds_1_to_9)"
    if essays is not None:
        _try(result, "appendix_D4_summary", lambda: summary_d4_essays(essays))
    else:
        result.skipped["appendix_D4_summary"] = "needs the essays (target ncds_essays)"
    if note:
        for table in result.tables.values():
            if "variables_note" not in table.columns:
                table["variables_note"] = note
    return result


def build_from_run(run_result, ncds_1_to_9: pd.DataFrame | None = None,
                   essays: pd.DataFrame | None = None) -> ReportingTables:
    """:func:`build_tables` from a :class:`pipeline.execute.RunResult`, taking the
    metrics, the fits and (unless given) the NCDS and essay frames from the run."""
    fits = {name: value for name, value in run_result.values.items()
            if isinstance(value, list) and value and isinstance(value[0], tuple)}
    if ncds_1_to_9 is None:
        ncds_1_to_9 = run_result.values.get("ncds_1_to_9")
    if essays is None:
        essays = run_result.values.get("ncds_essays")
    return build_tables(run_result.metrics, fits=fits, ncds_1_to_9=ncds_1_to_9, essays=essays,
                        note=getattr(run_result, "variables_note", None))


def write_tables(tables: ReportingTables, prefix: str = "") -> list[Path]:
    """Write every built table to ``$LCP_DATA_ROOT/reporting/<prefix><name>.csv``.

    There is deliberately no parameter for another location: a table leaves the data
    root only through the export guard (Task 2.7). ``prefix`` carries a run's label
    (``SMOKE_``), so a table from a smoke run cannot be taken for a real result.
    """
    directory = config.participant_output_dir("reporting")
    written = []
    for name, table in tables.tables.items():
        path = directory / f"{prefix}{name}.csv"
        table.to_csv(path, index=False)
        written.append(path)
    return written
