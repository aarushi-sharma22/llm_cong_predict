"""Reporting tables: the port of ``R/create_data.R``.

Every output file of the R, what it is built from and whether the R can produce it:
docs/reference/create_data_outputs.md. The deviations are PORTING_NOTES N1-N4.
"""

from .build import ReportingTables, build_from_run, build_tables, write_tables

__all__ = ["ReportingTables", "build_tables", "build_from_run", "write_tables"]
