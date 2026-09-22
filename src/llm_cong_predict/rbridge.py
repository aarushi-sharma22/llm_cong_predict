"""Calling R through rpy2 (optional; ``pip install -e '.[oracle]'``).

Used by the R reference implementations (``cleaning/factors.py::create_factors_r``,
``models/r_superlearner.py``) and by the oracle tests. Nothing here is needed for the
native pipeline. R and the R packages are never installed by this code; see
docs/REFERENCE_SOURCES.md for the installed versions.
"""

from __future__ import annotations

import contextlib
import importlib.util


class RUnavailable(RuntimeError):
    """R, rpy2 or a required R package is not available."""


def r_unavailable_reason(packages: tuple[str, ...] = ()) -> str | None:
    """None if rpy2 imports, R starts and every package in ``packages`` is installed;
    otherwise a sentence saying what is missing (used as a pytest skip reason)."""
    if importlib.util.find_spec("rpy2") is None:
        return "rpy2 is not installed (pip install -e '.[oracle]')"
    try:
        import rpy2.robjects as ro
    except Exception as exc:  # R missing or broken
        return f"R cannot be started through rpy2 ({type(exc).__name__}: {exc})"
    missing = [p for p in packages
               if not bool(ro.r(f'isTRUE(requireNamespace("{p}", quietly = TRUE))')[0])]  # visible value
    if missing:
        return f"R package(s) not installed: {', '.join(missing)}"
    return None


def require_r(packages: tuple[str, ...] = ()) -> None:
    reason = r_unavailable_reason(packages)
    if reason is not None:
        raise RUnavailable(reason)


@contextlib.contextmanager
def converter():
    """numpy and pandas <-> R conversion for the block (rpy2 >= 3.5 style)."""
    from rpy2.robjects import default_converter, numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(default_converter + numpy2ri.converter + pandas2ri.converter):
        yield


def r_package_version(package: str) -> str:
    import rpy2.robjects as ro

    return str(ro.r(f'as.character(packageVersion("{package}"))')[0])
