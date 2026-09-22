"""Point ``$LCP_DATA_ROOT`` at a directory for the duration of a block.

Used by the tests that write participant-level output or read restricted inputs, so
that everything they touch stays in a temporary directory.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from llm_cong_predict import config


@contextlib.contextmanager
def data_root(path):
    before = os.environ.get(config.LCP_DATA_ROOT_ENV)
    os.environ[config.LCP_DATA_ROOT_ENV] = str(path)
    try:
        yield Path(path)
    finally:
        if before is None:
            os.environ.pop(config.LCP_DATA_ROOT_ENV, None)
        else:
            os.environ[config.LCP_DATA_ROOT_ENV] = before
