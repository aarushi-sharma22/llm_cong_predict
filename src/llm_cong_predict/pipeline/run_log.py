"""The run log: which targets ran, which were skipped and why.

Written to ``$LCP_DATA_ROOT/logs/run_log.json`` (config.logs_dir). It holds target
names, counts and reasons only, never participant data.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .. import config


def write_run_log(entries: dict, name: str = "run_log.json") -> Path:
    """Write ``entries`` (JSON-serialisable) with a UTC timestamp; return the path."""
    record = {"written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entries}
    path = config.logs_dir() / name
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def skipped_targets_entry(skipped, reason: str) -> dict:
    """A run-log entry listing skipped model targets by name."""
    return {"skipped_targets": sorted(s.name for s in skipped), "skip_reason": reason,
            "n_skipped": len(skipped)}
