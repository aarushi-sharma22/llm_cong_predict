#!/usr/bin/env python
"""Refuse to commit restricted or participant-level data.

This repository must never contain NCDS survey data, essays, derived essay features,
embeddings, polygenic scores or per-person outputs. Those live
outside the repository, under ``$LCP_DATA_ROOT``. This checker is the last line of
defence behind ``.gitignore``, because ``git add -f`` bypasses ignore rules.

It checks the files staged for commit (default), or every tracked file (``--all``),
and fails with one line per problem when any file:

  1. has a data-like extension (.parquet .feather .npy .npz .pkl .joblib .rds .RData
     .sav .tab .dta) and is not on the allow-list;
  2. lies under ``data/`` and is not on the allow-list;
  3. is a .csv or .txt file outside ``tests/`` and ``docs/`` (``requirements*.txt``
     excepted);
  4. is larger than 5 MB (5,000,000 bytes, the size of the staged blob);
  5. lies under ``reference/`` (third-party source clones; see docs/REFERENCE_SOURCES.md).

The allow-list is exactly: ``data/variables.xlsx``,
``data/occupation_aspiration_mapping.xlsx`` and ``data/camsis/*.dta`` (no
subdirectories). These are public reference files shipped with the original R
repository.

Usage:
    python scripts/check_no_restricted_data.py          # staged files
    python scripts/check_no_restricted_data.py --all    # all tracked files

Run automatically before every commit by ``scripts/hooks/pre-commit`` once the hook
path is enabled with ``git config core.hooksPath scripts/hooks``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import PurePosixPath

RESTRICTED_EXTENSIONS = frozenset(
    {".parquet", ".feather", ".npy", ".npz", ".pkl", ".joblib", ".rds", ".rdata",
     ".sav", ".tab", ".dta"}
)
ALLOWED_DATA_FILES = frozenset({"data/variables.xlsx", "data/occupation_aspiration_mapping.xlsx"})
TABLE_TEXT_EXTENSIONS = frozenset({".csv", ".txt"})
TABLE_TEXT_ALLOWED_TOP_DIRS = frozenset({"tests", "docs"})
MAX_BYTES = 5_000_000


def is_allow_listed(path: str) -> bool:
    """True for the public reference files that may be tracked."""
    p = PurePosixPath(path)
    if path in ALLOWED_DATA_FILES:
        return True
    return p.parent == PurePosixPath("data/camsis") and p.suffix.lower() == ".dta"


def problems_for(path: str, size: int | None) -> list[str]:
    """Return the reasons ``path`` (repository-relative, POSIX) may not be committed."""
    p = PurePosixPath(path)
    suffix = p.suffix.lower()
    top = p.parts[0] if p.parts else ""
    allowed = is_allow_listed(path)
    out: list[str] = []

    if suffix in RESTRICTED_EXTENSIONS and not allowed:
        out.append(f"data-like file type '{p.suffix}' outside the allow-list")
    if top == "data" and not allowed:
        out.append("file under data/ that is not one of the public reference files")
    if suffix in TABLE_TEXT_EXTENSIONS and top not in TABLE_TEXT_ALLOWED_TOP_DIRS:
        if not (suffix == ".txt" and p.name.startswith("requirements")):
            out.append(f"'{p.suffix}' table/text file outside tests/ and docs/")
    if size is not None and size > MAX_BYTES:
        out.append(f"file is {size:,} bytes, over the {MAX_BYTES:,}-byte limit")
    if top == "reference":
        out.append("third-party reference source (reference/ must never be committed)")
    return out


def _git(args: list[str], cwd: str | None) -> bytes:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True).stdout


def _index_sizes(cwd: str | None) -> dict[str, int | None]:
    """Map every path in the index to the size of its staged blob (None for gitlinks)."""
    entries = [e for e in _git(["ls-files", "-s", "-z"], cwd).split(b"\0") if e]
    paths, objects = [], []
    for entry in entries:
        meta, path = entry.split(b"\t", 1)
        mode, obj, _stage = meta.split(b" ")
        paths.append(path.decode())
        objects.append(None if mode == b"160000" else obj.decode())
    blobs = [o for o in objects if o is not None]
    sizes: dict[str, int] = {}
    if blobs:
        out = subprocess.run(
            ["git", "cat-file", "--batch-check=%(objectname) %(objectsize)"],
            cwd=cwd, check=True, capture_output=True, input="\n".join(blobs).encode(),
        ).stdout.decode().split()
        sizes = {out[i]: int(out[i + 1]) for i in range(0, len(out), 2)}
    return {p: (sizes[o] if o is not None else None) for p, o in zip(paths, objects)}


def files_to_check(all_files: bool, cwd: str | None) -> list[str]:
    if all_files:
        raw = _git(["ls-files", "-z"], cwd)
    else:
        raw = _git(["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"], cwd)
    return [p.decode() for p in raw.split(b"\0") if p]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true", help="check every tracked file, not only staged ones")
    ap.add_argument("--repo", default=None, help="repository to check (default: current directory)")
    args = ap.parse_args(argv)

    root = _git(["rev-parse", "--show-toplevel"], args.repo).decode().strip()
    paths = files_to_check(args.all, root)
    sizes = _index_sizes(root)

    failures = [(p, why) for p in paths for why in problems_for(p, sizes.get(p))]
    scope = "tracked" if args.all else "staged"
    if failures:
        print(f"check_no_restricted_data: REFUSED — {len(failures)} problem(s) in {scope} files:",
              file=sys.stderr)
        for p, why in failures:
            print(f"  {p}: {why}", file=sys.stderr)
        print(
            "Restricted or participant-level data must stay under $LCP_DATA_ROOT, outside "
            "the repository. Unstage the file(s) with "
            "'git restore --staged <path>'.",
            file=sys.stderr,
        )
        return 1
    print(f"check_no_restricted_data: OK ({len(paths)} {scope} file(s) checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
