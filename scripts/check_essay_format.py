#!/usr/bin/env python
"""Check the format of the essay files, printing AGGREGATE COUNTS ONLY.

It prints
numbers and nothing else: never essay text, never file names, never IDs. It writes
no files.

Each essay file is expected to look like this (R: llm_paper/R/functions.R:L26–31):

    ID: <ncdsid>
    ----------------------
    <essay text>  Words: <count>

The file is read as ``readtext`` reads it and split exactly as ``read_essays`` splits
it (``llm_cong_predict.io.readers.parse_essay``). Counts:

  files found            regular files in the folder
  matching format        both separators exactly once and a word count R can parse
  missing separators     the "----------------------" line or "  Words: " absent
  extra separators       either separator more than once (read_essays keeps the first
                         two pieces, as tidyr::separate does)
  word count unparsable  the text after "  Words: " is not a number for R's
                         as.numeric() (includes files where it is missing)
  unreadable files       files that cannot be opened or decoded (--encoding)

A file can be counted under more than one of: missing separators, extra separators,
word count unparsable.

Usage:
    python scripts/check_essay_format.py                 # the essays under $LCP_DATA_ROOT
    python scripts/check_essay_format.py --essays DIR    # another folder
"""

from __future__ import annotations

import argparse
import glob
import os
import sys


def count_formats(folder: str, encoding: str = "utf-8") -> dict[str, int]:
    """Aggregate format counts for every regular file in ``folder``."""
    from llm_cong_predict.io.labels import r_as_numeric
    from llm_cong_predict.io.readers import _readtext_txt, parse_essay

    counts = {"files found": 0, "matching format": 0, "missing separators": 0,
              "extra separators": 0, "word count unparsable": 0, "unreadable files": 0}
    for path in glob.glob(os.path.join(folder, "*")):
        if not os.path.isfile(path):
            continue
        counts["files found"] += 1
        try:
            content = _readtext_txt(path, encoding)
        except (OSError, UnicodeDecodeError):  # never echo the error: it can quote content
            counts["unreadable files"] += 1
            continue
        fields, problems = parse_essay(content)
        bad_words = bool(r_as_numeric([fields["words"]]).isna().iloc[0])
        counts["missing separators"] += "missing" in problems
        counts["extra separators"] += "extra" in problems
        counts["word count unparsable"] += bad_words
        counts["matching format"] += not problems and not bad_words
    return counts


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--essays", default=None,
                    help="essay folder (default: the 'essays' entry under $LCP_DATA_ROOT)")
    ap.add_argument("--encoding", default="utf-8")
    args = ap.parse_args(argv)

    if args.essays is None:
        from llm_cong_predict import config

        folder = str(config.restricted_path("essays"))
    else:
        folder = args.essays
    if not os.path.isdir(folder):
        print("check_essay_format: the essay folder does not exist.", file=sys.stderr)
        return 2
    for name, n in count_formats(folder, args.encoding).items():
        print(f"{name}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
