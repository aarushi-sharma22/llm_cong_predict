"""Separate pipeline step: generate RoBERTa essay embeddings in their own process.

    python -m llm_cong_predict.features.roberta_step

Reads the essays from ``$LCP_DATA_ROOT`` (``config.RESTRICTED_INPUTS["essays"]``),
runs :func:`llm_cong_predict.features.embeddings.roberta_embeddings` (public
``roberta-base`` weights, run locally; no essay text leaves the machine) and writes
``$LCP_DATA_ROOT/derived/roberta_embeddings.csv`` (``config.DERIVED_FILES``). The
pipeline then reads that file (``read_roberta_embeddings``).

Why a separate process: torch and xgboost load different OpenMP runtimes and crash
when used in one process (``llm_cong_predict.isolation``,
docs/ORCHESTRATION.md). This module imports neither library at import time.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def write_roberta_embeddings(essays: pd.DataFrame, out_path, **kwargs) -> pd.DataFrame:
    """Compute the embeddings of ``essays`` and write them to ``out_path`` (CSV).

    Values are written in Python's shortest round-trip form, so reading the file back
    gives exactly the same float64 values.
    """
    from .embeddings import roberta_embeddings

    emb = roberta_embeddings(essays, **kwargs)
    emb.to_csv(out_path, index=False)
    return emb


def main(argv: list[str] | None = None) -> int:
    from .. import config
    from ..io.readers import read_essays

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)

    essays = read_essays(str(config.restricted_path("essays")))
    out = config.derived_path("roberta_embeddings")
    emb = write_roberta_embeddings(essays, out, batch_size=args.batch_size)
    print(f"roberta_step: wrote embeddings for {len(emb)} essays to $LCP_DATA_ROOT/derived/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
