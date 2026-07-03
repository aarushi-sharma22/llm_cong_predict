#!/usr/bin/env python
"""Generate GPT essay embeddings via the OpenAI API.

Port of ``R/get_gpt_embeddings.R``. Runnable once you have the real essays and an
OpenAI API key; not exercised in the dev sandbox (no key, no essays).

Faithful to the original:
  * batches essays in groups of 100 (the R used ``ceiling(seq/100)``);
  * generates two embedding sets — ``text-embedding-ada-002`` ("GPT 3.5") and
    ``text-embedding-3-large`` ("GPT 4"), the exact models the R used.

DEVIATION (documented, PORTING_NOTES G1): the R saved raw API responses as ``.rds``
(an R-only binary format). We instead save a Python-native Parquet with an
``ncdsid`` column plus ``embedding_*`` columns — the shape ``features.embeddings.
gpt_embeddings`` reads. This removes the R-only format and the original reshaper's
path-vs-frame contradiction.

Usage:
    export OPENAI_API_KEY=...          # never hard-code the key (the R had "<KEY>")
    python scripts/get_gpt_embeddings.py --essays data/essays --out data/embeddings

Requires: pip install -e '.[embeddings]'  (openai)
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# The two models, matching the original script exactly.
MODELS = {
    "gpt35": "text-embedding-ada-002",
    "gpt4": "text-embedding-3-large",
}
BATCH_SIZE = 100  # R: ceiling(seq_along(1:nrow(essays)) / 100)


def _batched(items: list, n: int):
    for i in range(0, len(items), n):
        yield i // n, items[i : i + n]


def generate(essays: pd.DataFrame, model: str, api_key: str) -> pd.DataFrame:
    """Return a frame: ncdsid + embedding_1..K for the given model."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    ids = essays["ncdsid"].tolist()
    texts = essays["text"].astype(str).tolist()

    vectors: list[list[float]] = []
    for _, chunk in _batched(texts, BATCH_SIZE):
        resp = client.embeddings.create(model=model, input=chunk)
        vectors.extend([d.embedding for d in resp.data])

    emb = pd.DataFrame(vectors, columns=[f"embedding_{i+1}" for i in range(len(vectors[0]))])
    emb.insert(0, "ncdsid", ids)
    return emb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--essays", required=True, help="folder of essay text files")
    ap.add_argument("--out", required=True, help="output folder for embedding parquet files")
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY in the environment (do not hard-code it).", file=sys.stderr)
        return 2

    # Read essays via the ported reader (same parsing as the pipeline).
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from llm_cong_predict.io.readers import read_essays

    essays = read_essays(args.essays)
    os.makedirs(args.out, exist_ok=True)

    for key in args.models:
        model = MODELS[key]
        print(f"Generating {key} embeddings ({model}) for {len(essays)} essays...")
        emb = generate(essays, model, api_key)
        out_path = os.path.join(args.out, f"embeddings_{key}.parquet")
        emb.to_parquet(out_path, index=False)
        print(f"  wrote {out_path}  ({emb.shape[1]-1} dims)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
