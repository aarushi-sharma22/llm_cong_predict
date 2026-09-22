#!/usr/bin/env python
"""Generate GPT essay embeddings via the OpenAI API — kept for provenance only.

Port of ``R/get_gpt_embeddings.R``. THIS SCRIPT SENDS ESSAY TEXT TO AN EXTERNAL
SERVICE (OpenAI). The project rule is that essays and any other participant data
must never be sent to an external API (PORTING_NOTES H2). The
script therefore refuses to run unless BOTH of these are given:

  * the command-line flag  --i-confirm-the-data-licence-permits-external-processing
  * the environment variable  LCP_ALLOW_EXTERNAL_API=1

It is not called by the pipeline or by any other code in this repository.

Faithful to the original:
  * batches essays in groups of 100 (R: llm_paper/R/get_gpt_embeddings.R:L14–17);
  * two embedding sets, ``text-embedding-ada-002`` ("GPT 3.5",
    R: llm_paper/R/get_gpt_embeddings.R:L22) and ``text-embedding-3-large``
    ("GPT 4", R: llm_paper/R/get_gpt_embeddings.R:L43).

DEVIATION (PORTING_NOTES G1): the R saved raw API responses as ``.rds``. This script
saves a CSV with an ``ncdsid`` column plus ``embedding_*`` columns, the shape
``features.embeddings.gpt_embeddings`` reads (CSV, so that reading the embeddings needs
no Parquet library). Essays are read from and embeddings
written to ``$LCP_DATA_ROOT`` (``config.RESTRICTED_INPUTS``), never the repository.

Requires the separate extra:  pip install -e '.[external-api]'  (openai)
"""

from __future__ import annotations

import argparse
import os
import sys

CONFIRM_FLAG = "--i-confirm-the-data-licence-permits-external-processing"
ALLOW_ENV = "LCP_ALLOW_EXTERNAL_API"

REFUSAL = (
    "get_gpt_embeddings.py: REFUSED. This script sends essay text to the OpenAI API, an "
    "external service. The project rule is that essays and any other participant data "
    "must never be sent to an external API. It runs only if you both "
    f"pass {CONFIRM_FLAG} and set {ALLOW_ENV}=1, after confirming that the data licence "
    "permits external processing."
)

# The two models, matching the original script exactly.
MODELS = {
    "gpt35": "text-embedding-ada-002",  # R: llm_paper/R/get_gpt_embeddings.R:L22
    "gpt4": "text-embedding-3-large",  # R: llm_paper/R/get_gpt_embeddings.R:L43
}
BATCH_SIZE = 100  # R: llm_paper/R/get_gpt_embeddings.R:L14 (ceiling(seq/100))


def external_processing_confirmed(argv: list[str], environ: dict[str, str]) -> bool:
    """True only when the flag AND the environment variable are both present."""
    return CONFIRM_FLAG in argv and environ.get(ALLOW_ENV) == "1"


def _batched(items: list, n: int):
    for i in range(0, len(items), n):
        yield i // n, items[i : i + n]


def generate(essays, model: str, api_key: str):
    """Return a frame: ncdsid + embedding_1..K for the given model."""
    import pandas as pd
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


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # The gate runs before anything else: no essay is read and openai is not imported.
    if not external_processing_confirmed(argv, dict(os.environ)):
        print(REFUSAL, file=sys.stderr)
        return 3

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from llm_cong_predict import config
    from llm_cong_predict.io.readers import read_essays

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(CONFIRM_FLAG, action="store_true", dest="confirmed")
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = ap.parse_args(argv)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY in the environment (do not hard-code it).", file=sys.stderr)
        return 2

    essays = read_essays(str(config.restricted_path("essays")))
    for key in args.models:
        model = MODELS[key]
        out_path = config.restricted_path(f"{key}_embeddings")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Generating {key} embeddings ({model}) for {len(essays)} essays...")
        emb = generate(essays, model, api_key)
        emb.to_csv(out_path, index=False)
        print(f"  wrote {out_path}  ({emb.shape[1]-1} dims)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
