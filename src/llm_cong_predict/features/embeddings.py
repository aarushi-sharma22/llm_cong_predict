"""Essay embeddings: RoBERTa (generated natively) and GPT (reshaped from a file).

Ports of ``get_roberta_embeddings`` and ``get_gpt_embeddings`` from
``R/functions.R``.

RUNNABLE-BUT-NOT-TESTED-HERE: ``roberta_embeddings`` needs the RoBERTa model
weights and the real essays, so it is not exercised in the dev sandbox. It IS a
full, faithful translation that runs once those inputs exist. Install extras with
``pip install -e '.[embeddings]'`` (torch + transformers).

VALIDATION: numerical agreement with the R output is deferred (needs real essays);
the RoBERTa pooling is matched to the R exactly (see note below).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ROBERTA_DIM = 768  # roberta-base hidden size (R output columns roberta_dim_1..768)


def roberta_embeddings(essays: pd.DataFrame, max_len: int = 250, text_col: str = "text") -> pd.DataFrame:
    """Port of ``get_roberta_embeddings(essays, max_len = 250)``.

    R behaviour reproduced:
      * tokenise each essay with the ``roberta-base`` tokenizer, ``max_length=250``,
        truncation on, padded to max length;
      * run the frozen ``roberta-base`` model;
      * take the MEAN of the last hidden state over ALL token positions
        (``tf$reduce_mean(..., axis=1)``) — note the R passes only input_ids and does
        NOT mask padding, so padding positions ARE included in the mean. We match
        that exactly (a mask-weighted mean would give different numbers).
      * return a frame ``id`` + ``roberta_dim_1 .. roberta_dim_768``.

    Implemented with PyTorch/transformers (the R used TF via reticulate); the model
    weights are identical, so this is a faithful translation modulo framework. The
    ``do_lower_case=True`` from the R is preserved.
    """
    try:
        import torch
        from transformers import RobertaModel, RobertaTokenizer
    except Exception as exc:  # pragma: no cover - optional heavy deps
        raise ImportError(
            "roberta_embeddings needs torch + transformers. Install with: "
            "pip install -e '.[embeddings]'"
        ) from exc

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
    model = RobertaModel.from_pretrained("roberta-base")
    model.eval()
    for p in model.parameters():  # trainable = FALSE
        p.requires_grad_(False)

    texts = essays[text_col].astype(str).tolist()
    enc = tokenizer(
        texts,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    with torch.no_grad():
        # Pass input_ids only, matching the R (no attention_mask -> padding attended).
        out = model(input_ids=enc["input_ids"])
        # mean over the token axis (dim=1), including padding positions, as in R.
        emb = out.last_hidden_state.mean(dim=1).cpu().numpy()

    cols = [f"roberta_dim_{i}" for i in range(1, ROBERTA_DIM + 1)]
    result = pd.DataFrame(emb, columns=cols)
    result.insert(0, "id", essays["ncdsid"].values)
    return result


def gpt_embeddings(path: str, id_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Port of ``get_gpt_embeddings`` (the reshaper), reading a saved embeddings file.

    THE ORIGINAL IS SELF-CONTRADICTORY (flagged, PORTING_NOTES G1): the R body does
    ``readRDS(essays)`` (treating the argument as a PATH to the raw OpenAI-response
    RDS saved by ``get_gpt_embeddings.R``) but then ``bind_cols(essays, .)`` (treating
    the same argument as the essays data frame). ``.rds`` is also an R-only binary
    format. We therefore replace the RDS round-trip with a Python-native embeddings
    file produced by ``scripts/get_gpt_embeddings.py`` (a Parquet with an ``ncdsid``
    column plus ``embedding_*`` columns), and return the reshaper's INTENDED output:
    a frame ``id`` + embedding columns.

    Parameters
    ----------
    path:
        Parquet/CSV written by the ported generation script, containing ``ncdsid``
        and one column per embedding dimension.
    id_frame:
        Unused; accepted for signature parity with callers that pass the essays.
    """
    if path.endswith(".parquet"):
        try:
            df = pd.read_parquet(path)
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "Reading .parquet needs pyarrow (pip install pyarrow), or save the "
                "embeddings as .csv instead."
            ) from exc
    else:
        df = pd.read_csv(path)

    if "ncdsid" not in df.columns:
        raise ValueError(
            "gpt_embeddings expects a saved embeddings file with an 'ncdsid' column "
            "plus embedding columns (see scripts/get_gpt_embeddings.py). "
            f"Got columns: {list(df.columns)[:8]}..."
        )
    return df.rename(columns={"ncdsid": "id"})
