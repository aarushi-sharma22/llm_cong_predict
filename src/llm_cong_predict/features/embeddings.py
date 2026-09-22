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


def roberta_pool(input_ids, model) -> np.ndarray:
    """Mean of RoBERTa's last hidden state over ALL token positions, padding included.

    R: llm_paper/R/functions.R:L480–482 — the keras model takes only input ids (no
    attention mask, so padding positions are attended) and outputs
    ``tf$reduce_mean(roberta_model(input)[[1]], axis = 1L)``. A mask-weighted mean would
    give different numbers and is deliberately not used. Each row depends only on its
    own ids, so any batching gives the same numbers (tests/test_features.py).
    """
    import torch

    with torch.no_grad():
        out = model(input_ids=input_ids)
        return out.last_hidden_state.mean(dim=1).cpu().numpy()


def roberta_embeddings(
    essays: pd.DataFrame,
    max_len: int = 250,
    text_col: str = "text",
    batch_size: int = 32,
    model=None,
    tokenizer=None,
) -> pd.DataFrame:
    """Port of ``get_roberta_embeddings(essays, max_len = 250)`` (R: llm_paper/R/functions.R:L446–490).

    R behaviour reproduced:
      * tokenise each essay with the ``roberta-base`` tokenizer, ``max_length = 250``,
        truncation on, padded to max length (L454–455);
      * run the frozen ``roberta-base`` model (L476–478);
      * :func:`roberta_pool` on every essay;
      * return a frame ``id`` + ``roberta_dim_1 .. roberta_dim_768`` (L487–489).

    Essays go through the model ``batch_size`` at a time (brief F8: a single forward
    pass over ~10,000 essays of 250 tokens runs out of memory). The R's keras
    ``predict()`` also runs in batches. ``model`` and ``tokenizer`` may be passed in;
    by default the public ``roberta-base`` weights are loaded locally. No essay text
    leaves the machine. Implemented with PyTorch (the R used TensorFlow through
    reticulate). ``do_lower_case=True`` from the R is passed through.
    """
    try:
        import torch  # noqa: F401
        from transformers import RobertaModel, RobertaTokenizer
    except Exception as exc:  # pragma: no cover - optional heavy deps
        raise ImportError(
            "roberta_embeddings needs torch + transformers. Install with: "
            "pip install -e '.[embeddings]'"
        ) from exc

    if tokenizer is None:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
    if model is None:
        model = RobertaModel.from_pretrained("roberta-base")
    model.eval()
    for p in model.parameters():  # trainable = FALSE (L478)
        p.requires_grad_(False)

    texts = essays[text_col].astype(str).tolist()
    enc = tokenizer(texts, max_length=max_len, truncation=True, padding="max_length",
                    return_tensors="pt")
    ids = enc["input_ids"]
    emb = np.concatenate([roberta_pool(ids[i:i + batch_size], model)
                          for i in range(0, ids.shape[0], batch_size)], axis=0)

    cols = [f"roberta_dim_{i}" for i in range(1, emb.shape[1] + 1)]
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
