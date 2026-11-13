"""Thin client for local embedding models served by Ollama."""
from __future__ import annotations

import requests

from . import config

# Each model was trained with task-specific prefixes; using them matters for retrieval quality.
QUERY_PREFIX = {
    "nomic-embed-text": "search_query: ",
    "mxbai-embed-large": "Represent this sentence for searching relevant passages: ",
}
DOC_PREFIX = {
    "nomic-embed-text": "search_document: ",
    "mxbai-embed-large": "",
}


def embed(texts: list[str], model: str, kind: str = "doc", batch_size: int = 32) -> list[list[float]]:
    prefix = (DOC_PREFIX if kind == "doc" else QUERY_PREFIX).get(model, "")
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[i : i + batch_size]]
        r = requests.post(
            f"{config.OLLAMA_URL}/api/embed",
            json={"model": model, "input": batch, "truncate": True},
            timeout=300,
        )
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
    return out
