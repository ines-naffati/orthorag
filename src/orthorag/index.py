"""Build and query the retrieval indexes: BM25 (lexical) and ChromaDB (dense), fused with RRF."""
from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass

import chromadb
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from . import config
from .chunking import Chunk, chunk_document
from .embed import embed
from .parse import load_parsed

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or that the
this to was were which with within device devices system systems""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def collection_name(model: str) -> str:
    return "chunks_" + re.sub(r"[^a-zA-Z0-9]", "_", model)


@dataclass
class Hit:
    chunk_id: str
    k_number: str
    section: str
    raw: str
    device_name: str
    score: float


class Retriever:
    """Lexical, dense or hybrid retrieval over the chunked corpus."""

    def __init__(self, embed_model: str = config.DEFAULT_EMBED_MODEL):
        self.embed_model = embed_model
        with (config.INDEX_DIR / "chunks.jsonl").open() as f:
            self.chunks = [Chunk(**json.loads(line)) for line in f]
        self.by_id = {c.chunk_id: c for c in self.chunks}
        with (config.INDEX_DIR / "bm25.pkl").open("rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)
        client = chromadb.PersistentClient(path=str(config.INDEX_DIR / "chroma"))
        self.collection = client.get_collection(collection_name(embed_model))

    # --- individual retrievers -------------------------------------------------
    def bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.chunks[i].chunk_id, float(scores[i])) for i in top]

    def dense_search(self, query: str, k: int, exclude_k: str | None = None) -> list[tuple[str, float]]:
        qvec = embed([query], self.embed_model, kind="query")[0]
        where = {"k_number": {"$ne": exclude_k}} if exclude_k else None
        res = self.collection.query(query_embeddings=[qvec], n_results=k, where=where)
        ids, dists = res["ids"][0], res["distances"][0]
        return [(i, 1.0 - d) for i, d in zip(ids, dists)]  # cosine similarity

    # --- public API ------------------------------------------------------------
    def search(self, query: str, k: int = 8, mode: str = "hybrid",
               exclude_k: str | None = None, pool: int = 100, rrf_k: int = 60) -> list[Hit]:
        """Return top-k chunks. `exclude_k` removes the query document itself (benchmark)."""
        def keep(cid: str) -> bool:
            return exclude_k is None or self.by_id[cid].k_number != exclude_k

        if mode == "bm25":
            ranked = [(c, s) for c, s in self.bm25_search(query, pool) if keep(c)]
        elif mode == "dense":
            ranked = self.dense_search(query, pool, exclude_k)
        elif mode == "hybrid":
            # Reciprocal Rank Fusion: robust to the different score scales of BM25 and cosine.
            fused: dict[str, float] = {}
            for results in (
                [(c, s) for c, s in self.bm25_search(query, pool) if keep(c)],
                self.dense_search(query, pool, exclude_k),
            ):
                for rank, (cid, _) in enumerate(results):
                    fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            ranked = sorted(fused.items(), key=lambda x: -x[1])
        else:
            raise ValueError(f"unknown mode {mode!r}")
        hits = []
        for cid, score in ranked[:k]:
            c = self.by_id[cid]
            hits.append(Hit(cid, c.k_number, c.section, c.raw, c.device_name, score))
        return hits


def build_index(models: list[str] | None = None) -> int:
    """Chunk all parsed documents, then build BM25 and one Chroma collection per embedding model."""
    models = models or config.EMBED_MODELS
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks: list[Chunk] = []
    for doc in load_parsed():
        chunks.extend(chunk_document(doc))

    with (config.INDEX_DIR / "chunks.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict()) + "\n")
    with (config.INDEX_DIR / "bm25.pkl").open("wb") as f:
        pickle.dump(BM25Okapi([tokenize(c.text) for c in chunks]), f)

    client = chromadb.PersistentClient(path=str(config.INDEX_DIR / "chroma"))
    for model in models:
        name = collection_name(model)
        try:
            client.delete_collection(name)
        except Exception:
            pass
        col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
        step = 64
        for i in tqdm(range(0, len(chunks), step), desc=f"embed {model}"):
            batch = chunks[i : i + step]
            col.add(
                ids=[c.chunk_id for c in batch],
                embeddings=embed([c.text for c in batch], model, kind="doc"),
                documents=[c.raw for c in batch],
                metadatas=[{"k_number": c.k_number, "section": c.section} for c in batch],
            )
    return len(chunks)
