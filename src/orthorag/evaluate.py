"""Predicate-retrieval benchmark.

Every 510(k) summary names the *predicate devices* it claims equivalence to. That gives
free, expert-made relevance labels: given a new device's description, a good retriever
should surface the summaries of its predicates. No manual annotation needed.

Query  = device description + indications for use of a held-out summary
         (K-numbers masked, predicate section never used -> no identifier leakage)
Labels = cited K-numbers that are present in the corpus
Corpus = all other summaries (the query document itself is excluded)
"""
from __future__ import annotations

import math
import random
import re

import pandas as pd
from tqdm import tqdm

from . import config
from .chunking import mask_k_numbers
from .index import Retriever
from .parse import load_parsed

QUERY_SECTIONS = ("device_description", "indications")
MAX_QUERY_CHARS = 1500


GENERIC_WORDS = set("""system systems plate plates plating screw screws fixation bone implant implants
device devices locking cannulated orthopaedic orthopedic orthopaedics orthopedics the and for with of inc llc
ltd corp corporation co gmbh sa medical surgical""".split())


def mask_names(text: str, doc: dict) -> str:
    """Remove brand and manufacturer words (e.g. 'Quick-Start', 'Innovate') from a query.

    Predicates are often earlier versions of the same product line, so brand names make
    retrieval easy. The 'hard' setting forces the retriever to match on technical content.
    """
    words = set()
    for field in ("device_name", "applicant"):
        for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", doc.get(field) or ""):
            if w.lower() not in GENERIC_WORDS and len(w) > 2:
                words.add(w)
    for w in sorted(words, key=len, reverse=True):
        text = re.sub(rf"(?i)\b{re.escape(w)}\b", "[NAME]", text)
    return text


def build_queries(docs: list[dict], max_queries: int | None = None, seed: int = 0,
                  hard: bool = False) -> list[dict]:
    corpus = {d["k_number"] for d in docs}
    queries = []
    for d in docs:
        relevant = [k for k in d["predicates"] if k in corpus]
        text = " ".join(d["sections"].get(s, "") for s in QUERY_SECTIONS).strip()
        if relevant and len(text) > 200:
            text = mask_k_numbers(text)
            if hard:
                text = mask_names(text, d)
            queries.append({
                "k_number": d["k_number"],
                "query": text[:MAX_QUERY_CHARS],
                "relevant": relevant,
            })
    random.Random(seed).shuffle(queries)
    return queries[:max_queries] if max_queries else queries


def rank_documents(hits) -> list[str]:
    """Collapse a chunk ranking into a document ranking (best chunk per document)."""
    seen: list[str] = []
    for h in hits:
        if h.k_number not in seen:
            seen.append(h.k_number)
    return seen


def metrics(ranking: list[str], relevant: list[str], ks=(1, 5, 10)) -> dict[str, float]:
    rel = set(relevant)
    out = {}
    for k in ks:
        out[f"recall@{k}"] = len(rel & set(ranking[:k])) / len(rel)
    out["hit@5"] = float(bool(rel & set(ranking[:5])))
    first = next((i for i, k in enumerate(ranking) if k in rel), None)
    out["mrr"] = 0.0 if first is None else 1.0 / (first + 1)
    dcg = sum(1.0 / math.log2(i + 2) for i, k in enumerate(ranking[:10]) if k in rel)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), 10)))
    out["ndcg@10"] = dcg / idcg
    return out


def run_benchmark(max_queries: int | None = None, models: list[str] | None = None,
                  pool: int = 100) -> pd.DataFrame:
    docs = load_parsed()
    models = models or config.EMBED_MODELS
    systems = [("bm25", models[0])] + [(m, e) for e in models for m in ("dense", "hybrid")]
    retrievers = {e: Retriever(e) for e in models}
    rows = []
    n_queries = 0
    for setting, hard in (("standard", False), ("hard (names masked)", True)):
        queries = build_queries(docs, max_queries, hard=hard)
        n_queries = len(queries)
        for mode, emb in systems:
            r = retrievers[emb]
            name = "BM25" if mode == "bm25" else f"{mode} ({emb})"
            for q in tqdm(queries, desc=f"{setting} · {name}"):
                hits = r.search(q["query"], k=pool, mode=mode, exclude_k=q["k_number"], pool=pool)
                rows.append({"setting": setting, "system": name, "query": q["k_number"],
                             "n_relevant": len(q["relevant"]),
                             **metrics(rank_documents(hits), q["relevant"])})
    per_query = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(config.RESULTS_DIR / "retrieval_per_query.csv", index=False)
    summary = summarize(per_query)
    summary.to_csv(config.RESULTS_DIR / "retrieval_benchmark.csv")
    (config.RESULTS_DIR / "retrieval_benchmark.md").write_text(
        f"Queries: {n_queries} per setting · corpus: {len(docs)} summaries\n\n"
        + summary.to_markdown(floatfmt=".3f")
    )
    plot(summary)
    return summary


def summarize(per_query: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Mean of each metric plus a bootstrap 95% CI on MRR."""
    cols = ["recall@1", "recall@5", "recall@10", "hit@5", "mrr", "ndcg@10"]
    summary = per_query.groupby(["setting", "system"], sort=False)[cols].mean()
    rng = random.Random(seed)
    lo, hi = [], []
    for setting, system in summary.index:
        mask = (per_query.setting == setting) & (per_query.system == system)
        vals = per_query.loc[mask, "mrr"].tolist()
        boots = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(n_boot))
        lo.append(boots[int(0.025 * n_boot)])
        hi.append(boots[int(0.975 * n_boot)])
    summary["mrr_ci_low"], summary["mrr_ci_high"] = lo, hi
    return summary


def plot(summary: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = summary.index.get_level_values(0).unique()
    systems = list(summary.index.get_level_values(1).unique())
    colors = ["#3b6ea8", "#d08a3c"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    height = 0.38
    for j, setting in enumerate(settings):
        s = summary.loc[setting].reindex(systems)
        ys = [i + (j - 0.5) * height for i in range(len(systems))]
        err = [s["mrr"] - s["mrr_ci_low"], s["mrr_ci_high"] - s["mrr"]]
        ax.barh(ys, s["mrr"], height=height, xerr=err, color=colors[j], capsize=2, label=setting)
    ax.set_yticks(range(len(systems)), systems)
    ax.invert_yaxis()
    ax.set_xlabel("MRR — predicate retrieval (95% bootstrap CI)")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "retrieval_benchmark.png", dpi=160)
