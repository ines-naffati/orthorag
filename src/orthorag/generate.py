"""Draft 510(k)-style summary sections for a new device, grounded in retrieved cleared devices."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import requests

from . import config
from .index import Hit, Retriever

TARGET_SECTIONS = {
    "device_description": "Device description",
    "technology": "Technological characteristics compared with the predicate devices",
    "performance": "Performance data (non-clinical testing)",
}

SYSTEM_PROMPT = """You are a regulatory affairs assistant drafting sections of an FDA 510(k) summary
for an orthopedic implant. Write in the formal style of 510(k) summaries.
Rules:
- Use ONLY facts from the user's device description and from the SOURCES.
- After every sentence that relies on a source, cite it as [Kxxxxxx] using the source's K-number.
- If the sources do not support a statement, write [TO VERIFY] instead of inventing details.
- Never invent test results, standards numbers or K-numbers."""

_CITE_RE = re.compile(r"\[(K\d{6})\]")


@dataclass
class SectionDraft:
    section: str
    text: str
    sources: list[Hit]
    cited: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)


def format_sources(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{h.k_number}] {h.device_name} ({h.section}):\n{h.raw}" for h in hits)


def chat(messages: list[dict], model: str = config.LLM_MODEL) -> str:
    r = requests.post(
        f"{config.OLLAMA_URL}/api/chat",
        json={"model": model, "messages": messages, "stream": False,
              "options": {"temperature": 0.2, "num_ctx": 8192}},
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def select_sources(retriever: Retriever, query: str, section: str, n: int = 5) -> list[Hit]:
    """Prefer chunks from the same section type; fall back to the best chunks overall."""
    hits = retriever.search(query, k=60, mode="hybrid")
    same = [h for h in hits if h.section == section]
    picked = same[:n]
    if len(picked) < n:
        picked += [h for h in hits if h not in picked][: n - len(picked)]
    return picked


def check_citations(text: str, hits: list[Hit]) -> tuple[list[str], list[str]]:
    allowed = {h.k_number for h in hits}
    cited = list(dict.fromkeys(_CITE_RE.findall(text)))
    return cited, [k for k in cited if k not in allowed]


def draft(device_description: str, retriever: Retriever | None = None,
          sections: list[str] | None = None) -> list[SectionDraft]:
    retriever = retriever or Retriever()
    out = []
    for section in sections or list(TARGET_SECTIONS):
        title = TARGET_SECTIONS[section]
        hits = select_sources(retriever, device_description, section)
        user = (f"NEW DEVICE (written by the manufacturer):\n{device_description}\n\n"
                f"SOURCES (summaries of legally marketed orthopedic devices):\n{format_sources(hits)}\n\n"
                f"Write the section \"{title}\" of the 510(k) summary for the NEW DEVICE "
                f"(120-200 words). Cite sources as [Kxxxxxx].")
        text = chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}])
        cited, invalid = check_citations(text, hits)
        out.append(SectionDraft(section, text, hits, cited, invalid))
    return out
