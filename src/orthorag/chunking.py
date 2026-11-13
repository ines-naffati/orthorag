"""Section-aware chunking with a contextual header on every chunk."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from . import config

SECTION_TITLES = {
    "device_description": "Device description",
    "indications": "Indications for use",
    "predicates": "Predicate devices",
    "technology": "Technological characteristics",
    "performance": "Performance data",
    "conclusion": "Conclusion",
    "body": "Summary",
}
_K_RE = re.compile(r"\bK\s?\d{6}\b")


@dataclass
class Chunk:
    chunk_id: str
    k_number: str
    section: str
    text: str            # text that is embedded / indexed (with header)
    raw: str             # original passage, shown to the LLM and the user
    device_name: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def split_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Split on paragraph/sentence boundaries into ~`size`-character windows with overlap."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    pieces = re.split(r"(?<=[.;:])\s+|\n{2,}", text)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if len(current) + len(piece) + 1 <= size:
            current = f"{current} {piece}".strip()
            continue
        if current:
            chunks.append(current)
            current = current[-overlap:] + " " + piece if overlap else piece
        else:  # single piece longer than size: hard-wrap
            for i in range(0, len(piece), size - overlap):
                chunks.append(piece[i : i + size])
            current = ""
    if current.strip():
        chunks.append(current.strip())
    return chunks


def mask_k_numbers(text: str) -> str:
    """Hide K-numbers so retrieval cannot match on identifiers (prevents benchmark leakage)."""
    return _K_RE.sub("K######", text)


def chunk_document(doc: dict, mask_ids: bool = False) -> list[Chunk]:
    sections = doc.get("sections") or {}
    if not sections:
        sections = {"body": doc["text"]}
    name = doc.get("device_name") or ""
    out: list[Chunk] = []
    for section, body in sections.items():
        for i, piece in enumerate(split_text(body)):
            raw = mask_k_numbers(piece) if mask_ids else piece
            header = f"{name} — {SECTION_TITLES.get(section, section)}\n"
            out.append(Chunk(
                chunk_id=f"{doc['k_number']}:{section}:{i}",
                k_number=doc["k_number"],
                section=section,
                text=header + raw,
                raw=raw,
                device_name=name,
            ))
    return out
