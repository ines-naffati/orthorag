"""Extract text, sections and cited predicate devices from 510(k) summary PDFs."""
from __future__ import annotations

import json
import re
from pathlib import Path

from pypdf import PdfReader
from tqdm import tqdm

from . import config

K_NUMBER_RE = re.compile(r"\bK\s?(\d{6})\b")

# Canonical section -> heading patterns seen in 510(k) summaries.
SECTION_PATTERNS: dict[str, str] = {
    "device_description": r"device\s+description|description\s+of\s+(?:the\s+)?device",
    "indications": r"indications?\s+for\s+use|intended\s+use",
    "predicates": r"predicate\s+devices?|legally\s+marketed\s+devices?|primary\s+predicate",
    "technology": r"technolog(?:ical|y)\s+characteristics|comparison\s+(?:to|with)\s+(?:the\s+)?predicate|substantial\s+equivalence",
    "performance": r"performance\s+(?:data|testing)|non-?clinical\s+(?:testing|data|performance)|bench\s+testing",
    "conclusion": r"conclusions?",
}
_HEADING_RE = re.compile(
    r"(?im)^[ \t]*(?:[0-9IVX]+[.)]\s*|[A-H][.)]\s*)?(" +
    "|".join(f"(?P<{name}>{pat})" for name, pat in SECTION_PATTERNS.items()) +
    r")\s*[:.\-]?"
)
SUMMARY_START_RE = re.compile(r"(?i)510\s*\(\s*k\s*\)\s+summary")


def normalize_k(raw: str) -> str:
    return "K" + raw


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n".join(pages)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_clearance_letter(text: str) -> str:
    """PDFs start with the FDA clearance letter + IFU form; keep from the 510(k) Summary on."""
    matches = list(SUMMARY_START_RE.finditer(text))
    # The phrase can also appear in the letter; take the last occurrence in the first 70% of the doc.
    candidates = [m for m in matches if m.start() < 0.7 * len(text)]
    if candidates:
        return text[candidates[-1].start():]
    return text


def split_sections(text: str) -> dict[str, str]:
    """Split summary text into canonical sections (first occurrence of each heading wins)."""
    hits = []
    seen = set()
    for m in _HEADING_RE.finditer(text):
        name = next(k for k, v in m.groupdict().items() if v and k in SECTION_PATTERNS)
        if name in seen:
            continue
        seen.add(name)
        hits.append((m.start(), m.end(), name))
    sections: dict[str, str] = {}
    for i, (start, end, name) in enumerate(hits):
        stop = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        body = text[end:stop].strip()
        if body:
            sections[name] = body
    return sections


def cited_k_numbers(text: str, own_k: str) -> list[str]:
    """All K-numbers cited in the summary other than the device's own, in order of appearance."""
    seen: list[str] = []
    for m in K_NUMBER_RE.finditer(text):
        k = normalize_k(m.group(1))
        if k != own_k and k not in seen:
            seen.append(k)
    return seen


def parse_pdf(pdf_path: Path, meta: dict | None = None) -> dict | None:
    k = pdf_path.stem
    try:
        full = extract_text(pdf_path)
    except Exception:  # corrupted / encrypted PDF
        return None
    if len(full) < 500:  # scanned image without a text layer
        return None
    summary = strip_clearance_letter(full)
    return {
        "k_number": k,
        "device_name": (meta or {}).get("device_name"),
        "applicant": (meta or {}).get("applicant"),
        "decision_date": (meta or {}).get("decision_date"),
        "product_code": (meta or {}).get("product_code"),
        "text": summary,
        "sections": split_sections(summary),
        "predicates": cited_k_numbers(summary, k),
    }


def parse_all() -> int:
    from .fetch import load_metadata

    meta = load_metadata()
    config.PARSED_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for pdf in tqdm(sorted(config.RAW_DIR.glob("K*.pdf")), desc="parse"):
        out = config.PARSED_DIR / f"{pdf.stem}.json"
        if out.exists():
            n += 1
            continue
        doc = parse_pdf(pdf, meta.get(pdf.stem))
        if doc:
            out.write_text(json.dumps(doc))
            n += 1
    return n


def load_parsed() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(config.PARSED_DIR.glob("K*.json"))]
