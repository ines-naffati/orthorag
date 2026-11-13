"""Download public 510(k) summaries for orthopedic devices from the FDA.

Metadata comes from the openFDA device/510k endpoint; PDFs come from
accessdata.fda.gov. Both are public U.S. government data.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

from . import config

OPENFDA_URL = "https://api.fda.gov/device/510k.json"
K_RE = re.compile(r"^K(\d{2})\d{4}$")
HEADERS = {"User-Agent": "orthorag-research/0.1 (educational project)"}


def pdf_url(k_number: str) -> str | None:
    """Return the FDA PDF URL for a K-number, or None for pre-2002 files (no PDFs)."""
    m = K_RE.match(k_number)
    if not m:
        return None
    yy = int(m.group(1))
    if yy >= 90:  # 1990s submissions have no online summary
        return None
    return f"https://www.accessdata.fda.gov/cdrh_docs/pdf{yy}/{k_number}.pdf"


def query_openfda(product_codes: list[str], since: str, limit: int) -> list[dict]:
    """Fetch 510(k) records that published a *Summary* (not a Statement), newest first."""
    codes = "+OR+".join(f"product_code:{c}" for c in product_codes)
    search = f"({codes})+AND+statement_or_summary:Summary+AND+decision_date:[{since}+TO+2099-12-31]"
    records: list[dict] = []
    skip = 0
    while len(records) < limit:
        page = min(100, limit - len(records))
        url = f"{OPENFDA_URL}?search={search}&sort=decision_date:desc&limit={page}&skip={skip}"
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 404:  # no more results
            break
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            break
        records.extend(results)
        skip += page
        time.sleep(0.3)
    return records


def lookup_k_numbers(k_numbers: list[str]) -> list[dict]:
    """Fetch openFDA records for specific K-numbers (used to add predicate devices)."""
    out = []
    for i in range(0, len(k_numbers), 25):
        batch = k_numbers[i : i + 25]
        search = "+OR+".join(f"k_number:{k}" for k in batch)
        r = requests.get(f"{OPENFDA_URL}?search={search}&limit=100", headers=HEADERS, timeout=30)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        out.extend(r.json().get("results", []))
        time.sleep(0.3)
    return out


def slim(record: dict) -> dict:
    keys = ["k_number", "device_name", "applicant", "decision_date",
            "product_code", "statement_or_summary", "decision_description"]
    return {k: record.get(k) for k in keys}


def download_pdf(k_number: str, dest_dir: Path) -> Path | None:
    dest = dest_dir / f"{k_number}.pdf"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = pdf_url(k_number)
    if url is None:
        return None
    r = requests.get(url, headers=HEADERS, timeout=60)
    if r.status_code != 200 or not r.content.startswith(b"%PDF"):
        return None
    dest.write_bytes(r.content)
    time.sleep(0.3)
    return dest


def load_metadata() -> dict[str, dict]:
    if not config.METADATA_FILE.exists():
        return {}
    with config.METADATA_FILE.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["k_number"]: r for r in rows}


def save_metadata(meta: dict[str, dict]) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with config.METADATA_FILE.open("w") as f:
        for rec in sorted(meta.values(), key=lambda r: r["k_number"]):
            f.write(json.dumps(rec) + "\n")


def fetch_seed(limit: int, since: str) -> dict[str, dict]:
    """Stage 1: the most recent `limit` orthopedic 510(k) summaries."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()
    records = query_openfda(config.PRODUCT_CODES, since, limit)
    for rec in tqdm(records, desc="seed PDFs"):
        k = rec["k_number"]
        if download_pdf(k, config.RAW_DIR):
            meta[k] = {**slim(rec), "source": "seed"}
    save_metadata(meta)
    return meta


def fetch_predicates(max_new: int) -> dict[str, dict]:
    """Stage 2: add predicate devices cited by the seed documents (one hop).

    This makes the predicate-retrieval benchmark meaningful: a query document's
    predicates must be in the corpus for retrieval to be able to find them.
    Requires `parse` to have run on the seed documents.
    """
    from .parse import load_parsed

    meta = load_metadata()
    cited: dict[str, int] = {}
    for doc in load_parsed():
        for k in doc["predicates"]:
            if k not in meta:
                cited[k] = cited.get(k, 0) + 1
    wanted = [k for k, _ in sorted(cited.items(), key=lambda x: -x[1])][:max_new]
    records = {r["k_number"]: r for r in lookup_k_numbers(wanted)}
    for k in tqdm(wanted, desc="predicate PDFs"):
        if download_pdf(k, config.RAW_DIR):
            rec = records.get(k, {"k_number": k})
            meta[k] = {**slim(rec), "k_number": k, "source": "predicate"}
    save_metadata(meta)
    return meta
