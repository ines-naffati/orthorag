"""Central configuration: paths, model names, defaults."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("ORTHORAG_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # downloaded PDFs
PARSED_DIR = DATA_DIR / "parsed"    # extracted text + sections (JSON)
INDEX_DIR = DATA_DIR / "index"      # ChromaDB + BM25
METADATA_FILE = DATA_DIR / "metadata.jsonl"
RESULTS_DIR = ROOT / "results"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODELS = ["nomic-embed-text", "mxbai-embed-large"]
DEFAULT_EMBED_MODEL = os.environ.get("ORTHORAG_EMBED", "nomic-embed-text")
LLM_MODEL = os.environ.get("ORTHORAG_LLM", "llama3.2:3b")

# Orthopedic product codes (FDA classification):
#   HRS = single/multiple component metallic bone fixation appliances (plates)
#   HWC = smooth or threaded metallic bone fixation fastener (screws)
PRODUCT_CODES = ["HRS", "HWC"]

CHUNK_SIZE = 900       # characters
CHUNK_OVERLAP = 150
