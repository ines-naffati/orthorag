"""Unit tests that run without network, Ollama or downloaded data."""
from orthorag.chunking import chunk_document, mask_k_numbers, split_text
from orthorag.evaluate import metrics
from orthorag.fetch import pdf_url
from orthorag.generate import check_citations
from orthorag.index import Hit, tokenize
from orthorag.parse import cited_k_numbers, split_sections, strip_clearance_letter

SAMPLE = """Dear Ms. Smith, we have reviewed your 510(k) premarket notification K231234.
510(k) Summary
Submitter: Example Ortho Inc.
Device Description:
The Example Distal Radius Plate System consists of titanium alloy plates and locking screws.
Indications for Use:
The system is indicated for fixation of fractures of the distal radius.
Predicate Devices:
Primary predicate: Acme Wrist Plate (K 201111). Reference device: K199999.
Performance Data:
Static and dynamic four-point bending tests were performed per ASTM F382.
Conclusion:
The subject device is substantially equivalent to the predicate devices."""


def test_pdf_url():
    assert pdf_url("K241234") == "https://www.accessdata.fda.gov/cdrh_docs/pdf24/K241234.pdf"
    assert pdf_url("K051747") == "https://www.accessdata.fda.gov/cdrh_docs/pdf5/K051747.pdf"
    assert pdf_url("K981234") is None
    assert pdf_url("DEN123") is None


def test_strip_letter_and_sections():
    summary = strip_clearance_letter(SAMPLE)
    assert summary.startswith("510(k) Summary")
    sections = split_sections(summary)
    assert "titanium alloy plates" in sections["device_description"]
    assert "distal radius" in sections["indications"]
    assert "ASTM F382" in sections["performance"]
    assert set(sections) >= {"device_description", "indications", "predicates", "performance", "conclusion"}


def test_cited_k_numbers_excludes_self():
    assert cited_k_numbers(SAMPLE, "K231234") == ["K201111", "K199999"]


def test_mask_k_numbers():
    assert mask_k_numbers("predicate K201111 and K 199999") == "predicate K###### and K######"


def test_split_text_respects_size():
    text = " ".join(f"Sentence number {i} about bone plates." for i in range(200))
    chunks = split_text(text, size=300, overlap=50)
    assert len(chunks) > 5
    assert all(len(c) <= 360 for c in chunks)


def test_chunk_document_headers():
    doc = {"k_number": "K231234", "device_name": "Example Plate", "text": "x",
           "sections": {"device_description": "Titanium plate."}}
    [c] = chunk_document(doc)
    assert c.chunk_id == "K231234:device_description:0"
    assert c.text.startswith("Example Plate — Device description")


def test_tokenize_drops_stopwords():
    assert tokenize("The Device is a Titanium plate") == ["titanium", "plate"]


def test_metrics():
    m = metrics(["A", "B", "C"], ["B", "Z"])
    assert m["recall@1"] == 0 and m["recall@5"] == 0.5
    assert m["mrr"] == 0.5 and m["hit@5"] == 1.0
    assert 0 < m["ndcg@10"] < 1


def test_check_citations():
    hits = [Hit("K1:a:0", "K111111", "a", "", "", 1.0)]
    cited, invalid = check_citations("Plate [K111111]. Screw [K222222].", hits)
    assert cited == ["K111111", "K222222"] and invalid == ["K222222"]


def test_mask_names_keeps_technical_words():
    from orthorag.evaluate import mask_names
    doc = {"device_name": "Quick-Start Orthopaedic Fixation Screw", "applicant": "Innovate Orthopaedics, LLC"}
    out = mask_names("The Innovate Quick-Start screw is a titanium fixation screw.", doc)
    assert out == "The [NAME] [NAME] screw is a titanium fixation screw."
