from langchain_core.documents import Document

from rag.answer import format_context, parse_sources
from rag.prompts import FALLBACK_TEXT


def make_doc(title="How to Apply", heading="Application Fee", url="https://x.test/apply/"):
    return Document(
        page_content="The application fee is $90.",
        metadata={
            "chunk_id": "chunk-00258",
            "page_title": title,
            "heading_path": heading,
            "url": url,
            "n_tokens": 10,
            "source": f"{title} ({url})",
        },
    )


def test_parse_sources_splits_on_final_marker():
    raw = "Tuition is $6,384 per course.\nSOURCES: Program Tuition (https://x.test/t/)"
    answer, sources = parse_sources(raw)
    assert answer == "Tuition is $6,384 per course."
    assert sources == "Program Tuition (https://x.test/t/)"


def test_parse_sources_dedupes_preserving_order():
    raw = (
        "Answer body.\n"
        "SOURCES: B (https://x.test/b/), A (https://x.test/a/), B (https://x.test/b/)"
    )
    _, sources = parse_sources(raw)
    assert sources == "B (https://x.test/b/), A (https://x.test/a/)"


def test_parse_sources_handles_missing_marker():
    raw = "Answer with no sources line."
    answer, sources = parse_sources(raw)
    assert answer == "Answer with no sources line."
    assert sources == ""


def test_parse_sources_suppresses_sources_on_fallback():
    raw = f"{FALLBACK_TEXT}\nSOURCES: Irrelevant Page (https://x.test/z/)"
    answer, sources = parse_sources(raw)
    assert FALLBACK_TEXT in answer
    assert sources == ""


def test_parse_sources_uses_last_marker_when_repeated():
    raw = "Discussing SOURCES: in prose.\nSOURCES: Real Page (https://x.test/r/)"
    answer, sources = parse_sources(raw)
    assert sources == "Real Page (https://x.test/r/)"
    assert "in prose" in answer


def test_format_context_includes_provenance():
    context = format_context([make_doc()])
    assert "How to Apply" in context
    assert "Application Fee" in context
    assert "https://x.test/apply/" in context
    assert "The application fee is $90." in context


def test_format_context_separates_multiple_docs():
    context = format_context([make_doc(title="A"), make_doc(title="B")])
    assert context.count("---") == 2
