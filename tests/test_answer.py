from langchain_core.documents import Document

from rag.answer import format_context, generate_stream, parse_sources
from rag.prompts import FALLBACK_TEXT
from tests.fakes import FakeStreamingLLM


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


# --- streaming ---------------------------------------------------------------


def test_stream_yields_the_answer_body():
    llm = FakeStreamingLLM(["The fee ", "is $90.", "\nSOURCES: A (https://x.test/a/)"])
    sink: dict = {}
    streamed = "".join(generate_stream(llm, "Fee?", [make_doc()], sink))
    assert streamed.strip() == "The fee is $90."


def test_stream_never_emits_the_sources_marker():
    llm = FakeStreamingLLM(["Body.", "\nSOURCES: A (https://x.test/a/)"])
    sink: dict = {}
    streamed = "".join(generate_stream(llm, "Q?", [make_doc()], sink))
    assert "SOURCES" not in streamed


def test_stream_hides_a_marker_split_across_chunks():
    """The marker can arrive as "SOU" + "RCES:" — a naive per-chunk check leaks it."""
    llm = FakeStreamingLLM(["Body.\n", "SOU", "RCES:", " A (https://x.test/a/)"])
    sink: dict = {}
    streamed = "".join(generate_stream(llm, "Q?", [make_doc()], sink))
    assert "SOURCES" not in streamed
    assert "SOU" not in streamed.replace("Body.", "")


def test_stream_populates_the_sink_with_the_parsed_result():
    llm = FakeStreamingLLM(["Body.", "\nSOURCES: A (https://x.test/a/)"])
    sink: dict = {}
    list(generate_stream(llm, "Q?", [make_doc()], sink))
    assert sink["answer"] == "Body."
    assert sink["sources"] == "A (https://x.test/a/)"


def test_stream_handles_a_response_with_no_sources_line():
    llm = FakeStreamingLLM(["Just an answer."])
    sink: dict = {}
    streamed = "".join(generate_stream(llm, "Q?", [make_doc()], sink))
    assert streamed.strip() == "Just an answer."
    assert sink["sources"] == ""


def test_stream_returns_the_fallback_without_calling_the_llm_when_no_docs():
    llm = FakeStreamingLLM(["should not be used"])
    sink: dict = {}
    streamed = "".join(generate_stream(llm, "Q?", [], sink))
    assert streamed == FALLBACK_TEXT
    assert sink == {"answer": FALLBACK_TEXT, "sources": ""}
    assert llm.prompts == []
