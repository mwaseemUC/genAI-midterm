"""Context assembly and grounded answer generation.

This is the ONLY place the answer LLM is called, which is what makes the
shared prompt structurally guaranteed rather than three copies that drift.
"""

from __future__ import annotations

from typing import Iterator

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from rag.prompts import ANSWER_SYSTEM_PROMPT, FALLBACK_TEXT

_MARKER = "SOURCES:"


def format_context(docs: list[Document]) -> str:
    """Render retrieved chunks with provenance headers the LLM can cite."""
    blocks = []
    for doc in docs:
        title = doc.metadata.get("page_title", "Untitled")
        heading = doc.metadata.get("heading_path", "")
        url = doc.metadata.get("url", "")
        blocks.append(f"--- {title} | {heading}\nURL: {url}\n{doc.page_content}")
    return "\n\n".join(blocks)


def parse_sources(raw: str) -> tuple[str, str]:
    """Split the answer body from its trailing SOURCES line.

    Splits on the LAST occurrence so the marker appearing in prose does not
    truncate the answer. Deduplicates while preserving order, and suppresses
    sources entirely when the model returned the no-answer fallback.
    """
    answer, marker, sources = raw.rpartition("SOURCES:")
    if not marker:
        return raw.strip(), ""

    answer = answer.strip()
    parts = [part.strip() for part in sources.split(",") if part.strip()]
    sources = ", ".join(dict.fromkeys(parts))

    if FALLBACK_TEXT in answer:
        sources = ""

    return answer, sources


def _build_messages(question: str, docs: list[Document]) -> list:
    """The one place the shared prompt is bound to a question and its context."""
    return [
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Context passages:\n\n{format_context(docs)}\n\nQuestion: {question}"
        ),
    ]


def generate(answer_llm, question: str, docs: list[Document]) -> dict:
    """Produce a grounded answer from retrieved documents.

    Returns {"answer": str, "sources": str}.
    """
    if not docs:
        return {"answer": FALLBACK_TEXT, "sources": ""}

    raw = answer_llm.invoke(_build_messages(question, docs)).content
    answer, sources = parse_sources(raw)
    return {"answer": answer, "sources": sources}


def generate_stream(
    answer_llm, question: str, docs: list[Document], sink: dict
) -> Iterator[str]:
    """Stream the answer body, then fill `sink` with the parsed result.

    Yields only display text: the trailing "SOURCES: ..." line is withheld so
    the citation markup never flickers into view mid-answer. Because a token
    boundary can split the marker itself, the last len("SOURCES:") characters
    are always held back until more text arrives or the stream ends.

    On completion `sink` holds {"answer": str, "sources": str} — the same shape
    generate() returns, so callers store one message format either way.
    """
    if not docs:
        sink.update({"answer": FALLBACK_TEXT, "sources": ""})
        yield FALLBACK_TEXT
        return

    raw = ""
    emitted = 0

    for chunk in answer_llm.stream(_build_messages(question, docs)):
        raw += chunk.content
        cut = raw.find(_MARKER)
        # No marker yet: everything but a possible partial marker is safe to show.
        safe = cut if cut != -1 else max(0, len(raw) - len(_MARKER))
        if safe > emitted:
            yield raw[emitted:safe]
            emitted = safe

    answer, sources = parse_sources(raw)
    sink.update({"answer": answer, "sources": sources})

    # Flush whatever the hold-back withheld, but never past the marker.
    tail_end = raw.find(_MARKER)
    tail_end = len(raw) if tail_end == -1 else tail_end
    if tail_end > emitted:
        yield raw[emitted:tail_end]
