"""Context assembly and grounded answer generation.

This is the ONLY place the answer LLM is called, which is what makes the
shared prompt structurally guaranteed rather than three copies that drift.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from rag.prompts import ANSWER_SYSTEM_PROMPT, FALLBACK_TEXT


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


def generate(answer_llm, question: str, docs: list[Document]) -> dict:
    """Produce a grounded answer from retrieved documents.

    Returns {"answer": str, "sources": str}.
    """
    if not docs:
        return {"answer": FALLBACK_TEXT, "sources": ""}

    messages = [
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Context passages:\n\n{format_context(docs)}\n\nQuestion: {question}"
        ),
    ]
    raw = answer_llm.invoke(messages).content
    answer, sources = parse_sources(raw)
    return {"answer": answer, "sources": sources}
