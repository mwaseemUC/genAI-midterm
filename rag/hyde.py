"""HyDE retrieval: Hypothetical Document Embeddings.

Ported from Midterm_Group_1_EN_v2.ipynb Section 9. The LLM writes a plausible
but not necessarily correct answer, and that text — not the user's question —
is embedded and searched. A hypothetical answer sits closer in embedding space
to real answer passages than a question does.

The fabricated content never reaches the user: it is used only as a retrieval
query. rag/answer.py generates the actual answer from real retrieved passages.
"""

from __future__ import annotations

from langchain_core.documents import Document

from rag.prompts import HYDE_PROMPT


def generate_hypothetical(llm, question: str) -> str:
    """Write a plausible, not-necessarily-correct answer used only for retrieval."""
    return llm.invoke(HYDE_PROMPT.format(question=question)).content


def retrieve(
    vectorstore,
    llm,
    question: str,
    k_per_query: int = 8,
    top_n: int = 8,
) -> tuple[list[Document], list[str]]:
    """Generate a hypothetical answer, then retrieve against its embedding.

    Returns (top_n documents, [the hypothetical document] — for display).
    Only one search is issued, so k_per_query and top_n are both honoured by
    requesting max(k_per_query, top_n) and truncating.
    """
    hypothetical = generate_hypothetical(llm, question)
    docs = vectorstore.similarity_search(hypothetical, k=max(k_per_query, top_n))
    return docs[:top_n], [hypothetical]
