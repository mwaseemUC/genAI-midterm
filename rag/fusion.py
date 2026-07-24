"""RAG-Fusion retrieval: multi-query expansion merged by Reciprocal Rank Fusion.

Ported from mid_project/src/rag_fusion.py, with answer generation removed —
that now lives in rag/answer.py so all pipelines share one prompt.
"""

from __future__ import annotations

from langchain_core.documents import Document

from rag.prompts import QUERY_GEN_PROMPT
from rag.queries import expand

N_VARIANTS = 4


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], k: int = 60
) -> list[tuple[Document, float]]:
    """Merge ranked Document lists into one list scored by RRF.

    Score of a doc = sum over lists of 1 / (k + rank), rank starting at 1.
    Documents are identified by chunk_id. k=60 is the constant from the
    original RRF paper (Cormack et al.).
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, Document] = {}

    for docs in ranked_lists:
        for rank, document in enumerate(docs, start=1):
            key = document.metadata.get("chunk_id") or document.page_content[:80]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(key, document)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(first_seen[key], score) for key, score in ordered]


def retrieve(
    vectorstore,
    llm,
    question: str,
    k_per_query: int = 8,
    top_n: int = 8,
) -> tuple[list[Document], list[str]]:
    """Expand the question, retrieve per query, merge by RRF.

    Returns (top_n documents, the queries used — for display).
    """
    queries = expand(llm, question, QUERY_GEN_PROMPT, N_VARIANTS)
    ranked_lists = [vectorstore.similarity_search(q, k=k_per_query) for q in queries]
    fused = reciprocal_rank_fusion(ranked_lists)
    return [document for document, _score in fused[:top_n]], queries
