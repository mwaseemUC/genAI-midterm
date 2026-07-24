"""Multi-Query retrieval: query expansion merged by round-robin union.

Ported from Midterm_Group_1_MultiQuery.ipynb. Unlike RAG-Fusion there is no
rank fusion and no reranking — this is the deduplicated union of each query's
results, interleaved so no single query monopolises the context.

Note the budget change from the notebook: k_per_query was 4 there, which left
as few as 2 chunks reaching the answer model. It is 8 here so all three
pipelines get an equal budget and retrieval strategy is the only variable.
"""

from __future__ import annotations

from langchain_core.documents import Document

from rag.prompts import MULTIQUERY_PROMPT
from rag.queries import expand

N_VARIANTS = 3


def round_robin_union(
    results_by_query: dict[str, list[Document]],
    queries: list[str],
    k_per_query: int,
    top_n: int,
) -> list[Document]:
    """Interleave per-query results, deduplicating on chunk_id.

    Takes each query's first-ranked result, then each query's second, and so
    on, so one query cannot fill every context slot.
    """
    unique: list[Document] = []
    seen: set[str] = set()

    for rank in range(k_per_query):
        for query in queries:
            docs = results_by_query.get(query, [])
            if rank >= len(docs):
                continue

            document = docs[rank]
            key = document.metadata.get("chunk_id") or document.page_content[:80]
            if key not in seen:
                seen.add(key)
                unique.append(document)

            if len(unique) >= top_n:
                return unique

    return unique


def retrieve(
    vectorstore,
    llm,
    question: str,
    k_per_query: int = 8,
    top_n: int = 8,
) -> tuple[list[Document], list[str]]:
    """Expand the question, retrieve per query, merge by round-robin union.

    Returns (top_n documents, the queries used — for display).
    """
    queries = expand(llm, question, MULTIQUERY_PROMPT, N_VARIANTS)
    results_by_query = {
        query: vectorstore.similarity_search(query, k=k_per_query) for query in queries
    }
    docs = round_robin_union(results_by_query, queries, k_per_query, top_n)
    return docs, queries
