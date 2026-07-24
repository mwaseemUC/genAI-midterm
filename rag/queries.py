"""Shared LLM query expansion for the RAG-Fusion and Multi-Query pipelines.

Both expand a question into variants and differ only in prompt and count, so
the implementation lives here once.
"""

from __future__ import annotations

import re

# Leading list markers only. The original rag_fusion.py used
# .strip("-*0123456789. "), which strips both ends and would turn
# "TOEFL minimum score 102" into "TOEFL minimum score".
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def clean_query_line(line: str) -> str:
    """Strip list markers and wrapping quotes from one generated query."""
    return _LIST_MARKER.sub("", line.strip()).strip().strip('"').strip("'").strip()


def expand(llm, question: str, prompt_template: str, n_variants: int) -> list[str]:
    """Return [original question] + up to n_variants unique LLM reformulations.

    The template must accept {n} and {question}. Deduplication is
    case-insensitive and drops any variant that merely echoes the original.
    """
    raw = llm.invoke(prompt_template.format(n=n_variants, question=question)).content

    question = question.strip()
    queries = [question]
    seen = {question.lower()}

    for line in raw.splitlines():
        query = clean_query_line(line)
        if not query or query.lower() in seen:
            continue
        seen.add(query.lower())
        queries.append(query)
        if len(queries) == n_variants + 1:
            break

    return queries
