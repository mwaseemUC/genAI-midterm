"""History-aware query rewriting.

Retrieval is stateless: "what about the online program?" embeds only those
five words and loses the topic. This condenses a follow-up plus recent history
into a standalone question before retrieval runs.
"""

from __future__ import annotations

from rag.prompts import CONDENSE_PROMPT


def _format_history(history: list[dict], max_turns: int) -> str:
    """Render the most recent exchanges as plain text for the prompt."""
    recent = history[-(max_turns * 2) :]
    lines = []
    for message in recent:
        speaker = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {message['content']}")
    return "\n".join(lines)


def condense(llm, question: str, history: list[dict], max_turns: int = 4) -> str:
    """Rewrite a follow-up into a standalone question.

    Returns the question unchanged when there is no history, so the first
    message of a session costs no extra LLM call.
    """
    if not history:
        return question

    prompt = CONDENSE_PROMPT.format(
        history=_format_history(history, max_turns), question=question
    )
    rewritten = llm.invoke(prompt).content.strip()
    return rewritten or question
