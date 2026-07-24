"""Minimal fakes so retrieval logic can be tested without network calls."""

from langchain_core.documents import Document


def doc(chunk_id: str, title: str = "Page", url: str = "https://x.test/p/") -> Document:
    """Build a Document with the metadata shape the app relies on."""
    return Document(
        page_content=f"Body of {chunk_id}.",
        metadata={
            "chunk_id": chunk_id,
            "page_title": title,
            "heading_path": "Section",
            "url": url,
            "n_tokens": 10,
            "source": f"{title} ({url})",
        },
    )


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Returns queued responses in order and records the prompts it received."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.prompts: list = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("FakeLLM ran out of queued responses")
        return FakeResponse(self._responses.pop(0))


class FakeVectorStore:
    """Returns a preset document list per query string.

    Unknown queries return the `default` list, so tests only need to specify
    the queries whose results actually matter.
    """

    def __init__(self, results_by_query: dict[str, list[Document]], default=None):
        self._results = results_by_query
        self._default = default if default is not None else []
        self.calls: list[tuple[str, int]] = []

    def similarity_search(self, query: str, k: int = 4):
        self.calls.append((query, k))
        return self._results.get(query, self._default)[:k]
