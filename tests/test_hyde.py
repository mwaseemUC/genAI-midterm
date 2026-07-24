from rag.hyde import generate_hypothetical, retrieve
from tests.fakes import FakeLLM, FakeVectorStore, doc


def test_generate_hypothetical_returns_llm_text():
    llm = FakeLLM(["The minimum TOEFL score is 100."])
    assert generate_hypothetical(llm, "TOEFL?") == "The minimum TOEFL score is 100."


def test_generate_hypothetical_embeds_question_in_prompt():
    llm = FakeLLM(["hypothetical text"])
    generate_hypothetical(llm, "What is tuition?")
    assert "What is tuition?" in llm.prompts[0]


def test_retrieve_searches_on_hypothetical_not_question():
    """The defining behaviour of HyDE: the hypothetical answer is the query."""
    hypothetical = "Tuition is approximately $60,000 for the program."
    llm = FakeLLM([hypothetical])
    store = FakeVectorStore({hypothetical: [doc("a"), doc("b")]}, default=[])
    docs, queries = retrieve(store, llm, "What is tuition?", top_n=2)
    assert store.calls[0][0] == hypothetical
    assert [d.metadata["chunk_id"] for d in docs] == ["a", "b"]
    assert queries == [hypothetical]


def test_retrieve_returns_hypothetical_for_display():
    llm = FakeLLM(["a plausible fabricated answer"])
    store = FakeVectorStore({}, default=[doc("a")])
    _, queries = retrieve(store, llm, "Q?")
    assert queries == ["a plausible fabricated answer"]


def test_retrieve_respects_top_n():
    llm = FakeLLM(["hypo"])
    store = FakeVectorStore({"hypo": [doc(f"c{i}") for i in range(10)]})
    docs, _ = retrieve(store, llm, "Q?", k_per_query=8, top_n=8)
    assert len(docs) == 8


def test_retrieve_issues_exactly_one_search():
    llm = FakeLLM(["hypo"])
    store = FakeVectorStore({}, default=[doc("a")])
    retrieve(store, llm, "Q?")
    assert len(store.calls) == 1
