from rag.fusion import reciprocal_rank_fusion, retrieve
from tests.fakes import FakeLLM, FakeVectorStore, doc


def test_rrf_rewards_consistency_across_lists():
    """A doc ranked 2nd in both lists beats one ranked 1st in only one."""
    a, b, c = doc("a"), doc("b"), doc("c")
    merged = reciprocal_rank_fusion([[a, b], [c, b]])
    assert [d.metadata["chunk_id"] for d, _ in merged][0] == "b"


def test_rrf_scores_match_formula():
    merged = reciprocal_rank_fusion([[doc("a")]], k=60)
    assert merged[0][1] == 1.0 / 61


def test_rrf_sums_scores_for_repeated_docs():
    a = doc("a")
    merged = reciprocal_rank_fusion([[a], [a]], k=60)
    assert len(merged) == 1
    assert merged[0][1] == 2 * (1.0 / 61)


def test_rrf_returns_each_document_once():
    a, b = doc("a"), doc("b")
    assert len(reciprocal_rank_fusion([[a, b], [b, a]])) == 2


def test_rrf_handles_empty_input():
    assert reciprocal_rank_fusion([]) == []


def test_retrieve_returns_top_n_and_queries():
    llm = FakeLLM(["alt one\nalt two"])
    store = FakeVectorStore({}, default=[doc("a"), doc("b"), doc("c")])
    docs, queries = retrieve(store, llm, "What is tuition?", k_per_query=3, top_n=2)
    assert len(docs) == 2
    assert queries[0] == "What is tuition?"
    assert len(queries) == 3


def test_retrieve_searches_once_per_query():
    llm = FakeLLM(["alt one\nalt two"])
    store = FakeVectorStore({}, default=[doc("a")])
    retrieve(store, llm, "Q?", k_per_query=8, top_n=8)
    assert len(store.calls) == 3
    assert all(k == 8 for _, k in store.calls)


def test_retrieve_defaults_to_eight_docs():
    llm = FakeLLM(["one\ntwo\nthree\nfour"])
    store = FakeVectorStore({}, default=[doc(f"c{i}") for i in range(10)])
    docs, queries = retrieve(store, llm, "Q?")
    assert len(docs) == 8
    assert len(queries) == 5
