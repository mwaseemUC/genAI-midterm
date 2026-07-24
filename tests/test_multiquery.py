from rag.multiquery import retrieve, round_robin_union
from tests.fakes import FakeLLM, FakeVectorStore, doc


def test_round_robin_takes_first_from_each_query_before_seconds():
    a1, a2, b1, b2 = doc("a1"), doc("a2"), doc("b1"), doc("b2")
    results = {"qa": [a1, a2], "qb": [b1, b2]}
    merged = round_robin_union(results, ["qa", "qb"], k_per_query=2, top_n=4)
    assert [d.metadata["chunk_id"] for d in merged] == ["a1", "b1", "a2", "b2"]


def test_round_robin_dedupes_by_chunk_id():
    shared = doc("shared")
    results = {"qa": [shared], "qb": [shared]}
    merged = round_robin_union(results, ["qa", "qb"], k_per_query=1, top_n=4)
    assert len(merged) == 1


def test_round_robin_respects_top_n():
    results = {"qa": [doc("a"), doc("b"), doc("c")]}
    merged = round_robin_union(results, ["qa"], k_per_query=3, top_n=2)
    assert len(merged) == 2


def test_round_robin_handles_uneven_result_lengths():
    results = {"qa": [doc("a1"), doc("a2")], "qb": [doc("b1")]}
    merged = round_robin_union(results, ["qa", "qb"], k_per_query=2, top_n=4)
    assert [d.metadata["chunk_id"] for d in merged] == ["a1", "b1", "a2"]


def test_retrieve_returns_four_queries_including_the_original():
    llm = FakeLLM(["paraphrase\nkeywords\ndetails"])
    store = FakeVectorStore({}, default=[doc("a")])
    _, queries = retrieve(store, llm, "Is there a fee waiver?")
    assert queries[0] == "Is there a fee waiver?"
    assert len(queries) == 4


def test_retrieve_returns_eight_docs_by_default():
    llm = FakeLLM(["one\ntwo\nthree"])
    pool = [doc(f"c{i}") for i in range(10)]
    store = FakeVectorStore({}, default=pool)
    docs, queries = retrieve(store, llm, "Q?")
    assert len(docs) == 8
    assert len(queries) == 4


def test_retrieve_uses_k_per_query_of_eight():
    llm = FakeLLM(["one\ntwo\nthree"])
    store = FakeVectorStore({}, default=[doc("a")])
    retrieve(store, llm, "Q?")
    assert all(k == 8 for _, k in store.calls)
