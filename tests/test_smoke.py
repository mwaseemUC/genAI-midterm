"""Live end-to-end tests. Skipped automatically when no API key is present.

Run explicitly:
    conda run -n msads-app python -m pytest tests/test_smoke.py -v
"""

import os

import pytest
from langchain_openai import ChatOpenAI

from rag import fusion, hyde, multiquery
from rag.answer import generate
from rag.store import CHUNKS_PATH, build_store, load_chunks

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping live tests",
)


@pytest.fixture(scope="module")
def store():
    return build_store(load_chunks(CHUNKS_PATH))


@pytest.fixture(scope="module")
def llms():
    return (
        ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=5),
        ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_retries=5),
    )


@pytest.mark.parametrize("pipeline", [fusion, hyde, multiquery])
def test_pipeline_answers_with_sources(store, llms, pipeline):
    answer_llm, query_llm = llms
    docs, queries = pipeline.retrieve(store, query_llm, "What is tuition for the program?")

    assert len(docs) == 8, "every pipeline must return the same context budget"
    assert queries

    result = generate(answer_llm, "What is tuition for the program?", docs)
    assert "6,384" in result["answer"]
    assert result["sources"]


@pytest.mark.parametrize("pipeline", [fusion, hyde, multiquery])
def test_no_pipeline_reports_the_toefl_typo(store, llms, pipeline):
    """The site says "102 or 5 (current requirement)". A score of 5 is impossible.

    In the notebooks HyDE and Multi-Query both repeated it. The shared hardened
    prompt must stop that in every pipeline.
    """
    question = "What are the minimum scores for the TOEFL and IELTS requirement?"
    answer_llm, query_llm = llms
    docs, _ = pipeline.retrieve(store, query_llm, question)
    answer = generate(answer_llm, question, docs)["answer"]

    assert "102" in answer
    assert "or 5 " not in answer.replace("(current requirement)", "")


def test_off_topic_question_is_declined(store, llms):
    answer_llm, query_llm = llms
    docs, _ = fusion.retrieve(store, query_llm, "Who won the 2022 World Cup?")
    answer = generate(answer_llm, "Who won the 2022 World Cup?", docs)["answer"]
    assert "Argentina" not in answer
