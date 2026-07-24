from rag.queries import clean_query_line, expand
from tests.fakes import FakeLLM

TEMPLATE = "Write {n} queries for: {question}"


def test_clean_strips_leading_numbering():
    assert clean_query_line("1. first query") == "first query"
    assert clean_query_line("2) second query") == "second query"


def test_clean_strips_leading_bullets():
    assert clean_query_line("- bulleted query") == "bulleted query"
    assert clean_query_line("* starred query") == "starred query"


def test_clean_preserves_trailing_digits():
    """Regression: the original strip() removed trailing digits, so a query
    like this lost the number that matters most."""
    assert clean_query_line("TOEFL minimum score 102") == "TOEFL minimum score 102"


def test_clean_strips_wrapping_quotes():
    assert clean_query_line('"quoted query"') == "quoted query"


def test_expand_prepends_original():
    llm = FakeLLM(["paraphrase one\nkeyword two"])
    queries = expand(llm, "What is tuition?", TEMPLATE, 2)
    assert queries == ["What is tuition?", "paraphrase one", "keyword two"]


def test_expand_drops_blank_lines():
    llm = FakeLLM(["alpha\n\n\nbeta"])
    assert expand(llm, "Q?", TEMPLATE, 4)[1:] == ["alpha", "beta"]


def test_expand_dedupes_case_insensitively():
    llm = FakeLLM(["Alpha\nalpha\nbeta"])
    assert expand(llm, "Q?", TEMPLATE, 4)[1:] == ["Alpha", "beta"]


def test_expand_excludes_echo_of_the_original_question():
    llm = FakeLLM(["What is tuition?\nreal variant"])
    queries = expand(llm, "What is tuition?", TEMPLATE, 3)
    assert queries == ["What is tuition?", "real variant"]


def test_expand_caps_at_n_variants():
    llm = FakeLLM(["one\ntwo\nthree\nfour\nfive"])
    assert len(expand(llm, "Q?", TEMPLATE, 2)) == 3


def test_expand_formats_the_prompt_template():
    llm = FakeLLM(["variant"])
    expand(llm, "What is tuition?", TEMPLATE, 4)
    assert llm.prompts[0] == "Write 4 queries for: What is tuition?"
