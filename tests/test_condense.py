from rag.condense import condense
from tests.fakes import FakeLLM


def test_condense_returns_question_unchanged_when_no_history():
    """First message of a session must not spend an LLM call."""
    llm = FakeLLM([])
    assert condense(llm, "What is tuition?", []) == "What is tuition?"
    assert llm.prompts == []


def test_condense_rewrites_follow_up_using_history():
    history = [
        {"role": "user", "content": "What is tuition?"},
        {"role": "assistant", "content": "$6,384 per course."},
    ]
    llm = FakeLLM(["What is the tuition for the MS-ADS Online program?"])
    result = condense(llm, "what about the online program?", history)
    assert result == "What is the tuition for the MS-ADS Online program?"


def test_condense_includes_history_in_prompt():
    history = [
        {"role": "user", "content": "What is tuition?"},
        {"role": "assistant", "content": "$6,384 per course."},
    ]
    llm = FakeLLM(["rewritten"])
    condense(llm, "what about online?", history)
    prompt = llm.prompts[0]
    assert "What is tuition?" in prompt
    assert "$6,384 per course." in prompt
    assert "what about online?" in prompt


def test_condense_limits_history_to_max_turns():
    history = []
    for i in range(10):
        history.append({"role": "user", "content": f"question {i}"})
        history.append({"role": "assistant", "content": f"answer {i}"})
    llm = FakeLLM(["rewritten"])
    condense(llm, "follow up", history, max_turns=2)
    prompt = llm.prompts[0]
    assert "question 9" in prompt
    assert "question 0" not in prompt


def test_condense_falls_back_to_original_on_empty_response():
    history = [{"role": "user", "content": "What is tuition?"}]
    llm = FakeLLM(["   "])
    assert condense(llm, "what about online?", history) == "what about online?"
