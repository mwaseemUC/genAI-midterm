"""Streamlit chat UI for the UChicago MS-ADS RAG chatbot.

This module is UI wiring only. All retrieval and answering logic lives in rag/.
"""

from __future__ import annotations

import os

import streamlit as st
from langchain_openai import ChatOpenAI

from rag import condense as condense_module
from rag import fusion, hyde, multiquery
from rag.answer import generate
from rag.store import CHUNKS_PATH, build_store, load_chunks

ANSWER_MODEL = "gpt-4o-mini"
QUERY_MODEL = "gpt-4o-mini"

PIPELINES = {
    "RAG-Fusion": {
        "retrieve": fusion.retrieve,
        "caption": (
            "Expands your question into 5 phrasings, searches each, and merges "
            "the results with Reciprocal Rank Fusion. Passages that rank well "
            "across several phrasings win."
        ),
        "queries_label": "Queries searched",
    },
    "HyDE": {
        "retrieve": hyde.retrieve,
        "caption": (
            "Writes a hypothetical answer first, then searches using that text "
            "instead of your question. The hypothetical is never shown as fact — "
            "it is only a search key."
        ),
        "queries_label": "Hypothetical document used",
    },
    "Multi-Query": {
        "retrieve": multiquery.retrieve,
        "caption": (
            "Expands your question into 4 phrasings and takes the deduplicated "
            "union of their results, interleaved. No rank fusion."
        ),
        "queries_label": "Queries searched",
    },
}

STARTERS = [
    "What does the program cost?",
    "How do I apply?",
    "What are the English language requirements?",
    "Is there an application fee waiver?",
]

st.set_page_config(
    page_title="MS-ADS Assistant — UChicago",
    page_icon="🎓",
    layout="centered",
)


def resolve_api_key() -> str | None:
    """Streamlit secrets first, then the environment. Never hardcoded."""
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except FileNotFoundError:
        pass  # no secrets.toml locally — fall through to the env var
    return os.environ.get("OPENAI_API_KEY")


@st.cache_resource(show_spinner="Building the knowledge base…")
def get_store():
    """Embed the corpus once per container. Cached for the process lifetime."""
    return build_store(load_chunks(CHUNKS_PATH))


@st.cache_resource
def get_llms():
    """Answer LLM is deterministic; query LLM is slightly creative for variety."""
    return (
        ChatOpenAI(model=ANSWER_MODEL, temperature=0, max_retries=5),
        ChatOpenAI(model=QUERY_MODEL, temperature=0.3, max_retries=5),
    )


def render_sources(sources: str) -> None:
    if not sources:
        return
    st.markdown("**Sources**")
    for part in [s.strip() for s in sources.split(",") if s.strip()]:
        if "(" in part and part.endswith(")"):
            title, _, url = part.rpartition("(")
            st.markdown(f"- [{title.strip()}]({url.rstrip(')')})")
        else:
            st.markdown(f"- {part}")


def render_details(message: dict) -> None:
    """Retrieved passages and the queries used — the rubric's 'visually present
    retrieved information' requirement, and how each pipeline stays inspectable."""
    docs = message.get("docs", [])
    if docs:
        with st.expander(f"Retrieved passages ({len(docs)})"):
            for i, doc in enumerate(docs, start=1):
                meta = doc.metadata
                st.markdown(
                    f"**{i}. {meta.get('page_title', 'Untitled')}** — "
                    f"`{meta.get('heading_path', '')}`"
                )
                st.markdown(f"[{meta.get('url', '')}]({meta.get('url', '')})")
                st.text(doc.page_content)
                st.divider()

    queries = message.get("queries", [])
    if queries:
        label = message.get("queries_label", "Queries searched")
        with st.expander(f"{label} ({len(queries)})"):
            for query in queries:
                st.markdown(f"- {query}")


def answer_question(question: str, pipeline_name: str) -> dict:
    """Condense against history, retrieve with the selected pipeline, answer."""
    store = get_store()
    answer_llm, query_llm = get_llms()
    pipeline = PIPELINES[pipeline_name]

    standalone = condense_module.condense(
        query_llm, question, st.session_state.messages
    )
    docs, queries = pipeline["retrieve"](store, query_llm, standalone)
    result = generate(answer_llm, standalone, docs)

    return {
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "docs": docs,
        "queries": queries,
        "queries_label": pipeline["queries_label"],
        "pipeline": pipeline_name,
    }


# --- Session state -----------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

# --- Sidebar -----------------------------------------------------------------

with st.sidebar:
    st.title("MS-ADS Assistant")
    st.caption(
        "Answers questions about the University of Chicago MS in Applied "
        "Data Science program, grounded in the program website."
    )
    st.divider()

    pipeline_name = st.radio(
        "Retrieval method",
        list(PIPELINES),
        index=0,
        help="How the assistant searches the knowledge base before answering.",
    )
    st.caption(PIPELINES[pipeline_name]["caption"])

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()

    st.divider()
    st.caption(
        "**Scope:** program site, Physical Sciences Division policies, the "
        "Booth MBA/MS joint degree, and university English-language requirements."
    )

# --- Guard: API key ----------------------------------------------------------

api_key = resolve_api_key()
if not api_key:
    st.error(
        "**No OpenAI API key found.**\n\n"
        "- **Deployed:** add `OPENAI_API_KEY` in the Streamlit Cloud app settings "
        "under *Secrets*.\n"
        "- **Local:** set the `OPENAI_API_KEY` environment variable before "
        "running `streamlit run app.py`."
    )
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# --- Guard: corpus -----------------------------------------------------------

try:
    load_chunks(CHUNKS_PATH)
except Exception as error:  # noqa: BLE001 — any corpus fault is fatal; make it legible
    st.error(
        "**Could not load the knowledge base.**\n\n"
        f"Expected a valid JSONL corpus at `{CHUNKS_PATH}`.\n\n"
        f"```\n{error}\n```\n\n"
        "Regenerate it with `mid_project/src/preprocess.py` and copy the result "
        "to `data/chunks.jsonl`."
    )
    st.stop()

# --- Main pane ---------------------------------------------------------------

st.title("Ask about the MS in Applied Data Science")
st.caption(
    "Admissions, tuition, curriculum, deadlines, and student life — with a "
    "citation for every answer."
)

if not st.session_state.messages:
    st.markdown("**Try one of these:**")
    columns = st.columns(2)
    for i, starter in enumerate(STARTERS):
        if columns[i % 2].button(starter, use_container_width=True, key=f"start{i}"):
            st.session_state.pending = starter
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", ""))
            render_details(message)
            st.caption(f"Answered using {message.get('pipeline', 'unknown')}")

# --- Handle input ------------------------------------------------------------

typed = st.chat_input("Ask a question…")
question = typed or st.session_state.pending
st.session_state.pending = None

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the program site…"):
                message = answer_question(question, pipeline_name)
        except Exception as error:  # noqa: BLE001 — surface any failure, keep the session alive
            st.error(f"Something went wrong answering that question: {error}")
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append(message)
            st.rerun()
