"""Streamlit chat UI for the UChicago MS-ADS RAG chatbot.

This module is UI wiring only. All retrieval and answering logic lives in rag/.
"""

from __future__ import annotations

import html
import os
import sys
import time

# ChromaDB hard-raises at import time when sqlite3 < 3.35, and its own hot-swap
# only runs under Colab. Streamlit Cloud's image can ship an older sqlite3, which
# would kill the app before it renders anything. pysqlite3-binary is a Linux-only
# requirement, so this is a no-op on Windows and macOS. It must run before any
# import that reaches chromadb — that is, before rag.store.
try:  # pragma: no cover - platform-dependent
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

from rag import condense as condense_module
from rag import fusion, hyde, multiquery
from rag.answer import citation_pairs, generate, generate_stream
from rag.store import CHUNKS_PATH, build_store, load_chunks

ANSWER_MODEL = "gpt-4o-mini"
QUERY_MODEL = "gpt-4o-mini"

PIPELINES = {
    "RAG-Fusion": {
        "retrieve": fusion.retrieve,
        "short": "Fusion",
        "description": "Multi-query, merged by rank",
        "caption": (
            "Rewrites your question into several phrasings, searches each one, then "
            "merges the results with Reciprocal Rank Fusion. Passages that rank well "
            "across several phrasings rise to the top."
        ),
        "queries_label": "Queries searched",
        "accent": ("#9A1B1B", "#F08A8A"),  # (light, dark)
    },
    "HyDE": {
        "retrieve": hyde.retrieve,
        "short": "HyDE",
        "description": "Hypothetical document embeddings",
        "caption": (
            "Writes a plausible answer first, then searches using that text instead of "
            "your question — a draft answer sits closer in embedding space to real "
            "answer passages than a question does. The draft is never shown as fact."
        ),
        "queries_label": "Hypothetical document used",
        "accent": ("#2B4ACB", "#8FB0FF"),
    },
    "Multi-Query": {
        "retrieve": multiquery.retrieve,
        "short": "Multi-Query",
        "description": "Multi-query, interleaved union",
        "caption": (
            "Rewrites your question into several phrasings and takes the deduplicated "
            "union of their results, interleaved so no single phrasing fills the "
            "context. No rank fusion."
        ),
        "queries_label": "Queries searched",
        "accent": ("#136B3A", "#6FD69B"),
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
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session state -----------------------------------------------------------

st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending", None)
st.session_state.setdefault("theme", "light")
st.session_state.setdefault("pipeline", "RAG-Fusion")
st.session_state.setdefault("compare", False)

IS_DARK = st.session_state.theme == "dark"

# --- Theme -------------------------------------------------------------------

if IS_DARK:
    BG, SURFACE, RAISED = "#0E1117", "#171A23", "#1F2430"
    TEXT, MUTED, BORDER = "#E8EAED", "#9AA0A6", "#2C313D"
else:
    BG, SURFACE, RAISED = "#FFFFFF", "#F7F7F8", "#FFFFFF"
    TEXT, MUTED, BORDER = "#1A1A1A", "#6B6B70", "#E4E4E7"

ACCENT = "#9A1B1B" if not IS_DARK else "#F08A8A"


def accent_for(pipeline_name: str) -> str:
    """The pipeline's accent colour for the current theme."""
    light, dark = PIPELINES[pipeline_name]["accent"]
    return dark if IS_DARK else light


st.markdown(
    f"""
<style>
  /* ---- Streamlit chrome: these render OUTSIDE .stApp, so they need naming
         explicitly or the page keeps a light header and footer in dark mode. */
  [data-testid="stHeader"],
  [data-testid="stToolbar"],
  [data-testid="stBottom"],
  [data-testid="stBottomBlockContainer"],
  [data-testid="stAppViewContainer"],
  .stApp {{
      background: {BG} !important;
  }}
  [data-testid="stHeader"] {{ border-bottom: 1px solid {BORDER}; }}

  [data-testid="stSidebar"],
  [data-testid="stSidebarContent"] {{
      background: {SURFACE} !important;
      border-right: 1px solid {BORDER};
  }}

  /* ---- Typography */
  .stApp, .stMarkdown, .stMarkdown p, .stMarkdown li,
  .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
      color: {TEXT};
  }}
  [data-testid="stCaptionContainer"], .stCaption, .stCaption p {{
      color: {MUTED} !important;
  }}
  .stMarkdown h1 {{ font-size: 2rem; font-weight: 650; letter-spacing: -0.02em; }}
  .stMarkdown a {{ color: {ACCENT}; }}

  /* Answers contain indented blocks (the mailing address), which markdown
     renders as code. Untouched, those stay light in dark mode. */
  .stMarkdown code, .stMarkdown pre, .stMarkdown pre code,
  [data-testid="stCode"], [data-testid="stCode"] pre {{
      background: {SURFACE} !important;
      color: {TEXT} !important;
      border-radius: 8px;
  }}
  .stMarkdown pre {{ border: 1px solid {BORDER}; }}

  /* ---- Chat */
  [data-testid="stChatMessage"] {{
      background: transparent;
      padding: 0.25rem 0 1.25rem 0;
  }}

  /* The chat bar is several nested elements deep and Base Web paints the inner
     one, so styling only the outer test id leaves a light pill in dark mode. */
  [data-testid="stBottom"],
  [data-testid="stBottom"] > div,
  [data-testid="stBottomBlockContainer"] {{
      background: {BG} !important;
  }}
  [data-testid="stChatInput"],
  [data-testid="stChatInputContainer"],
  [data-testid="stChatInput"] > div,
  [data-testid="stChatInput"] div[data-baseweb="textarea"],
  [data-testid="stChatInput"] div[data-baseweb="base-input"] {{
      background: {RAISED} !important;
      border-color: {BORDER} !important;
      border-radius: 12px;
  }}
  [data-testid="stChatInput"] textarea {{
      background: transparent !important;
      color: {TEXT} !important;
      -webkit-text-fill-color: {TEXT};
  }}
  [data-testid="stChatInput"] textarea::placeholder {{ color: {MUTED} !important; }}

  /* ---- Controls */
  .stButton button {{
      background: {RAISED};
      color: {TEXT};
      border: 1px solid {BORDER};
      border-radius: 10px;
      font-weight: 500;
      transition: border-color .15s ease, background .15s ease;
  }}
  .stButton button:hover {{ border-color: {ACCENT}; color: {TEXT}; }}
  .stButton button[kind="primary"] {{
      background: {ACCENT};
      border-color: {ACCENT};
      color: {"#1A1A1A" if IS_DARK else "#FFFFFF"};
  }}
  .stButton button[kind="primary"]:hover {{
      background: {ACCENT}; filter: brightness(1.05);
      color: {"#1A1A1A" if IS_DARK else "#FFFFFF"};
  }}

  /* An expander is <details><summary>; the summary paints its own background,
     which is why the header stayed light while the body went dark. */
  [data-testid="stExpander"],
  [data-testid="stExpander"] details,
  [data-testid="stExpander"] summary,
  [data-testid="stExpanderHeader"],
  [data-testid="stExpanderDetails"] {{
      background: {RAISED} !important;
      color: {TEXT} !important;
  }}
  [data-testid="stExpander"] details {{
      border: 1px solid {BORDER};
      border-radius: 10px;
      overflow: hidden;
  }}
  [data-testid="stExpander"] summary:hover {{ color: {ACCENT} !important; }}
  [data-testid="stExpander"] summary p,
  [data-testid="stExpander"] summary svg {{ color: inherit !important; fill: currentColor; }}

  /* Widget labels: the toggle caption stayed dark-on-dark without this. */
  [data-testid="stWidgetLabel"],
  [data-testid="stWidgetLabel"] p,
  [data-testid="stWidgetLabel"] label,
  label[data-baseweb="checkbox"] span {{
      color: {TEXT} !important;
  }}

  /* Popover panel (citation detail) */
  [data-baseweb="popover"] > div,
  [data-testid="stPopoverBody"] {{
      background: {RAISED} !important;
      border: 1px solid {BORDER};
  }}

  hr, [data-testid="stSidebar"] hr {{ border-color: {BORDER}; }}

  /* ---- Custom components */
  .chip {{
      display: inline-flex; align-items: center; gap: 7px;
      padding: 3px 11px 3px 9px;
      border-radius: 999px;
      font-size: 12px; font-weight: 550; letter-spacing: .01em;
      border: 1px solid; margin-bottom: 9px;
  }}
  .chip .dot {{
      width: 7px; height: 7px; border-radius: 50%;
      background: currentColor; flex: none;
  }}
  .chip .elapsed {{ color: {MUTED}; font-weight: 450; }}

  .active-pipeline {{
      border: 1px solid {BORDER}; border-left-width: 3px;
      border-radius: 10px; padding: 11px 13px; margin-bottom: 12px;
      background: {RAISED};
  }}
  .active-pipeline .label {{
      font-size: 10px; letter-spacing: .09em; text-transform: uppercase;
      color: {MUTED};
  }}
  .active-pipeline .name {{ font-size: 15px; font-weight: 600; margin-top: 3px; }}
  .active-pipeline .desc {{ font-size: 12px; color: {MUTED}; margin-top: 2px; }}

  /* ---- Citations: a popover trigger styled as a numbered source chip. */
  [data-testid="stPopover"] button {{
      background: {RAISED};
      border: 1px solid {BORDER};
      border-radius: 8px;
      padding: 5px 11px;
      font-size: 12.5px; font-weight: 500;
      color: {MUTED};
      text-align: left;
      justify-content: flex-start;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  [data-testid="stPopover"] button:hover {{
      border-color: {ACCENT}; color: {TEXT};
  }}
  [data-testid="stPopover"] button p {{
      font-size: 12.5px; color: inherit; margin: 0;
  }}

  .quote {{
      border-left: 2px solid {ACCENT};
      padding: 2px 0 2px 11px; margin: 6px 0 12px 0;
      font-size: 13px; line-height: 1.55; color: {TEXT};
      white-space: pre-wrap; word-break: break-word;
  }}

  /* The passage body lives INSIDE the card — rendering it with st.text()
     dropped it outside the border and broke the card in half. */
  .passage-card {{
      border: 1px solid {BORDER}; border-radius: 8px;
      padding: 11px 13px; margin-bottom: 10px; background: {RAISED};
  }}
  .passage-card .title {{ font-weight: 560; font-size: 13.5px; color: {TEXT}; }}
  .passage-card .crumb {{ font-size: 11.5px; color: {MUTED}; margin: 2px 0 7px 0; }}
  .passage-card .body {{
      font-size: 13px; line-height: 1.55; color: {TEXT};
      white-space: pre-wrap; word-break: break-word;
  }}

  /* Columns must be allowed to shrink or long words push content into the
     neighbouring column instead of wrapping. */
  [data-testid="stColumn"] {{
      min-width: 0;
      overflow-wrap: break-word;
  }}
  [data-testid="stVerticalBlockBorderWrapper"] {{
      border-color: {BORDER} !important;
      border-radius: 10px;
  }}
</style>
""",
    unsafe_allow_html=True,
)


# --- Resources ---------------------------------------------------------------


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


# --- Rendering ---------------------------------------------------------------


def render_chip(pipeline_name: str, elapsed: float | None = None) -> None:
    """Small pipeline tag above an answer. Deliberately quiet: it repeats on
    every message, so it must read as metadata, not as a heading."""
    colour = accent_for(pipeline_name)
    timing = (
        f'<span class="elapsed">· {elapsed:.1f}s</span>' if elapsed is not None else ""
    )
    st.markdown(
        f'<span class="chip" style="color:{colour};border-color:{colour}44;'
        f'background:{colour}14;"><span class="dot"></span>{pipeline_name}'
        f"{timing}</span>",
        unsafe_allow_html=True,
    )


def passages_from(docs: list, url: str) -> list:
    """The retrieved passages that came from one cited page."""
    return [doc for doc in docs if doc.metadata.get("url") == url]


def recover_citations(sources: str, docs: list) -> list[tuple[str, str]]:
    """Rescue citations from a SOURCES line that named pages without linking them.

    Any page title mentioned in the line that also appears among the retrieved
    documents gets its URL back, so the chip is still clickable. The URL comes
    from our own retrieval, never invented.
    """
    lowered = (sources or "").lower()
    recovered: list[tuple[str, str]] = []
    seen: set[str] = set()
    for doc in docs:
        title = str(doc.metadata.get("page_title", "")).strip()
        url = str(doc.metadata.get("url", "")).strip()
        if title and url and url not in seen and title.lower() in lowered:
            seen.add(url)
            recovered.append((title, url))
    return recovered


def render_citations(message: dict, key_prefix: str) -> None:
    """Numbered citation chips. Clicking one opens the evidence behind it.

    Prospective students are the audience, so a citation should answer "where
    did this come from?" in one click, not require reading a passage dump.
    """
    raw_sources = message.get("sources", "") or ""
    docs = message.get("docs", [])

    citations = citation_pairs(raw_sources) or recover_citations(raw_sources, docs)

    if not citations:
        # The model cited something we could not turn into links. Show its own
        # words rather than nothing — silently dropping a citation is worse than
        # an unstyled one.
        if raw_sources.strip():
            st.caption("Sources")
            st.markdown(raw_sources.strip())
        return

    st.caption("Sources")

    per_row = 3
    for start in range(0, len(citations), per_row):
        row = citations[start : start + per_row]
        columns = st.columns(per_row, gap="small")
        for offset, (title, url) in enumerate(row):
            number = start + offset + 1
            label = title if len(title) <= 26 else f"{title[:25]}…"
            with columns[offset]:
                with st.popover(
                    f"{number}  {label}",
                    use_container_width=True,
                ):
                    st.markdown(f"**{title}**")
                    supporting = passages_from(docs, url)
                    for doc in supporting:
                        heading = doc.metadata.get("heading_path", "")
                        if heading:
                            st.caption(heading)
                        st.markdown(
                            f'<div class="quote">{html.escape(doc.page_content)}</div>',
                            unsafe_allow_html=True,
                        )
                    if not supporting:
                        st.caption(
                            "Cited by the assistant; the passage came from this page."
                        )
                    st.markdown(f"[Open page ↗]({url})")


def render_provenance(message: dict, key_prefix: str) -> None:
    """The retrieval trace: what was searched and everything it returned.

    Kept behind one quiet expander. It satisfies the rubric's "visually present
    retrieved information" requirement and drives the pipeline demo, but a
    student asking about tuition should not have to scroll past it.
    """
    docs = message.get("docs", [])
    queries = message.get("queries", [])
    if not docs and not queries:
        return

    with st.expander("How this answer was found"):
        if queries:
            st.caption(message.get("queries_label", "Queries searched"))
            for i, query in enumerate(queries, start=1):
                st.markdown(f"{i}. {query}")
            st.write("")

        if docs:
            st.caption(f"Retrieved passages ({len(docs)})")
            for i, doc in enumerate(docs, start=1):
                meta = doc.metadata
                st.markdown(
                    f'<div class="passage-card">'
                    f'<div class="title">{i}. '
                    f'{html.escape(str(meta.get("page_title", "Untitled")))}</div>'
                    f'<div class="crumb">'
                    f'{html.escape(str(meta.get("heading_path", "")))}</div>'
                    f'<div class="body">{html.escape(doc.page_content)}</div>'
                    f'<a href="{html.escape(str(meta.get("url", "")))}" '
                    f'target="_blank" style="font-size:12px;color:{ACCENT};'
                    f'text-decoration:none;">Open page ↗</a></div>',
                    unsafe_allow_html=True,
                )


def render_assistant_message(message: dict, key_prefix: str = "") -> None:
    """One assistant turn: either a single answer or a three-way comparison."""
    comparisons = message.get("comparisons")

    if not comparisons:
        render_chip(message.get("pipeline", "unknown"), message.get("elapsed"))
        st.markdown(message["content"])
        render_citations(message, key_prefix)
        render_provenance(message, key_prefix)
        return

    st.caption(
        "Same question, same context budget, same answer prompt — only the "
        "retrieval strategy differs."
    )
    for column, result in zip(st.columns(len(comparisons), gap="medium"), comparisons):
        with column:
            render_chip(result["pipeline"], result.get("elapsed"))
            # A fixed-height scroll pane per column. Answers differ in length by
            # hundreds of words, and without this the shortest column strands the
            # citations of the other two a screen further down.
            with st.container(height=420, border=True):
                st.markdown(result["content"])
            key = f"{key_prefix}-{result['pipeline']}"
            render_citations(result, key)
            render_provenance(result, key)


# --- Pipeline orchestration --------------------------------------------------


def condense_question(question: str) -> str:
    """Rewrite a follow-up into a standalone question using recent history."""
    _, query_llm = get_llms()
    return condense_module.condense(query_llm, question, st.session_state.messages)


def retrieve_with(pipeline_name: str, standalone: str):
    """Run one pipeline's retrieval. Returns (docs, queries)."""
    _, query_llm = get_llms()
    return PIPELINES[pipeline_name]["retrieve"](get_store(), query_llm, standalone)


def build_message(pipeline_name: str, result: dict, docs, queries, elapsed) -> dict:
    return {
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "docs": docs,
        "queries": queries,
        "queries_label": PIPELINES[pipeline_name]["queries_label"],
        "pipeline": pipeline_name,
        "elapsed": elapsed,
    }


def answer_all_pipelines(question: str) -> dict:
    """Run every pipeline over one question for side-by-side comparison.

    Not a fourth pipeline — a view over the three. The matched context budget
    and shared answer prompt are what make the columns comparable at all.
    """
    answer_llm, _ = get_llms()
    standalone = condense_question(question)

    comparisons = []
    progress = st.progress(0.0, text="Running all three pipelines…")
    for i, name in enumerate(PIPELINES, start=1):
        progress.progress((i - 1) / len(PIPELINES), text=f"Running {name}…")
        started = time.perf_counter()
        docs, queries = retrieve_with(name, standalone)
        result = generate(answer_llm, standalone, docs)
        elapsed = time.perf_counter() - started
        comparisons.append(build_message(name, result, docs, queries, elapsed))
    progress.empty()

    active = st.session_state.pipeline
    canonical = next(c for c in comparisons if c["pipeline"] == active)
    return {
        "role": "assistant",
        "content": canonical["content"],  # keeps follow-up history coherent
        "sources": canonical["sources"],
        "comparisons": comparisons,
    }


# --- Sidebar -----------------------------------------------------------------

with st.sidebar:
    st.markdown("### MS-ADS Assistant")
    st.caption(
        "Grounded answers about the University of Chicago MS in Applied Data "
        "Science program."
    )

    st.divider()

    active = st.session_state.pipeline
    colour = accent_for(active)
    st.markdown(
        f'<div class="active-pipeline" style="border-left-color:{colour};">'
        f'<div class="label">Active pipeline</div>'
        f'<div class="name" style="color:{colour};">{active}</div>'
        f'<div class="desc">{PIPELINES[active]["description"]}</div></div>',
        unsafe_allow_html=True,
    )

    for column, name in zip(st.columns(len(PIPELINES)), PIPELINES):
        with column:
            if st.button(
                PIPELINES[name]["short"],
                use_container_width=True,
                key=f"pick-{name}",
                type="primary" if name == active else "secondary",
            ):
                st.session_state.pipeline = name
                st.rerun()

    with st.expander("How it works"):
        st.write(PIPELINES[active]["caption"])

    st.toggle(
        "Compare all three",
        key="compare",
        help="Answer with every pipeline at once and show the results side by side.",
    )

    st.divider()

    with st.expander("Example questions"):
        for i, starter in enumerate(STARTERS):
            if st.button(starter, use_container_width=True, key=f"side-start-{i}"):
                st.session_state.pending = starter
                st.rerun()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()

    st.divider()

    st.caption("Appearance")
    light_col, dark_col = st.columns(2)
    with light_col:
        if st.button(
            "Light",
            use_container_width=True,
            key="theme-light",
            type="primary" if not IS_DARK else "secondary",
        ):
            st.session_state.theme = "light"
            st.rerun()
    with dark_col:
        if st.button(
            "Dark",
            use_container_width=True,
            key="theme-dark",
            type="primary" if IS_DARK else "secondary",
        ):
            st.session_state.theme = "dark"
            st.rerun()

    st.divider()
    st.caption(
        "All three pipelines share one answer prompt and an identical 8-passage "
        "budget, so the selector isolates retrieval strategy."
    )

# --- Guard: API key ----------------------------------------------------------

api_key = resolve_api_key()
if not api_key:
    st.error(
        "**No OpenAI API key found.**\n\n"
        "- **Deployed:** add `OPENAI_API_KEY` in the Streamlit Cloud app settings "
        "under *Secrets*.\n"
        "- **Local:** put `OPENAI_API_KEY=sk-...` in a `.env` file, or set the "
        "environment variable before running `streamlit run app.py`."
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

st.markdown("# MS in Applied Data Science")
st.caption("By Afnan Waseem, Eric Nelson, and Kennedy Damtse")
st.write(
    "Ask about admissions, tuition, curriculum, deadlines, or student life. "
    "Every answer cites the pages it came from."
)

if not st.session_state.messages:
    st.write("")
    columns = st.columns(2)
    for i, starter in enumerate(STARTERS):
        if columns[i % 2].button(
            starter, use_container_width=True, key=f"start-{i}"
        ):
            st.session_state.pending = starter
            st.rerun()

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_assistant_message(message, key_prefix=f"m{index}")

# --- Handle input ------------------------------------------------------------

typed = st.chat_input("Ask a question about the program…")
question = typed or st.session_state.pending
st.session_state.pending = None

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        pipeline_name = st.session_state.pipeline
        try:
            if st.session_state.compare:
                message = answer_all_pipelines(question)
            else:
                # Retrieval is silent; the answer streams so the wait is visible.
                render_chip(pipeline_name)
                with st.spinner("Searching the program site…"):
                    started = time.perf_counter()
                    standalone = condense_question(question)
                    docs, queries = retrieve_with(pipeline_name, standalone)

                answer_llm, _ = get_llms()
                sink: dict = {}
                st.write_stream(
                    generate_stream(answer_llm, standalone, docs, sink)
                )
                elapsed = time.perf_counter() - started
                message = build_message(
                    pipeline_name, sink, docs, queries, elapsed
                )
        except Exception as error:  # noqa: BLE001 — surface it, keep the session alive
            st.error(f"Something went wrong answering that question: {error}")
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append(message)
            st.rerun()
