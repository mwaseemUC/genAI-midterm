# MS-ADS RAG Chatbot — Streamlit UI Design

**Date:** 2026-07-21
**Status:** Approved
**Project:** GenAI Principles midterm, Group 1 (Eric Nelson, Afnan Waseem, Kennedy Damtse)
**Deliverable addressed:** PDF Step 4 — "Developing a User Interface", plus the instructor's hosted-URL requirement.

---

## 1. Goal

A deployed, publicly reachable Streamlit chat application that answers questions about the University of Chicago MS in Applied Data Science program, grounded in the 473-chunk corpus already built in `mid_project/`.

Users can switch between three retrieval pipelines — RAG-Fusion, HyDE, and Multi-Query — all sharing one hardened answer prompt.

**Success criteria:**

1. Publicly reachable URL, no login required.
2. Answers all 12 evaluation questions correctly, including Q3 (TOEFL 102 / IELTS 7.0, not the site's "5" typo).
3. Every answer carries source citations linking to the originating page.
4. Retrieved passages and the queries used are inspectable in the UI.
5. Follow-up questions resolve against conversation context.
6. No secret material in the repository.

---

## 2. Decisions and rationale

| # | Decision | Rationale |
|---|---|---|
| 1 | New sibling repo `msads-chatbot/`, fully self-contained | Streamlit Community Cloud clones only the app repo. Any reference to `../mid_project` breaks on deploy. `data/chunks.jsonl` is copied in. |
| 2 | Three pipelines: RAG-Fusion (default), HyDE, Multi-Query | RAG-Fusion is the evaluated winner; the other two are real team work worth demonstrating. Baseline is excluded — it adds a fourth arm with no new insight. |
| 3 | One shared hardened prompt across all three | In the notebooks only RAG-Fusion had it, so HyDE and Multi-Query both answered "TOEFL: 102 or 5 (current requirement)" — incorrect admissions information. On a bot DSI staff will evaluate, no arm may do that. Side benefit: retrieval becomes the only variable. |
| 4 | Matched retrieval budgets — 8 chunks to the LLM in every arm | Removes the largest confound identified in `mid_project/reports/approach_comparison.md` §6. The deployed app becomes the controlled experiment proposed in §7 of that document. |
| 5 | History-aware query rewriting | The PDF asks for "interactive, conversational Q&A sessions." Without it, "what about the online program?" retrieves on five words and loses the topic. |
| 6 | Retrieval logic separated from answer generation | Makes the shared prompt structurally guaranteed rather than three copies that drift, and makes each retriever independently testable. |
| 7 | In-memory Chroma, rebuilt at startup | Container filesystems are ephemeral, so persistence buys nothing. Also avoids the Windows file-lock problem the notebook worked around with `rebuild_vector_store`. |

### 2.1 Deviation from notebook configurations — intentional

Decisions 3 and 4 mean app answers will **not** reproduce notebook outputs exactly. This is deliberate. The notebooks compared three pipelines that differed in prompt *and* retrieval budget simultaneously; the app holds both constant so the pipeline selector isolates retrieval strategy.

This must be stated in the final documentation so the app and the notebooks are not read as contradicting each other.

---

## 3. Repository layout

```
msads-chatbot/
├── app.py                     # Streamlit UI. No RAG logic.
├── rag/
│   ├── __init__.py
│   ├── prompts.py             # shared hardened system prompt, fallback text
│   ├── store.py               # chunks.jsonl -> in-memory Chroma
│   ├── answer.py              # context assembly, answer LLM, SOURCES parsing
│   ├── condense.py            # history-aware query rewriting
│   ├── queries.py             # shared LLM query expansion (fusion + multiquery)
│   ├── fusion.py              # RAG-Fusion retrieval (RRF)
│   ├── hyde.py                # HyDE retrieval
│   └── multiquery.py          # Multi-Query retrieval
├── data/
│   └── chunks.jsonl           # copied from mid_project/data/processed/
├── tests/
│   ├── test_fusion.py
│   ├── test_multiquery.py
│   ├── test_answer.py
│   ├── test_store.py
│   └── test_smoke.py          # live; skips without an API key
├── docs/superpowers/specs/
├── .streamlit/config.toml     # theme only — never secrets
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 4. Module contracts

### 4.1 Retrieval modules — uniform interface

`rag/fusion.py`, `rag/hyde.py`, and `rag/multiquery.py` each expose exactly:

```python
def retrieve(
    vectorstore,
    llm: ChatOpenAI,
    question: str,
    k_per_query: int = 8,
    top_n: int = 8,
) -> tuple[list[Document], list[str]]:
    """Return (ranked documents, queries used) — the latter for display."""
```

The second return value differs in meaning per pipeline, and the UI labels it accordingly:

| Pipeline | `queries` contains | UI label |
|---|---|---|
| RAG-Fusion | original question + 4 LLM reformulations | "Queries searched (5)" |
| HyDE | the generated hypothetical document | "Hypothetical document used" |
| Multi-Query | original question + 3 LLM variants | "Queries searched (4)" |

**Behaviour per pipeline:**

- **`fusion.retrieve`** — port of `mid_project/src/rag_fusion.py`. Generates 4 variants, retrieves `k_per_query` for each of the 5 queries, merges by Reciprocal Rank Fusion (`score = Σ 1/(60 + rank)`), returns `top_n`.
- **`hyde.retrieve`** — generates a hypothetical answer with the HyDE prompt from `Midterm_Group_1_EN_v2.ipynb`, embeds and searches on that text, returns `top_n`. Note `k_per_query` and `top_n` are equal for this pipeline since there is a single query.
- **`multiquery.retrieve`** — generates 3 variants, retrieves `k_per_query` per query, merges by round-robin union deduped on `chunk_id`, returns `top_n`.

### 4.2 `rag/answer.py`

```python
def format_context(docs: list[Document]) -> str
def generate(answer_llm, question: str, docs: list[Document]) -> dict
    # -> {"answer": str, "sources": str}
def parse_sources(raw: str) -> tuple[str, str]
    # splits on the final "SOURCES:" line; dedupes, preserves order
```

Uses `prompts.ANSWER_SYSTEM_PROMPT` for every pipeline. This is the only place the answer LLM is called.

### 4.3 `rag/condense.py`

```python
def condense(llm, question: str, history: list[dict], max_turns: int = 4) -> str
```

Returns `question` unchanged when `history` is empty. Otherwise rewrites the question into a standalone form using the last `max_turns` exchanges. Skips the LLM call entirely on the first message of a session, so single-question sessions cost nothing extra.

### 4.4 `rag/store.py`

```python
def load_chunks(path: Path) -> list[dict]
def to_documents(chunks: list[dict]) -> list[Document]
def build_store(chunks: list[dict])   # in-memory Chroma
```

Document metadata mirrors the notebook exactly: `chunk_id`, `page_title`, `heading_path`, `url`, `n_tokens`, `source`.

### 4.5 `rag/queries.py`

```python
def clean_query_line(line: str) -> str
def expand(llm, question: str, prompt_template: str, n_variants: int) -> list[str]
```

RAG-Fusion and Multi-Query both expand a question into LLM-generated variants, differing only in prompt text and variant count. The implementation lives here once rather than being duplicated.

This also fixes a defect in the original `rag_fusion.py`, which cleaned generated queries with `.strip("-*0123456789. ")`. That strips both ends, so a keyword query such as `"TOEFL minimum score 102"` would lose its trailing `102` — precisely the value being searched for. `clean_query_line` strips leading list markers only.

### 4.6 `rag/prompts.py`

Holds `ANSWER_SYSTEM_PROMPT` (ported verbatim from `rag_fusion.py`, including grounding rules, program disambiguation, data-quality/typo handling, Responsible-AI guardrails, and the SOURCES contract), `FALLBACK_TEXT`, `QUERY_GEN_PROMPT`, `HYDE_PROMPT`, `MULTIQUERY_PROMPT`, and `CONDENSE_PROMPT`.

---

## 5. Retrieval budgets

| Pipeline | Notebook | App | Change |
|---|---|---|---|
| RAG-Fusion | 5 queries × k=8 → RRF → top 8 | identical | none |
| HyDE | k=10 → 10 chunks | k=8 → top 8 | −2 chunks |
| Multi-Query | 4 × k=4 → 2–6 chunks | 4 × k=8 → top 8 | substantially more context |

Multi-Query's notebook configuration fed the answer model as few as 2 chunks because its round-robin dedupe exited early, well under its own `MAX_CONTEXT_DOCS = 12`. The app corrects this.

---

## 6. Data flow

```
user submits message
  │
  ├─ history non-empty? ── yes ─→ condense.condense() ─→ standalone question
  │                        no  ─→ question unchanged
  │
  ├─ retrieve() from the selected pipeline module
  │     └─ returns (docs, queries)
  │
  ├─ answer.generate(answer_llm, question, docs)
  │     └─ shared hardened prompt → raw → parse_sources() → {answer, sources}
  │
  └─ render: answer body, source links, expander(passages), expander(queries)
```

**LLM calls per question:** 1 (answer) + 1 (expansion or hypothetical) + 1 (condense, follow-ups only) = 2–3 on `gpt-4o-mini`, plus one embedding call per search query.

---

## 7. UI specification

**Sidebar**

- Title and one-line description.
- `st.radio` pipeline selector: RAG-Fusion (default), HyDE, Multi-Query.
- Caption describing the selected pipeline's mechanism, updating on change.
- "Clear chat" button resetting session state.
- Scope note: program site, PSD, Booth joint degree, university English-language policy.

**Switching pipeline mid-conversation:** the transcript is retained and the new pipeline applies from the next question onward. Prior answers are not recomputed. Each assistant message records which pipeline produced it and displays that label, so a transcript containing answers from more than one pipeline stays unambiguous.

**Main pane**

- Header and short description of what the assistant covers.
- Four starter-question buttons, shown only while the transcript is empty.
- Chat transcript via `st.chat_message`.
- `st.chat_input` pinned at the bottom.

**Per assistant message**

- Answer body, rendered as markdown.
- **Sources** — bulleted, each a clickable link to its page URL.
- Expander **"Retrieved passages (N)"** — per passage: page title, heading path, source link, passage text.
- Expander labelled per §4.1 — the queries or hypothetical document used.

Both expanders are collapsed by default. They satisfy the PDF requirement to "visually present retrieved information," and make each pipeline's mechanism observable — selecting HyDE reveals the fabricated hypothetical document alongside the grounded answer that ignored it.

**Starter questions** (natural phrasing, not the evaluation set verbatim):

1. "What does the program cost?"
2. "How do I apply?"
3. "What are the English language requirements?"
4. "Is there an application fee waiver?"

---

## 8. Error handling

| Condition | Behaviour |
|---|---|
| No API key at startup | `st.error` with setup instructions, then `st.stop()` |
| OpenAI error during a question | Caught; rendered as an error message in the transcript; session survives and accepts the next question |
| Retrieval returns nothing | `FALLBACK_TEXT`, no sources |
| Transient network failure | `max_retries=5` on all clients, matching `rag_fusion.py` |
| `data/chunks.jsonl` missing or malformed | `st.error` naming the file, then `st.stop()` |

---

## 9. Secrets and deployment

- Key resolution order: `st.secrets["OPENAI_API_KEY"]` → `OPENAI_API_KEY` env var → error and stop.
- The key is entered in the Streamlit Cloud secrets UI. It never appears in the repository.
- `.gitignore` excludes `.env`, `.streamlit/secrets.toml`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`.
- `.streamlit/config.toml` holds theme configuration only.
- **Prerequisite:** the OpenAI key previously committed in the midterm notebook must be rotated before deployment. The new key is used only in Streamlit secrets.

**Deployment steps:** `git init` in `msads-chatbot/` → push to GitHub → connect the repo on Streamlit Community Cloud → set `OPENAI_API_KEY` in secrets → deploy.

---

## 10. Testing

Unit tests require no API key, because the logic worth testing is pure:

| Test | Asserts |
|---|---|
| `test_fusion.py` | RRF scoring — known ranked lists produce a known merged order; a doc ranking well across several lists outranks one ranking first in a single list |
| `test_multiquery.py` | Round-robin union preserves one document per `chunk_id` and respects `top_n` |
| `test_answer.py` | `parse_sources` splits on the final `SOURCES:` line, dedupes while preserving order, and handles a missing line without raising |
| `test_store.py` | Loader parses `chunks.jsonl`, required fields present, document count equals line count |
| `test_smoke.py` | One live question per pipeline returns a non-empty answer with sources. Auto-skips when no API key is present. |

---

## 11. Out of scope

Deliberately excluded to keep this shippable:

- User accounts, authentication, persistence across sessions.
- Feedback capture or answer rating.
- Re-scraping or re-preprocessing — `chunks.jsonl` is an input, produced by `mid_project/src/`.
- Streaming token output.
- The baseline `RetrievalQAWithSourcesChain` as a fourth selectable pipeline.
- Answer caching across users.

---

## 12. Follow-ups outside this spec

1. Rotate the previously committed OpenAI key — **blocks deployment**.
2. Re-scrape and re-run `eval_rag.py` shortly before submission; regenerate `data/chunks.jsonl` if the corpus changes.
3. Document the intentional prompt/budget deviation (§2.1) in the final write-up.
4. The stress-Q4 Booth mis-citation is a corpus/prompt issue, not a UI issue; fix or disclose separately.
