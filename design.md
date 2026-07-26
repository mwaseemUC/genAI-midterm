# Application Design — MS-ADS Assistant

**Purpose of this file:** source material for the *Application Design* slide(s) of the
GenAI Principles midterm deck. Everything below is verified against the shipped code
and corpus, not recalled. Figures were measured on 2026-07-25.

**Project:** RAG chatbot for the University of Chicago MS in Applied Data Science program
**Group 1:** Afnan Waseem, Eric Nelson, Kennedy Damtse
**Stack:** Python 3.11 · Streamlit · LangChain · ChromaDB · OpenAI

---

## 1. The one-sentence thesis

> The application is not just a chatbot — it is a **controlled experiment**. Three
> retrieval strategies compete under identical conditions, and the interface makes
> the comparison visible to the user.

If the slide says one thing, say that. Everything else is supporting detail.

---

## 2. Why the app exists in this shape

The research phase produced three notebooks, one per retrieval approach: RAG-Fusion,
HyDE, and Multi-Query. RAG-Fusion won — but the comparison had a flaw that we documented
rather than hid.

**The three were never run under matched conditions.**

| Confound in the notebooks | Effect |
|---|---|
| RAG-Fusion saw ~40 candidate documents; Multi-Query saw ~16 | Unequal evidence |
| Only RAG-Fusion had the hardened answer prompt | Unequal instructions |

Every confound favoured the winner. So the recommendation could not honestly claim
"RRF is the better algorithm" — only that RAG-Fusion produced correct answers under the
conditions it was given.

**The application resolves both confounds by construction.** All three pipelines now
share one answer prompt and one context budget. Retrieval strategy is the only remaining
variable. That design choice is the intellectual contribution of the app, and it is
worth a slide on its own.

---

## 3. Architecture

### Request lifecycle

```
User question
     │
     ▼
┌─────────────────────────────────────────────┐
│ condense.py — history-aware rewriting       │
│ "what about the online one?"                │
│   → "What is tuition for the MS-ADS Online  │
│      program?"                              │
│ Skipped on the first turn (saves an LLM call)│
└─────────────────────────────────────────────┘
     │  standalone question
     ▼
┌─────────────────────────────────────────────┐
│ RETRIEVAL — exactly one of three            │
│                                             │
│  fusion.py   │  hyde.py    │ multiquery.py  │
│  RRF merge   │  hypothetical│ round-robin   │
│              │  doc search  │ union         │
│                                             │
│  All three return EXACTLY 8 documents       │
│  All three retrieve only — none calls the   │
│  answer LLM                                 │
└─────────────────────────────────────────────┘
     │  8 passages + the queries used (for display)
     ▼
┌─────────────────────────────────────────────┐
│ answer.py — the single answer LLM call      │
│ • assembles context with provenance headers │
│ • applies ANSWER_SYSTEM_PROMPT (shared)     │
│ • streams the response                      │
│ • parses the trailing SOURCES: line         │
└─────────────────────────────────────────────┘
     │  grounded answer + citations
     ▼
┌─────────────────────────────────────────────┐
│ app.py — UI only. No retrieval, no prompts. │
└─────────────────────────────────────────────┘
```

### Module map

| Module | Responsibility |
|---|---|
| `rag/store.py` | Load `chunks.jsonl`, validate schema, build the in-memory Chroma store |
| `rag/prompts.py` | Every prompt string in the system — one file, so sharing is provable |
| `rag/queries.py` | Shared LLM query expansion (used by Fusion and Multi-Query) |
| `rag/fusion.py` | RAG-Fusion retrieval + Reciprocal Rank Fusion |
| `rag/hyde.py` | HyDE retrieval |
| `rag/multiquery.py` | Multi-Query retrieval + round-robin union |
| `rag/condense.py` | Follow-up → standalone question rewriting |
| `rag/answer.py` | Context assembly, the shared prompt, the answer call, citation parsing |
| `app.py` | Streamlit UI wiring |

### The five architectural invariants

These are enforced, not aspirational. Breaking one silently regresses correctness, so
each has a reason and most have a test.

1. **Every pipeline uses `prompts.ANSWER_SYSTEM_PROMPT`.** No pipeline defines its own.
   *Why it matters:* in the notebooks only RAG-Fusion had the hardened prompt, and HyDE
   and Multi-Query consequently told users the TOEFL minimum was **5**. A live smoke test
   (`test_no_pipeline_reports_the_toefl_typo`) guards this for all three.
2. **Retrieval modules retrieve only.** They never call the answer LLM. This is what makes
   the shared prompt *structural* rather than three copies that drift apart over time.
3. **Uniform signature across all three:**
   `retrieve(vectorstore, llm, question, k_per_query=8, top_n=8) -> (list[Document], list[str])`
   Swapping pipelines is a dictionary lookup in the UI, not a branch.
4. **All three return exactly 8 documents.** Equal budgets are the whole point.
5. **`app.py` contains no retrieval or prompt logic.** The UI can be rewritten without
   touching the research.

---

## 4. The three pipelines

All share: `k_per_query=8`, `top_n=8`, the same answer prompt, the same embedding model.

### RAG-Fusion — *default*

1. Expand the question into **4 variants** → 5 queries total (original + 4)
2. Run 5 similarity searches, 8 results each → up to 40 candidates
3. Merge with **Reciprocal Rank Fusion**: `score(d) = Σ 1 / (60 + rank)` across lists
4. Take the top 8

*Intuition:* a passage that ranks reasonably well across several phrasings beats one that
ranks first for a single phrasing. Consensus over peak score. `k=60` is the constant from
the original RRF paper (Cormack et al.).

### HyDE — Hypothetical Document Embeddings

1. Ask the LLM to write a *plausible but not necessarily correct* answer
2. Embed **that text**, not the question
3. One similarity search, take 8

*Intuition:* a question and an answer are different kinds of text. A draft answer sits
closer in embedding space to real answer passages than a question does.

*Important framing point:* the fabricated text is **never shown to the user as fact**. It
is a search key only. The interface displays it under "Hypothetical document used" so the
mechanism is inspectable.

### Multi-Query

1. Expand into **3 variants** → 4 queries total
2. Run 4 searches, 8 results each
3. **Round-robin union**: take each query's 1st result, then each query's 2nd, and so on,
   deduplicating by `chunk_id`, until 8 are collected

*Intuition:* interleaving guarantees no single phrasing monopolises the context window.
No rank fusion, no reranking — deliberately the simplest of the three.

### At a glance

| | RAG-Fusion | HyDE | Multi-Query |
|---|---|---|---|
| Queries issued | 5 | 1 | 4 |
| LLM calls before retrieval | 1 | 1 | 1 |
| Merge strategy | Reciprocal Rank Fusion | none (single list) | Round-robin union |
| Candidates seen | up to 40 | 8 | up to 32 |
| Passages to the answer model | **8** | **8** | **8** |

**A finding worth a bullet:** HyDE issues the *fewest* searches but is often the *slowest*,
because it must generate a full hypothetical document before it can search at all. Fewer
queries does not mean faster. Live timings from the app: HyDE ~7.5s, RAG-Fusion ~6.4s,
Multi-Query ~5.1s on the same question.

---

## 5. Data layer

**Corpus** — a build artefact of the research repo's scraper and preprocessor, copied in
so the app is self-contained (Streamlit Cloud clones only this repository).

| Measure | Value |
|---|---|
| Chunks | **473** |
| Unique source pages | **28** |
| Mean chunk size | **138 tokens** |
| Total corpus | ~65,400 tokens |

Pages by source:

| Host | Pages | What it covers |
|---|---|---|
| `datascience.uchicago.edu` | 15 | The program itself — DSI |
| `physicalsciences.uchicago.edu` | 9 | Division-wide policy — PSD administers the program |
| `chicagobooth.edu` | 2 | The MBA/MS joint degree |
| `grad.uchicago.edu` | 1 | University English-language requirements |
| `apply-psd.uchicago.edu` | 1 | Application portal |

*Scope decision worth mentioning:* the crawl deliberately crossed host boundaries. Fee
waivers, English-language minimums, and the joint degree are documented by PSD, the
university, and Booth respectively — not by the program site. A single-host crawl would
have produced a bot that could not answer common admissions questions. The answer prompt
requires the model to attribute such policies to their actual owner.

**Vector store:** Chroma, **built in memory at startup**, never persisted.

*Why:* container filesystems are ephemeral, so persistence buys nothing on Streamlit
Cloud; it also sidesteps a Windows file-lock problem that the research notebooks had to
work around. Cost is one embedding pass per container start, cached for the process
lifetime via `st.cache_resource`.

---

## 6. Interface design

The rubric asks for a *user-friendly interface* and for retrieved information to be
*visually presented*. Those two pull in opposite directions: a prospective student wants a
clean answer, an evaluator wants the machinery. The design resolves it by **layering**.

**Layer 1 — the answer.** Streams token by token. No wall of technical panels.

**Layer 2 — citations.** A row of numbered source chips beneath the answer. Clicking one
opens a popover with the exact passage that page contributed and a link to the live page.
One click from claim to evidence.

**Layer 3 — the retrieval trace.** A single collapsed panel, *"How this answer was found"*,
holding the queries issued (or the hypothetical document, for HyDE) and all 8 retrieved
passages. Satisfies the rubric; invisible to a student who doesn't want it.

**Other decisions:**

| Decision | Rationale |
|---|---|
| Pipeline selector in the sidebar | Makes the experiment interactive rather than described |
| **Compare all three** mode | One question → three answers side by side. The controlled experiment made visible in a single click. |
| Elapsed time on every answer | Surfaces the real cost difference between strategies |
| Light/dark theme | Ordinary product expectation |
| Example questions, persistent | New users don't face an empty box |
| Errors caught per-turn | A failed question never kills the session |

**Key-handling:** `st.secrets` → `OPENAI_API_KEY` environment variable → explicit error
with setup instructions. No key is ever committed; `.env` and `.streamlit/secrets.toml`
are both git-ignored.

---

## 7. Configuration

| Component | Choice | Why |
|---|---|---|
| Answer model | `gpt-4o-mini`, **temperature 0** | Determinism — the same question should not produce different facts |
| Query-generation model | `gpt-4o-mini`, **temperature 0.3** | Variety is the point of query expansion |
| Embeddings | `text-embedding-3-small` | **Must** match the model that built the corpus |
| Retries | `max_retries=5` on every OpenAI client | Transient API failures shouldn't surface to a user |
| History window | 4 turns | Enough for pronoun resolution without unbounded prompt growth |

---

## 8. Engineering practice — optional slide

- **Test-driven**: 55 unit tests, all network-free via fakes, plus live smoke tests that
  skip automatically when no API key is present.
- **A regression fixed on the way in:** the research code cleaned generated queries with
  `.strip("-*0123456789. ")`, which strips *both* ends — so the query
  `"TOEFL minimum score 102"` silently lost the `102`, the single most important token in
  it. The shared cleaner strips leading list markers only.
- **A second one:** citation parsing originally split the sources line on commas. The
  corpus contains a page titled *"Tuition, Fees, & Aid"*, which was being shredded into
  three fake citations. Parsing now anchors on the bracketed URL.
- **Streaming without leaking:** the answer ends with a `SOURCES:` line that should never
  flicker into view. Because that marker can arrive split across token boundaries, the
  streamer always withholds the last 8 characters until more text arrives.

---

## 9. Challenges — feeds the *Challenges* slide

**Source-data problems.** These affect all three pipelines equally and are not defects in
our retrieval:

1. **The TOEFL typo.** The live *How to Apply* page reads "102 (prior requirement) or 5
   (current requirement)". A score of 5 is impossible on a 0–120 scale. In the notebooks,
   HyDE and Multi-Query both repeated it as fact. **The fix was architectural, not
   per-pipeline:** one hardened prompt for everyone, with an explicit data-quality rule —
   report the plausible documented value and flag the apparent typo. A test enforces it.
2. **Unpublished deadlines.** The 2027 in-person application portal opens September 2026.
   No retrieval method can surface dates that do not exist. The prompt instructs the model
   to say so and offer the adjacent useful fact rather than fall back to "I don't know".
3. **A stale address in the sample answers.** The live site says Suite **2800**; older
   material says Suite 950. The bot correctly reports what the site says.

**Design challenges.**

4. **Conversational retrieval is stateless.** "What about the online program?" embeds five
   words and loses the topic entirely. Solved by condensing follow-ups against recent
   history *before* retrieval runs.
5. **The comparison confound.** Discussed in §2 — the honest problem, and the one the
   application architecture exists to solve.

---

## 10. Deliberately out of scope

Worth a line, because knowing what you did *not* build is a design statement:

- No fourth pipeline
- No authentication
- No feedback capture
- No answer caching
- No re-scraping from the application repo — that belongs to the research repo

---

## 11. Suggested slide cuts

If you want one slide: **§3 architecture diagram** + the thesis from §1.

If you want three:

1. **Architecture** — the lifecycle diagram, the layer separation, the invariant that
   retrieval never generates
2. **Three pipelines, one budget** — the comparison table from §4 and the controlled-
   experiment argument from §2
3. **Interface** — the three-layer disclosure model from §6, ideally a screenshot of
   Compare mode

**Visuals worth building:** the lifecycle diagram (§3); the pipeline comparison table
(§4); a screenshot of compare mode with all three answers visible; a screenshot of a
citation popover open over an answer.

---

## Note on one figure

`CONTEXT.md` and the data-quality report cite **29 pages**. The shipped `chunks.jsonl`
contains **28 unique source URLs** (measured directly). The difference is one page that
produced no chunks or shares a URL with another. Use 28 if the slide quotes the corpus you
actually deployed; either is defensible, but be consistent across the deck and the report.
