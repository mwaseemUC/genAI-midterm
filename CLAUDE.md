# CLAUDE.md — msads-chatbot

Operating instructions for this repository. Read `CONTEXT.md` for project state and what to do next.

## What this is

A Streamlit RAG chatbot for the UChicago MS in Applied Data Science program. It is the UI deliverable for the GenAI Principles midterm. The research that produced the corpus and chose the retrieval approach lives in the sibling directory `../mid_project/`.

**Status: specced and planned, not yet implemented.** No code exists beyond `docs/`. Follow the plan.

## Read these before writing code

| Document | Purpose |
|---|---|
| `docs/superpowers/specs/2026-07-21-msads-chatbot-ui-design.md` | Approved design. Decisions and rationale. |
| `docs/superpowers/plans/2026-07-21-msads-chatbot-ui.md` | 8-task implementation plan with complete code. |
| `../mid_project/reports/approach_comparison.md` | Why RAG-Fusion is the default. |

Implement via `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Do not freelance around the plan.

## Environment

Conda is **not on PATH**. Always use the full path:

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n msads-app <command>
```

- App env is **`msads-app`** (Python 3.11), created in Task 1. It does not exist yet.
- Do **not** use `langchain-env` — it lacks `streamlit` and `pytest`, and its site-packages are read-only so pip installs fail.
- `C:\Python314` is on PATH but is untested with chromadb. Use the conda env.

```powershell
# tests
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n msads-app python -m pytest tests/ -v

# app
$env:OPENAI_API_KEY = "sk-..."
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n msads-app streamlit run app.py
```

## Architecture invariants

These are the point of the design. Breaking one silently regresses correctness.

1. **Every pipeline uses `prompts.ANSWER_SYSTEM_PROMPT`.** No pipeline defines its own answer prompt. In the notebooks only RAG-Fusion had the hardened prompt, and HyDE and Multi-Query consequently told users the TOEFL minimum was "5". `tests/test_smoke.py::test_no_pipeline_reports_the_toefl_typo` guards this.
2. **Retrieval modules retrieve only.** `fusion.py`, `hyde.py`, `multiquery.py` never call the answer LLM. `answer.py` is the single place that happens.
3. **Uniform signature** across all three:
   ```python
   def retrieve(vectorstore, llm, question, k_per_query=8, top_n=8) -> tuple[list[Document], list[str]]
   ```
   The second value is display-only: queries for fusion/multiquery, the hypothetical document for HyDE.
4. **All three return exactly 8 documents.** Equal budgets make retrieval strategy the only variable — this is deliberate and is what makes the app a controlled experiment. Do not tune one pipeline's `k` without the others.
5. **`app.py` is UI wiring only.** No retrieval or prompt logic.
6. **Chroma is in-memory.** Never pass `persist_directory`.
7. **No secrets in the repo.** Key resolution is `st.secrets` → `OPENAI_API_KEY` env var → error and stop.

## Conventions

- **TDD**: failing test → run it and see it fail → minimal implementation → run it and see it pass → commit.
- One commit per plan task, message prefixed `feat:` / `test:` / `docs:` / `fix:`.
- Models: `gpt-4o-mini` (answers temp 0, query generation temp 0.3), `text-embedding-3-small` for embeddings — must match the model that built the corpus.
- `max_retries=5` on every OpenAI client.
- All file reads pass `encoding="utf-8"`.

## Gotchas

- **`conda run` cannot take newlines in `python -c`.** It raises `AssertionError: Support for scripts where arguments contain newlines not implemented.` Write a script file instead.
- **`tests/__init__.py` is required**, not incidental. Without it pytest prepends only `tests/` to `sys.path` and every `import rag` fails at collection.
- **Do not copy `.strip("-*0123456789. ")` from `../mid_project/src/rag_fusion.py`.** It strips both ends, so `"TOEFL minimum score 102"` loses the `102`. Use `rag.queries.clean_query_line`.
- `data/chunks.jsonl` is a **build artifact** of `../mid_project/src/preprocess.py`. Do not hand-edit it. To refresh, re-run the scraper and preprocessor there and copy the result.
- PowerShell here is 5.1: no `&&`, no `??`, no ternary. Use `;` and `if ($?) { }`.

## Do not

- Deploy before the OpenAI key committed in the original notebook is rotated.
- Add a fourth pipeline, auth, feedback capture, or answer caching — explicitly out of scope in spec §11.
- Re-run scraping or preprocessing from this repo. That belongs to `../mid_project/`.
- Change `ANSWER_SYSTEM_PROMPT` without re-running the smoke tests.
