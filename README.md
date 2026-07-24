# MS-ADS Assistant

A Retrieval-Augmented Generation chatbot for the University of Chicago
**MS in Applied Data Science** program. Ask about admissions, tuition,
curriculum, deadlines, or student life and get an answer grounded in the
program website, with a citation for every claim.

Built for the GenAI Principles midterm project by **Group 1** — Eric Nelson,
Afnan Waseem, Kennedy Damtse.

## Three retrieval pipelines

Selectable in the sidebar. All three share one hardened answer prompt and an
identical 8-passage context budget, so the selector isolates retrieval strategy.

| Pipeline | How it searches |
|---|---|
| **RAG-Fusion** (default) | Expands the question into 5 phrasings, searches each, merges by Reciprocal Rank Fusion |
| **HyDE** | Writes a hypothetical answer and searches using that text instead of the question |
| **Multi-Query** | Expands into 4 phrasings and takes the deduplicated interleaved union |

RAG-Fusion is the default because it won the evaluation documented in
`../mid_project/reports/approach_comparison.md`.

> **Note:** answers differ slightly from the research notebooks. There, only
> RAG-Fusion had the hardened prompt and each pipeline used a different context
> budget. This app holds both constant, which corrects a confound in the
> original comparison — see §2.1 of the design spec.

## Architecture

```
question
  └─ condense.py    rewrite follow-ups into standalone questions
       └─ fusion.py | hyde.py | multiquery.py    retrieval only → 8 passages
            └─ answer.py    shared hardened prompt → grounded answer + SOURCES
```

Retrieval is separated from answer generation so the shared prompt is
structurally guaranteed rather than three copies that drift.

| Component | Choice |
|---|---|
| Corpus | 473 chunks from 29 pages |
| Embeddings | `text-embedding-3-small` |
| Vector store | Chroma, in memory, built at startup |
| LLM | `gpt-4o-mini` |

## Running locally

```bash
conda create -n msads-app python=3.11 -y
conda run -n msads-app pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."        # PowerShell: $env:OPENAI_API_KEY="sk-..."
conda run -n msads-app streamlit run app.py
```

## Tests

```bash
conda run -n msads-app python -m pytest tests/ -v
```

Unit tests need no API key. `tests/test_smoke.py` runs live pipelines and skips
automatically when `OPENAI_API_KEY` is unset.

## Deployment

Hosted on Streamlit Community Cloud.

1. Push this repository to GitHub.
2. Create an app at [share.streamlit.io](https://share.streamlit.io) pointing at `app.py`.
3. Under **Settings → Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
4. Deploy.

The key is never committed. `.gitignore` excludes `.env` and
`.streamlit/secrets.toml`.

## Corpus

`data/chunks.jsonl` is produced by the pipeline in `../mid_project/src/`
(`scraper.py` → `preprocess.py`). To refresh, re-run those and copy the
regenerated file here.

## Known limitations

- In-person application deadlines are not published on the live site; the
  portal for 2027 entry opens September 2026. No retrieval method can
  surface dates that do not exist.
- The program site contains a typo stating the TOEFL requirement is
  "102 or 5". The assistant reports 102 and flags the typo.
- The mailing address is Suite **2800**; some older materials say Suite 950.
