# PROMPTS.md — session starters

Copy-pasteable prompts for resuming this project in a new Claude Code session.

---

## Before you start

**Open Claude Code in the right directory.** It matters:

| Working directory | Use it for |
|---|---|
| `msads-chatbot/` | Building the app — Tasks 1–8 |
| `mid_project/` | Re-scraping, re-running evaluation, the documentation rewrite |

**Two things to know:**

1. **Memory does not follow you between the two directories.** Claude Code namespaces memory by project path, so memories saved while working in `mid_project/` will not load in `msads-chatbot/`. This is handled — the critical environment details are duplicated into `CLAUDE.md` — but don't expect the memory entries themselves.
2. **Only `CLAUDE.md` auto-loads.** `CONTEXT.md` is a normal file. Naming it explicitly in your prompt is the reliable way to get it read.

---

## Building the app — in `msads-chatbot/`

### Start implementation

```
Read CLAUDE.md and CONTEXT.md, then implement
docs/superpowers/plans/2026-07-21-msads-chatbot-ui.md starting at Task 1.
Use subagent-driven-development.
```

### Get oriented without doing anything

```
Read CLAUDE.md and CONTEXT.md and tell me where this project stands
and what Task 1 involves before doing anything.
```

### Resume part-way through

```
Read CLAUDE.md and CONTEXT.md. Check which tasks in
docs/superpowers/plans/2026-07-21-msads-chatbot-ui.md are already
committed, then continue from the first incomplete task.
```

### Run a single task

```
Read CLAUDE.md, then execute only Task 4 of
docs/superpowers/plans/2026-07-21-msads-chatbot-ui.md.
Stop and show me the test output before committing.
```

### What to expect on Task 1

Permission will be requested for all three — each is intended:

- `git init` in this directory
- `conda create -n msads-app python=3.11`
- reading `../mid_project/data/processed/chunks.jsonl` (outside the working directory)

Tasks 1–7 need **no API key**; all ~49 unit tests are pure logic. The key is only needed for Task 7's manual checks and Task 8's smoke tests and deploy.

---

## Deploying — in `msads-chatbot/`, after Task 7

**Rotate the leaked OpenAI key first.** One was committed in the original notebook. Revoke it before anything goes public; the replacement goes only into Streamlit Cloud secrets, never into the repo.

```
Read CLAUDE.md, then execute Task 8 of
docs/superpowers/plans/2026-07-21-msads-chatbot-ui.md.
I have rotated the API key and created an empty GitHub repo at <URL>.
```

### Verify a deployed app

```
The app is live at <URL>. Walk me through verifying it against all 12
questions in ../mid_project/evaluation/eval_questions.json, and confirm
Question 3 returns TOEFL 102 / IELTS 7.0 under all three pipelines.
```

---

## Remaining deliverables — in `mid_project/`

### Documentation rewrite

```
Read claudework.md, README.md, and reports/approach_comparison.md.
The PDF asks for at least 5 pages of "medium article like" documentation
covering preprocessing, model architecture, and system design. README.md
is currently framed as a Step-1 technical README. Restructure it into a
narrative article weighting those three areas equally. Do not invent new
results — everything needed is in the existing reports.
```

### PowerPoint deck

```
Read reports/approach_comparison.md, reports/rag_eval_report.md, and
../msads-chatbot/CONTEXT.md. Draft a 10-minute deck outline covering
implementation, challenges, evaluation, and future improvements.
Use the known issues in CONTEXT.md section 6 for the challenges slide,
and be honest about the confounds in the approach comparison.
```

### Video script

```
Draft a 10-minute video script matching the deck. Include a live demo
segment: ask the same question under each of the three pipelines and show
the "Queries searched" / "Hypothetical document used" panels to make each
retrieval strategy visible.
```

### Re-scrape before submission

```
Read claudework.md, then re-run the pipeline: src/scraper.py,
src/preprocess.py, src/quality_check.py. Report whether the in-person
deadline dates have been published yet, and whether the TOEFL typo on
the how-to-apply page is still there. If chunks.jsonl changed, copy it
to ../msads-chatbot/data/chunks.jsonl.
```

### Re-run the scored evaluation

```
Read claudework.md, then re-run src/eval_rag.py and summarise what
changed versus the current reports/rag_eval_report.md.
```

### Optional — the matched-conditions experiment

Converts the approach comparison from judgment into measurement, and removes the objection that the winning pipeline was simply the best-resourced one.

```
Read reports/approach_comparison.md, section 7. Extend src/eval_rag.py to
score all three pipelines — RAG-Fusion, HyDE, Multi-Query — under matched
conditions: the same hardened prompt, 8 chunks of context each, and
comparable candidate pools. Run the 12 evaluation questions and the 8
stress questions through all three, then update the comparison document
with measured results in place of the judged Tier 2 table.
```

---

## Useful mid-session prompts

**Check state before trusting anything:**

```
Before we continue, verify what actually exists: list the files in rag/
and tests/, and show me the git log.
```

**When a test fails:**

```
Use systematic-debugging on this failure. Do not propose a fix until
you can explain the root cause.
```

**Before claiming completion:**

```
Use verification-before-completion. Run the full test suite and show me
the output before telling me this is done.
```
