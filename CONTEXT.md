# CONTEXT.md — GenAI Midterm, Group 1

Project state as of **2026-07-21**. Read `CLAUDE.md` for how to work in this repo.

**Group 1:** Eric Nelson, Afnan Waseem, Kennedy Damtse

--- 

## 1. The assignment

Build a RAG-based conversational AI for the UChicago **MS in Applied Data Science** website.
Source: <https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/>
Brief: `../mid_project/Class project-1 Midterm Project-1.pdf`

### Deliverables and status

| # | Deliverable (from the PDF) | Status |
|---|---|---|
| 1 | Functional RAG chatbot with accurate, context-aware answers | ✅ Done — working end-to-end, 12/12 |
| 2 | Documentation, min 5 pages, "medium article like" | ⚠️ Partial — `../mid_project/README.md` is ~2,500 words but framed as a Step-1 technical README, not a narrative article |
| 3 | User-friendly interface | 🔷 Specced + planned, **not implemented** |
| 4 | PowerPoint deck (10 min) — implementation, challenges, future improvements | ❌ Not started |
| 5 | Evaluation metrics (retrieval accuracy, response relevance) | ✅ Done — `../mid_project/reports/rag_eval_report.md` |

The instructor's announcement adds two items that override the PDF's submission section:
- A **hosted public URL** for the chatbot. DSI staff will evaluate it against the 12-question set.
- A **10-minute recorded Zoom video** alongside the deck.

---

## 2. What exists

### `../mid_project/` — research, complete

```
src/scraper.py        multi-host BFS crawler → data/raw/ + manifest.json
src/preprocess.py     clean → markdown → chunks
src/quality_check.py  eval-question keyword coverage
src/rag_fusion.py     RAG-Fusion + RRF + hardened prompt + groundedness check
src/eval_rag.py       12-question scored eval, baseline vs fusion
```

**Corpus:** 29 pages (15 DSI + 11 PSD + 2 Booth + 1 grad.uchicago), 36,787 words, **473 chunks**, mean 138 tokens. Quality check **12/12 PASS**.

**Three notebooks, one per retrieval approach:**

| Notebook | Approach |
|---|---|
| `Midterm_Group_1_Afnan.ipynb` | RAG-Fusion + Reciprocal Rank Fusion |
| `Midterm_Group_1_EN_v2.ipynb` | HyDE (hypothetical document embeddings) |
| `Midterm_Group_1_MultiQuery.ipynb` | Multi-Query (union, no rank fusion) |

**Reports:**
- `reports/data_quality_report.md` — corpus stats, 12/12 coverage
- `reports/rag_eval_report.md` — scored eval, gpt-4.1 judge
- `reports/approach_comparison.md` — the three-way comparison (written 2026-07-21)

### `msads-chatbot/` — this repo, design only

```
docs/superpowers/specs/2026-07-21-msads-chatbot-ui-design.md    approved design
docs/superpowers/plans/2026-07-21-msads-chatbot-ui.md           8-task plan
CLAUDE.md      operating rules and architecture invariants
CONTEXT.md     this file — project state
PROMPTS.md     copy-pasteable session starters for resuming work
```

No application code yet. No git repo yet.

---

## 3. Which approach was chosen, and why

**RAG-Fusion**, documented in `../mid_project/reports/approach_comparison.md`.

On the 12 evaluation questions: RAG-Fusion 6 outright wins, 5 ties, 1 loss. Multi-Query 1 win. HyDE 0.
On the 8 stress questions: RAG-Fusion 3, HyDE 1, 4 ties. Multi-Query was never stress-tested.

**The decisive finding** is factual rather than stylistic. The live *How to Apply* page contains a typo reading "TOEFL 102 (prior requirement) or 5 (current requirement)". A score of 5 is impossible on a 0–120 scale.

| Approach | Answer |
|---|---|
| RAG-Fusion | TOEFL 102 / IELTS 7.0 — correct |
| HyDE | "102 or 5 (current requirement)" — wrong |
| Multi-Query | "102 or 5 (current requirement)" — wrong |

**Important caveat, and it is documented in the comparison:** the three were never run under matched conditions. RAG-Fusion had ~40 candidate documents, Multi-Query ~16, and only RAG-Fusion had the hardened prompt. Every confound favours RAG-Fusion. The recommendation therefore rests on verifiable facts — correctness on Q3, rubric coverage, deployability — not on a claim that RRF is intrinsically the best algorithm.

The app resolves those confounds by design (see §4).

---

## 4. Design decisions for the app

Full rationale in the spec; summary here.

| Decision | Why |
|---|---|
| Self-contained sibling repo | Streamlit Cloud clones only this repo; any `../mid_project` reference breaks on deploy |
| Three pipelines: RAG-Fusion (default), HyDE, Multi-Query | Winner as default; the others are real team work worth demonstrating |
| **One shared hardened prompt for all three** | Otherwise HyDE and Multi-Query tell visitors the TOEFL minimum is 5. Unacceptable on a bot DSI staff will evaluate. |
| **Matched budgets — 8 chunks in every arm** | Removes the biggest confound from the comparison. The deployed app becomes the controlled experiment. |
| History-aware query rewriting | The PDF asks for conversational Q&A; without it "what about the online program?" retrieves on five words |
| Retrieval separated from answer generation | Makes the shared prompt structural rather than three copies that drift |
| In-memory Chroma, rebuilt at startup | Containers are ephemeral; also avoids the Windows file-lock issue |

**Consequence to document:** app answers will *not* match the notebook outputs, because the notebooks varied prompt and budget together. This is intentional and must be stated in the write-up, or the two will look contradictory.

---

## 5. What's next

### Immediate — implement the app

Execute `docs/superpowers/plans/2026-07-21-msads-chatbot-ui.md`, 8 tasks:

1. Scaffold, `msads-app` conda env, corpus store
2. Shared prompts + answer generation
3. Shared query expansion + RAG-Fusion (RRF)
4. Multi-Query (round-robin union)
5. HyDE
6. History-aware query rewriting
7. Streamlit UI
8. Smoke tests, README, deploy

Tasks 1–7 are self-contained. Task 8 needs the rotated key and a GitHub repo.

### Blocking

**Rotate the OpenAI API key.** A key was committed in the original notebook. It must be revoked before anything is deployed publicly, and the replacement goes only in Streamlit Cloud secrets.

### Then, the remaining deliverables

| Item | Notes |
|---|---|
| Documentation rewrite | Restructure `../mid_project/README.md` into narrative prose weighting preprocessing, architecture, and system design equally. Material exists — it needs reframing, not new research. |
| PowerPoint deck | Challenges slide is well supplied: TOEFL typo, Suite 2800 vs the sample answer's 950, unpublished deadlines, PSD/Booth scope decisions, and the matched-budget confound |
| 10-minute video | The pipeline selector is a live demo — same question under each pipeline, showing the "Queries searched" / "Hypothetical document used" panels |
| Re-scrape before submission | Deadline dates may be published; the quality report's notes flip automatically |

### Optional, strengthens the evaluation

Run all three pipelines through `eval_rag.py` and the stress set under matched conditions (spec §7 of the comparison doc). About an hour of compute. It converts the approach comparison from judgment into measurement and removes the objection that the winner was simply the best-resourced arm.

---

## 6. Known issues

### Site-side — affect all approaches equally, mention as challenges

1. **In-person deadlines unpublished.** The portal for 2027 entry opens September 2026. No retrieval method can surface dates that do not exist.
2. **TOEFL typo** — "102 or 5". A live-site editing error, not a pipeline defect.
3. **Mailing address** — the live site says Suite **2800**; the sample answer's "Suite 950" is outdated. 2800 is correct.

### Pipeline defects

4. **Stress Q4 mis-citation** — both RAG-Fusion and HyDE cite the Booth MBA/MS page for a general admissions-background question. Fix or disclose.
5. **Q7 groundedness** — judge marked PARTIALLY_GROUNDED for attributing a scheduling link to Jose Alvarado where the context only links Matthew Harris-Ridker. Multi-Query makes the same claim.

### Repo hygiene

6. `../mid_project/claudework.md` is **stale** — its header still says "Next step: Step 4 (UI)" and predates the approach comparison, this spec, and this plan. Treat this file as the current state.

---

## 7. Environment quick reference

- Windows 11, PowerShell 5.1. Conda **not on PATH** — use `& "C:\ProgramData\anaconda3\Scripts\conda.exe"`.
- `langchain-env` (Python 3.11) runs the notebooks and `../mid_project/src/`. **Read-only site-packages — pip installs fail.** No `streamlit`, no `pytest`.
- `msads-app` (Python 3.11) is the app env, created in plan Task 1. Does not exist yet.
- Project folders are OneDrive-synced, so `data/` follows across machines.
