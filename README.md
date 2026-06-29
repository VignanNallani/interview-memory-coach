# 🎯 Interview Memory Coach

> AI interview coaching that knows **your** history — grounded in your resume, target role, and every past interview Q&A you've ever given.

Most interview-prep tools give the same advice to everyone. This one reads your actual track record — your resume, the specific job you want, the questions you've been asked before and how you answered them — and builds a **personal knowledge graph** that every coaching answer must be grounded in. Ask it about system design, it cites the exact past interview where you proposed SQLite for a 10 M req/day workload and couldn't explain distributed scale. Ask about your strengths, it points to the FastAPI work you already know cold.

---

## Demo

<!-- Replace with your actual screenshot / GIF -->
![Demo screenshot](docs/demo.png)

**Live walkthrough video:** [link to your recording]

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        Streamlit UI  (app.py)                      │
│   Tab 1: Coach me │ Tab 2: Memory graph │ Tab 3: Add your own      │
└──────────────────────────────┬─────────────────────────────────────┘
                               │ direct Python calls (no HTTP)
┌──────────────────────────────▼─────────────────────────────────────┐
│                    Coach logic  (src/coach/loop.py)                │
│   ingest_session()  ──►  store.add() + store.cognify()            │
│   coach()           ──►  store.recall()                           │
│   FastAPI app       ──►  POST /ingest, POST /coach, GET /health   │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────────┐
│                 Memory store  (src/memory/store.py)                │
│                  thin async wrapper around cognee 1.2              │
│                                                                    │
│   add()      ── stages text, writes metadata ──► SQLite           │
│   cognify()  ── LLM entity extraction ──────────► KuzuDB (graph)  │
│              ── chunk embedding ────────────────► LanceDB (vectors)│
│   recall()   ── GRAPH_COMPLETION search ────────► 70B LLM answer  │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               │                               │
    ┌──────────▼──────────┐        ┌──────────▼──────────┐
    │  Groq (extraction)  │        │  fastembed (local)  │
    │  llama-3.1-8b-inst  │        │  BAAI/bge-small-en  │
    │  ~2 LLM calls/ingest│        │  no API key needed  │
    └─────────────────────┘        └─────────────────────┘
    ┌─────────────────────┐
    │  Groq (answers)     │
    │  llama-3.3-70b-ver  │
    │  1 call per question│
    └─────────────────────┘
```

**Local storage** — all three databases live in `./data/databases/`:

| Database | Purpose |
|---|---|
| `cognee_db` (SQLite) | Dataset registry, document metadata, task status |
| `cognee_graph_ladybug` (KuzuDB) | Entity/relationship graph — the coaching brain |
| `cognee.lancedb` (LanceDB) | Chunk embedding vectors for semantic search |

---

## The key design decisions

### 1. Graph-completion retrieval, not plain RAG

Most "AI with memory" demos do vector similarity search: embed the question, find the nearest chunks, shove them in a prompt. That misses cross-document reasoning.

This project uses cognee's `GRAPH_COMPLETION` search: the query is first matched against the LanceDB vector index, then the retrieved nodes are used as seeds to **traverse the KuzuDB entity graph**, collecting related facts across documents. The LLM sees not just "here is a chunk about system design" but "Jane has Python/FastAPI experience [edge: used\_by] URL shortener design question [edge: struggled\_with] distributed scale [edge: required\_by] target JD". That's what makes the coaching answer reference her *specific* past failure, not generic advice.

### 2. Two-model split: 8B for extraction, 70B for answers

Cognee's `cognify()` pipeline calls the LLM **twice per chunk** — once to extract named entities, once to summarise. These calls need structured JSON output (function-calling), not eloquence. `llama-3.1-8b-instant` is fast and cheap enough to handle this within Groq's 6000 TPM free tier.

The coaching answer (`store.recall()`) fires **one** LLM call that needs actual reasoning quality. That call uses `llama-3.3-70b-versatile` via a per-call `LLMConfig` override — better advice, zero impact on ingestion cost.

### 3. One merged document per session ingest

Naively adding four documents (resume + JD + Q&A×2) and calling `cognify()` spawns four parallel LLM extraction chains in the same minute, blowing past the 6000 TPM limit. The fix: merge all session materials into one labelled document before `add()`. Section headers (`[RESUME]`, `[JOB DESCRIPTION]`, `[INTERVIEW HISTORY]`) preserve semantic boundaries inside the single chunk while keeping cognify to one LLM-call burst.

### 4. KuzuDB is single-process by design

KuzuDB is an embedded graph database (like SQLite, but for graphs). It holds an exclusive write lock on the database file — only one process can open it at a time. For a per-user coaching tool, this is the right trade: zero operational overhead, no server to run, no credentials to manage. The constraint is: run `streamlit run app.py` *or* `uvicorn src.coach.loop:app`, never both simultaneously.

---

## Quick start

### Prerequisites

- Python 3.10+
- A [Groq](https://console.groq.com) API key (free tier is enough)

### Setup

```bash
git clone https://github.com/<you>/interview-memory-coach
cd interview-memory-coach

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -e .
```

### Configure

```bash
cp .env.template .env
# Edit .env and set LLM_API_KEY=gsk_...
# Everything else in .env.template is already correct for Groq + local storage.
```

### Run the Streamlit demo

```bash
# Windows
$env:PYTHONUTF8=1
streamlit run app.py

# macOS/Linux
PYTHONUTF8=1 streamlit run app.py
```

Open http://localhost:8501. The first run ingests the sample session (~30 s); subsequent runs are instant.

### Run the FastAPI server (optional, alternative to Streamlit)

```bash
uvicorn src.coach.loop:app --port 8000
```

```bash
# Ingest a session
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"resume": "...", "job_description": "...", "past_qa": ["Past interview: ..."]}'

# Get coaching
curl -X POST http://localhost:8000/coach \
  -H "Content-Type: application/json" \
  -d '{"question": "How should I prepare for system design questions?"}'
```

**Do not run both Streamlit and FastAPI at the same time** — they share the KuzuDB file lock.

---

## Project layout

```
interview-memory-coach/
├── app.py                  # Streamlit demo UI
├── smoke_test.py           # End-to-end stack verification (no LLM coaching)
├── test_coach.py           # Graph-grounded coaching pipeline test
├── src/
│   ├── memory/
│   │   └── store.py        # Thin async wrapper: configure/add/cognify/recall/search
│   └── coach/
│       └── loop.py         # ingest_session(), coach(), FastAPI app
├── .env.template           # Credentials template (copy to .env)
└── pyproject.toml
```

---

## Running the smoke test

Verifies the full local stack (SQLite + LanceDB + KuzuDB) without spending tokens on the coaching model:

```bash
python smoke_test.py
```

Verifies the graph-grounded coaching pipeline end-to-end:

```bash
python test_coach.py
```

---

## Stack

| Component | Library | Role |
|---|---|---|
| Memory layer | [cognee](https://github.com/topoteretes/cognee) 1.2 | Add / cognify / search orchestration |
| Graph DB | KuzuDB via ladybug | Entity-relationship storage, graph traversal |
| Vector DB | LanceDB | Chunk embedding + ANN search |
| Relational DB | SQLite (aiosqlite) | Dataset registry, task tracking |
| Embeddings | fastembed + BAAI/bge-small-en-v1.5 | Local, no API key |
| LLM (extraction) | Groq llama-3.1-8b-instant | Fast structured entity extraction |
| LLM (coaching) | Groq llama-3.3-70b-versatile | High-quality coaching answers |
| UI | Streamlit 1.58 | Three-tab demo with live graph viz |
| HTTP API | FastAPI + uvicorn | REST alternative to the UI |
