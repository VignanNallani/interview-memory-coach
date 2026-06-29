"""
Thin async wrapper around cognee's three core verbs: add, cognify, search.

Why a wrapper?
- Isolates all cognee imports so the rest of the codebase never calls cognee directly,
  making the backend swappable without touching coach logic.
- Centralises configuration — one place to enforce the local-only stack.
- Inline comments on every cognee call explain WHAT it does internally and WHY we call
  it at that stage, useful for interview defence.

Startup contract
----------------
Call `configure()` ONCE before any other function, AFTER `load_dotenv()`.
cognee resolves its Pydantic settings lazily — calling configure() sets config objects
in memory before the first real API call picks them up.

Storage layout produced by configure(data_dir="./data")
--------------------------------------------------------
  ./data/
    databases/
      cognee_db.db            ← SQLite (dataset registry, task tracking)
      cognee_graph_ladybug/   ← KuzuDB via ladybug (entity/relationship graph)
      cognee.lancedb/         ← LanceDB (chunk embedding vectors)
"""

import os
import uuid
import cognee

# SearchType is accessible at cognee top-level (1.x) and cognee.api.v1.search.
# The v1 path is kept for backward-compatibility so either import works.
from cognee.api.v1.search import SearchType
from cognee.infrastructure.llm.config import LLMConfig


async def configure(data_dir: str = "./data") -> None:
    """
    Point ALL three cognee storage backends at a local directory so no data
    leaves the machine.

    This function is idempotent and safe to call on every startup — it only
    sets path config and creates the root directory; it never touches stored data.
    Call reset() separately when a clean slate is needed (e.g. tests, explicit
    user-initiated wipe).

    How cognee.config.system_root_directory() works internally:
    - Sets base_config.system_root_directory to abs_data_dir.
    - Sets relational_config.db_path      → {abs_data_dir}/databases   (SQLite)
    - Sets graph_config.graph_file_path   → {abs_data_dir}/databases/cognee_graph_ladybug (KuzuDB/ladybug)
    - Sets vector_config.vector_db_url    → {abs_data_dir}/databases/cognee.lancedb       (LanceDB)
    All three storage locations cascade from a single call — no need to set them individually.

    Why NOT pre-create subdirectories:
    KuzuDB (via ladybug) and LanceDB create their own subdirs on first write.
    Passing a pre-existing directory to ladybug's Database() raises a RuntimeError —
    it expects to create the path itself.
    """
    abs_data_dir = os.path.abspath(data_dir)
    # Create only the root dir; cognee creates databases/ and its children.
    os.makedirs(abs_data_dir, exist_ok=True)

    # system_root_directory() cascades to SQLite, KuzuDB (ladybug), and LanceDB
    # in one call.  This is the canonical way in cognee 1.x to configure a custom
    # storage path without touching individual provider configs.
    cognee.config.system_root_directory(abs_data_dir)


async def reset() -> None:
    """
    DESTRUCTIVE — wipes all stored memory (graph, vectors, SQLite metadata, cache).

    Use for a clean test slate or an explicit user-initiated reset, never on
    normal startup.  Must be called AFTER configure() so cognee's path config
    points at the correct directory before the wipe runs.
    """
    await cognee.prune.prune_system(graph=True, vector=True, metadata=True, cache=True)


async def add(text: str, dataset_name: str = "interview_memory") -> None:
    """
    Stage raw text for later processing.

    What cognee.add() does:
    - Validates LLM connectivity (cognee 1.x runs setup_and_check_environment even
      during add() — so LLM_API_KEY must be set before calling add(), not just before cognify()).
    - Registers the text as a new document in the named dataset.
    - Writes lightweight metadata (document hash, dataset association) to SQLite.
    - Does NOT embed or extract entities yet — that heavy work happens in cognify().

    Why stage-then-process?
    Lets us batch multiple docs (resume, JD, Q&A transcripts) into one dataset
    before running the expensive embedding + entity-graph pass in a single cognify() call.
    """
    # cognee.add accepts str, list[str], file-like objects, or DataItem.
    # A list is used here for consistency with multi-doc batching later.
    await cognee.add([text], dataset_name=dataset_name)


async def cognify(datasets: list[str] | None = None) -> None:
    """
    Run the full ingestion pipeline on all staged documents.

    What cognee.cognify() does (in order):
      1. Chunk each document into overlapping text windows.
      2. Call the configured LLM (Groq / Mistral) to extract named entities and
         relationships from each chunk — the ONLY step that makes outbound LLM calls.
      3. Embed each chunk with the embedding model → write vectors to LanceDB.
      4. Write entity nodes and relationship edges → KuzuDB (via ladybug).
      5. Mark pipeline tasks complete in SQLite.

    Why this matters for interview coaching:
    After cognify() we can answer "what did I learn about X across ALL documents?"
    rather than doing a per-document keyword search.

    datasets: list of dataset name strings to process a subset.
              Pass None to process everything staged since the last cognify().
    """
    # 70B for extraction: 8B intermittently omits required 'description' fields in the
    # KnowledgeGraph JSON schema, causing cognify to exhaust all retries and fail.
    await cognee.cognify(datasets=datasets, llm_config=_llm_70b())


def _llm_70b() -> LLMConfig:
    # 70B for both extraction and coaching: 8B intermittently violates the strict
    # KnowledgeGraph JSON schema cognee requires during cognify (missing 'description'
    # fields cause all retries to fail). Ingest is one-time so the extra tokens are fine.
    return LLMConfig(
        llm_provider="openai",
        llm_model="groq/llama-3.3-70b-versatile",
        llm_api_key=os.getenv("LLM_API_KEY"),
    )


COACH_SYSTEM_PROMPT = """You are an interview coach speaking DIRECTLY to the candidate in second person ("you"). Use ONLY their history in context (resume, target JD, past interview Q&A). Ground every point in what their history shows. Be concrete and direct. If their history reveals a weakness on the asked topic, name it plainly and give 2-3 specific, actionable steps. NEVER write a fake interview question, NEVER refer to the candidate in third person by name, NEVER add meta-commentary about the question. Begin with the coaching directly. Never invent experience they don't have."""


async def recall(question: str, top_k: int = 5) -> str:
    """
    Return a synthesised coaching answer grounded in the candidate's knowledge graph.

    GRAPH_COMPLETION traverses KuzuDB entity/relationship edges before calling the LLM,
    so the answer can reason over *connections between facts* across documents rather
    than just returning the nearest text chunk.  CHUNKS would miss those inferences.

    70B handles both cognify extraction and the coaching answer — extraction reliability
    outweighs TPM cost since ingest is one-time.  top_k=5 keeps graph context small so
    each search call stays well under Groq's free-tier 6000 TPM.
    """
    # session_id=None defaults to 'default_session' — cognee treats repeated calls as a
    # multi-turn conversation and short-circuits with "I'll wait for clarification".
    # A fresh UUID per call forces an independent session so every question is answered fresh.
    results = await cognee.search(
        query_text=question,
        query_type=SearchType.GRAPH_COMPLETION,
        system_prompt=COACH_SYSTEM_PROMPT,
        top_k=top_k,
        llm_config=_llm_70b(),
        session_id=str(uuid.uuid4()),
    )
    return results[0] if results else ""


async def improve(dataset_name: str = "session") -> None:
    """
    Run cognee's post-ingestion enrichment pass (memify Stage 3) over the dataset graph.

    What cognee.improve() does without session_ids:
    - Extracts triplet embeddings from existing KuzuDB nodes and indexes them in LanceDB.
    - Re-indexes vector data points for improved ANN retrieval quality.
    - Does NOT replay Q&A feedback or modify node feedback_weight values — that path
      requires an active session cache (CACHING=true) with stored used_graph_element_ids,
      which we keep off to prevent the "I'll wait for clarification" deferral bug.

    Why call it here:
    After the candidate marks a topic as practiced, this deepens the triplet index so
    future GRAPH_COMPLETION searches traverse more richly connected edges, surfacing
    finer-grained coaching context on subsequent questions.
    """
    # dataset= is keyword-only in cognee.improve() — positional passing raises TypeError.
    await cognee.improve(dataset=dataset_name)


async def forget(dataset_name: str) -> None:
    """
    Surgically remove an entire dataset from all three storage backends.

    What cognee.forget(dataset=...) does:
    - Deletes all KuzuDB entity nodes and relationship edges for the named dataset.
    - Deletes all LanceDB chunk embeddings for the dataset.
    - Removes relational metadata (dataset row + data records) from SQLite.
    - Leaves all other datasets untouched.

    All args to cognee.forget() are keyword-only (bare * in the signature) — the
    dataset= keyword is required; positional passing raises TypeError.

    Use this to surgically remove a stale interview session so it no longer
    pollutes coaching answers, without resetting the entire knowledge graph.
    """
    await cognee.forget(dataset=dataset_name)


async def search(query: str, top_k: int = 5) -> list:
    """
    Retrieve semantically relevant chunks using vector similarity.

    What cognee.search() does:
    - Embeds the query string using the same model used during cognify().
    - Runs ANN (approximate nearest-neighbour) over LanceDB to find similar chunks.

    SearchType.CHUNKS returns raw text chunks ranked by vector similarity — fast
    and reliable because it only needs the LanceDB index (no KuzuDB traversal).
    Switch to SearchType.GRAPH_COMPLETION (the default) once the knowledge graph
    is populated for richer, synthesised answers drawn from both stores.

    Returns list[SearchResult]; each result has attributes like .id and .payload.
    """
    # query_text is the correct parameter name in cognee 1.x (not 'query').
    results = await cognee.search(query_text=query, query_type=SearchType.CHUNKS)
    return results[:top_k]
