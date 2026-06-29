"""
smoke_test.py

Proves the local SQLite + LanceDB + KuzuDB stack works end-to-end.

Prerequisites
-------------
1. Copy .env.template to .env and fill in your LLM credentials:
     LLM_API_KEY=gsk_...         (Groq key)
     LLM_MODEL=llama3-8b-8192
     LLM_ENDPOINT=https://api.groq.com/openai/v1
     ENABLE_BACKEND_ACCESS_CONTROL=false

2. Install dependencies:
     pip install -e .

Run:
     python smoke_test.py
"""

import asyncio
import sys

# load_dotenv() must run before any cognee import so that cognee's Pydantic
# settings objects pick up LLM_API_KEY / LLM_ENDPOINT from .env at init time.
from dotenv import load_dotenv
load_dotenv()

from src.memory.store import configure, reset, add, cognify, search

SAMPLE_DOC = """\
Alice is a senior software engineer specialising in distributed systems.
She built a real-time event-streaming pipeline at DataFlow using Kafka and Apache Flink.
In her last interview at TechCorp she was asked about the CAP theorem, eventual consistency,
and the Raft consensus algorithm. She struggled to explain leader election clearly.
Her resume lists Python, Go, and three years of Kubernetes experience.
"""

QUERY = "What distributed systems topics did Alice struggle with in interviews?"


async def main() -> None:
    print("=" * 60)
    print("  Interview Memory Coach — Smoke Test")
    print("  cognee 1.2.x  |  SQLite + LanceDB + KuzuDB")
    print("=" * 60)

    # ── Step 1: configure local storage ──────────────────────────────────────
    print("\n[1/4] Configuring local storage (SQLite + LanceDB + KuzuDB)...")
    # reset() before configure() so the wipe targets the correct path, not the
    # cognee default.  configure() is now path-only and never touches stored data.
    await configure(data_dir="./smoke_data")
    await reset()
    print("      ✓ Local stack configured at ./smoke_data/")

    # ── Step 2: stage the sample document ────────────────────────────────────
    print("\n[2/4] Staging sample document via cognee.add()...")
    # cognee.add() writes document metadata to SQLite.
    # No LLM call happens here — this is purely local I/O.
    await add(SAMPLE_DOC, dataset_name="smoke_test")
    print("      ✓ Document staged (metadata in SQLite, not yet embedded)")

    # ── Step 3: cognify ───────────────────────────────────────────────────────
    print("\n[3/4] Running cognify() — LLM entity extraction + LanceDB embedding...")
    print("      (Makes outbound call to your Groq/Mistral endpoint — ~10-30s)")
    # cognee.cognify() is the pipeline step that:
    #   • calls the LLM to extract entities/relationships → writes to KuzuDB
    #   • embeds text chunks → writes vectors to LanceDB
    #   • marks tasks complete in SQLite
    await cognify(datasets=["smoke_test"])
    print("      ✓ Cognify complete — entity graph + embeddings on disk")

    # ── Step 4: search ────────────────────────────────────────────────────────
    print(f"\n[4/4] Searching with SearchType.CHUNKS (pure vector similarity)...")
    print(f"      Query: \"{QUERY}\"")
    # cognee.search() embeds the query and runs ANN over LanceDB.
    # SearchType.CHUNKS bypasses the KuzuDB graph and returns raw chunk matches —
    # the simplest retrieval mode; ideal for smoke-testing the vector index.
    results = await search(QUERY, top_k=3)

    print(f"\n{'─' * 60}")
    print(f"  Results ({len(results)} returned)")
    print(f"{'─' * 60}")

    if not results:
        print("  [!] No results — LanceDB index may be empty.")
        print("  Check that cognify() above did not error silently.")
        sys.exit(1)

    for i, result in enumerate(results, 1):
        print(f"\n  ── Result {i} ──")
        # cognee SearchResult objects expose .id and .payload at minimum.
        # Print the full object so we can see exactly what 1.2.x returns.
        if hasattr(result, "__dict__"):
            for k, v in vars(result).items():
                print(f"  {k}: {v}")
        elif isinstance(result, dict):
            for k, v in result.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {result}")

    print(f"\n{'=' * 60}")
    print("  SMOKE TEST PASSED")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
