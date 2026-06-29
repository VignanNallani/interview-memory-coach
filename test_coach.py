"""
test_coach.py

End-to-end test of the graph-grounded coaching pipeline.

Prerequisites: same as smoke_test.py — copy .env.template to .env and fill in
LLM_API_KEY / LLM_MODEL / LLM_ENDPOINT / ENABLE_BACKEND_ACCESS_CONTROL=false.

Run:
    $env:PYTHONUTF8=1
    python test_coach.py
"""

import asyncio
import sys

from dotenv import load_dotenv
load_dotenv()

from src.memory.store import configure, reset
from src.coach.loop import ingest_session, coach

# ── Minimal session fixtures ──────────────────────────────────────────────────

RESUME = """\
Jane is a backend engineer with 2 years of Python and FastAPI experience.
She has built REST APIs but has never worked on system design or distributed databases.
She is interviewing for a senior role that requires both."""

JD = """\
Senior backend engineer role.
Requires strong system design skills and experience with distributed databases."""

# Note: cognee on Windows treats "X: ..." strings where X is one letter as Windows drive
# paths (data_item[1] == ":"). Using "Past interview:" avoids that false positive in
# save_data_item_to_storage.py. This applies to "Q:", "A:", etc.
PAST_QA = [
    (
        "Past interview: Jane was asked to design a URL shortener at 10M req/day. "
        "She proposed SQLite and admitted she had no idea how to handle distributed scale."
    ),
    (
        "Past interview: Jane was asked to structure a FastAPI application. "
        "She answered well — routers, dependency injection, Pydantic models."
    ),
]

COACHING_QUESTION = (
    "How should I prepare for system design questions given my past interviews?"
)


async def main() -> None:
    print("=" * 60)
    print("  Interview Memory Coach — Coaching Test")
    print("  GRAPH_COMPLETION  |  SQLite + LanceDB + KuzuDB")
    print("=" * 60)

    print("\n[1/3] Configuring and resetting local storage...")
    await configure(data_dir="./coach_test_data")
    await reset()
    print("      ✓ Clean slate at ./coach_test_data/")

    print("\n[2/3] Ingesting session (resume + JD + 2 Q&A pairs)...")
    print("      (LLM entity extraction via Groq — ~20-60s)")
    await ingest_session(RESUME, JD, PAST_QA)
    print("      ✓ Session ingested — entity graph built")

    print(f"\n[3/3] Asking coaching question:")
    print(f"      \"{COACHING_QUESTION}\"")
    print("      (GRAPH_COMPLETION traverses KuzuDB then calls LLM — ~10-30s)\n")
    answer = await coach(COACHING_QUESTION)

    print("─" * 60)
    print("  COACHING ANSWER")
    print("─" * 60)
    print(answer)
    print("─" * 60)

    if not answer or not answer.strip():
        print("\n[FAIL] Empty response — check cognify logs above for errors.")
        sys.exit(1)

    print("\n[PASS] Non-empty graph-grounded coaching answer returned.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
