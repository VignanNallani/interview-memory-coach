"""
app.py — Streamlit demo for Interview Memory Coach

Run:  streamlit run app.py
"""

import os
import asyncio

# load_dotenv BEFORE cognee imports — cognee resolves Pydantic settings at import time.
from dotenv import load_dotenv
load_dotenv()

import nest_asyncio
nest_asyncio.apply()          # lets asyncio.run() work inside Streamlit's sync context

import streamlit as st

from src.memory import store
from src.coach import loop

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_DIR = "./data"
FLAG_FILE = os.path.join(DATA_DIR, ".demo_ingested")

RESUME = """\
Jane is a backend engineer with 2 years of Python and FastAPI experience.
She has built REST APIs but has never worked on system design or distributed databases.
She is interviewing for a senior role that requires both."""

JD = """\
Senior backend engineer role.
Requires strong system design skills and experience with distributed databases."""

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

EXAMPLE_QUESTIONS = [
    "How should I prepare for system design questions given my past interviews?",
    "What are my biggest weaknesses based on my interview history?",
    "How do I bridge the gap between my FastAPI experience and the distributed DB requirement?",
]


# ── Async helper ──────────────────────────────────────────────────────────────

def run(coro):
    """Run an async coroutine from Streamlit's synchronous context."""
    return asyncio.run(coro)


# ── One-time startup ──────────────────────────────────────────────────────────

def ensure_configured():
    """Set storage paths once per process — idempotent, never resets data."""
    if "configured" not in st.session_state:
        run(store.configure(data_dir=DATA_DIR))
        st.session_state.configured = True


def ensure_demo_ingested():
    """
    Ingest the Jane fixture exactly once, ever.

    The flag file persists on disk so the expensive cognify call is skipped
    on every subsequent run — including after Streamlit hot-reloads and server
    restarts.  The session_state guard skips the os.path check on reruns within
    the same browser session.
    """
    if st.session_state.get("demo_ready"):
        return

    if os.path.exists(FLAG_FILE):
        st.session_state.demo_ready = True
        return

    with st.spinner("Loading sample session — one-time setup (~30 s)…"):
        run(loop.ingest_session(RESUME, JD, PAST_QA))

    os.makedirs(DATA_DIR, exist_ok=True)
    open(FLAG_FILE, "w").write("ingested")
    st.session_state.demo_ready = True


# ── Page config (must come before any other st call) ──────────────────────────

st.set_page_config(
    page_title="Interview Memory Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ensure_configured()
ensure_demo_ingested()

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🎯 Interview Memory Coach")
st.markdown(
    "_AI interview coaching that knows **YOUR** history — "
    "grounded in your resume, target role, and past Q&A._"
)

if st.session_state.get("demo_ready"):
    st.success("Sample session loaded ✓  (Jane — backend engineer targeting a senior role)", icon="✅")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_coach, tab_graph, tab_add = st.tabs(
    ["💬  Coach me", "🔗  The memory graph", "➕  Add your own"]
)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Coach me
# ─────────────────────────────────────────────────────────────────────────────
with tab_coach:
    st.subheader("Ask your interview coach")

    # Example question buttons fill the text box (session_state key pattern).
    st.caption("Quick-start examples:")
    ex_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        label = example[:55] + "…" if len(example) > 55 else example
        if ex_cols[i].button(label, key=f"eq_{i}", use_container_width=True):
            st.session_state["question_box"] = example

    question = st.text_input(
        "Your question",
        key="question_box",
        placeholder="e.g. How should I prepare for system design questions?",
        label_visibility="collapsed",
    )

    if st.button("Ask →", type="primary", disabled=not bool(question)):
        with st.spinner("Traversing knowledge graph + 70B model (~10 s)…"):
            try:
                answer = run(loop.coach(question))
            except Exception as exc:
                st.error(f"Error: {exc}")
                answer = ""

        if answer and answer.strip():
            st.session_state["coaching_answer"] = answer
            st.session_state.pop("improve_done", None)
        else:
            st.session_state.pop("coaching_answer", None)
            st.warning(
                "No relevant context found. "
                "Make sure a session has been ingested (check the **Add your own** tab)."
            )

    if st.session_state.get("coaching_answer"):
        st.markdown("---")
        st.markdown(st.session_state["coaching_answer"])
        st.markdown("")

        if not st.session_state.get("improve_done"):
            if st.button(
                "🧠 Deepen my memory graph",
                key="improve_btn",
                help=(
                    "Runs a graph enrichment pass (triplet re-indexing) over the session "
                    "dataset. This strengthens the connections cognee can traverse for "
                    "future questions — it does not store feedback scores."
                ),
            ):
                with st.spinner("Re-indexing graph triplets (~15 s)…"):
                    try:
                        run(loop.mark_practiced())
                        st.session_state["improve_done"] = True
                        st.toast("Memory graph deepened — triplet index rebuilt.", icon="🧠")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Enrichment failed: {exc}")
        else:
            st.caption("✓ Memory graph deepened for this answer.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — The memory graph
# ─────────────────────────────────────────────────────────────────────────────
with tab_graph:
    st.subheader("Knowledge graph extracted from your history")
    st.caption(
        "cognee parsed your resume, job description, and past Q&A into an entity-relationship "
        "graph stored in KuzuDB. Every coaching answer traverses this graph."
    )

    render_col, _ = st.columns([1, 3])
    render_btn = render_col.button("🔄  Render graph", key="render_graph", use_container_width=True)

    if render_btn:
        st.session_state.pop("graph_html", None)
        st.session_state.pop("graph_error", None)

        with st.spinner("Loading graph from KuzuDB…"):
            try:
                import cognee
                html = run(cognee.visualize_graph(dataset="session"))
                st.session_state.graph_html = html
            except Exception as primary_err:
                st.session_state.graph_error = str(primary_err)

                # Fallback: memory provenance graph (reads relational DB, not KuzuDB)
                try:
                    html = run(cognee.visualize_memory_provenance())
                    st.session_state.graph_html = html
                    st.session_state.graph_fallback = True
                except Exception as fallback_err:
                    st.session_state.graph_fallback_error = str(fallback_err)

    if "graph_html" in st.session_state:
        if st.session_state.get("graph_fallback"):
            st.info(
                "Primary graph unavailable — showing memory provenance graph "
                "(tenant → user → dataset). "
                f"Primary error: {st.session_state.get('graph_error', '')}",
                icon="ℹ️",
            )
        st.components.v1.html(st.session_state.graph_html, height=720, scrolling=True)

    elif "graph_fallback_error" in st.session_state:
        st.error(f"Primary graph error: {st.session_state.get('graph_error')}")
        st.error(f"Fallback graph error: {st.session_state.graph_fallback_error}")
        st.info("Click **Render graph** after confirming a session has been ingested.")

    else:
        st.info("Click **Render graph** to visualise the knowledge graph.", icon="👆")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Add your own
# ─────────────────────────────────────────────────────────────────────────────
with tab_add:
    st.subheader("Ingest your own session")
    st.caption(
        "⚠️ **This takes ~30–60 s** (entity extraction via Groq llama-3.1-8b-instant). "
        "The demo sample is already loaded — use this to add YOUR real data."
    )

    resume_in = st.text_area(
        "Resume *",
        height=150,
        placeholder="Paste your resume text here…",
    )
    jd_in = st.text_area(
        "Job Description *",
        height=100,
        placeholder="Paste the target job description here…",
    )
    qa_in = st.text_area(
        "Past interview Q&A  (one exchange per line; start with 'Past interview:')",
        height=150,
        placeholder=(
            "Past interview: Asked to design a distributed cache. "
            "I described an LRU cache but couldn't explain consistent hashing.\n"
            "Past interview: Asked about my Python experience. Went well."
        ),
    )

    if st.button("Ingest session", type="primary"):
        if not resume_in.strip() or not jd_in.strip():
            st.warning("Resume and Job Description are both required.")
        else:
            past_qa_lines = [l.strip() for l in qa_in.strip().splitlines() if l.strip()]
            with st.spinner("Ingesting… please wait (~30–60 s)"):
                try:
                    run(loop.ingest_session(resume_in, jd_in, past_qa_lines))
                    st.success(
                        "Session ingested! "
                        "Switch to **💬 Coach me** to ask questions about your new data."
                    )
                    # Invalidate cached graph so it re-renders with new data.
                    st.session_state.pop("graph_html", None)
                    st.session_state.pop("graph_error", None)
                    st.session_state.pop("graph_fallback", None)
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

    # ── Sample session lifecycle ───────────────────────────────────────────────
    st.divider()
    st.subheader("Sample session management")
    st.caption(
        "The Jane fixture is pre-loaded on startup. "
        "Use **Forget** to remove it from the knowledge graph; "
        "use **Reload** to restore it."
    )

    col_forget, col_reload = st.columns(2)

    with col_forget:
        st.warning("⚠️ This deletes the sample — use **Reload** to restore.", icon="⚠️")
        if st.button(
            "🗑️ Forget the sample session",
            key="forget_btn",
            use_container_width=True,
        ):
            with st.spinner("Removing sample session from knowledge graph…"):
                try:
                    run(loop.forget_session("session"))
                    if os.path.exists(FLAG_FILE):
                        os.remove(FLAG_FILE)
                    for key in ("demo_ready", "graph_html", "graph_error",
                                "graph_fallback", "coaching_answer", "improve_done"):
                        st.session_state.pop(key, None)
                    st.success("Sample session removed from knowledge graph.")
                except Exception as exc:
                    st.error(f"Forget failed: {exc}")

    with col_reload:
        st.info("Re-ingests the Jane fixture and rebuilds the graph (~30 s).", icon="🔄")
        if st.button(
            "↺ Reload sample session",
            key="reload_btn",
            use_container_width=True,
        ):
            with st.spinner("Re-ingesting Jane fixture — rebuilding knowledge graph (~30 s)…"):
                try:
                    run(loop.ingest_session(RESUME, JD, PAST_QA))
                    os.makedirs(DATA_DIR, exist_ok=True)
                    open(FLAG_FILE, "w").write("ingested")
                    st.session_state["demo_ready"] = True
                    for key in ("graph_html", "graph_error", "graph_fallback",
                                "coaching_answer", "improve_done"):
                        st.session_state.pop(key, None)
                    st.success(
                        "Sample session restored! "
                        "Switch to **💬 Coach me** to ask questions."
                    )
                except Exception as exc:
                    st.error(f"Reload failed: {exc}")
