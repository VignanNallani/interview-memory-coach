"""
Interview Memory Coach — FastAPI service.

Lifecycle:
  POST /ingest → ingest_session() → cognify (slow, ~30-60s)
  POST /coach  → coach()          → GRAPH_COMPLETION answer (fast, ~5-10s)
  GET  /health → {"status": "ok"}

Run:
  uvicorn src.coach.loop:app --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from src.memory import store


# ── Pydantic models ───────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    resume: str
    job_description: str
    past_qa: list[str]


class IngestResponse(BaseModel):
    status: str
    message: str


class CoachRequest(BaseModel):
    question: str


class CoachResponse(BaseModel):
    answer: str


class ImproveResponse(BaseModel):
    status: str
    message: str


class ForgetRequest(BaseModel):
    dataset_name: str


class ForgetResponse(BaseModel):
    status: str
    message: str


# ── Startup ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # configure() is idempotent — sets storage paths without wiping data.
    # Do NOT call reset() here; that would destroy any already-ingested sessions.
    await store.configure(data_dir="./data")
    yield


app = FastAPI(title="Interview Memory Coach", lifespan=lifespan)


# ── Core functions (unchanged) ────────────────────────────────────────────────

async def ingest_session(resume: str, job_description: str, past_qa: list[str]) -> None:
    """
    Stage all session documents then run ONE cognify pass.

    All materials are merged into a single labelled document before add() so cognify
    processes one chunk rather than N parallel chunks.  With N separate add() calls,
    cognify would spawn N parallel entity-extraction requests that collectively exceed
    Groq's 6000 TPM free-tier limit.  One document = one chunk = ~2 LLM calls, which
    comfortably fits in the TPM budget while still giving KuzuDB cross-document links
    (the section labels preserve semantic boundaries inside the single chunk).
    """
    qa_block = "\n\n".join(past_qa)
    merged = (
        f"[RESUME]\n{resume}\n\n"
        f"[JOB DESCRIPTION]\n{job_description}\n\n"
        f"[INTERVIEW HISTORY]\n{qa_block}"
    )
    await store.add(merged, dataset_name="session")
    await store.cognify(datasets=["session"])


async def coach(question: str) -> str:
    """
    Return a graph-grounded coaching answer for the given interview question.
    """
    return await store.recall(question)


async def mark_practiced() -> None:
    """Run the graph enrichment pass over the session dataset."""
    await store.improve(dataset_name="session")


async def forget_session(dataset_name: str) -> None:
    """Surgically remove a named dataset from the knowledge graph."""
    await store.forget(dataset_name)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    try:
        await ingest_session(req.resume, req.job_description, req.past_qa)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return IngestResponse(status="ok", message="Session ingested and knowledge graph built.")


@app.post("/coach", response_model=CoachResponse)
async def coach_endpoint(req: CoachRequest) -> CoachResponse:
    try:
        answer = await coach(req.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if not answer or not answer.strip():
        raise HTTPException(status_code=404, detail="No relevant context found — ingest a session first.")
    return CoachResponse(answer=answer)


@app.post("/improve", response_model=ImproveResponse)
async def improve_endpoint() -> ImproveResponse:
    try:
        await mark_practiced()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return ImproveResponse(status="ok", message="Graph enrichment pass complete.")


@app.post("/forget", response_model=ForgetResponse)
async def forget_endpoint(req: ForgetRequest) -> ForgetResponse:
    try:
        await forget_session(req.dataset_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return ForgetResponse(
        status="ok",
        message=f"Dataset '{req.dataset_name}' removed from knowledge graph.",
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
