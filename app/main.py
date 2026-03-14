"""
main.py — FastAPI application exposing the RAG pipeline as a REST API.

Endpoints:
  POST /ingest          — kick off PDF ingestion (async background task)
  GET  /ingest/status   — check ingestion progress
  POST /query           — ask a question
  GET  /sources         — list ingested documents
  GET  /health          — liveness check
"""

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

import db

# ──────────────────────────────────────────────────────────────────────────────
# Startup — pre-load the embedding model so the first query isn't slow
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the embedding model in the background at startup
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _warm_model)
    yield


def _warm_model():
    try:
        import rag
        rag._get_model()
        print("[startup] Embedding model warmed up.")
    except Exception as e:
        print(f"[startup] Warning: could not pre-load model: {e}")


app = FastAPI(
    title       = "Db2 for i SQL Reference RAG",
    description = "Ask questions about IBM Db2 for i SQL using retrieval-augmented generation.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Simple in-process task tracker (fine for single-instance deployment)
_ingest_status: dict[str, str] = {}   # source → "running" | "done" | "error: ..."


# ──────────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────────

class IngestURLRequest(BaseModel):
    url:   str
    force: bool = False


class IngestFileRequest(BaseModel):
    file_path: str           # path inside the container, e.g. /app/pdfs/doc.pdf
    force:     bool = False


class QueryRequest(BaseModel):
    question:    str
    top_k:       int  = 8
    max_tokens:  int  = 2048
    execute_sql: bool = False


class SourceInfo(BaseModel):
    source:      str
    chunk_count: int
    ingested_at: str


class IBMiExecRequest(BaseModel):
    sql: str


# ──────────────────────────────────────────────────────────────────────────────
# Background ingestion helpers
# ──────────────────────────────────────────────────────────────────────────────

def _run_ingest_url(url: str, force: bool):
    try:
        _ingest_status[url] = "running"
        from ingest import download_pdf, ingest
        pdf_path = download_pdf(url)
        ingest(url, pdf_path, force=force)
        _ingest_status[url] = "done"
    except Exception as e:
        _ingest_status[url] = f"error: {e}"
        raise


def _run_ingest_file(file_path: str, force: bool):
    try:
        _ingest_status[file_path] = "running"
        from ingest import ingest
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        ingest(path.name, path, force=force)
        _ingest_status[file_path] = "done"
    except Exception as e:
        _ingest_status[file_path] = f"error: {e}"
        raise


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/sources", response_model=list[dict])
def list_sources():
    """List all documents that have been ingested."""
    return db.list_sources()


@app.post("/ingest/url", status_code=202)
def ingest_url(req: IngestURLRequest, background_tasks: BackgroundTasks):
    """
    Kick off ingestion of a PDF from a URL.
    Returns immediately; poll /ingest/status?source=<url> to track progress.
    """
    if _ingest_status.get(req.url) == "running":
        return {"message": "Already running", "source": req.url}

    background_tasks.add_task(_run_ingest_url, req.url, req.force)
    return {"message": "Ingestion started", "source": req.url}


@app.post("/ingest/file", status_code=202)
def ingest_file(req: IngestFileRequest, background_tasks: BackgroundTasks):
    """
    Kick off ingestion of a local PDF file (must be accessible inside the container).
    """
    if _ingest_status.get(req.file_path) == "running":
        return {"message": "Already running", "source": req.file_path}

    background_tasks.add_task(_run_ingest_file, req.file_path, req.force)
    return {"message": "Ingestion started", "source": req.file_path}


@app.get("/ingest/status")
def ingest_status(source: str):
    """Check the status of an ingestion job."""
    status = _ingest_status.get(source)
    if status is None:
        # Fall back to checking the DB
        if db.is_already_ingested(source):
            return {"source": source, "status": "done"}
        return {"source": source, "status": "not_started"}
    return {"source": source, "status": status}


@app.post("/ibmi/exec")
def ibmi_exec(req: IBMiExecRequest):
    """Execute any SQL statement directly against IBM i (DDL, DML, SELECT)."""
    import ibmi
    result = ibmi.run_statement(req.sql)
    return result


@app.post("/query")
def query(req: QueryRequest):
    """
    Ask a question about the ingested documentation.
    Returns the answer and the source chunks used to generate it.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY is not set in the environment.",
        )

    import rag
    result = rag.ask(req.question, top_k=req.top_k, max_tokens=req.max_tokens, execute_sql=req.execute_sql)

    return {
        "question": req.question,
        "answer":   result.answer,
        "sources": [
            {
                "chunk_id": s.chunk_id,
                "source":   s.source,
                "page_num": s.page_num,
                "score":    round(s.score, 4),
                "excerpt":  s.content[:300] + ("…" if len(s.content) > 300 else ""),
            }
            for s in result.sources
        ],
        "usage": {
            "prompt_tokens":   result.prompt_tokens,
            "response_tokens": result.response_tokens,
        },
        "sql_results": result.sql_results,
    }
