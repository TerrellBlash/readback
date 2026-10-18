"""
FastAPI server for the document pipeline.

Endpoints:
  POST /process       Start processing a document sent as text
  POST /process/file  Start processing an uploaded PDF, PNG or JPEG
  POST /review        Submit a human review decision for a paused document
  GET  /status/{id}   Check a document's current state

Any client (webhook, dashboard, mobile app) integrates through these
endpoints. The Dockerfile runs this app under uvicorn.
"""

import os
import secrets
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from langgraph.types import Command

from app.documents import (
    DocumentError,
    UnsupportedDocumentError,
    check_transcriber,
    document_to_text,
    vision_transcriber,
)
from app.graph import graph, close_checkpointer
from app.nodes import UnprocessableDocumentError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
MULTIPART_OVERHEAD_BYTES = 64 * 1024  # form boundaries and fields around the file
# About 30k tokens. Every extraction prompt carries the whole text, so this
# bounds the cost of a single document.
MAX_TEXT_CHARS = 120_000
OPEN_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}  # reachable without an API key


# --- Request/Response Models ---

class ProcessRequest(BaseModel):
    """What the client sends to start processing a document."""
    document_text: str = Field(max_length=MAX_TEXT_CHARS)
    document_filename: str = "unknown.pdf"
    document_source: str = "api_upload"


class ReviewRequest(BaseModel):
    """What a human reviewer sends to approve/reject."""
    thread_id: str
    approved: bool
    corrections: dict | None = None


class StatusResponse(BaseModel):
    """What we send back about the current state of processing."""
    thread_id: str
    status: str
    classification: dict | None = None
    extraction_preview: dict | None = None
    needs_review: bool = False
    review_context: dict | None = None
    final_output: dict | None = None


# --- App Setup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Document Processing Agent starting up")
    if not os.environ.get("DOC_AGENT_API_KEY"):
        logger.warning("DOC_AGENT_API_KEY is not set: the API accepts requests from anyone.")
    yield
    logger.info("Document Processing Agent shutting down")
    close_checkpointer(graph.checkpointer)


app = FastAPI(
    title="Intelligent Document Processing Agent",
    description=(
        "LangGraph-powered pipeline that classifies, extracts, "
        "validates, and routes documents with human-in-the-loop review."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    """
    Refuse requests without the deployment's API key, before the body is
    read or a model is called: every accepted request spends OpenAI
    credits. The health probe and the API docs stay open.

    The key comes from DOC_AGENT_API_KEY. Unset, the API is open, which is
    only meant for local development.
    """
    expected = os.environ.get("DOC_AGENT_API_KEY")
    if expected and request.url.path not in OPEN_PATHS:
        given = request.headers.get("x-api-key", "")
        if not secrets.compare_digest(given.encode(), expected.encode()):
            return JSONResponse({"detail": "Missing or invalid X-API-Key header"}, status_code=401)
    return await call_next(request)


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    """
    Refuse an oversized upload from its declared size, before the body is
    read. The endpoint's own size check only runs after the whole body has
    been received and spooled to disk.
    """
    if request.url.path == "/process/file":
        declared = request.headers.get("content-length")
        if declared is None:
            return JSONResponse({"detail": "Content-Length is required"}, status_code=411)
        if not declared.isdigit():
            return JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
        if int(declared) > MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES:
            return JSONResponse(
                {"detail": f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"},
                status_code=413,
            )
    return await call_next(request)


# --- Endpoints ---

@app.post("/process", response_model=StatusResponse)
async def process_document(request: ProcessRequest):
    """
    Start processing a new document.
    
    The graph will run until it either:
    - Completes successfully (high confidence extraction)
    - Pauses at human_review (needs approval via /review)
    """
    return _start_processing(
        {
            "document_text": request.document_text,
            "document_filename": request.document_filename,
            "document_source": request.document_source,
        }
    )


@app.post("/process/file", response_model=StatusResponse)
def process_file(
    file: UploadFile = File(...),
    document_source: str = Form("api_upload"),
):
    """
    Start processing an uploaded PDF, PNG or JPEG.

    PDF pages with a text layer are read directly; scanned pages and images
    are transcribed by the vision model first, and read again by a second
    model to catch misreads. The state records which pages came from which
    and whether the two reads agreed, for the reviewer.

    This is a plain `def`, so FastAPI runs it in a worker thread and slow
    vision calls don't block the event loop.
    """
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )

    try:
        document = document_to_text(data, vision_transcriber(), check_transcriber())
    except UnsupportedDocumentError as e:
        raise HTTPException(status_code=415, detail=str(e))
    except DocumentError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if len(document.text.strip()) < 10:
        raise HTTPException(status_code=422, detail="No readable text found in the document")
    if len(document.text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"The document has more than {MAX_TEXT_CHARS:,} characters of text",
        )

    return _start_processing(
        {
            "document_text": document.text,
            "document_filename": file.filename or "upload",
            "document_source": document_source,
            "document_pages": document.page_summary(),
        }
    )


@app.post("/review", response_model=StatusResponse)
async def submit_review(request: ReviewRequest):
    """
    Submit a human review decision for a paused document.
    
    This is the other half of interrupt() — the graph has been
    sleeping since it hit the human_review node. This wakes it up.
    """
    config = {"configurable": {"thread_id": request.thread_id}}

    # Build the Command that resumes the graph
    human_response = Command(
        resume={
            "approved": request.approved,
            "corrections": request.corrections or {},
        }
    )

    logger.info(
        f"Resuming review for thread {request.thread_id} "
        f"(approved: {request.approved})"
    )

    try:
        result = graph.invoke(human_response, config)
    except Exception:
        logger.exception(f"Review processing failed (thread {request.thread_id})")
        raise HTTPException(status_code=500, detail="Review processing failed")

    return _build_status_response(request.thread_id, result)


@app.get("/status/{thread_id}", response_model=StatusResponse)
async def get_status(thread_id: str):
    """
    Check the current status of a document being processed.
    
    Reads from the checkpointer — the saved state from the
    last time the graph ran for this thread.
    """
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = graph.get_state(config)
        if not state or not state.values:
            raise HTTPException(
                status_code=404,
                detail=f"No document found for thread {thread_id}",
            )
        return _build_status_response(thread_id, state.values)
    except HTTPException:
        raise
    except Exception:
        logger.exception(f"Status lookup failed (thread {thread_id})")
        raise HTTPException(status_code=500, detail="Could not read the document's status")


@app.get("/health")
async def health_check():
    """Azure Container Apps uses this to know the service is alive."""
    return {"status": "healthy", "service": "readback"}


# --- Helpers ---

def _start_processing(initial_state: dict) -> StatusResponse:
    """Run a new document through the graph on a fresh thread."""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    logger.info(
        f"Processing document: {initial_state['document_filename']} (thread: {thread_id})"
    )

    try:
        result = graph.invoke(initial_state, config)
    except UnprocessableDocumentError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        # The full error goes to the log, not the client: it can carry
        # model output, connection strings or stack details.
        logger.exception(f"Processing failed (thread {thread_id})")
        raise HTTPException(status_code=500, detail="Processing failed")

    return _build_status_response(thread_id, result)


def _build_status_response(thread_id: str, state: dict) -> StatusResponse:
    """Build a clean status response from raw graph state."""
    status = state.get("processing_status", "unknown")
    classification = state.get("classification")
    extraction = state.get("extraction")

    # Check if the graph is paused waiting for human review
    needs_review = status == "reviewing"
    review_context = None
    if needs_review:
        review_context = {
            "issues": state.get("validation_issues", []),
            "classification": classification,
            "extraction_preview": (
                extraction.get("fields", {}) if extraction else None
            ),
        }

    return StatusResponse(
        thread_id=thread_id,
        status=status,
        classification=classification,
        extraction_preview=(
            extraction.get("fields", {}) if extraction else None
        ),
        needs_review=needs_review,
        review_context=review_context,
        final_output=state.get("final_output"),
    )
