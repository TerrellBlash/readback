"""
FastAPI server for the document pipeline.

Endpoints:
  POST /process       Start processing a document sent as text
  POST /process/file  Start processing an uploaded PDF, PNG or JPEG
  POST /review        Submit a human review decision for a paused document
  GET  /status/{id}   Check a document's current state

The Dockerfile runs this app under uvicorn. Every endpoint that runs the
graph is a plain `def`: model calls take seconds, and FastAPI runs plain
functions in a worker thread so they don't block the event loop (and with
it the health probe).
"""

import logging
import os
import secrets
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from langgraph.types import Command
from pydantic import BaseModel, Field, ValidationError

from app.documents import (
    DocumentError,
    UnsupportedDocumentError,
    check_transcriber,
    document_to_text,
    vision_transcriber,
)
from app.graph import close_checkpointer, graph
from app.nodes import UnprocessableDocumentError
from app.schemas import EXTRACTION_SCHEMAS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
MULTIPART_OVERHEAD_BYTES = 64 * 1024  # form boundaries and fields around the file
# About 30k tokens. Every extraction prompt carries the whole text, so this
# bounds the cost of a single document.
MAX_TEXT_CHARS = 120_000
# JSON can escape a character as \uXXXX, six bytes, so a body at the text
# limit can reach about 720 KB.
MAX_JSON_BYTES = 1024 * 1024
# The filename and source go into the classifier prompt and the logs.
MAX_LABEL_CHARS = 255
OPEN_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}  # reachable without an API key
BODY_LIMITS = {"/process": MAX_JSON_BYTES, "/process/file": MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES}


# --- Request/Response Models ---

class ProcessRequest(BaseModel):
    """A document sent as text."""
    document_text: str = Field(max_length=MAX_TEXT_CHARS)
    document_filename: str = Field("unknown.pdf", max_length=MAX_LABEL_CHARS)
    document_source: str = Field("api_upload", max_length=MAX_LABEL_CHARS)


class ReviewRequest(BaseModel):
    """A reviewer's decision on a paused document."""
    thread_id: str
    approved: bool
    corrections: dict | None = None


class StatusResponse(BaseModel):
    """A document's state after the graph ran or paused."""
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
    logger.info("readback starting up")
    if not os.environ.get("DOC_AGENT_API_KEY"):
        logger.warning("DOC_AGENT_API_KEY is not set: the API accepts requests from anyone.")
    yield
    logger.info("readback shutting down")
    close_checkpointer(graph.checkpointer)


app = FastAPI(
    title="readback",
    description=(
        "Extracts invoices, resumes and contracts from text, PDFs and scans. "
        "A second model re-reads every scanned page, and a document whose "
        "reads disagree or whose required fields are missing pauses for a "
        "human reviewer. Send the API key in the X-API-Key header."
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
async def limit_body_size(request: Request, call_next):
    """
    Refuse an oversized body from its declared size, before it is read.
    The endpoints' own checks only run after the whole body has been
    received and parsed.
    """
    limit = BODY_LIMITS.get(request.url.path)
    if limit is not None and request.method == "POST":
        declared = request.headers.get("content-length")
        if declared is None:
            return JSONResponse({"detail": "Content-Length is required"}, status_code=411)
        if not declared.isdigit():
            return JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
        if int(declared) > limit:
            return JSONResponse(
                {"detail": f"Request body is larger than {limit // (1024 * 1024)} MB"},
                status_code=413,
            )
    return await call_next(request)


# --- Endpoints ---

@app.post("/process", response_model=StatusResponse)
def process_document(request: ProcessRequest):
    """
    Start processing a document sent as text.

    The response either carries the final output, or has needs_review set
    and the issues the reviewer should look at; resume it with /review.
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
    document_source: str = Form("api_upload", max_length=MAX_LABEL_CHARS),
):
    """
    Start processing an uploaded PDF, PNG or JPEG.

    PDF pages with a text layer are read directly; scanned pages and images
    are transcribed by the vision model first, and read again by a second
    model to catch misreads. The state records which pages came from which
    and whether the two reads agreed, for the reviewer.
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
        raise HTTPException(status_code=415, detail=str(e)) from e
    except DocumentError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    # Checked here as well as in ingest, so a blank scan is refused before
    # a graph run is started for it.
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
            "document_filename": (file.filename or "upload")[:MAX_LABEL_CHARS],
            "document_source": document_source,
            "document_pages": document.page_summary(),
        }
    )


@app.post("/review", response_model=StatusResponse)
def submit_review(request: ReviewRequest):
    """
    Resume a document paused for review with the reviewer's decision.

    Approving stores the extraction, with any corrections applied; rejecting
    marks it failed. Returns 404 for an unknown thread and 409 for one that
    isn't waiting for review.
    """
    config = {"configurable": {"thread_id": request.thread_id}}

    try:
        state = graph.get_state(config)
    except Exception as e:
        logger.exception("Status lookup failed (thread %s)", request.thread_id)
        raise HTTPException(status_code=500, detail="Could not read the document's status") from e
    if not state.values:
        raise HTTPException(status_code=404, detail=f"No document found for thread {request.thread_id}")
    if "human_review" not in state.next:
        raise HTTPException(status_code=409, detail="This document isn't waiting for review")
    if request.approved and request.corrections:
        _check_corrections(state.values, request.corrections)

    logger.info("Resuming review for thread %s (approved: %s)", request.thread_id, request.approved)
    resume = Command(resume={"approved": request.approved, "corrections": request.corrections or {}})
    try:
        result = graph.invoke(resume, config)
    except Exception as e:
        logger.exception("Review processing failed (thread %s)", request.thread_id)
        raise HTTPException(status_code=500, detail="Review processing failed") from e

    return _build_status_response(request.thread_id, result)


@app.get("/status/{thread_id}", response_model=StatusResponse)
def get_status(thread_id: str):
    """Return a document's state as last saved by the checkpointer."""
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = graph.get_state(config)
    except Exception as e:
        logger.exception("Status lookup failed (thread %s)", thread_id)
        raise HTTPException(status_code=500, detail="Could not read the document's status") from e
    if not state.values:
        raise HTTPException(status_code=404, detail=f"No document found for thread {thread_id}")
    return _build_status_response(thread_id, state.values)


@app.get("/health")
async def health_check():
    """Liveness probe for Docker and Azure Container Apps."""
    return {"status": "healthy", "service": "readback"}


# --- Helpers ---

def _start_processing(initial_state: dict) -> StatusResponse:
    """Run a new document through the graph on a fresh thread."""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # %r quotes the client-supplied name, so a newline in it can't forge a log line.
    logger.info("Processing document %r (thread %s)", initial_state["document_filename"], thread_id)

    try:
        result = graph.invoke(initial_state, config)
    except UnprocessableDocumentError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        # The full error goes to the log, not the client: it can carry
        # model output, connection strings or stack details.
        logger.exception("Processing failed (thread %s)", thread_id)
        raise HTTPException(status_code=500, detail="Processing failed") from e

    return _build_status_response(thread_id, result)


def _check_corrections(state: dict, corrections: dict) -> None:
    """
    Refuse corrections the extraction schema wouldn't accept, before the
    graph resumes: a resumed run can't be paused again to fix a typo.

    - Parameter state: the paused graph's values.
    - Parameter corrections: field name to corrected value, from the reviewer.
    - Throws: HTTPException 422 when nothing was extracted to correct, a
        field isn't in the document type's schema, or a value has the
        wrong type.
    """
    extraction = state.get("extraction")
    doc_type = (state.get("classification") or {}).get("doc_type")
    schema = EXTRACTION_SCHEMAS.get(doc_type or "")
    if not extraction or schema is None:
        raise HTTPException(status_code=422, detail="Nothing was extracted, so there is nothing to correct")
    unknown = sorted(set(corrections) - set(schema.model_fields))
    if unknown:
        raise HTTPException(status_code=422, detail=f"Not {doc_type} fields: {', '.join(unknown)}")
    try:
        schema.model_validate({**extraction["fields"], **corrections})
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        raise HTTPException(status_code=422, detail=f"Invalid corrections: {problems}") from e


def _build_status_response(thread_id: str, state: dict) -> StatusResponse:
    """Shape graph state into the API response."""
    status = state.get("processing_status", "unknown")
    classification = state.get("classification")
    extraction = state.get("extraction")
    preview = extraction.get("fields", {}) if extraction else None

    # "reviewing" is set by every route into human_review, so it means the
    # graph is paused there.
    needs_review = status == "reviewing"
    review_context = None
    if needs_review:
        review_context = {
            "issues": state.get("validation_issues", []),
            "classification": classification,
            "extraction_preview": preview,
        }

    return StatusResponse(
        thread_id=thread_id,
        status=status,
        classification=classification,
        extraction_preview=preview,
        needs_review=needs_review,
        review_context=review_context,
        final_output=state.get("final_output"),
    )
