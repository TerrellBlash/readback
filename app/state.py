"""
Graph state shared by every node.

Each stage writes its own slice:
- ingest: the input document
- classify: document type and classifier confidence
- extract: fields, completeness-based confidence, missing fields
- validate: the issues that send a document to review
- human_review: the reviewer's decision and corrections

State holds data, never formatted prompts; each node builds its own prompt.
"""

from typing import Literal, TypedDict

# --- Sub-structures ---

class DocumentClassification(TypedDict):
    """What the classifier figured out about the document."""
    doc_type: Literal["invoice", "resume", "contract", "unknown"]
    confidence: float  # 0.0 to 1.0, as reported by the classifier
    language: str      # e.g. "english", "spanish"
    summary: str       # one-line description


class ExtractionResult(TypedDict):
    """The structured data pulled from the document."""
    fields: dict           # the actual extracted key-value pairs
    confidence: float      # share of required fields present (0.0 to 1.0)
    missing_fields: list[str]  # required fields absent or empty


# --- Main state ---

class DocAgentState(TypedDict):
    # Inputs (written once at the start)
    document_text: str          # raw text content of the document
    document_filename: str      # original filename for context
    document_source: str        # where it came from (email, upload, etc.)
    document_pages: list[dict] | None  # uploads only: per page, text_layer or vision

    # Classification (written by classify node)
    classification: DocumentClassification | None

    # Extraction (written by extract nodes)
    extraction: ExtractionResult | None

    # Validation (written by classify and validate)
    validation_issues: list[str] | None  # why the document is in review

    # Human review (written by human review node)
    human_approved: bool | None
    human_corrections: dict | None  # any edits the human made

    # Enrichment (written by enrich_extraction node)
    enrichment: dict | None  # vendor data added via tool call, if any

    # Final output
    final_output: dict | None       # the cleaned, approved result
    processing_status: Literal[
        "ingested", "classified", "extracted",
        "validated", "reviewing", "complete", "failed"
    ] | None
