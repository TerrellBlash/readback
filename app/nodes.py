"""
Graph nodes.

Each node reads the state and returns the slice it changed. Nodes that
branch return a Command naming the next node, so the routing lives next
to the decision that drives it: classify picks the extractor, validate
picks auto-store or human review.
"""

import json
import logging
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt

from app.llm import get_llm
from app.schemas import EXTRACTION_SCHEMAS, REQUIRED_FIELDS
from app.state import DocAgentState, DocumentClassification, ExtractionResult
from app.tools import vendor_lookup

logger = logging.getLogger(__name__)

# Below this classifier confidence, a person confirms the document type
# before any extraction runs.
CLASSIFICATION_THRESHOLD = 0.7


class UnprocessableDocumentError(ValueError):
    """The document can't be processed as sent; the client should fix it."""


def ingest_document(state: DocAgentState) -> dict:
    """
    Reject empty or unusable input before any model call.

    - Parameter state: graph state; reads document_text.
    - Returns: the status update.
    - Throws: UnprocessableDocumentError when the text has fewer than 10
        non-blank characters.
    """
    doc_text = state.get("document_text", "")

    if len(doc_text.strip()) < 10:
        raise UnprocessableDocumentError("No readable text found in the document")

    logger.info("Ingested %r (%d chars)", state.get("document_filename", "unknown"), len(doc_text))

    return {"processing_status": "ingested"}


def classify_document(
    state: DocAgentState, config: RunnableConfig
) -> Command[Literal["extract_invoice", "extract_resume", "extract_contract", "human_review"]]:
    """
    Classify the document and route it to that type's extractor.

    - Parameter state: graph state; reads the text, filename and source.
    - Parameter config: the run config, passed to get_llm.
    - Returns: a Command to the type's extractor, or to human_review when
        the type is unknown or the classifier's confidence is below
        CLASSIFICATION_THRESHOLD.
    """
    structured_llm = get_llm(config).with_structured_output(DocumentClassification)

    prompt = f"""Analyze this document and classify it.

Document filename: {state.get('document_filename', 'unknown')}
Source: {state.get('document_source', 'unknown')}

Document content (first 3000 chars):
{state['document_text'][:3000]}

Classify as one of: invoice, resume, contract, unknown.
Provide your confidence (0.0 to 1.0), the language, and a one-line summary.
"""

    classification = structured_llm.invoke(prompt)
    doc_type = classification["doc_type"]
    confidence = classification["confidence"]

    logger.info("Classified as %s (confidence %.2f)", doc_type, confidence)

    if doc_type == "unknown" or confidence < CLASSIFICATION_THRESHOLD:
        issue = (
            "Not an invoice, resume or contract"
            if doc_type == "unknown"
            else f"Classified as {doc_type} with low confidence ({confidence:.2f})"
        )
        return Command(
            update={
                "classification": classification,
                "validation_issues": [issue],
                "processing_status": "reviewing",
            },
            goto="human_review",
        )

    route_map = {
        "invoice": "extract_invoice",
        "resume": "extract_resume",
        "contract": "extract_contract",
    }

    return Command(
        update={
            "classification": classification,
            "processing_status": "classified",
        },
        goto=route_map[doc_type],
    )


_EXTRACTION_ROLES = {
    "invoice": "You are an expert invoice processor.",
    "resume": "You are an expert resume parser.",
    "contract": "You are an expert contract analyst.",
}


def _extraction_prompt(doc_type: str, document_text: str) -> str:
    return f"""{_EXTRACTION_ROLES[doc_type]}
Extract the fields in the response schema from this {doc_type}.

Rules:
- Use only what the document states. Do not guess or invent values.
- If the document doesn't contain a field, return null for it, or an empty
  list for a list field.
- Write amounts as plain numbers, without currency symbols or thousands
  separators.

{doc_type.capitalize()} text:
{document_text}
"""


def _extract(
    doc_type: str, state: DocAgentState, config: RunnableConfig
) -> Command[Literal["validate_extraction"]]:
    """
    Fill the document type's schema with structured output, then score how
    complete the result is.

    - Parameter doc_type: "invoice", "resume" or "contract".
    - Parameter state: graph state; reads document_text.
    - Parameter config: the run config, passed to get_llm.
    - Returns: a Command to validate_extraction carrying the extraction. If
        the model's output can't be parsed into the schema, confidence is
        0.0 and missing_fields is ["all_fields"], so it can't pass the gate.
    """
    structured_llm = get_llm(config).with_structured_output(
        EXTRACTION_SCHEMAS[doc_type], include_raw=True
    )
    result = structured_llm.invoke(_extraction_prompt(doc_type, state["document_text"]))
    parsed = result.get("parsed")

    if parsed is None:
        logger.warning(
            "Structured extraction failed for %s: %s", doc_type, result.get("parsing_error")
        )
        raw = result.get("raw")
        fields = {"raw_response": raw.text if raw is not None else ""}
        missing = ["all_fields"]
        confidence = 0.0  # zero verified fields were extracted
    else:
        fields = parsed.model_dump()
        confidence, missing = compute_extraction_confidence(doc_type, fields)

    extraction = ExtractionResult(
        fields=fields,
        confidence=confidence,
        missing_fields=missing,
    )
    return Command(
        update={"extraction": extraction, "processing_status": "extracted"},
        goto="validate_extraction",
    )


def extract_invoice(
    state: DocAgentState, config: RunnableConfig
) -> Command[Literal["validate_extraction"]]:
    """Extract vendor, dates, line items and totals from an invoice."""
    return _extract("invoice", state, config)


def extract_resume(
    state: DocAgentState, config: RunnableConfig
) -> Command[Literal["validate_extraction"]]:
    """Extract contact details, skills, experience and education from a resume."""
    return _extract("resume", state, config)


def extract_contract(
    state: DocAgentState, config: RunnableConfig
) -> Command[Literal["validate_extraction"]]:
    """Extract parties, dates, terms and governing law from a contract."""
    return _extract("contract", state, config)


def compute_extraction_confidence(
    doc_type: str | None, fields: dict
) -> tuple[float, list[str]]:
    """
    Measure how complete an extraction is against the required fields for
    its document type.

    Pure function: no side effects, no network calls. A field counts as
    missing if its key is absent from `fields`, or its value is None, an
    empty string, an empty list, or an empty dict.

    - Parameter doc_type: the classified document type (e.g. "invoice").
        If it is None or not a key in REQUIRED_FIELDS (e.g. "unknown"), there is
        nothing to check completeness against.
    - Parameter fields: the extracted field dict to check.
    - Returns: a tuple of (confidence, missing_fields) where confidence is
        the fraction of required fields present, rounded to 4 decimal
        places, and missing_fields lists the required field names that are
        absent or empty. When doc_type is not in REQUIRED_FIELDS, returns
        (1.0, []).
    """
    required = REQUIRED_FIELDS.get(doc_type) if doc_type else None
    if required is None:
        return 1.0, []

    empty_values = ("", [], {})
    missing = [
        field
        for field in required
        if field not in fields
        or fields[field] is None
        or fields[field] in empty_values
    ]

    confidence = round((len(required) - len(missing)) / len(required), 4)
    return confidence, missing


def _page_check_issues(pages: list[dict]) -> list[str]:
    """
    Review issues for vision pages whose second read didn't confirm the first.

    - Parameter pages: the state's document_pages.
    - Returns: one issue per disagreeing or unchecked page, naming the
        words that differ so the reviewer knows what to look at.
    """
    issues = []
    for page in pages:
        check = page.get("check")
        if not check or check["agrees"]:
            continue
        if check.get("error"):
            issues.append(f"Page {page['page']} is unchecked: the second read failed ({check['error']})")
        else:
            issues.append(
                f"Page {page['page']}: the two reads disagree. First read only: "
                f"{', '.join(check['first_read_only']) or 'nothing'}; second read only: "
                f"{', '.join(check['second_read_only']) or 'nothing'}"
            )
    return issues


def validate_extraction(
    state: DocAgentState,
) -> Command[Literal["enrich_extraction", "human_review"]]:
    """
    Decide whether the extraction can be stored without a person looking.

    A document passes only with every required field present and, for
    scanned pages, both reads agreeing. Anything else goes to review.

    - Parameter state: graph state; reads extraction and document_pages.
    - Returns: a Command to enrich_extraction when it passes, otherwise to
        human_review with the issues the reviewer should check.
    """
    extraction = state.get("extraction")
    issues = []

    if not extraction:
        return Command(
            update={"validation_issues": ["No extraction result found"], "processing_status": "reviewing"},
            goto="human_review",
        )

    # Confidence is computed from missing_fields alone, so a separate
    # threshold on it would fire exactly when this check does.
    if extraction["missing_fields"]:
        issues.append(
            f"Missing fields: {', '.join(extraction['missing_fields'])} "
            f"(confidence: {extraction['confidence']:.2f})"
        )

    # A vision page read twice by different models. A disagreement is the
    # one signal of a misread: a misread value is present and well-formed,
    # so the missing-fields check above can't see it.
    issues.extend(_page_check_issues(state.get("document_pages") or []))

    if not issues:
        logger.info("Validation passed; storing without review")
        return Command(
            update={"validation_issues": [], "processing_status": "validated"},
            goto="enrich_extraction",
        )
    logger.info("Validation found %d issues; sending to review", len(issues))
    return Command(
        update={"validation_issues": issues, "processing_status": "reviewing"},
        goto="human_review",
    )


def human_review(
    state: DocAgentState,
) -> Command[Literal["store_results"]]:
    """
    Pause until a reviewer responds, then apply their decision.

    interrupt() saves the state through the checkpointer and stops the run;
    /review resumes it, possibly in another worker or after a restart. On
    resume the node runs again from the top, so nothing before interrupt()
    may have side effects.

    - Parameter state: graph state.
    - Returns: a Command to store_results. Approval applies any corrections
        and recomputes confidence from the corrected fields; rejection marks
        the document failed.
    """
    classification = state.get("classification") or {}
    extraction = state.get("extraction") or {}

    human_decision = interrupt({
        "message": "Document needs human review",
        "document_filename": state.get("document_filename", "unknown"),
        "classification": classification,
        "extraction_preview": {
            k: v for k, v in extraction.get("fields", {}).items()
            if not isinstance(v, list)  # show simple fields only
        },
        "issues": state.get("validation_issues", []),
        "document_pages": state.get("document_pages"),
        "action_needed": "Approve, correct or reject the extraction",
    })

    if not human_decision.get("approved", False):
        return Command(
            update={"human_approved": False, "processing_status": "failed"},
            goto="store_results",
        )

    update = {"human_approved": True, "processing_status": "validated"}
    corrections = human_decision.get("corrections") or {}
    if corrections and extraction:
        fields = {**extraction["fields"], **corrections}
        confidence, missing = compute_extraction_confidence(classification.get("doc_type"), fields)
        update["extraction"] = {**extraction, "fields": fields, "confidence": confidence, "missing_fields": missing}
        update["human_corrections"] = corrections
    return Command(update=update, goto="store_results")


def store_results(state: DocAgentState) -> dict:
    """
    Assemble the final output returned by the API and kept in the
    checkpoint. Writing it to a system of record is left to the caller.

    - Parameter state: graph state.
    - Returns: final_output, and processing_status "complete" or "failed".
    """
    extraction = state.get("extraction") or {}
    classification = state.get("classification") or {}
    status = "failed" if state.get("processing_status") == "failed" else "complete"

    final_output = {
        "document_filename": state.get("document_filename", "unknown"),
        "document_type": classification.get("doc_type", "unknown"),
        "classification_confidence": classification.get("confidence", 0),
        "extracted_data": extraction.get("fields", {}),
        "extraction_confidence": extraction.get("confidence", 0),
        "document_pages": state.get("document_pages"),
        "human_reviewed": state.get("human_approved") is not None,
        "human_approved": state.get("human_approved"),
        "human_corrections": state.get("human_corrections"),
        "processing_status": status,
    }

    logger.info("Stored %r (type %s, status %s)", final_output["document_filename"],
                final_output["document_type"], status)
    return {"final_output": final_output, "processing_status": status}


def enrich_extraction(
    state: DocAgentState, config: RunnableConfig
) -> Command[Literal["store_results"]]:
    """
    The one agentic node. Binds the vendor_lookup tool to the LLM and
    lets the MODEL decide whether a lookup applies to this document.

    Invoices with a named vendor: the model calls the tool.
    Resumes and most other docs: the model declines, enrichment stays None.

    Either way, this node routes to store_results. It never changes
    which nodes run; its only discretion is whether to call the tool.
    """
    extraction = state.get("extraction") or {}
    fields = extraction.get("fields", {})

    llm_with_tools = get_llm(config).bind_tools([vendor_lookup])

    prompt = (
        "You are enriching extracted document data. If the document names a "
        "vendor or supplier and a vendor lookup would add useful detail, call "
        "the vendor_lookup tool with the vendor name. If there is no vendor "
        "(for example a resume), do not call any tool.\n\n"
        f"Extracted fields:\n{json.dumps(fields, indent=2)}"
    )

    response = llm_with_tools.invoke(prompt)

    enrichment = None
    if response.tool_calls:
        vendor_name = response.tool_calls[0]["args"].get("vendor_name", "")
        enrichment = vendor_lookup.invoke({"vendor_name": vendor_name})
        logger.info("enrich_extraction: looked up vendor %r", vendor_name)
    else:
        logger.info("enrich_extraction: no vendor lookup needed")

    return Command(
        update={"enrichment": enrichment},
        goto="store_results",
    )
