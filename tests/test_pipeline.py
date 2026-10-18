"""
TESTS — the whole graph against the real model
================================================
Why these matter: the offline tests prove the wiring with a scripted model;
only a real model shows the prompts and schemas actually get the values
out. Each test checks a value printed in the document, so a prompt or
schema change that breaks extraction fails here.

The live tests call the OpenAI API and cost a few cents, so they run only
with RUN_LIVE_TESTS=1 and OPENAI_API_KEY set:

    RUN_LIVE_TESTS=1 pytest tests/test_pipeline.py -v

All names, companies and numbers are fictional.
"""

import uuid

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.graph import build_graph
from app.nodes import UnprocessableDocumentError
from tests.markers import live

INVOICE = """
INVOICE #INV-2024-0847

From: CloudTech Solutions LLC
123 Innovation Drive, San Francisco, CA 94105

To: Acme Corporation
456 Business Blvd, Austin, TX 78701

Date: 2024-03-15
Due Date: 2024-04-15
Payment Terms: Net 30

Line Items:
1. Cloud Hosting (Annual) - Qty: 1 - $12,000.00
2. Premium Support Package - Qty: 1 - $3,600.00
3. Data Migration Service - Qty: 1 - $2,500.00

Subtotal: $18,100.00
Tax (8.25%): $1,493.25
Total: $19,593.25
"""

RESUME = """
DANA WHITFIELD
Data Engineer
Denver, CO | dana.whitfield@example.com | (303) 555-0148

SUMMARY
Data engineer with six years of experience building batch and streaming
pipelines for logistics companies.

SKILLS
Python, SQL, Spark, Airflow, dbt, Kafka, PostgreSQL

EXPERIENCE
Senior Data Engineer, Northwind Freight (2021-2025)
- Moved nightly shipment reporting from cron scripts to Airflow

Data Engineer, Bluebird Logistics (2019-2021)
- Built the Kafka feed for real-time truck locations

EDUCATION
B.S. Information Systems, Front Range State University (2019)
"""

CONTRACT = """
SERVICES AGREEMENT

This Services Agreement is entered into as of March 1, 2024 by and between
Harbor Analytics Inc., a Delaware corporation ("Provider"), and Juniper
Retail LLC, a New York limited liability company ("Client").

1. SERVICES. Provider will deliver monthly sales forecasting reports.
2. PAYMENT. Client will pay $4,000 per month, due within 30 days of invoice.
3. TERM. This Agreement runs until February 28, 2026.
4. TERMINATION. Either party may terminate on 30 days' written notice.
5. GOVERNING LAW. This Agreement is governed by the laws of the State of Delaware.
"""

NOTE = "Hello, this is just a random note. Nothing special here."


@pytest.fixture
def run():
    """Run a document through a fresh graph; returns (graph, config, result)."""
    graph = build_graph(MemorySaver())

    def _run(text: str):
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = graph.invoke({"document_text": text, "document_filename": "document.txt",
                               "document_source": "test"}, config)
        return graph, config, result

    return _run


def test_an_empty_document_is_refused_before_any_model_call(run):
    with pytest.raises(UnprocessableDocumentError):
        run("   ")


@live
def test_invoice_total_is_read_exactly(run):
    _, _, result = run(INVOICE)
    assert result["classification"]["doc_type"] == "invoice"
    fields = result["extraction"]["fields"]
    assert fields["total_amount"] == pytest.approx(19593.25)
    assert fields["invoice_number"] == "INV-2024-0847"


@live
def test_resume_contact_details_are_read(run):
    _, _, result = run(RESUME)
    assert result["classification"]["doc_type"] == "resume"
    fields = result["extraction"]["fields"]
    assert fields["full_name"].lower() == "dana whitfield"
    assert fields["email"] == "dana.whitfield@example.com"


@live
def test_contract_parties_and_law_are_read(run):
    _, _, result = run(CONTRACT)
    assert result["classification"]["doc_type"] == "contract"
    fields = result["extraction"]["fields"]
    assert fields["governing_law"] and "delaware" in fields["governing_law"].lower()
    party_names = " ".join(str(party) for party in fields["parties"]).lower()
    assert "harbor analytics" in party_names and "juniper retail" in party_names


@live
def test_a_note_pauses_for_review_and_approval_completes_it(run):
    graph, config, result = run(NOTE)
    assert result["processing_status"] == "reviewing"
    assert graph.get_state(config).next == ("human_review",)

    resumed = graph.invoke(Command(resume={"approved": True, "corrections": {}}), config)
    assert resumed["processing_status"] == "complete"
    assert resumed["final_output"]["human_approved"] is True
