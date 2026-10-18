"""
TESTS — the review loop through the API
=========================================
The real graph runs with an in-memory checkpointer and a scripted model,
so these need no API key and no network.

Why these matter: the API is the reviewer's only way in. A document that
pauses must say so (needs_review and the issues to check), or a client
trusting the flag stores it unreviewed. A review must reach a document
that is actually waiting, and a correction must be one the schema accepts
and must update the confidence it ends up stored with, because a resumed
run can't be paused again to fix a typo.

Run with: pytest tests/test_review_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

import app.main as main
import app.nodes as nodes
from app.graph import build_graph
from tests.scripted_model import ScriptedModel

DOCUMENT = {"document_text": "INVOICE INV-1001\nVendor: Acme Corp\nTotal due: $1,250.00"}


@pytest.fixture
def use_model(monkeypatch):
    graph = build_graph(MemorySaver())
    monkeypatch.setattr(main, "graph", graph)
    monkeypatch.delenv("DOC_AGENT_API_KEY", raising=False)

    def install(**kwargs):
        model = ScriptedModel(**kwargs)
        monkeypatch.setattr(nodes, "get_llm", lambda config=None: model)

    return install


@pytest.fixture
def client():
    return TestClient(main.app)


def test_a_document_of_no_known_type_says_it_needs_review(client, use_model):
    use_model(doc_type="unknown", confidence=0.9)
    body = client.post("/process", json=DOCUMENT).json()
    assert body["status"] == "reviewing"
    assert body["needs_review"] is True
    assert body["review_context"]["issues"] == ["Not an invoice, resume or contract"]


def test_a_low_confidence_classification_says_it_needs_review(client, use_model):
    use_model(doc_type="invoice", confidence=0.5)
    body = client.post("/process", json=DOCUMENT).json()
    assert body["needs_review"] is True
    assert "low confidence (0.50)" in body["review_context"]["issues"][0]


def test_a_review_for_an_unknown_document_is_404(client, use_model):
    use_model()
    response = client.post("/review", json={"thread_id": "no-such-thread", "approved": True})
    assert response.status_code == 404


def test_a_second_review_of_the_same_document_is_409(client, use_model):
    use_model()
    thread_id = client.post("/process", json=DOCUMENT).json()["thread_id"]
    assert client.post("/review", json={"thread_id": thread_id, "approved": False}).status_code == 200
    assert client.post("/review", json={"thread_id": thread_id, "approved": True}).status_code == 409


def test_a_correction_fills_the_missing_field_and_the_stored_confidence_follows(client, use_model):
    use_model()
    paused = client.post("/process", json=DOCUMENT).json()
    assert paused["needs_review"] is True
    assert "Missing fields: due_date" in paused["review_context"]["issues"][0]

    review = {"thread_id": paused["thread_id"], "approved": True, "corrections": {"due_date": "2026-03-31"}}
    final = client.post("/review", json=review).json()["final_output"]
    assert final["extracted_data"]["due_date"] == "2026-03-31"
    assert final["extraction_confidence"] == 1.0
    assert final["human_corrections"] == {"due_date": "2026-03-31"}


@pytest.mark.parametrize("corrections, detail", [
    ({"due_dte": "2026-03-31"}, "Not invoice fields: due_dte"),
    ({"total_amount": "twelve fifty"}, "Invalid corrections: total_amount"),
])
def test_a_bad_correction_is_refused_and_the_document_stays_paused(client, use_model, corrections, detail):
    use_model()
    thread_id = client.post("/process", json=DOCUMENT).json()["thread_id"]

    response = client.post("/review", json={"thread_id": thread_id, "approved": True, "corrections": corrections})
    assert response.status_code == 422
    assert response.json()["detail"].startswith(detail)
    assert client.get(f"/status/{thread_id}").json()["needs_review"] is True
