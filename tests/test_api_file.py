"""
TESTS — POST /process/file
============================
The graph and the vision model are stubbed, so these run offline.

Why these matter: uploads are the boundary where untrusted bytes enter.
Wrong types, oversized files and unreadable scans must get a clear 4xx
before any model call, and a good upload must reach the graph with the
page provenance a reviewer relies on.

Run with: pytest tests/test_api_file.py -v
"""

import pytest
from fastapi.testclient import TestClient

import app.main as main
from evals.pdfgen import scanned_pdf, text_pdf

INVOICE_TEXT = "INVOICE INV-1001\nVendor: Acme Corp\nTotal due: $1,250.00"


class StubGraph:
    """Records what the endpoint sends to the graph."""

    def __init__(self):
        self.state = None

    def invoke(self, state, config):
        self.state = state
        return {"processing_status": "complete", "final_output": {"ok": True}}


@pytest.fixture
def graph(monkeypatch):
    stub = StubGraph()
    monkeypatch.setattr(main, "graph", stub)
    return stub


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture(autouse=True)
def _no_second_read(monkeypatch):
    """No real check model in offline tests; the one test that needs a
    check stubs it."""
    monkeypatch.setattr(main, "check_transcriber", lambda: None)


def upload(client, name, data, content_type="application/pdf"):
    return client.post("/process/file", files={"file": (name, data, content_type)})


def test_text_pdf_reaches_the_graph_with_page_sources(client, graph):
    response = upload(client, "invoice.pdf", text_pdf(INVOICE_TEXT))
    assert response.status_code == 200
    assert graph.state["document_text"] == INVOICE_TEXT
    assert graph.state["document_filename"] == "invoice.pdf"
    assert graph.state["document_pages"] == [
        {"page": 1, "source": "text_layer", "chars": len(INVOICE_TEXT)}
    ]


def test_unsupported_type_is_415_and_never_reaches_the_graph(client, graph):
    response = upload(client, "notes.txt", b"plain text notes", "text/plain")
    assert response.status_code == 415
    assert graph.state is None


def test_oversized_upload_is_413(client, graph, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1000)
    response = upload(client, "big.pdf", text_pdf(INVOICE_TEXT * 40))
    assert response.status_code == 413
    assert graph.state is None


def test_scan_with_no_readable_text_is_422(client, graph, monkeypatch):
    monkeypatch.setattr(main, "vision_transcriber", lambda: lambda image, mime_type: "")
    response = upload(client, "blank-scan.pdf", scanned_pdf("", seed=1))
    assert response.status_code == 422
    assert graph.state is None


def test_corrupt_pdf_is_422(client, graph):
    response = upload(client, "broken.pdf", b"%PDF-1.4 not really a pdf")
    assert response.status_code == 422


def test_declared_size_over_the_limit_is_refused_before_reading(client, graph, monkeypatch):
    """The middleware refuses from Content-Length alone, so a huge body is
    never spooled to disk."""
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1000)
    monkeypatch.setattr(main, "MULTIPART_OVERHEAD_BYTES", 0)
    response = upload(client, "big.pdf", text_pdf(INVOICE_TEXT * 40))
    assert response.status_code == 413
    assert graph.state is None


def test_text_over_the_cost_limit_is_422(client, graph, monkeypatch):
    """Every extraction prompt carries the whole text, so its length bounds
    the cost of one document."""
    monkeypatch.setattr(main, "MAX_TEXT_CHARS", 20)
    response = upload(client, "invoice.pdf", text_pdf(INVOICE_TEXT))
    assert response.status_code == 422
    assert graph.state is None


def test_text_endpoint_enforces_the_same_limit(client, graph):
    response = client.post("/process", json={"document_text": "x" * (main.MAX_TEXT_CHARS + 1)})
    assert response.status_code == 422
    assert graph.state is None


def test_internal_errors_are_logged_not_returned(client, monkeypatch):
    class FailingGraph:
        def invoke(self, state, config):
            raise RuntimeError("connection to postgresql://user:secret@db failed")

    monkeypatch.setattr(main, "graph", FailingGraph())
    response = client.post("/process", json={"document_text": INVOICE_TEXT})
    assert response.status_code == 500
    assert "secret" not in response.text


def test_empty_document_is_the_clients_error(client, monkeypatch):
    class EmptyDocumentGraph:
        def invoke(self, state, config):
            raise main.UnprocessableDocumentError("Document is empty or too short to process.")

    monkeypatch.setattr(main, "graph", EmptyDocumentGraph())
    response = client.post("/process", json={"document_text": "   "})
    assert response.status_code == 422
    assert "too short" in response.text


def test_scanned_upload_is_read_twice_and_a_disagreement_reaches_the_graph(client, graph, monkeypatch):
    monkeypatch.setattr(main, "vision_transcriber", lambda: lambda image, mime_type: "Invoice 91830 dated 2025-04-23")
    monkeypatch.setattr(main, "check_transcriber", lambda: lambda image, mime_type: "Invoice 91630 dated 2026-04-23")
    response = upload(client, "scan.pdf", scanned_pdf(INVOICE_TEXT, seed=1))
    assert response.status_code == 200
    check = graph.state["document_pages"][0]["check"]
    assert check["agrees"] is False
    assert set(check["first_read_only"]) == {"91830", "2025"}


def test_a_page_the_model_declines_is_422(client, graph, monkeypatch):
    refusal = "I'm sorry, but I can't assist with that."
    monkeypatch.setattr(main, "vision_transcriber", lambda: lambda image, mime_type: refusal)
    response = upload(client, "scan.pdf", scanned_pdf(INVOICE_TEXT, seed=1))
    assert response.status_code == 422
    assert "declined" in response.text
    assert graph.state is None
