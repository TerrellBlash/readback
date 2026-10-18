"""
A scripted stand-in for ChatOpenAI that answers each node's call: the
classifier, the invoice extractor and the enrichment tool call. Tests use
it to run the real graph offline.
"""

from langchain_core.messages import AIMessage

from app.schemas import InvoiceFields

INVOICE_FIELDS = {
    "vendor_name": "Acme Corp",
    "invoice_number": "INV-1001",
    "invoice_date": "2026-03-01",
    "due_date": None,  # missing, so the document pauses for review
    "line_items": [{"description": "Widgets", "quantity": 10, "unit_price": 125.0, "total": 1250.0}],
    "subtotal": 1250.0,
    "tax": 0.0,
    "total_amount": 1250.0,
    "currency": "USD",
}


class ScriptedModel:
    """Answers the classifier, the extractor and the enrichment call."""

    def __init__(self, doc_type="invoice", confidence=0.95):
        self.classification = {"doc_type": doc_type, "confidence": confidence,
                               "language": "english", "summary": "test document"}
        self.schema = None

    def with_structured_output(self, schema, include_raw=False):
        clone = ScriptedModel()
        clone.classification, clone.schema = self.classification, schema
        return clone

    def bind_tools(self, tools):
        return self

    def invoke(self, prompt):
        if self.schema is InvoiceFields:
            return {"parsed": InvoiceFields(**INVOICE_FIELDS), "raw": None, "parsing_error": None}
        if self.schema is not None:
            return self.classification
        return AIMessage(content="no lookup needed")
