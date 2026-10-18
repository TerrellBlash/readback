"""
TESTS — second read of vision pages
=====================================
Stub transcribers, no network.

Why these matter: the eval showed every silent error came from a misread
on a poor scan, and that asking the same model again repeats its mistakes
(2026 read as 2025 twice). A different model reading the same page caught
every misread in the experiment. These tests pin the parts that make that
check trustworthy: a disagreement must reach the reviewer with the words
that differ, a failed second read must never let a page through unchecked,
the two reads must run at the same time, and a model that declines to
transcribe must not have its apology processed as the document.

Run with: pytest tests/test_read_check.py -v
"""

import logging
import time

import pytest
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app import llm
from app.documents import DocumentError, PageCheck, document_to_text
from app.nodes import validate_extraction
from evals.pdfgen import scanned_pdf, text_pdf

PAGE = "INVOICE 91630\nInvoice Date: 2026-04-23\nTotal Due: $6,197.01"


def reads(text):
    return lambda image, mime_type: text


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    for name in ("DOC_AGENT_MODEL", "DOC_AGENT_CHECK_MODEL", "DOC_AGENT_CHECK_REASONING_EFFORT",
                 "DOC_AGENT_REASONING_EFFORT"):
        monkeypatch.delenv(name, raising=False)
    llm._build.cache_clear()
    yield
    llm._build.cache_clear()


# --- Which model does the second read ---

def test_second_read_uses_a_different_model_at_its_own_effort():
    """gpt-6.1-sol rejects effort "none", the main reader's setting, so the
    check model needs its own effort or every second read fails."""
    client = llm.get_check_llm()
    assert client is not None
    assert client.model_name == "gpt-6.1-sol"
    assert client.reasoning_effort == "low"


def test_check_is_skipped_when_it_would_be_the_same_model():
    """A model re-reading its own page repeats its mistakes, so a
    same-model check would only look like protection."""
    assert llm.resolve_check_model({"configurable": {"model": "gpt-6.1-sol"}}) is None


def test_check_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("DOC_AGENT_CHECK_MODEL", "off")
    assert llm.get_check_llm() is None


def test_run_config_picks_the_check_model():
    config = {"configurable": {"check_model": "gpt-6-sol"}}
    assert llm.resolve_check_model(config) == "gpt-6-sol"


# --- Comparing the two reads ---

def test_matching_reads_mark_the_page_checked():
    document = document_to_text(scanned_pdf(PAGE, seed=1), reads(PAGE), reads(PAGE))
    page = document.page_summary()[0]
    assert page["check"]["agrees"] is True


def test_formatting_differences_are_not_disagreements():
    reformatted = "INVOICE 91630 Invoice Date: 2026-04-23   Total Due: $6197.01"
    document = document_to_text(scanned_pdf(PAGE, seed=1), reads(PAGE), reads(reformatted.replace(",", "")))
    # "6,197.01" and "6197.01" tokenize differently; commas inside numbers
    # are dropped before comparing.
    assert document.page_summary()[0]["check"]["agrees"] is True


def test_a_misread_is_flagged_with_the_words_that_differ():
    misread = PAGE.replace("2026", "2025").replace("91630", "91830")
    document = document_to_text(scanned_pdf(PAGE, seed=1), reads(misread), reads(PAGE))
    check = document.page_summary()[0]["check"]
    assert check["agrees"] is False
    assert set(check["first_read_only"]) == {"2025", "91830"}
    assert set(check["second_read_only"]) == {"2026", "91630"}
    # The document still carries the main read; the reviewer decides.
    assert "91830" in document.text


def test_a_failed_second_read_leaves_the_page_unchecked_not_passed():
    def broken(image, mime_type):
        raise TimeoutError("check model timed out")

    document = document_to_text(scanned_pdf(PAGE, seed=1), reads(PAGE), broken)
    check = document.page_summary()[0]["check"]
    assert check["agrees"] is False
    assert "TimeoutError" in check["error"]


def test_text_layer_pages_are_not_sent_for_a_second_read():
    def must_not_run(image, mime_type):
        raise AssertionError("text-layer page sent to the vision model")

    document = document_to_text(text_pdf(PAGE), must_not_run, must_not_run)
    assert "check" not in document.page_summary()[0]


def test_both_reads_run_at_the_same_time():
    """Run one after the other, the check would add a full vision call to
    every scanned page's latency."""
    def slow(text):
        def transcribe(image, mime_type):
            time.sleep(1.0)
            return text
        return transcribe

    started = time.perf_counter()
    document_to_text(scanned_pdf(PAGE, seed=1), slow(PAGE), slow(PAGE))
    assert time.perf_counter() - started < 1.8  # one after the other would be 2.0+


def test_no_check_configured_leaves_pages_without_a_check():
    document = document_to_text(scanned_pdf(PAGE, seed=1), reads(PAGE), None)
    assert "check" not in document.page_summary()[0]


# --- A model that declines ---

@pytest.mark.parametrize("reply", [
    "I'm sorry, but I can't assist with that.",
    "I’m sorry, I cannot help with transcribing this image.",
    "I am unable to transcribe this page.",
])
def test_a_declined_page_is_unreadable_not_text(reply):
    with pytest.raises(DocumentError, match="Page 1"):
        document_to_text(scanned_pdf(PAGE, seed=1), reads(reply), reads(PAGE))


def test_a_letter_that_apologizes_is_still_a_document():
    letter = ("Dear Ms. Park,\nI'm sorry for the delay on invoice 91630. "
              "Payment of $6,197.01 was sent on 2026-05-01 by wire transfer to the account on file.\n"
              "Regards,\nAccounts Payable, Summit Ridge Manufacturing")
    document = document_to_text(scanned_pdf(letter, seed=1), reads(letter), reads(letter))
    assert "I'm sorry" in document.text


# --- Routing ---

def _state(pages):
    return {
        "extraction": {"fields": {"invoice_number": "91830"}, "confidence": 1.0, "missing_fields": []},
        "document_pages": pages,
    }


def test_a_disagreement_sends_the_document_to_review_with_the_words():
    pages = [{"page": 1, "source": "vision", "chars": 60,
              "check": {"agrees": False, "first_read_only": ["2025", "91830"],
                        "second_read_only": ["2026", "91630"], "error": None}}]
    command = validate_extraction(_state(pages))
    assert command.goto == "human_review"
    issue = " ".join(command.update["validation_issues"])
    assert "91830" in issue and "91630" in issue


def test_an_unchecked_page_sends_the_document_to_review():
    pages = [{"page": 1, "source": "vision", "chars": 60,
              "check": {"agrees": False, "first_read_only": [], "second_read_only": [],
                        "error": "TimeoutError: check model timed out"}}]
    command = validate_extraction(_state(pages))
    assert command.goto == "human_review"
    assert "unchecked" in " ".join(command.update["validation_issues"])


def test_agreeing_reads_do_not_block_auto_storage():
    pages = [{"page": 1, "source": "vision", "chars": 60,
              "check": {"agrees": True, "first_read_only": [], "second_read_only": [], "error": None}}]
    assert validate_extraction(_state(pages)).goto == "store_results"



# --- Found in review ---

@pytest.mark.parametrize("first, second", [
    ("Subtotal $50.10 Tax $10.50", "Subtotal $10.50 Tax $50.10"),  # values swapped
    ("Invoice Date: 04/23/2026", "Invoice Date: 23/04/2026"),  # day and month swapped
    ("Credit: -500.00", "Credit: 500.00"),  # a dropped minus sign
    ("Client: Иванов", "Client: Иванова"),  # a misread in a non-Latin name
])
def test_misreads_the_word_bag_missed_are_flagged(first, second):
    document = document_to_text(scanned_pdf(PAGE, seed=1), reads(first), reads(second))
    assert document.page_summary()[0]["check"]["agrees"] is False


@pytest.mark.parametrize("first, second", [
    ("Café Lumière\n____________", "Café Lumière\n________________________"),  # rule lines differ in length
    ("Total: 6,197.01", "Total: 6197.01"),
])
def test_layout_and_formatting_still_agree(first, second):
    document = document_to_text(scanned_pdf(PAGE, seed=1), reads(first), reads(second))
    assert document.page_summary()[0]["check"]["agrees"] is True


def test_both_reads_are_counted_in_token_usage():
    """The eval's cost figures come from LangChain's usage callback, which
    lives in a context variable. Reads on worker threads must still be
    counted, or a checked run looks cheaper than an unchecked one."""
    def billed(model_name):
        def transcribe(image, mime_type):
            reply = AIMessage(content=PAGE, response_metadata={"model_name": model_name},
                              usage_metadata={"input_tokens": 1000, "output_tokens": 10, "total_tokens": 1010})
            return FakeMessagesListChatModel(responses=[reply]).invoke("read").text
        return transcribe

    with get_usage_metadata_callback() as usage:
        document_to_text(scanned_pdf(PAGE, seed=1), billed("gpt-6-luna"), billed("gpt-6.1-sol"))
    assert set(usage.usage_metadata) == {"gpt-6-luna", "gpt-6.1-sol"}


def test_a_failed_first_read_does_not_wait_for_the_second():
    def broken(image, mime_type):
        raise ConnectionError("vision API down")

    def slow(image, mime_type):
        time.sleep(2.0)
        return PAGE

    started = time.perf_counter()
    with pytest.raises(ConnectionError):
        document_to_text(scanned_pdf(PAGE, seed=1), broken, slow)
    assert time.perf_counter() - started < 1.5


@pytest.mark.parametrize("reply", [
    "I'm unable to transcribe this image.",
    "I apologize, but I can't help with this request.",
    "I'm not able to help with that.",
])
def test_more_ways_of_declining(reply):
    with pytest.raises(DocumentError, match="declined"):
        document_to_text(scanned_pdf(PAGE, seed=1), reads(reply), reads(PAGE))


def test_a_short_letter_saying_we_cant_help_is_a_document():
    letter = "Unfortunately, we are unable to help with your claim. Please contact your insurer."
    document = document_to_text(scanned_pdf(letter, seed=1), reads(letter), reads(letter))
    assert document.text == letter


def test_a_check_model_that_declines_leaves_the_page_unchecked():
    document = document_to_text(scanned_pdf(PAGE, seed=1), reads(PAGE),
                                reads("I'm sorry, but I can't assist with that."))
    check = document.page_summary()[0]["check"]
    assert check["agrees"] is False and "declined" in check["error"]


def test_failure_details_stay_in_the_log_not_the_review_issue():
    """Issues are returned to API clients, so a raw exception message (which
    can carry URLs or request details) must not end up there."""
    def broken(image, mime_type):
        raise TimeoutError("POST https://internal.example/v1 timed out")

    document = document_to_text(scanned_pdf(PAGE, seed=1), reads(PAGE), broken)
    assert document.page_summary()[0]["check"]["error"] == "TimeoutError"


def test_real_check_output_routes_through_the_gate():
    misread = PAGE.replace("2026", "2025")
    pages = document_to_text(scanned_pdf(PAGE, seed=1), reads(misread), reads(PAGE)).page_summary()
    command = validate_extraction(_state(pages))
    assert command.goto == "human_review"
    assert "2025" in " ".join(command.update["validation_issues"])


@pytest.mark.parametrize("main_model", ["GPT-6.1-sol", "ft:gpt-6.1-sol:acme::abc123"])
def test_same_model_skip_sees_through_case_and_fine_tunes(main_model, caplog):
    with caplog.at_level(logging.WARNING, logger="app.llm"):
        assert llm.resolve_check_model({"configurable": {"model": main_model}}) is None
    assert "second read is off" in caplog.text


def test_model_calls_have_a_timeout():
    """A hung call would otherwise hold the request open forever."""
    assert llm.get_check_llm().request_timeout == llm.REQUEST_TIMEOUT_SECONDS


def test_page_check_summary_round_trips():
    check = PageCheck(agrees=False, first_read_only=("2025",), second_read_only=("2026",))
    assert check.summary() == {"agrees": False, "first_read_only": ["2025"],
                               "second_read_only": ["2026"], "error": None}
