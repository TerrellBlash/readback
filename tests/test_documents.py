"""
TESTS — turning uploads into text
===================================
Offline except the last test: a recording stub stands in for the vision
model.

Why these matter: a PDF's own text layer is exact and free, while every
vision call costs money and can misread digits. So a page with a text
layer must never reach the vision model, a scan must, and each page has
to say which one produced its text so a reviewer knows what to trust.

Run with: pytest tests/test_documents.py -v
"""

import io
import subprocess
import sys
import textwrap

import pypdfium2 as pdfium
import pytest
from PIL import Image

import app.documents as documents
from app.documents import (
    MAX_IMAGE_SIDE,
    MAX_PAGES,
    DocumentError,
    UnsupportedDocumentError,
    document_to_text,
    vision_transcriber,
)
from evals.pdfgen import LINES_PER_PAGE, scanned_pdf, text_pdf
from tests.markers import live

INVOICE_TEXT = "INVOICE INV-1001\nVendor: Acme Corp\nTotal due: $1,250.00"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class RecordingTranscriber:
    def __init__(self, reply: str = "TRANSCRIBED TEXT"):
        self.reply = reply
        self.calls: list[tuple[bytes, str]] = []

    def __call__(self, image: bytes, mime_type: str) -> str:
        self.calls.append((image, mime_type))
        return self.reply


def merge_pdfs(*pdfs: bytes) -> bytes:
    merged = pdfium.PdfDocument.new()
    for data in pdfs:
        merged.import_pages(pdfium.PdfDocument(data))
    out = io.BytesIO()
    merged.save(out)
    return out.getvalue()


def image_bytes(size: tuple[int, int], fmt: str) -> bytes:
    out = io.BytesIO()
    Image.new("L", size, 255).save(out, format=fmt)
    return out.getvalue()


def test_text_layer_pdf_never_reaches_the_vision_model():
    transcriber = RecordingTranscriber()
    document = document_to_text(text_pdf(INVOICE_TEXT), transcriber)
    assert document.text == INVOICE_TEXT
    assert [page.source for page in document.pages] == ["text_layer"]
    assert transcriber.calls == []


def test_scanned_page_is_sent_to_the_vision_model_as_png():
    transcriber = RecordingTranscriber("INVOICE INV-1001")
    document = document_to_text(scanned_pdf(INVOICE_TEXT, seed=1), transcriber)
    assert [page.source for page in document.pages] == ["vision"]
    assert len(transcriber.calls) == 1
    image, mime_type = transcriber.calls[0]
    assert image.startswith(PNG_MAGIC) and mime_type == "image/png"
    assert document.text == "INVOICE INV-1001"


def test_mixed_pdf_reads_each_page_its_own_way():
    transcriber = RecordingTranscriber("page two from the scan")
    document = document_to_text(
        merge_pdfs(text_pdf(INVOICE_TEXT), scanned_pdf("Page two", seed=2)), transcriber
    )
    assert [(page.page_number, page.source) for page in document.pages] == [
        (1, "text_layer"),
        (2, "vision"),
    ]
    assert len(transcriber.calls) == 1
    assert document.page_summary() == [
        {"page": 1, "source": "text_layer", "chars": len(INVOICE_TEXT)},
        {"page": 2, "source": "vision", "chars": len("page two from the scan")},
    ]


def test_multi_page_text_keeps_page_order():
    lines = [f"Line {number}" for number in range(LINES_PER_PAGE * 2 + 5)]
    document = document_to_text(text_pdf("\n".join(lines)), RecordingTranscriber())
    assert [page.page_number for page in document.pages] == [1, 2, 3]
    assert document.text.splitlines() == [line for line in lines]


@pytest.mark.parametrize("fmt, mime_type", [("PNG", "image/png"), ("JPEG", "image/jpeg")])
def test_image_upload_goes_straight_to_the_vision_model(fmt, mime_type):
    transcriber = RecordingTranscriber("receipt text")
    document = document_to_text(image_bytes((800, 600), fmt), transcriber)
    assert [page.source for page in document.pages] == ["vision"]
    assert transcriber.calls[0][1] == mime_type


def test_large_photo_is_downscaled_before_the_vision_call():
    """A 12 MP phone photo would cost far more image tokens for no gain."""
    transcriber = RecordingTranscriber()
    document_to_text(image_bytes((4000, 3000), "JPEG"), transcriber)
    sent = Image.open(io.BytesIO(transcriber.calls[0][0]))
    assert max(sent.size) == MAX_IMAGE_SIDE


def test_unknown_file_type_is_rejected():
    with pytest.raises(UnsupportedDocumentError):
        document_to_text(b"just some plain text", RecordingTranscriber())


def test_corrupt_pdf_is_rejected():
    with pytest.raises(DocumentError):
        document_to_text(b"%PDF-1.4 this is not really a pdf", RecordingTranscriber())


def test_too_many_pages_is_rejected_before_any_vision_call():
    """Page count is checked first, so an oversized scan can't run up a bill.
    Blank pages have no text layer, so each would otherwise go to vision."""
    blank = pdfium.PdfDocument.new()
    for _ in range(MAX_PAGES + 1):
        blank.new_page(612, 792)
    out = io.BytesIO()
    blank.save(out)
    transcriber = RecordingTranscriber()
    with pytest.raises(DocumentError):
        document_to_text(out.getvalue(), transcriber)
    assert transcriber.calls == []


def test_concurrent_reads_do_not_crash_the_process():
    """PDFium isn't thread-safe; unguarded, this workload aborts the
    interpreter (SIGABRT) within seconds. Runs in a subprocess so a crash
    fails this test instead of killing the test run."""
    script = textwrap.dedent(
        """
        from concurrent.futures import ThreadPoolExecutor
        from app.documents import document_to_text
        from evals.pdfgen import text_pdf
        pdfs = [text_pdf("INVOICE INV-1001 total due 1,250.00\\n" * 30)] * 400
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda data: document_to_text(data, lambda image, mime: "text"), pdfs))
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr.decode()[-500:]


def test_oversized_page_renders_at_capped_size():
    """A page can declare any size; a tiny PDF with a huge blank page must
    not turn into a gigapixel render."""
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(4000, 4000)  # points; 2x would be 8000 x 8000 pixels
    out = io.BytesIO()
    pdf.save(out)
    transcriber = RecordingTranscriber()
    document_to_text(out.getvalue(), transcriber)
    sent = Image.open(io.BytesIO(transcriber.calls[0][0]))
    assert max(sent.size) <= MAX_IMAGE_SIDE


def test_pdf_with_a_broken_page_is_a_document_error():
    data = text_pdf(INVOICE_TEXT).replace(b"/Kids [4 0 R]", b"/Kids [9 0 R]")
    with pytest.raises(DocumentError):
        document_to_text(data, RecordingTranscriber())


def test_sideways_phone_photo_is_turned_upright():
    image = Image.new("RGB", (400, 300), "white")
    exif = image.getexif()
    exif[0x0112] = 6  # stored sideways; viewers rotate it 90 degrees
    out = io.BytesIO()
    image.save(out, format="JPEG", exif=exif)
    transcriber = RecordingTranscriber()
    document_to_text(out.getvalue(), transcriber)
    sent = Image.open(io.BytesIO(transcriber.calls[0][0]))
    assert sent.size == (300, 400)


def test_image_over_the_pixel_cap_is_refused_before_decoding():
    out = io.BytesIO()
    Image.new("1", (8000, 8000)).save(out, format="PNG")  # 64 MP, small file
    transcriber = RecordingTranscriber()
    with pytest.raises(DocumentError):
        document_to_text(out.getvalue(), transcriber)
    assert transcriber.calls == []


def test_decompression_bomb_is_a_document_error(monkeypatch):
    monkeypatch.setattr(documents, "MAX_IMAGE_PIXELS", 10**9)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)  # Pillow refuses beyond 2x this
    with pytest.raises(DocumentError):
        document_to_text(image_bytes((100, 100), "PNG"), RecordingTranscriber())


@live
def test_vision_model_reads_a_scanned_invoice():
    document = document_to_text(scanned_pdf(INVOICE_TEXT, seed=7), vision_transcriber())
    assert document.pages[0].source == "vision"
    assert "INV-1001" in document.text
    assert "1,250.00" in document.text
