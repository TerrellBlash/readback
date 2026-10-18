"""
Which detector catches vision misreads on scanned pages?

For each of the 22 scanned eval documents (11 office scans, 11 poor scans),
read the page five ways and compare every read with the text that was
actually printed (regenerated from the seeded dataset builder):

  luna1, luna2  gpt-6-luna, effort none, the production read, twice
  luna_med      gpt-6-luna, effort medium
  sol           gpt-6.1-sol, effort low
  luna_marked   gpt-6-luna asked to mark characters it can't read with [?]

A read "has an error" when its words and numbers differ from the source
(case-insensitive, punctuation ignored). Detectors are judged against
luna1, the read production would use:

  A  luna1 and luna2 disagree
  A' luna1 and luna_med disagree
  B  luna1 and sol disagree
  C  page resolution below a threshold
  D  luna_marked contains [?]

This is the experiment behind the table in evals/README.md, "How the
second read was chosen". It ran once, on 2026-10-16, before the held-out
poor scans existed; its reads are in second_read_results.json.

    python -m evals.experiments.second_read --summary   # the table, from the saved reads
    python -m evals.experiments.second_read             # read again (real API calls, about $0.25)
"""

import base64
import collections
import concurrent.futures as cf
import json
import random
import re
import sys
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import HumanMessage

from app.documents import TRANSCRIBE_PROMPT, _clean, _read_pdf
from app.llm import _build
from evals import build_dataset as bd
from evals.pricing import cost_usd

OUT = Path(__file__).with_name("second_read_results.json")
MARKED_PROMPT = TRANSCRIBE_PROMPT + (
    " If you are not sure what a character is, write your best guess and put"
    " [?] right after the word that contains it."
)
READS = {
    "luna1": ("gpt-6-luna", "none", TRANSCRIBE_PROMPT),
    "luna2": ("gpt-6-luna", "none", TRANSCRIBE_PROMPT),
    "luna_med": ("gpt-6-luna", "medium", TRANSCRIBE_PROMPT),
    "sol": ("gpt-6.1-sol", "low", TRANSCRIBE_PROMPT),
    "luna_marked": ("gpt-6-luna", "none", MARKED_PROMPT),
}


def source_texts() -> dict[str, str]:
    texts = {}
    for doc_type, count, absent_by_index in bd.PLAN:
        for index in range(count):
            doc_id = f"{bd.PREFIX[doc_type]}-{index + 1:02d}"
            rng = random.Random(f"{bd.SEED}-{doc_id}")
            texts[doc_id], _ = bd.MAKERS[doc_type](rng, index, absent_by_index.get(index))
    for index, text in enumerate(bd.OTHER_DOCUMENTS):
        texts[f"unk-{index + 1:02d}"] = text
    return texts


def tokens(text: str) -> collections.Counter:
    return collections.Counter(re.findall(r"[a-z0-9]+", text.lower().replace("[?]", "")))


def differences(a: str, b: str) -> list[str]:
    ta, tb = tokens(a), tokens(b)
    return sorted((ta - tb).elements()) + [f"+{t}" for t in sorted((tb - ta).elements())]


def page_dpi(data: bytes) -> float:
    pdf = pdfium.PdfDocument(data)
    try:
        page = pdf[0]
        width_pt = page.get_size()[0]
        for obj in page.get_objects():
            if obj.type == pdfium_c.FPDF_PAGEOBJ_IMAGE:
                return obj.get_px_size()[0] / (width_pt / 72)
        return 0.0
    finally:
        pdf.close()


def read(name: str, png: bytes) -> tuple[str, float]:
    model, effort, prompt = READS[name]
    image = base64.b64encode(png).decode()
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
    ])
    with get_usage_metadata_callback() as usage:
        text = _build(model, effort).invoke([message]).text
    cost = sum(cost_usd(m, u) or 0.0 for m, u in usage.usage_metadata.items())
    return _clean(text), cost


def run_document(label: dict, sources: dict[str, str]) -> dict:
    data = (bd.DATA_DIR / label["file"]).read_bytes()
    pages = _read_pdf(data)
    assert len(pages) == 1 and pages[0].png is not None, label["id"]
    reads, costs = {}, {}
    for name in READS:
        reads[name], costs[name] = read(name, pages[0].png)
    source = sources[label["id"].removesuffix("-poor")]
    return {
        "id": label["id"], "format": label["format"], "dpi": round(page_dpi(data)),
        "reads": reads, "costs": costs,
        "errors": {name: differences(source, text) for name, text in reads.items()},
        "luna1_vs_luna2": differences(reads["luna1"], reads["luna2"]),
        "luna1_vs_med": differences(reads["luna1"], reads["luna_med"]),
        "luna1_vs_sol": differences(reads["luna1"], reads["sol"]),
        "marked": "[?]" in reads["luna_marked"],
    }


# Each detector: when it fires, and what it adds per page on top of luna1.
DETECTORS = {
    "luna1 vs luna2": (lambda r: bool(r["luna1_vs_luna2"]), lambda c: c["luna2"]),
    "luna1 vs luna_med": (lambda r: bool(r["luna1_vs_med"]), lambda c: c["luna_med"]),
    "luna1 vs sol": (lambda r: bool(r["luna1_vs_sol"]), lambda c: c["sol"]),
    "page under 100 DPI": (lambda r: r["dpi"] < 100, lambda c: 0.0),
    # The marked read would replace luna1, so only the difference counts.
    "[?] marks": (lambda r: r["marked"], lambda c: c["luna_marked"] - c["luna1"]),
}


def summarize(results: list[dict]) -> str:
    """The detector table: misreads caught, false alarms, extra cost per page."""
    misread = [bool(r["errors"]["luna1"]) for r in results]
    lines = [f"luna1 misread {sum(misread)} of {len(results)} pages",
             "| Detector | Misreads caught | False alarms | Extra cost per page |", "|---|---|---|---|"]
    for name, (fires, extra_cost) in DETECTORS.items():
        caught = sum(m and fires(r) for m, r in zip(misread, results, strict=True))
        false = sum(not m and fires(r) for m, r in zip(misread, results, strict=True))
        cost = sum(extra_cost(r["costs"]) for r in results) / len(results)
        lines.append(f"| {name} | {caught} of {sum(misread)} | {false} | ${cost:.4f} |")
    return "\n".join(lines)


def main() -> None:
    if "--summary" in sys.argv:
        print(summarize(json.loads(OUT.read_text())))
        return
    labels = json.loads((bd.DATA_DIR / "labels.json").read_text())
    # The held-out scans were added after this experiment, to test its pick.
    docs = [label for label in labels
            if label["format"] in ("pdf_scan", "pdf_scan_poor") and label.get("split") != "holdout"]
    sources = source_texts()
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda label: run_document(label, sources), docs))
    OUT.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
