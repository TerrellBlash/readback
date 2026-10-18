"""
Run the pipeline over the labeled dataset and score it.

Each document goes through the same path a real upload takes: a text file
straight into the graph, a PDF through app.documents first (text layer, or
the vision model for scans). Every document runs on its own graph with an
in-memory checkpointer. A document that pauses at human review counts as
routed to review; nothing is resumed.

Usage:
    python -m evals.run_eval --model gpt-4o-mini --model gpt-6-luna
    python -m evals.run_eval --model gpt-6-luna --reasoning-effort low --limit 5
    python -m evals.run_eval --model gpt-6-luna --check-model off

Writes evals/results/<run>.json (every document's extraction, outcomes,
tokens and timing, plus the summary) and regenerates evals/results/README.md
from all result files.
"""

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from app.documents import check_transcriber, document_to_text, vision_transcriber
from app.graph import build_graph
from app.llm import DEFAULT_MODEL, DEFAULT_REASONING_EFFORT, is_reasoning_model, resolve_check_model
from evals.pricing import cost_usd
from evals.scoring import score_document

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
FORMAT_NAMES = {"text": "Plain text", "pdf_text": "Text PDF", "pdf_scan": "Scanned PDF", "pdf_scan_poor": "Poor scan"}


def run_label(model: str, effort: str, check_model: str | None = None) -> str:
    """Name a run: the model, its reasoning effort for reasoning models, and
    the model that reads vision pages a second time, if any."""
    label = f"{model}@{effort}" if is_reasoning_model(model) else model
    return f"{label}+{check_model}-check" if check_model else label


def run_document(label: dict, model: str, check_model: str | None = None) -> dict:
    """
    Process one dataset document and score it against its label.

    - Parameter label: one record from labels.json.
    - Parameter model: the OpenAI model for every step, vision included.
    - Parameter check_model: the model that reads vision pages a second
        time, or None for no second read.
    - Returns: the document's record: route, classification, extracted
        fields, field outcomes, tokens, cost and seconds. An exception is
        recorded as the error, not raised. A failed document still counts:
        its fields are scored as not extracted, and the tokens it used
        before failing are in its cost.
    """
    path = DATA_DIR / label["file"]
    config: RunnableConfig = {"configurable": {
        "thread_id": f"eval-{label['id']}", "model": model, "check_model": check_model or "off",
    }}
    graph = build_graph(MemorySaver())
    text, pages, error = None, None, None
    started = time.perf_counter()
    with get_usage_metadata_callback() as usage:
        try:
            if path.suffix == ".txt":
                text = path.read_text()
            else:
                document = document_to_text(
                    path.read_bytes(), vision_transcriber(config), check_transcriber(config)
                )
                text, pages = document.text, document.page_summary()
            # A neutral name: the dataset's file names (inv-01.txt) would
            # give the classifier the answer.
            graph.invoke(
                {"document_text": text, "document_filename": f"document{path.suffix}",
                 "document_source": "eval", "document_pages": pages},
                config,
            )
        except Exception as exc:  # one bad document shouldn't stop the run
            error = f"{type(exc).__name__}: {exc}"
    seconds = round(time.perf_counter() - started, 2)

    state = None if error else graph.get_state(config)
    values = state.values if state else {}
    classified_as = (values.get("classification") or {}).get("doc_type")
    extraction = values.get("extraction") or {}
    fields = extraction.get("fields") if classified_as == label["doc_type"] else None
    tokens = usage.usage_metadata
    costs = [cost_usd(name, used) for name, used in tokens.items()]
    priced = [cost for cost in costs if cost is not None]
    return {
        **{key: label[key] for key in ("id", "doc_type", "format", "expected_route", "absent_field")},
        "split": label.get("split", "main"),
        "error": error,
        "route": "error" if state is None else "review" if state.next else "auto",
        "classified_as": classified_as,
        "extraction_confidence": extraction.get("confidence"),
        "missing_fields": extraction.get("missing_fields"),
        "document_pages": pages,
        "text_read": None if pages is None else text,  # what the PDF path produced, for auditing
        "extracted": extraction.get("fields"),
        "outcomes": score_document(label["doc_type"], label["fields"], fields)
        if label["doc_type"] != "unknown" else {},
        "seconds": seconds,
        "input_tokens": sum(used.get("input_tokens", 0) for used in tokens.values()),
        "output_tokens": sum(used.get("output_tokens", 0) for used in tokens.values()),
        "cost_usd": sum(priced) if len(priced) == len(costs) else None,  # None if any model is unpriced
    }


def _rate(hits: int, total: int) -> float | None:
    return hits / total if total else None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct * (len(ordered) - 1)))]


def summarize(records: list[dict]) -> dict:
    """
    Aggregate document records into the published metrics.

    A clean auto-store is a document stored without review that should have
    been, classified correctly, with every field correct. Any other
    auto-stored document is a silent error.

    - Parameter records: run_document() results.
    - Returns: the metrics, rates as fractions.
    """
    outcomes = [o for r in records for o in r["outcomes"].values()]
    by_format: dict[str, list[str]] = defaultdict(list)
    by_type: dict[str, list[str]] = defaultdict(list)
    for r in records:
        by_format[r["format"]] += r["outcomes"].values()
        by_type[r["doc_type"]] += r["outcomes"].values()
    absent = [r["outcomes"].get(r["absent_field"]) for r in records if r["absent_field"]]

    def clean(r: dict) -> bool:
        return (r["expected_route"] == "auto" and r["classified_as"] == r["doc_type"]
                and all(o == "correct" for o in r["outcomes"].values()))

    auto = [r for r in records if r["route"] == "auto"]
    expected_auto = [r for r in records if r["expected_route"] == "auto"]
    expected_review = [r for r in records if r["expected_route"] == "review"]
    costs = [r["cost_usd"] for r in records if r["cost_usd"] is not None]
    checks = [page["check"] for r in records for page in (r.get("document_pages") or []) if "check" in page]
    return {
        "documents": len(records),
        "errors": sum(1 for r in records if r["error"]),
        "classification_accuracy": _rate(sum(r["classified_as"] == r["doc_type"] for r in records), len(records)),
        "field_accuracy": _rate(outcomes.count("correct"), len(outcomes)),
        "field_accuracy_by_format": {k: _rate(v.count("correct"), len(v)) for k, v in by_format.items() if v},
        "field_accuracy_by_type": {k: _rate(v.count("correct"), len(v)) for k, v in by_type.items() if v},
        "field_outcomes": {k: outcomes.count(k) for k in sorted(set(outcomes))},
        "absent_fields_left_empty": _rate(absent.count("correct"), len(absent)),
        "straight_through_rate": _rate(sum(r["route"] == "auto" for r in expected_auto), len(expected_auto)),
        "review_catch_rate": _rate(sum(r["route"] == "review" for r in expected_review), len(expected_review)),
        "auto_stored": len(auto),
        "silent_errors": sum(not clean(r) for r in auto),
        "silent_error_ids": [r["id"] for r in auto if not clean(r)],
        "auto_store_precision": _rate(sum(clean(r) for r in auto), len(auto)),
        "seconds_p50": _percentile([r["seconds"] for r in records], 0.5),
        "seconds_p95": _percentile([r["seconds"] for r in records], 0.95),
        "mean_input_tokens": statistics.mean(r["input_tokens"] for r in records) if records else None,
        "mean_output_tokens": statistics.mean(r["output_tokens"] for r in records) if records else None,
        "cost_per_1000_docs_usd": statistics.mean(costs) * 1000 if costs else None,
        "pages_checked": len(checks),
        "pages_flagged_by_check": sum(not check["agrees"] and not check.get("error") for check in checks),
        "pages_unchecked": sum(bool(check.get("error")) for check in checks),
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _what_went_wrong(record: dict) -> str:
    """One line on why an auto-stored document counts as a silent error."""
    if record["classified_as"] != record["doc_type"]:
        return f"classified as {record['classified_as']}, not {record['doc_type']}"
    problems = [f"{field} {outcome.replace('_', ' ')}"
                for field, outcome in record["outcomes"].items() if outcome != "correct"]
    return ", ".join(problems) or "should have gone to review"


def markdown_report(results: list[dict]) -> str:
    """Comparison tables for every saved run, then each run's silent errors."""
    accuracy = ["| Run | Fields correct | " + " | ".join(FORMAT_NAMES.values())
                + " | Classification | Absent fields left empty |",
                "|---|---|" + "---|" * len(FORMAT_NAMES) + "---|---|"]
    routing = ["| Run | Auto-stored | Silent errors | Auto-store precision | Straight-through "
               "| Review catch rate | p50 / p95 seconds | Cost per 1,000 docs | Errors |",
               "|---|---|---|---|---|---|---|---|---|"]
    holdout = ["| Run | Fields correct | Auto-stored | Silent errors | Straight-through "
               "| Pages flagged by check | Cost per 1,000 docs |",
               "|---|---|---|---|---|---|---|"]
    silent = ["Documents stored without review that had something wrong.", ""]
    for result in results:
        s, name = result["summary"], result["run"]
        by_format = " | ".join(_pct(s["field_accuracy_by_format"].get(fmt)) for fmt in FORMAT_NAMES)
        accuracy.append(f"| {name} | {_pct(s['field_accuracy'])} | {by_format} "
                        f"| {_pct(s['classification_accuracy'])} | {_pct(s['absent_fields_left_empty'])} |")
        cost = "n/a" if s["cost_per_1000_docs_usd"] is None else f"${s['cost_per_1000_docs_usd']:.2f}"
        routing.append(f"| {name} | {s['auto_stored']} of {s['documents']} | {s['silent_errors']} "
                       f"| {_pct(s['auto_store_precision'])} | {_pct(s['straight_through_rate'])} "
                       f"| {_pct(s['review_catch_rate'])} | {s['seconds_p50']:.1f} / {s['seconds_p95']:.1f} "
                       f"| {cost} | {s['errors']} |")
        h = result.get("holdout_summary")
        if h:
            h_cost = "n/a" if h["cost_per_1000_docs_usd"] is None else f"${h['cost_per_1000_docs_usd']:.2f}"
            holdout.append(f"| {name} | {_pct(h['field_accuracy'])} | {h['auto_stored']} of {h['documents']} "
                           f"| {h['silent_errors']} | {_pct(h['straight_through_rate'])} "
                           f"| {h['pages_flagged_by_check']} of {h['pages_checked']} | {h_cost} |")
        records = {r["id"]: r for r in result["documents"]}
        ids = s["silent_error_ids"] + (h["silent_error_ids"] if h else [])
        silent.append(f"- **{name}**" + ("" if ids else ": none"))
        silent += [f"  - {doc_id}, {FORMAT_NAMES[records[doc_id]['format']]}"
                   f"{' (held out)' if records[doc_id].get('split') == 'holdout' else ''}: "
                   f"{_what_went_wrong(records[doc_id])}" for doc_id in ids]
    sections = ["\n".join(accuracy), "\n".join(routing)]
    if len(holdout) > 2:
        sections += ["## Fresh poor scans (held out)", "\n".join(holdout)]
    return "\n\n".join(sections + ["## Silent errors", "\n".join(silent)]) + "\n"


def write_results_readme() -> None:
    """Regenerate results/README.md from every saved run."""
    results = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
    header = ("# Eval results\n\nGenerated by `python -m evals.run_eval` from the JSON files in this folder. "
              "See [../README.md](../README.md) for the dataset and what each number means.\n\n")
    runs = "\n".join(f"- **{r['run']}**: {r['run_at'][:10]}, {len(r['documents'])} documents, "
                     f"{sum(1 for d in r['documents'] if d['error'])} errors" for r in results)
    (RESULTS_DIR / "README.md").write_text(header + runs + "\n\n" + markdown_report(results))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", help=f"model to run (repeatable; default {DEFAULT_MODEL})")
    parser.add_argument("--reasoning-effort", default=None, help="reasoning effort for reasoning models")
    parser.add_argument("--check-model", default=None,
                        help="model that reads vision pages a second time (default: the app's; 'off' for none)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="run only the first N documents")
    args = parser.parse_args()

    if args.reasoning_effort:
        os.environ["DOC_AGENT_REASONING_EFFORT"] = args.reasoning_effort
    effort = os.environ.get("DOC_AGENT_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
    labels = json.loads((DATA_DIR / "labels.json").read_text())[: args.limit]
    RESULTS_DIR.mkdir(exist_ok=True)

    for model in args.model or [DEFAULT_MODEL]:
        check = resolve_check_model({"configurable": {"model": model, "check_model": args.check_model}})
        run = run_label(model, effort, check)
        print(f"Running {run} on {len(labels)} documents...", flush=True)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            records = list(pool.map(partial(run_document, model=model, check_model=check), labels))
        main_records = [r for r in records if r["split"] == "main"]
        holdout_records = [r for r in records if r["split"] == "holdout"]
        result = {"run": run, "model": model,
                  "reasoning_effort": effort if is_reasoning_model(model) else None, "check_model": check,
                  "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
                  "summary": summarize(main_records),
                  "holdout_summary": summarize(holdout_records) if holdout_records else None,
                  "documents": records}
        (RESULTS_DIR / f"{run}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(markdown_report([result]), flush=True)
        for r in records:
            if r["error"]:
                print(f"  error on {r['id']}: {r['error']}")
    write_results_readme()


if __name__ == "__main__":
    main()
