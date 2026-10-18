# readback

> LangGraph pipeline that reads invoices, resumes and contracts from text, PDFs and scans. A second model re-reads every scanned page, anything doubtful goes to a human reviewer, and a published eval measures how often a wrong value gets through. Live on Azure Container Apps.

The name comes from radio readback: a pilot repeats an instruction back, and a mismatch gets caught before anyone acts on it.



## What This Does

Feed it any document (invoice, resume, or contract), as text, a PDF, or a photo, and the agent:

1. **Reads** — a PDF's own text layer when it has one; scanned pages and PNG/JPEG photos go to the vision model and are read again by a second model, and each page records which source produced its text and whether the two reads agreed
2. **Ingests** — validates the document is processable
3. **Classifies** — LLM determines document type + confidence score
4. **Extracts** — specialist node fills a per-type schema with structured output; fields the document lacks come back null
5. **Validates** — confidence gate checks for missing fields and for pages where the two reads disagree
6. **Routes** — high confidence → auto-store, low confidence → human review
7. **Stores** — packages clean JSON output ready for any database

The human-in-the-loop pattern uses LangGraph's `interrupt()` — the pipeline literally pauses, saves all state, and waits for a human to approve or correct the extraction. Could be 5 seconds or 5 days.

## Architecture

```
PDF / image → Read (text layer, or vision model for scans)
                    │
Text ───────────────┴→ Ingest → Classify ─┬─→ Extract Invoice  ─┐
                                           ├─→ Extract Resume   ─┼─→ Validate ─┬─→ Store Results
                                           └─→ Extract Contract ─┘             │
                                                                                └─→ Human Review → Store Results
```

## Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key

### Local Development

```bash
# Clone and install
git clone https://github.com/TerrellBlash/readback.git
cd readback
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY="sk-..."

# Run the server
uvicorn app.main:app --reload --port 8000

# Health check
curl http://localhost:8000/health
```

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | none | Required. |
| `DOC_AGENT_MODEL` | `gpt-6-luna` | OpenAI model for every step. A run can override it with `{"configurable": {"model": ...}}`. |
| `DOC_AGENT_REASONING_EFFORT` | `none` | Reasoning effort for reasoning models (GPT-5 and later). Older models run at temperature 0 instead. Some models don't accept `none` (gpt-6.1-sol needs `low` or higher) and fail every call with a 400 until this is set. |
| `DOC_AGENT_CHECK_MODEL` | `gpt-6.1-sol` | A second model that reads every scanned page and photo again; a disagreement sends the document to review. `off` disables it. It's skipped when it would be the run's own model. |
| `DOC_AGENT_CHECK_REASONING_EFFORT` | `low` | Reasoning effort for the check model. |
| `DATABASE_URL` | unset | Postgres for paused reviews. Unset falls back to in-memory checkpoints. |
| `DOC_AGENT_API_KEY` | unset | When set, every endpoint except `/health` and the API docs requires it in an `X-API-Key` header. Leave unset only for local development. |

### Process a Document

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "document_text": "INVOICE #INV-2024-0847\nFrom: CloudTech Solutions\nTotal: $19,593.25",
    "document_filename": "invoice_march.pdf",
    "document_source": "email_attachment"
  }'
```

### Process a PDF or Image

```bash
curl -X POST http://localhost:8000/process/file \
  -F "file=@invoice.pdf" \
  -F "document_source=email_attachment"
```

PDF, PNG and JPEG. The type is detected from the file's bytes, not its name.
`final_output.document_pages` shows each page's source (`text_layer` or
`vision`), so a reviewer knows which pages a model transcribed.

Uploads are untrusted, so each size is checked before the work it would cost:

| Limit | Value | Checked |
|---|---|---|
| Upload size | 20 MB | From `Content-Length`, before the body is read (411 without one) |
| Pages | 20 | Before any page is read or sent to the vision model |
| Rendered page | 2048 px on the long side | Before a scanned page is drawn, whatever size the PDF declares |
| Image | 50 megapixels | Before the image is decoded |
| Text | 120,000 characters | Before the graph runs, since every extraction prompt carries the full text |

Errors return a short message. Details go to the server log, not the client.

### Submit Human Review

```bash
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "YOUR_THREAD_ID_HERE",
    "approved": true,
    "corrections": {"total_amount": 19593.25}
  }'
```

### Run Tests

```bash
pytest tests/test_pipeline.py -v
```

#### Local Postgres for testing

The checkpointer test that proves paused reviews survive a restart
(`tests/test_checkpointer_restart.py`) needs a real Postgres instance —
it's skipped automatically if `DATABASE_URL` isn't set. To run it:

```bash
docker compose up -d postgres
export DATABASE_URL=postgresql://docagent:docagent@localhost:5433/docagent
pytest tests/test_checkpointer_restart.py -v
```

## Deploy to Azure

Every push to main that changes the app builds its image with GitHub
Actions and publishes it to GitHub's container registry
([.github/workflows/image.yml](.github/workflows/image.yml)). `deploy.sh`
runs that image on Azure Container Apps:

```bash
brew install azure-cli
az login
export OPENAI_API_KEY="sk-..."
# DATABASE_URL='postgresql://...' (direct, not pooled) in ~/.config/doc-agent/neon.env
./deploy.sh
```

- The app scales to zero when idle, so it costs nothing between
  requests. The first request after a quiet spell takes about 25 seconds
  while it starts.
- Paused reviews live in the Postgres database you point it at. A free
  [Neon](https://neon.com) project works. It suspends when idle too; the
  connection pool checks each connection before use and reconnects.
- The deployed API requires an `X-API-Key` header, since every accepted
  request spends OpenAI credits. The script generates the key on its
  first run, keeps it in `~/.config/doc-agent/azure.env` (mode 600), and
  never prints it.

Within Azure's monthly free allowance this setup costs nothing: there's no
Azure registry and no Azure database to pay for.

## Project Structure

```
readback/
├── app/
│   ├── __init__.py
│   ├── state.py          # State definitions
│   ├── nodes.py          # All node functions (each does one job)
│   ├── schemas.py        # Extraction schemas per document type
│   ├── llm.py            # Model selection (DOC_AGENT_MODEL, per-run override)
│   ├── documents.py      # PDF/image to text: text layer first, vision model for scans
│   ├── graph.py          # Graph wiring (connects nodes with edges), build_checkpointer()
│   └── main.py           # FastAPI server (API endpoints)
├── tests/
│   ├── test_pipeline.py              # Full pipeline tests with sample docs
│   ├── test_confidence.py            # Unit tests for extraction confidence scoring
│   ├── test_extraction.py            # Schema-enforced extraction against a stub model
│   ├── test_llm.py                   # Model selection and per-run override
│   ├── test_documents.py             # PDF/image reading, page sources, limits
│   ├── test_api_file.py              # Upload endpoint: errors and page provenance
│   ├── test_scoring.py               # Eval field-comparison rules
│   ├── test_eval_summary.py          # Eval routing and accuracy metrics
│   ├── test_checkpointer_restart.py  # Proves paused reviews survive a worker restart (needs Postgres)
│   └── test_checkpointer_infra.py    # LLM-free Postgres persistence and setup tests (needs Postgres)
├── evals/
│   ├── README.md         # Dataset, scoring rules, metric definitions, limits
│   ├── build_dataset.py  # Seeded synthetic dataset: 56 docs across text, PDFs, scans
│   ├── run_eval.py       # Runs the pipeline per model; accuracy, routing, latency, cost
│   ├── scoring.py        # Field comparison rules
│   ├── pricing.py        # Token prices for cost estimates
│   ├── pdfgen.py         # Builds text-layer and scanned PDFs for tests and the eval set
│   ├── data/             # The dataset and labels.json
│   └── results/          # One JSON per run, plus a generated summary
├── .github/workflows/image.yml  # Builds and publishes the image on push to main
├── Dockerfile            # Container config for Azure
├── docker-compose.yml    # Local Postgres for development/testing only
├── deploy.sh             # One-command Azure deployment
├── requirements.txt
└── README.md
```

## Key LangGraph Concepts Demonstrated

| Concept | Where It's Used |
|---------|----------------|
| **State design** | `state.py` — raw data, never formatted prompts |
| **Node functions** | `nodes.py` — each function does one thing |
| **Command routing** | `classify_document()` — LLM decides next node |
| **Structured output** | `schemas.py` — every field required and nullable, so the model must state each one |
| **Confidence gating** | `validate_extraction()` — auto-store vs human review |
| **interrupt()** | `human_review()` — pause/resume across sessions |
| **Retry policies** | `graph.py` — extraction nodes retry on failure |
| **Checkpointing** | `graph.py` — `build_checkpointer()` picks Postgres or MemorySaver, enabling durable execution |

## Checkpointing & Persistence

The pipeline's `human_review` step pauses the graph with LangGraph's
`interrupt()` — state has to live somewhere for that pause to survive
until a human responds. Where it lives depends on `DATABASE_URL`:

- **`DATABASE_URL` set** — `build_checkpointer()` (in `app/graph.py`) builds
  a `psycopg_pool.ConnectionPool` and a `PostgresSaver` on top of it. Every
  worker process's `.setup()` call is idempotent, so all 4 uvicorn workers
  can run it concurrently on startup. A review submitted to a different
  worker than the one that paused it still resolves, and a full container
  restart doesn't lose pending reviews.
- **`DATABASE_URL` not set** — falls back to an in-memory `MemorySaver`,
  logs a `WARNING`, and keeps running. This is fine for local development
  and tests, but **not restart-safe**: a container restart or a request
  landing on a different worker silently loses any paused review. Never
  run it this way in a multi-worker deployment.

For local Postgres setup, see "Local Postgres for testing" above.

## Evaluation

The models and the second read were chosen by running the pipeline over
labeled synthetic documents: 44 invoices, resumes, contracts and others as plain
text, text-layer PDFs, scanned PDFs and poor scans (80 DPI, heavy blur), plus
12 fresh poor scans held out until the second read had been chosen.
[evals/README.md](evals/README.md) covers the dataset, scoring and how the
second read was picked. Every run's per-document output is in
[evals/results/](evals/results/README.md).

| Run | Silent errors, main set (44) | Silent errors, fresh poor scans (12) | Stored without review when it should be, main / fresh | Cost per 1,000 docs | p50 seconds |
|---|---|---|---|---|---|
| gpt-4o-mini | 3 | 4 | 77% / 75% | $1.56 | 3.7 |
| gpt-6-luna | 3 | 8 | 100% / 100% | $0.41 | 4.7 |
| **gpt-6-luna, read again by gpt-6.1-sol** | **0** | **0** | 84% / 25% | $1.77 | 5.0 |
| gpt-6-luna, effort low | 2 | 9 | 100% / 100% | $0.55 | 5.8 |
| gpt-6.1-sol | 3 | 2 | 100% / 100% | $7.25 | 9.8 |

A silent error is a document stored without review that had something
wrong in it. Those are the expensive ones, because nobody looks at them.

What the numbers say:

- **Clean input is solved.** On plain text, text PDFs and ordinary scans,
  every gpt-6 run extracted every field correctly, and the second read
  flagged none of the ordinary scans.
- **Every silent error came from a poor scan.** A misread looks like a valid
  value: 2026 read as 2025, invoice 91630 as 91830, a (312) area code as
  (912). The confidence gate only checks that fields are present. On its
  own, gpt-6-luna stored 8 of the 12 fresh poor scans with a wrong field.
- **A second model reading the page closes the gap.** Asking the same model
  again repeats its misreads, and even gpt-6.1-sol alone made silent
  errors. Disagreement between the two is the signal: with it, the run had no
  silent errors, on the main set or on the fresh poor scans it was never
  tuned on.
- **The price is cost and reviews, not speed.** The reads run at the same
  time, so the median document takes 0.3 seconds longer. The second read
  costs about $0.0026 per scanned page, which raises this mix from $0.41 to
  $1.77 per 1,000 documents. It sent 14 poor scans to review: 8 had a wrong
  field, and 6 differed only in words outside the extracted fields (signer
  names, a year, an hourly rate) or in gpt-6.1-sol's own misread.
- **gpt-4o-mini declines some poor scans.** It refused to transcribe 6 of
  them. These are now rejected as unreadable instead of processed as the
  document's text, which is also why its cost fell.

The default is gpt-6-luna at effort none with the gpt-6.1-sol second read.
Set `DOC_AGENT_CHECK_MODEL=off` to trade the protection for cost. These are
one run each on a small synthetic set, good for comparing models and
changes, not a production accuracy figure.

## Known Limitations

- **Two models can misread the same way.** The second read only catches a
  misread the other model doesn't share. It didn't happen on these 23 poor
  scans, but in the experiment both models read invoice 91630 as 91830; that
  page was flagged only because they disagreed on its dates.

- **Some reviews aren't needed.** The two reads are compared word for word,
  so a difference in a signer's name sends a document to review even though
  no extracted field changed. Comparing extracted fields instead would cut
  these, at the price of a second extraction per page.

- **PDF reading is serialized per process.** PDFium isn't thread-safe, so one
  lock covers every PDFium call. Vision calls, the slow part, still run in
  parallel.

- **Every schema field counts toward confidence.** A document type's fields
  are all treated as required, so an NDA with no payment clause has a null
  `payment_terms` and goes to human review. Per-type optional fields would
  fix this.

## Production Upgrades

For a real production deployment, consider:

- **Authentication**: Replace the single shared API key with per-client keys or OAuth
- **Observability**: Connect LangSmith for tracing and debugging
- **Streaming**: Add SSE endpoints to stream progress updates to a frontend

## Tech Stack

- **LangGraph** — agent orchestration with state machines
- **LangChain** — LLM integration (OpenAI, swappable)
- **FastAPI** — async API server
- **Docker** — containerization
- **Azure Container Apps** — cloud deployment

## License

MIT
