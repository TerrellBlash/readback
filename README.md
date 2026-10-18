# readback

> LangGraph pipeline that reads invoices, resumes and contracts from text, PDFs and scans. A second model re-reads every scanned page, anything doubtful goes to a human reviewer, and a published eval measures how often a wrong value gets through. Live on Azure Container Apps.

The name comes from radio readback: a pilot repeats an instruction back, and a mismatch gets caught before anyone acts on it.

## What It Does

Send it an invoice, resume or contract as text, a PDF or a photo:

1. **Read.** A PDF page with a text layer is read directly. A scanned page or
   a PNG/JPEG goes to a vision model, and a second model reads it again;
   each page records where its text came from and whether the two reads
   agreed.
2. **Classify.** The model names the document type with a confidence. An
   unknown type, or a confidence below 0.7, goes to review.
3. **Extract.** A per-type schema is filled with structured output. A field
   the document doesn't have comes back null.
4. **Validate.** A document with every required field present and, for
   scanned pages, both reads agreeing is stored. Anything else goes to
   review, with the reasons.
5. **Review.** LangGraph's `interrupt()` pauses the run and saves it to
   Postgres. A reviewer approves, corrects or rejects it through `/review`,
   from any worker, after any restart.
6. **Return.** The API returns the extracted fields, the confidence, each
   page's source, and the review decision. Writing them to a system of
   record is left to the caller.

## Architecture

```
PDF / image → Read (text layer, or vision model for scans)
                    │
Text ───────────────┴→ Ingest → Classify ─┬─→ Extract Invoice  ─┐
                                   │       ├─→ Extract Resume   ─┼─→ Validate ─┬─→ Store Results
                                   │       └─→ Extract Contract ─┘             │
                                   └───────────────────────────────────────────┴─→ Human Review → Store Results
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
pip install -r requirements-dev.txt   # requirements.txt alone to only run the server

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

### Review a Paused Document

A response with `needs_review: true` lists the reasons in
`review_context.issues`. Approve it, optionally with corrections:

```bash
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "THREAD_ID_FROM_THE_RESPONSE",
    "approved": true,
    "corrections": {"due_date": "2024-04-15"}
  }'
```

Corrections are checked against the document type's schema before the run
resumes (422 for an unknown field or a wrong type), and the stored
confidence is recomputed from the corrected fields. A review of an unknown
thread is a 404, and of one that isn't waiting, a 409.
`GET /status/{thread_id}` returns a document's current state.

### Run Tests

```bash
pytest                       # offline: models are stubbed or scripted
RUN_LIVE_TESTS=1 pytest      # also runs the few tests that call the OpenAI API
```

The Postgres tests (checkpointer setup, and paused reviews surviving a
restart) are skipped unless `DATABASE_URL` is set:

```bash
docker compose up -d postgres
export DATABASE_URL=postgresql://docagent:docagent@localhost:5433/docagent
pytest
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
│   ├── state.py          # Graph state
│   ├── nodes.py          # Graph nodes: classify, extract, validate, review, store
│   ├── schemas.py        # Extraction schemas per document type
│   ├── llm.py            # Model selection, the second-read model, timeouts
│   ├── documents.py      # PDF/image to text: text layer, vision read, second read
│   ├── graph.py          # Graph wiring and the Postgres checkpointer
│   └── main.py           # FastAPI endpoints, API key, body limits
├── tests/
│   ├── test_pipeline.py              # The whole graph against the real model (RUN_LIVE_TESTS=1)
│   ├── test_review_api.py            # Review loop through the API, scripted model
│   ├── test_read_check.py            # Second read: comparison, flags, refusals
│   ├── test_api_auth.py              # API key check
│   ├── test_api_file.py              # Upload endpoint: errors and page provenance
│   ├── test_documents.py             # PDF/image reading, page sources, limits
│   ├── test_extraction.py            # Schema-enforced extraction against a stub model
│   ├── test_confidence.py            # Extraction confidence scoring
│   ├── test_llm.py                   # Model selection and per-run override
│   ├── test_scoring.py               # Eval field-comparison rules
│   ├── test_eval_summary.py          # Eval routing and accuracy metrics
│   ├── test_checkpointer_restart.py  # Paused reviews survive a worker restart (needs Postgres)
│   └── test_checkpointer_infra.py    # Postgres pool, setup lock, fresh database (needs Postgres)
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
├── Dockerfile
├── docker-compose.yml    # Local Postgres for tests
├── deploy.sh             # Azure Container Apps deployment
├── requirements.txt      # Runtime dependencies
├── requirements-dev.txt  # Plus pytest and ruff
└── pyproject.toml        # ruff settings
```

## Paused Reviews and Postgres

`interrupt()` saves the paused run through the graph's checkpointer, so
where a pause lives depends on `DATABASE_URL`:

- **Set:** `build_checkpointer()` in `app/graph.py` builds a connection pool
  and a `PostgresSaver`. A review sent to a different worker than the one
  that paused it still resolves, and a container restart loses nothing. On
  an empty database the 4 uvicorn workers would race to create the schema,
  so an advisory lock lets one run the setup while the others wait.
- **Unset:** an in-memory `MemorySaver`, with a warning in the log. Fine for
  local development and tests; with more than one worker, or after a
  restart, a paused review is lost.

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
- **The price is cost and reviews, more than speed.** The reads run at the
  same time, so the median document took 0.4 seconds longer on the main set
  and 2.2 seconds longer on the fresh poor scans, where every page is read
  twice. The second read costs about $0.003 per scanned page, which raises
  this mix from $0.41 to $1.77 per 1,000 documents. It sent 14 poor scans to review: 8 had a wrong
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
  misread the other model doesn't share. It happened once in the eval: on
  res-06-poor both models read the (512) area code as (312), so the reads
  agreed, and the document went to review only because its summary was
  missing. In the experiment that chose the second read, both models read
  invoice 91630 as 91830; that page was flagged only because they disagreed
  on its dates.

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

## Not Built Yet

- Per-client API keys or OAuth instead of one shared key.
- Tracing (LangSmith or OpenTelemetry).
- Writing results to a system of record, and a reviewer UI.

## License

MIT
