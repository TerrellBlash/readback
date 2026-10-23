## Project state (2026-10-23)

- **What it is:** readback, a FastAPI + LangGraph service that extracts structured fields from invoices, resumes and contracts (text, PDFs, scans) and sends doubtful documents to a human reviewer.
- **Default branch:** `main`.
- **Tests:** `python -m pytest -rs` (what CI runs). Offline by default: leave `RUN_LIVE_TESTS` unset (live tests need both `RUN_LIVE_TESTS=1` and `OPENAI_API_KEY`); `OPENAI_API_KEY` alone runs nothing live. The Postgres tests skip unless `DATABASE_URL` is set: `docker compose up -d postgres`, then `DATABASE_URL=postgresql://docagent:docagent@localhost:5433/docagent`.
- **CI:** `.github/workflows/ci.yml` runs on every push and PR: `ruff check .`, then the test suite with `DATABASE_URL` on a `postgres:16-alpine` service container, no OpenAI key. `image.yml` calls `ci.yml` as its `test` job and builds and pushes `ghcr.io/victorblash/readback` only after it passes.

Decisions in force:
- **Second-model re-read of scanned pages** (`DOC_AGENT_CHECK_MODEL`, default gpt-6.1-sol; `app/documents.py`, `app/llm.py`). Why: misreads look like valid values, so the presence-only confidence gate can't see them; reading twice with the same model repeats its misreads (commit 14589c9, `evals/experiments/`).
- **Missing fields or disagreeing reads route to human review** (`validate` in `app/nodes.py`; a failed second read counts as disagreement). Why: a document stored without review is never looked at, so silent errors are the expensive ones.
- **`X-API-Key` required when `DOC_AGENT_API_KEY` is set**, on every endpoint except `/health` and the API docs (`app/main.py`). Why: every accepted request spends OpenAI credits.
- **`store=False` on OpenAI calls** (`app/llm.py`). Why: the Responses API keeps requests (documents, page images) by default, and nothing here needs OpenAI to retain them.
- **Measured-claims wording in the README** (commit 65ae16a): results are stated as one run each on a small synthetic set, not a production accuracy figure. Why: that is all the eval measured.

Known open items:
- `app/retrieval.py` is untracked practice code, not on GitHub; `ruff check .` reports 2 errors from it locally (CI never sees it).
- Dependencies are unpinned (`>=` floors in `requirements*.txt`) with no lockfile.
- No per-request cost or latency telemetry in the app; cost and latency are measured only by `evals/run_eval.py`.
- The eval set is small (56 synthetic documents: 44 main + 12 held-out poor scans), and each configuration ran once.
- invoice-intelligence (VictorBlash/invoice-intelligence, local `~/dev/invoice-intelligence`) is a separate repo that also extracts invoices; neither README yet says how the two differ.

Working rule for Claude Code sessions: show a plan and wait for approval before editing; show each diff and wait; never commit, push or switch branches; the owner commits.
