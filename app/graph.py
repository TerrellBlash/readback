"""
Graph wiring and the checkpointer that persists paused reviews.

Only the fixed edges are declared here (START -> ingest -> classify, and
store_results -> END). Every branch is a Command returned by the node that
makes the decision; see app/nodes.py.

Nodes have no retry policy of their own: the OpenAI client already retries
rate limits, timeouts and server errors twice (see app/llm.py), and a
graph-level retry on top would multiply a 90-second timeout ninefold.
"""

import logging
import os
import time

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.nodes import (
    classify_document,
    extract_contract,
    extract_invoice,
    extract_resume,
    human_review,
    ingest_document,
    store_results,
    validate_extraction,
)
from app.state import DocAgentState

logger = logging.getLogger(__name__)



def build_checkpointer():
    """
    Build the checkpointer the graph will use to persist state.

    Why this matters: a paused `human_review` interrupt only lives inside
    whichever checkpointer backs the graph. The API server runs multiple
    uvicorn workers (separate processes, separate memory), and a container
    restart wipes process memory entirely. An in-memory checkpointer means
    a review submitted to a different worker than the one that paused it
    404s, and any restart silently drops every pending review.

    If DATABASE_URL is set, this builds a connection pool and a
    PostgresSaver backed by it, so paused state lives in the database
    instead of a single process's RAM — any worker, and a restart, can
    pick it back up. If it's not set, this falls back to MemorySaver,
    which is fine for local development and tests, but NOT fine for a
    multi-worker or restart-prone deployment.

    - Returns: a `PostgresSaver` when `DATABASE_URL` is configured,
        otherwise a `MemorySaver`.
    - Throws: psycopg errors or `PoolTimeout` when `DATABASE_URL` is set
        but the database can't be reached.
    """
    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        logger.warning(
            "No DATABASE_URL configured — falling back to in-memory "
            "checkpointing. Paused reviews will NOT survive a restart or "
            "a different worker process. This is only acceptable for "
            "local development or tests."
        )
        return MemorySaver()

    # Imported lazily so environments without DATABASE_URL (and without
    # these optional dependencies installed) never need them.
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg import Connection
    from psycopg.rows import DictRow, dict_row
    from psycopg_pool import ConnectionPool

    # check= tests each connection before handing it out and replaces a
    # dead one. Serverless Postgres (Neon) closes every connection when it
    # suspends after a few idle minutes, and networks drop idle ones too;
    # without the check, the next request after a quiet spell fails.
    pool: ConnectionPool[Connection[DictRow]] = ConnectionPool(
        conninfo=database_url,
        kwargs={"autocommit": True, "row_factory": dict_row},
        open=True,
        check=ConnectionPool[Connection[DictRow]].check_connection,
    )
    checkpointer = PostgresSaver(pool)

    # .setup() is idempotent ONCE the schema exists — but on a freshly
    # provisioned, not-yet-migrated database it is not safe to call
    # concurrently. It runs with autocommit (each statement commits as it
    # runs, not as one transaction), so two workers can both read "no
    # migrations applied yet" before either one's insert commits, and the
    # second insert then collides on a primary key. The Dockerfile starts
    # 4 uvicorn workers, each calling build_checkpointer() independently on
    # startup, so this race is real on first deploy against an empty
    # database. A Postgres advisory lock serializes it: whichever worker
    # gets here first runs the migration, the rest wait their turn, then
    # find there's nothing left to do.
    #
    # The waiting workers poll with pg_try_advisory_lock instead of blocking
    # in pg_advisory_lock. On an empty database, setup() builds indexes with
    # CREATE INDEX CONCURRENTLY, which waits for every running statement to
    # finish, and a blocked pg_advisory_lock call is a running statement.
    # The index build and the waiting workers would then wait on each other
    # forever.
    _SETUP_LOCK_KEY = 0x646F6361  # arbitrary fixed key, spells "doca"
    with pool.connection() as conn:
        while True:
            row = conn.execute("SELECT pg_try_advisory_lock(%s) AS got", (_SETUP_LOCK_KEY,)).fetchone()
            if row and row["got"]:
                break
            time.sleep(0.5)
        try:
            checkpointer.setup()
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_SETUP_LOCK_KEY,))

    # PostgresSaver doesn't own the pool, so keep a reference for
    # close_checkpointer(); otherwise every build_graph() leaks a pool.
    checkpointer._doc_agent_pool = pool

    logger.info("Persistent Postgres checkpointing is active (DATABASE_URL configured).")

    return checkpointer


def close_checkpointer(checkpointer):
    """
    Release whatever resources `checkpointer` is holding, if any.

    A MemorySaver holds nothing and this is a no-op. A PostgresSaver built
    by `build_checkpointer()` carries a live connection pool stashed at
    `_doc_agent_pool`; this closes it. Call this on app shutdown (see
    app/main.py's lifespan) and in any test that builds its own graph
    directly, so pools don't accumulate for the life of a long-running
    test session or a dev server that restarts often.
    """
    pool = getattr(checkpointer, "_doc_agent_pool", None)
    if pool is not None:
        pool.close()


def build_graph(checkpointer=None):
    """
    Build and compile the document processing graph.

    - Parameter checkpointer: the checkpointer to compile the graph with.
        When omitted (None), `build_checkpointer()` is called to pick one
        based on whether DATABASE_URL is configured. Tests that want to
        simulate a worker restart build two independent graphs this way —
        each call to `build_checkpointer()` (and, for Postgres, each new
        connection pool) stands in for a separate worker process, while
        the underlying Postgres database is what actually persists state
        across them.
    - Returns: the compiled graph ready to invoke.
    """
    workflow = StateGraph(DocAgentState)

    workflow.add_node("ingest", ingest_document)
    workflow.add_node("classify", classify_document)
    workflow.add_node("extract_invoice", extract_invoice)
    workflow.add_node("extract_resume", extract_resume)
    workflow.add_node("extract_contract", extract_contract)
    workflow.add_node("validate_extraction", validate_extraction)
    workflow.add_node("human_review", human_review)
    workflow.add_node("store_results", store_results)

    workflow.add_edge(START, "ingest")
    workflow.add_edge("ingest", "classify")
    workflow.add_edge("store_results", END)

    # interrupt() needs a checkpointer to save the paused run.
    if checkpointer is None:
        checkpointer = build_checkpointer()
    return workflow.compile(checkpointer=checkpointer)


# The API server's graph, built once per worker process.
graph = build_graph()
