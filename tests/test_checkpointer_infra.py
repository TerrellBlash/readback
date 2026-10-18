"""
TESTS — Postgres checkpointer persistence, isolated from the LLM
====================================================================
These test the checkpointer itself with a two-line graph that pauses
immediately: the connection pool, the advisory lock around .setup() on a
fresh database, a database that drops idle connections, and persistence
across two independent instances. test_checkpointer_restart.py covers the
same guarantee through the real pipeline.

Needs a real Postgres instance, so it's skipped unless DATABASE_URL is
set. See README.md "Run Tests".

Run with: pytest tests/test_checkpointer_infra.py -v
"""

import os
from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.graph import build_checkpointer, close_checkpointer

requires_postgres = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="requires a running Postgres instance; see README",
)


class MiniState(TypedDict):
    value: str


def _pause_node(state: MiniState) -> dict:
    decision = interrupt({"ask": "approve?"})
    return {"value": f"resumed-with-{decision}"}


def _build_mini_graph(checkpointer):
    """A two-node graph whose only job is to pause and resume. No LLM,
    no app.nodes dependency — this isolates the checkpointer itself."""
    workflow = StateGraph(MiniState)
    workflow.add_node("pause", _pause_node)
    workflow.add_edge(START, "pause")
    workflow.add_edge("pause", END)
    return workflow.compile(checkpointer=checkpointer)


@requires_postgres
def test_postgres_checkpointer_persists_across_independent_instances():
    """Two independent build_checkpointer() calls must share state via
    Postgres, with no LLM call anywhere in this test."""
    thread_id = "infra-checkpointer-restart-001"
    config = {"configurable": {"thread_id": thread_id}}

    # --- "Worker process #1" ---
    checkpointer_1 = build_checkpointer()
    graph_1 = _build_mini_graph(checkpointer_1)
    graph_1.invoke({"value": "start"}, config)

    state_1 = graph_1.get_state(config)
    assert state_1 is not None
    assert state_1.next, "Expected the mini graph to be paused at 'pause'."
    assert state_1.tasks
    assert any(task.interrupts for task in state_1.tasks)

    # --- "Worker process #2": a fully independent checkpointer + graph ---
    checkpointer_2 = build_checkpointer()
    graph_2 = _build_mini_graph(checkpointer_2)
    assert graph_2 is not graph_1
    assert checkpointer_2 is not checkpointer_1

    state_2 = graph_2.get_state(config)
    assert state_2 is not None
    assert state_2.next, (
        "A second, independent checkpointer instance should still see "
        "the pending interrupt — this is what proves persistence."
    )
    assert state_2.tasks
    assert any(task.interrupts for task in state_2.tasks)

    # --- Resume the SECOND instance ---
    result = graph_2.invoke(Command(resume="yes"), config)
    assert result["value"] == "resumed-with-yes"

    final_state = graph_2.get_state(config)
    assert not final_state.next, "Pipeline should have finished."

    close_checkpointer(checkpointer_1)
    close_checkpointer(checkpointer_2)


@requires_postgres
def test_build_checkpointer_setup_is_safe_to_call_concurrently():
    """Several threads calling build_checkpointer() at once — standing in
    for the Dockerfile's 4 concurrently-starting uvicorn workers — must
    all succeed, even against a database that may not have its schema
    yet. This is the advisory-lock fix: without it, concurrent first-time
    .setup() calls can race on inserting the same migration row."""
    import concurrent.futures

    def _build_and_close():
        checkpointer = build_checkpointer()
        close_checkpointer(checkpointer)
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: _build_and_close(), range(4)))

    assert results == [True, True, True, True]


@requires_postgres
def test_paused_review_survives_the_database_dropping_idle_connections():
    """Serverless Postgres (Neon) closes every connection when it suspends
    after a few idle minutes, and networks drop idle connections too. The
    pool must notice and reconnect, or the reviewer's next request fails
    on a dead connection."""
    import psycopg

    checkpointer = build_checkpointer()
    try:
        graph = _build_mini_graph(checkpointer)
        config = {"configurable": {"thread_id": "idle-drop-test"}}
        graph.invoke({"value": "start"}, config)

        # Kill every pooled connection from outside, as a suspend would.
        pool = checkpointer._doc_agent_pool
        pool.wait()
        with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as admin:
            admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = current_database() AND pid <> pg_backend_pid()"
            )

        result = graph.invoke(Command(resume="yes"), config)
        assert result["value"] == "resumed-with-yes"
    finally:
        close_checkpointer(checkpointer)


@requires_postgres
def test_concurrent_first_start_on_an_empty_database_does_not_deadlock(monkeypatch):
    """The first deploy against a brand-new database is the case that
    matters: setup() then builds indexes with CREATE INDEX CONCURRENTLY,
    which waits for every other running statement to finish. Workers that
    block inside SELECT pg_advisory_lock() are running statements, so the
    index build and the waiting workers wait on each other forever. This
    hung the first Azure deploy; every worker must start within seconds."""
    import concurrent.futures
    import uuid

    import psycopg
    from psycopg.conninfo import make_conninfo

    base = os.environ["DATABASE_URL"]
    fresh_name = f"docagent_fresh_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(base, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{fresh_name}"')
    monkeypatch.setenv("DATABASE_URL", make_conninfo(base, dbname=fresh_name))

    def _build_and_close(_):
        close_checkpointer(build_checkpointer())
        return True

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    try:
        futures = [pool.submit(_build_and_close, i) for i in range(4)]
        done, not_done = concurrent.futures.wait(futures, timeout=30)
        assert not not_done, "workers deadlocked on first start"
        assert all(f.result() for f in done)
    finally:
        # Ending the sessions unblocks any deadlocked threads so the
        # executor can shut down and the database can be dropped.
        with psycopg.connect(base, autocommit=True) as admin:
            admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s",
                (fresh_name,),
            )
        pool.shutdown(wait=True, cancel_futures=True)
        with psycopg.connect(base, autocommit=True) as admin:
            admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s",
                (fresh_name,),
            )
            admin.execute(f'DROP DATABASE IF EXISTS "{fresh_name}"')
