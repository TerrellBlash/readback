"""
TESTS — Postgres checkpointer survives a worker restart
==========================================================
Every other test in this suite builds one graph and keeps using that
same object. That proves interrupt()/resume works, but it can't prove
persistence: a single Python object holding onto MemorySaver's RAM would
pass those tests too, restart-safety or not.

This test is the one that actually proves restart survival. It builds
TWO independent graphs via build_graph() — standing in for two separate
uvicorn worker processes — and shows that the second one, which never
saw the pause happen, can still read and resume it because the paused
state lives in Postgres, not in either process's memory.

Needs a real Postgres instance, so it's skipped unless DATABASE_URL is
set. See README.md "Run Tests". The model is scripted, so
it needs no API key.

Run with: pytest tests/test_checkpointer_restart.py -v
"""

import os
import uuid

import pytest
from langgraph.types import Command

import app.nodes as nodes
from app.graph import build_graph, close_checkpointer
from tests.scripted_model import ScriptedModel

AMBIGUOUS_DOCUMENT_TEXT = "Hello, this is just a random note. Nothing special here."

requires_postgres = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="requires a running Postgres instance; see README",
)


@requires_postgres
def test_second_graph_instance_sees_pending_interrupt_after_restart(monkeypatch):
    """A second, independent build_graph() call must see the first's pause.

    Why this matters: app/graph.py's build_checkpointer() only persists
    paused reviews correctly if PostgresSaver (not MemorySaver) is what
    gets wired up when DATABASE_URL is set, and if that state is actually
    readable by a brand new graph/pool rather than cached in the first
    one's process. If this test used the same graph object for both
    halves, a MemorySaver would pass it too — reusing the object is
    exactly the bug this test needs to catch.
    """
    # The classifier calls this unknown, so the run pauses at human_review.
    model = ScriptedModel(doc_type="unknown", confidence=0.9)
    monkeypatch.setattr(nodes, "get_llm", lambda config=None: model)
    thread_id = f"test-restart-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    # --- "Worker process #1": pauses at human_review ---
    first_graph = build_graph()  # picks up DATABASE_URL from the environment
    first_graph.invoke(
        {
            "document_text": AMBIGUOUS_DOCUMENT_TEXT,
            "document_filename": "mystery.txt",
            "document_source": "unknown",
        },
        config,
    )

    state_from_first = first_graph.get_state(config)
    assert state_from_first is not None
    assert state_from_first.next, (
        "Expected the graph to be paused (a pending next step) at "
        "human_review after an ambiguous document."
    )
    assert state_from_first.tasks, "Expected a pending interrupt task."
    assert any(
        task.interrupts for task in state_from_first.tasks
    ), "Expected the pending task to carry an interrupt payload."

    # --- "Worker process #2": a fully independent graph, same thread_id ---
    # This simulates a restart / a different uvicorn worker picking up the
    # review. It must NOT reuse first_graph or its checkpointer/pool.
    second_graph = build_graph()
    assert second_graph is not first_graph

    state_from_second = second_graph.get_state(config)
    assert state_from_second is not None
    assert state_from_second.next, (
        "The second, independent graph instance should still see the "
        "pending interrupt via Postgres — this is what proves restart "
        "survival."
    )
    assert state_from_second.tasks
    assert any(task.interrupts for task in state_from_second.tasks)

    # --- Resume the SECOND instance with a human approval ---
    result = second_graph.invoke(
        Command(resume={"approved": True, "corrections": {}}),
        config,
    )

    assert result["processing_status"] == "complete"
    assert result["final_output"]["human_approved"] is True
    final_state = second_graph.get_state(config)
    assert not final_state.next, "Pipeline should have finished, nothing pending."

    # This test is the only one that builds two real connection pools
    # against Postgres in a single test function — close both rather than
    # leaking them for the rest of the pytest session.
    close_checkpointer(first_graph.checkpointer)
    close_checkpointer(second_graph.checkpointer)
