"""
TESTS — get_llm model selection
================================
No network calls: building a ChatOpenAI client doesn't contact the API.

Why these matter: current OpenAI reasoning models reject temperature=0
with a 400 error, so sending the old setting to them would fail every
call in the pipeline. And the evaluation harness compares models by
passing one per run, which only works if that choice actually reaches
the client.

Run with: pytest tests/test_llm.py -v
"""

import pytest

from app.llm import DEFAULT_MODEL, _build, get_llm


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """A dummy key so the client can be built offline; no model override.

    Clients are cached per model, so the cache is cleared on both sides:
    otherwise a client holding this dummy key would be handed to later
    tests that call the real API.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.delenv("DOC_AGENT_MODEL", raising=False)
    monkeypatch.delenv("DOC_AGENT_REASONING_EFFORT", raising=False)
    _build.cache_clear()
    yield
    _build.cache_clear()


def test_default_model_is_used_when_nothing_is_configured():
    assert get_llm().model_name == DEFAULT_MODEL


def test_default_is_the_model_the_eval_chose():
    """gpt-6-luna at effort none had the fewest silent errors of the cheap
    runs in evals/results, at a fifth of gpt-4o-mini's cost. Changing the
    default should mean re-running the eval, not just editing a string."""
    llm = get_llm()
    assert llm.model_name == "gpt-6-luna"
    assert llm.reasoning_effort == "none"
    assert llm.use_responses_api is False  # tool calls work on Chat Completions only at effort none


def test_reasoning_model_gets_reasoning_effort_and_no_temperature():
    """gpt-6 models only accept the default temperature."""
    llm = get_llm({"configurable": {"model": "gpt-6-luna"}})
    assert llm.model_name == "gpt-6-luna"
    assert llm.temperature is None
    assert llm.reasoning_effort == "none"
    assert not llm.use_responses_api


def test_older_model_keeps_temperature_zero():
    """gpt-4o-era models predate reasoning effort; temperature 0 keeps them steady."""
    llm = get_llm({"configurable": {"model": "gpt-4o-mini"}})
    assert llm.temperature == 0
    assert llm.reasoning_effort is None


@pytest.mark.parametrize("model", ["gpt-4", "gpt-3.5-turbo", "ft:gpt-4o-mini:acme::abc123"])
def test_models_outside_reasoning_families_keep_temperature(model):
    """Sending reasoning_effort to these returns a 400."""
    llm = get_llm({"configurable": {"model": model}})
    assert llm.temperature == 0
    assert llm.reasoning_effort is None


def test_fine_tuned_reasoning_model_follows_its_base():
    assert get_llm({"configurable": {"model": "ft:gpt-5.4-mini:acme::abc123"}}).reasoning_effort == "none"


def test_documents_are_not_stored_by_openai(monkeypatch):
    """The Responses API retains requests unless told not to; uploads can be
    invoices and resumes with personal data."""
    monkeypatch.setenv("DOC_AGENT_REASONING_EFFORT", "low")
    assert get_llm({"configurable": {"model": "gpt-6-luna"}}).store is False
    assert get_llm({"configurable": {"model": "gpt-4o-mini"}}).store is False


def test_env_var_sets_the_deployment_model(monkeypatch):
    monkeypatch.setenv("DOC_AGENT_MODEL", "gpt-5.4-mini")
    assert get_llm().model_name == "gpt-5.4-mini"


def test_per_run_config_overrides_the_env_var(monkeypatch):
    """The eval harness picks the model per run; that must win over the deployment default."""
    monkeypatch.setenv("DOC_AGENT_MODEL", "gpt-5.4-mini")
    assert get_llm({"configurable": {"model": "gpt-4o-mini"}}).model_name == "gpt-4o-mini"


def test_reasoning_effort_above_none_uses_the_responses_api(monkeypatch):
    """Chat Completions returns a 400 for tools plus a reasoning effort on
    gpt-6 models, which would break structured output in function-calling
    mode."""
    monkeypatch.setenv("DOC_AGENT_REASONING_EFFORT", "medium")
    llm = get_llm({"configurable": {"model": "gpt-6-luna"}})
    assert llm.reasoning_effort == "medium"
    assert llm.use_responses_api


def test_graph_passes_the_run_model_to_its_nodes(monkeypatch):
    """If the run's model never reached the nodes, an eval "comparing"
    models would silently run the same model every time."""
    from langgraph.checkpoint.memory import MemorySaver

    import app.nodes as nodes
    from app.graph import build_graph
    from app.llm import resolve_model

    seen = []

    class StopBeforeNetwork(Exception):
        pass

    def recording_get_llm(config=None):
        seen.append(resolve_model(config))
        raise StopBeforeNetwork

    monkeypatch.setattr(nodes, "get_llm", recording_get_llm)
    graph = build_graph(MemorySaver())
    with pytest.raises(StopBeforeNetwork):
        graph.invoke(
            {
                "document_text": "INVOICE #1 from Acme Corp. Total due: $5.00",
                "document_filename": "invoice.txt",
                "document_source": "test",
            },
            {"configurable": {"thread_id": "model-wiring", "model": "gpt-6-luna"}},
        )
    assert seen == ["gpt-6-luna"]
