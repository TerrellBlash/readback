"""
Chat model construction.

Every node gets its model from get_llm(), so the model can be chosen per
run (graph config) or per deployment (DOC_AGENT_MODEL) without code
changes. The evaluation harness relies on this to compare models on the
same dataset.
"""

import logging
import os
from functools import lru_cache

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "gpt-6-luna"  # chosen by the eval; see evals/results
DEFAULT_REASONING_EFFORT = "none"
# Reads every vision page a second time; see resolve_check_model.
DEFAULT_CHECK_MODEL = "gpt-6.1-sol"
DEFAULT_CHECK_REASONING_EFFORT = "low"
# A hung call would otherwise hold a request open forever. Vision reads of
# a page take seconds; the API client retries twice after a timeout.
REQUEST_TIMEOUT_SECONDS = 90

logger = logging.getLogger(__name__)

# Reasoning models reject temperature=0 with a 400 error and take
# reasoning_effort instead. Every other model samples with a temperature,
# and 0 keeps extraction steady. The default effort is "none": extraction is
# reading, not multi-step reasoning, and the eval measures what that costs.
_REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")


def is_reasoning_model(model: str) -> bool:
    """
    True for models that take reasoning_effort instead of temperature.

    - Parameter model: an OpenAI model name. A fine-tuned model
        (ft:<base>:...) follows its base model.
    - Returns: whether the model is in a reasoning family.
    """
    return model.removeprefix("ft:").startswith(_REASONING_PREFIXES)


def resolve_model(config: RunnableConfig | None = None) -> str:
    """
    Pick the model for this run.

    - Parameter config: the graph's run config. A `model` key under
        `configurable` wins; otherwise DOC_AGENT_MODEL, then DEFAULT_MODEL.
    - Returns: the OpenAI model name.
    """
    configurable = (config or {}).get("configurable") or {}
    return configurable.get("model") or os.environ.get("DOC_AGENT_MODEL") or DEFAULT_MODEL


@lru_cache(maxsize=8)
def _build(model: str, reasoning_effort: str) -> ChatOpenAI:
    # store=False: the Responses API keeps requests (documents, page images)
    # by default, and nothing here needs OpenAI to retain them.
    if not is_reasoning_model(model):
        return ChatOpenAI(model=model, temperature=0, store=False, timeout=REQUEST_TIMEOUT_SECONDS)
    # Any effort other than "none" goes through the Responses API: Chat
    # Completions rejects function tools combined with a reasoning effort on
    # current models, so structured output in function-calling mode would
    # fail there.
    return ChatOpenAI(
        model=model,
        reasoning_effort=reasoning_effort,
        use_responses_api=reasoning_effort != "none",
        store=False,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def resolve_check_model(config: RunnableConfig | None = None) -> str | None:
    """
    Pick the model that reads vision pages a second time.

    The eval showed a model re-reading its own page repeats its misreads,
    while a different model catches them, so the check never uses the
    run's own model.

    - Parameter config: the graph's run config. A `check_model` key under
        `configurable` wins; otherwise DOC_AGENT_CHECK_MODEL, then
        DEFAULT_CHECK_MODEL. "off" disables the check.
    - Returns: the model name, or None when the check is off or would be
        the run's own model.
    """
    configurable = (config or {}).get("configurable") or {}
    model = (configurable.get("check_model") or os.environ.get("DOC_AGENT_CHECK_MODEL")
             or DEFAULT_CHECK_MODEL)
    if model.lower() == "off":
        return None
    if _base_model(model) == _base_model(resolve_model(config)):
        logger.warning("The check model is the run's own model (%s), so the second read is off.", model)
        return None
    return model


def _base_model(model: str) -> str:
    """The model a name refers to: case-insensitive, fine-tunes as their base."""
    name = model.casefold()
    return name.removeprefix("ft:").split(":")[0] if name.startswith("ft:") else name


def get_check_llm(config: RunnableConfig | None = None) -> ChatOpenAI | None:
    """
    Return a chat client for the second read, or None when there isn't one.

    - Parameter config: the graph's run config.
    - Returns: a ChatOpenAI client at DOC_AGENT_CHECK_REASONING_EFFORT
        (default "low"; gpt-6.1-sol rejects "none"), or None.
    """
    model = resolve_check_model(config)
    if model is None:
        return None
    effort = os.environ.get("DOC_AGENT_CHECK_REASONING_EFFORT", DEFAULT_CHECK_REASONING_EFFORT)
    return _build(model, effort)


def get_llm(config: RunnableConfig | None = None) -> ChatOpenAI:
    """
    Return a chat client for the run's model, built once per model.

    - Parameter config: the graph's run config, passed through from the node.
    - Returns: a ChatOpenAI client. Reasoning models get
        DOC_AGENT_REASONING_EFFORT (default "none"); older models get
        temperature 0.
    """
    effort = os.environ.get("DOC_AGENT_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
    return _build(resolve_model(config), effort)
