"""Shared pytest markers."""

import os

import pytest

# Tests that call the OpenAI API cost money, so they run only on request:
#     RUN_LIVE_TESTS=1 pytest
live = pytest.mark.skipif(
    not (os.environ.get("RUN_LIVE_TESTS") == "1" and os.environ.get("OPENAI_API_KEY")),
    reason="calls the OpenAI API; set RUN_LIVE_TESTS=1 and OPENAI_API_KEY",
)
