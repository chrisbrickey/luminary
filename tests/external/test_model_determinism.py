"""Test determinism of responses

This pins the determinism promise of `DEFAULT_TEMPERATURE` and `DEFAULT_LLM_SEED`
end-to-end against real Ollama. Using a fixed temperature and seedin the chat path
used produces stable output in the UI and terminal app as well as the evaluation harness.

REQUIREMENTS:
- Ollama must be running: `ollama serve`
- Model must be pulled: `ollama pull mistral`
- ChromaDB must have ingested data: `uv run python scripts/ingest.py --author voltaire`

Run these tests explicitly:
    uv run pytest -m external tests/external/test_model_determinism.py -v
"""

import pytest

from src.chains.chat_chain import build_chain

pytestmark = pytest.mark.external

DETERMINISM_QUESTION = "What is tolerance?"


def test_default_chain_produces_identical_responses_on_repeat_invocation() -> None:
    """Two invocations of the same default chain on the same question should
    produce byte-identical text given that seed and temperature are fixed by configuration.

    If this ever fails, do NOT relax the assertion before investigating: it
    means langchain-ollama, Ollama itself, or the model has regressed on the
    seed/temperature determinism contract.
    """
    chain = build_chain(author="voltaire")

    response_a = chain.invoke(DETERMINISM_QUESTION)
    response_b = chain.invoke(DETERMINISM_QUESTION)

    assert response_a.text == response_b.text
