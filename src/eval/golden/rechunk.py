"""Re-generate `expected_chunk_ids` on an existing GoldenExample.

When tuning chunking parameters (and after re-ingesting the corpus, use this module to
update the expected chunk ids in a golden dataset. It runs a
single LLM pass per example whose only job is to pick the most relevant
chunk_ids out of k candidates retrieved from the newly-ingested vector store.

Why a single-purpose module?
- Keeps the rechunk workflow separate from full golden-dataset regeneration,
  so a chunk-boundary change does not accidentally redraw the LLM's judgment
  on keywords or source titles.
- Testable in isolation with mocked LLM/retriever.
"""

import json
import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.eval.golden.dataset_generation import retrieve_candidate_chunks
from src.schemas.eval import GoldenExample

logger = logging.getLogger(__name__)


def extract_json_object(content: str) -> str:
    """Strip an optional markdown fence and return the JSON-bearing substring."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    return match.group(1).strip() if match else content.strip()


def build_chunk_id_prompt(
    question: str,
    author: str,
    chunks: list[dict[str, Any]],
) -> str:
    """Build the LLM prompt for selecting relevant chunk_ids only.

    The prompt is intentionally scoped: no keywords, no synonyms, no source
    titles. The LLM's only job is to pick 3-7 of the provided candidates.
    """
    chunks_section = ""
    for i, chunk in enumerate(chunks, 1):
        chunks_section += (
            f"\n{i}. [ID: {chunk['chunk_id']}] [Source: {chunk['source']}]\n{chunk['text']}\n"
        )

    return f"""You are labeling a golden evaluation example for a RAG system where Enlightenment philosophers answer questions grounded in their historical texts.

**Task**: From the candidate chunks below, select the 3-7 chunks most directly relevant to the question. Return ONLY their chunk IDs.

**Question**: {question}
**Author**: {author}

**Selection criteria**:
- Prefer chunks that directly address the question over chunks that are merely on an adjacent theme.
- Include a chunk only if a well-grounded answer would cite or paraphrase from it.
- Use the EXACT chunk_id values shown in brackets.

**Output format**: Return ONLY a single JSON object with one field:
{{
  "expected_chunk_ids": ["abc123def456", "xyz789uvw012", "mno345pqr678"]
}}

Do NOT add prose, analysis, or any other fields.

**Candidate Chunks**:
{chunks_section}
"""


def regenerate_expected_chunk_ids(
    example: GoldenExample,
    llm: BaseChatModel,
    k: int = 15,
) -> list[str]:
    """Run a single LLM pass to pick relevant chunk_ids from k candidates.

    Args:
        example: Source example. Only `question` and `author` are used.
        llm: Low-variance LLM instance.
        k: Number of candidates to retrieve (default matches the full generator).

    Returns:
        List of chunk_id strings selected by the LLM. May be empty if the LLM
        judged none of the candidates relevant. In that case a warning is
        emitted; the caller decides whether to accept, retry, or fail.

    Raises:
        ValueError: If the LLM response is not valid JSON, lacks the
            expected field, or the field is not a list of strings.
    """
    chunks = retrieve_candidate_chunks(question=example.question, author=example.author, k=k)

    prompt = build_chunk_id_prompt(example.question, example.author, chunks)

    system_msg = SystemMessage(content="""You are a JSON generator for a RAG evaluation system.
You MUST return ONLY a valid JSON object with a single 'expected_chunk_ids' field.
Do NOT add explanations, analysis, other fields, or commentary.""")
    human_msg = HumanMessage(content=prompt)

    response = llm.invoke([system_msg, human_msg])
    raw_content = response.content if hasattr(response, "content") else response
    content = str(raw_content)
    logger.debug(f"LLM raw response for {example.id}:\n{content}")

    json_str = extract_json_object(content)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON for {example.id}. Raw content:\n{content}")
        raise ValueError(
            f"LLM returned invalid JSON for example {example.id}: {e}\n"
            f"First 200 chars of response: {content[:200]}"
        ) from e

    if not isinstance(data, dict) or "expected_chunk_ids" not in data:
        raise ValueError(
            f"LLM response for {example.id} is missing 'expected_chunk_ids': {data!r}"
        )

    ids = data["expected_chunk_ids"]
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        raise ValueError(
            f"LLM response for {example.id} has non-string chunk_ids: {ids!r}"
        )
    if not ids:
        logger.warning(
            f"LLM returned empty expected_chunk_ids for {example.id}. "
            f"The retrieval_relevance metric will be skipped for this example. "
            f"Consider re-running or manually inspecting candidates."
        )
    return ids


def rechunk_example(example: GoldenExample, new_chunk_ids: list[str]) -> GoldenExample:
    """Return a copy of `example` with only `expected_chunk_ids` replaced."""
    return example.model_copy(update={"expected_chunk_ids": new_chunk_ids})
