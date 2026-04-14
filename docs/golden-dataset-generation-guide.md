# Golden Dataset Generation Guide

## Overview

This guide explains how to generate high-quality golden datasets for the evaluation harness using LLM-based judgment instead of circular evaluation.

### The Problem with Circular Evaluation (initial approach)
- Script `src/eval/golden/scripts/query_for_golden_dataset.py` captured what the current system retrieves
- Golden dataset = snapshot of current system's output
- Eval = comparing current system against that snapshot
- Perfect scores = system matches its own baseline ✓
- **Limitation**: Perfect scores don't validate absolute quality—they only show the system hasn't changed since baseline creation

**What this means:**
- ✓ Useful for regression testing (detect if future changes break retrieval)
- ✗ Can't tell if retrieval is actually good or consistently retrieving wrong chunks
- ✗ Won't catch systemic issues that existed when baseline was created

### LLM-Based Solution (new approach)
- Use an LLM to judge chunk relevance by reading actual source texts
- LLM evaluates chunks against philosophical grounding criteria
- Generates golden dataset expectations that represent true quality, not just current system behavior
- Standard prompt template that auto-discovers schema requirements

**Benefits:**
1. **Non-circular evaluation**: LLM judges actual quality, not just current system behavior
2. **Standardized approach**: Same prompt template for all metrics
3. **Automatic adaptation**: Adding new metrics = update schema + guidance, no prompt rewriting
4. **Reproducible**: Deterministic LLM (temperature=0) produces consistent judgments
5. **Traceable**: Version golden datasets as requirements evolve
6. **Scalable**: Easy to generate datasets for new authors or languages

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Schema Introspection                                         │
│    - Read GoldenExample schema fields                           │
│    - Discover METRIC_REGISTRY required_example_fields           │
│    - Build dynamic field requirements list                      │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Retrieve Candidate Chunks                                    │
│    - Query ChromaDB with question                               │
│    - Get top K chunks (K > DEFAULT_K, e.g., K=15)               │
│    - Retrieve full text + metadata for each chunk               │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. LLM Judgment (using standard prompt template)                │
│    Input:                                                       │
│    - Question                                                   │
│    - Author context                                             │
│    - All candidate chunks with IDs + full text                  │
│    - Schema requirements (auto-discovered)                      │
│                                                                 │
│    Output:                                                      │
│    - Valid GoldenExample JSON with all required fields          │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Validation & Iteration                                       │
│    - Parse LLM output as GoldenExample (Pydantic validation)    │
│    - If invalid: show errors, retry with corrections            │
│    - If valid: save to golden dataset file                      │ 
└─────────────────────────────────────────────────────────────────┘
```

---

## Standard Prompt Template

The template adapts automatically as new metrics are added. It uses schema introspection to discover required fields.

```python
# src/eval/golden/dataset_generation.py

STANDARD_PROMPT_TEMPLATE = """You are an expert evaluator creating high-quality test data for a RAG (Retrieval-Augmented Generation) system about Enlightenment philosophers.

## Your Task

Analyze the provided chunks and create a golden example (ground truth test case) for the evaluation harness.

## Context

**Question:** {question}
**Author:** {author}
**Language:** {language}
**Philosophical Topics:** {topics}

## Candidate Chunks

Below are {num_chunks} chunks retrieved from the corpus. Each has:
- chunk_id: Unique identifier (12-char hash)
- source: Document title and page
- text: Full chunk content

{chunks_with_metadata}

## Your Evaluation Criteria

For each chunk, judge:

1. **Relevance**: Does this chunk contain information directly relevant to answering the question?
2. **Philosophical Grounding**: Does it contain {author}'s actual ideas/arguments on the topic?
3. **Source Quality**: Is it from a primary source (not commentary)?
4. **Sufficient Detail**: Does it provide enough context to support a grounded answer?

## Required Output

Generate a valid JSON object matching this schema:

{{
  "id": "{example_id}",
  "question": "{question}",
  "author": "{author}",
  "language": "{language}",

  // REQUIRED FIELDS (auto-discovered from metrics):
{required_fields_template}
}}

## Field-Specific Guidance

{field_guidance}

## Constraints

- **expected_chunk_ids**: Select 3-7 most relevant chunks (not all {num_chunks}!)
  - Prioritize chunks with direct philosophical arguments
  - Prefer primary sources over secondary commentary
  - Include chunks that provide different aspects of the answer

- **Quality over quantity**: Better to have fewer highly-relevant chunks than many marginal ones

- **DO NOT include chunks that**:
  - Only mention the topic in passing
  - Are from unrelated philosophical discussions
  - Contain biographical info without philosophical content

## Output Format

Return ONLY the JSON object. No explanations, no markdown code blocks, just raw JSON.
"""
```

---

## Implementation Components

### 1. Schema Introspection

```python
# src/eval/golden/dataset_generation.py

from src.schemas.eval import GoldenExample
from src.eval.metrics.base import METRIC_REGISTRY


def discover_required_fields() -> dict[str, str]:
    """Auto-discover all required fields from schema and metrics.

    Returns:
        Dict mapping field_name -> description
    """
    required_fields = {}

    # 1. Get fields from GoldenExample schema
    for field_name, field_info in GoldenExample.model_fields.items():
        # Skip core fields (id, question, author, language)
        if field_name in ['id', 'question', 'author', 'language']:
            continue

        # Include fields that are used by registered metrics
        required_fields[field_name] = field_info.description

    # 2. Cross-reference with METRIC_REGISTRY
    for spec in METRIC_REGISTRY:
        for field in spec.required_example_fields:
            if field not in required_fields:
                # Add fields required by metrics but not yet in schema
                required_fields[field] = f"Required by {spec.name} metric"

    return required_fields
```

### 2. Field Guidance Map

**Important**: The `guidance_map` is built incrementally. Add entries ONLY when implementing the corresponding metric. Do not pre-populate guidance for future metrics.

```python
def build_field_guidance(required_fields: dict[str, str]) -> str:
    """Generate detailed guidance for each field type.

    INCREMENTAL DESIGN: Add entries to guidance_map only when implementing
    the corresponding metric. Start with minimal guidance, expand as needed.
    """

    # START with only currently-implemented metrics
    guidance_map = {
        "expected_chunk_ids": """
**expected_chunk_ids**: List of chunk IDs (strings)
- Include chunks that directly address the question
- Typical range: 3-7 chunks (quality over quantity)
- Example: ["0a997c69b8c4", "f92b8fde600f", "2a5808bb9731"]
""",
        "expected_source_titles": """
**expected_source_titles**: List of source titles (strings)
- Extract from chunk metadata (source field)
- Include document title only, not page numbers
- Example: ["Lettres Philosophiques 1734"]
""",
        # ADD MORE ENTRIES HERE as you implement new metrics
        # See "Adding Guidance for New Metrics" section below
    }

    # Build guidance only for fields that exist in schema
    guidance_parts = []
    for field_name in required_fields.keys():
        if field_name in guidance_map:
            guidance_parts.append(guidance_map[field_name])
        else:
            # Generic guidance for unknown fields
            # This fallback ensures generation works even without specific guidance
            guidance_parts.append(f"""
**{field_name}**: {required_fields[field_name]}
- Populate based on chunk analysis
""")

    return "\n".join(guidance_parts)
```

### 3. Core Generation Functions

```python
def retrieve_candidate_chunks(
    question: str,
    author: str,
    k: int = 15,  # Retrieve more than DEFAULT_K for LLM to judge
) -> list[dict[str, Any]]:
    """Retrieve candidate chunks for LLM evaluation."""
    embeddings = OllamaEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
    collection = Chroma(
        persist_directory=str(VECTOR_DB_PATH),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    results = collection.similarity_search(
        question,
        k=k,
        filter={"author": author}
    )

    chunks = []
    for doc in results:
        chunks.append({
            "chunk_id": doc.metadata.get("chunk_id"),
            "source": doc.metadata.get("source"),
            "text": doc.page_content,
        })

    return chunks


def generate_golden_example_with_llm(
    question: str,
    author: str,
    language: str,
    topics: list[str],
    model: str = "mistral",  # Use same model as production for consistency
) -> GoldenExample:
    """Generate a golden example using LLM judgment.

    Returns:
        Validated GoldenExample object

    Raises:
        ValidationError: If LLM output doesn't match schema
        JSONDecodeError: If LLM output is invalid JSON
    """
    # Step 1: Retrieve candidate chunks
    chunks = retrieve_candidate_chunks(question, author, k=15)

    # Step 2: Build prompt with auto-discovered schema
    prompt = build_prompt(question, author, language, chunks, topics)

    # Step 3: Get LLM judgment
    llm = ChatOllama(model=model, temperature=0.0)  # Deterministic
    response = llm.invoke(prompt)

    # Step 4: Parse and validate
    try:
        data = json.loads(response.content)
        example = GoldenExample(**data)
        return example
    except Exception as e:
        # Log error and potentially retry with corrections
        print(f"LLM output validation failed: {e}")
        print(f"Raw output:\n{response.content}")
        raise
```

---

## Usage

### 1. Create Examples Configuration

Create a JSON file defining the questions and topics you want to evaluate:

```json
// src/eval/golden/configs/voltaire_examples.json
[
  {
    "question": "Que pensez-vous de la tolérance religieuse?",
    "author": "voltaire",
    "language": "fr",
    "topics": ["tolerance"]
  },
  {
    "question": "What do you think about religious tolerance?",
    "author": "voltaire",
    "language": "en",
    "topics": ["tolerance"]
  },
  {
    "question": "What would you post on social media about tolerance?",
    "author": "voltaire",
    "language": "en",
    "topics": ["anachronism_trap"]
  }
]
```

### 2. Generate Golden Dataset

```bash
# Generate golden dataset using LLM judgment
uv run python src/eval/golden/scripts/generate_dataset_using_LLM_judgement.py \
  --config src/eval/golden/configs/voltaire_examples.json \
  --output evals/golden/persona_voltaire_v2.0_{current_date}.json \
  --model mistral \
  --verbose
```

### 3. Validate and Use

The generated dataset will:
- Automatically include all fields required by registered metrics
- Be validated against the GoldenExample schema
- Be ready to use with `scripts/run_eval.py`

---

## Evolution with New Metrics

**Incremental Design Principle**: Add guidance entries ONLY when implementing the corresponding metric. Do not pre-populate guidance for future metrics.

### Process Overview
When you add a new metric, follow these steps.

  When a metric needs NEW GoldenExample fields:
  1. Add tests for the metric
  2. Implement the metric, including registration with `required_example_fields` in METRIC_REGISTRY
  3. Check in
  4. Add tests for schema updates
  5. Add new field(s) to `GoldenExample` schema with description
  6. Add field to `guidance_map` in `build_field_guidance()` for better LLM guidance
   - Without guidance: LLM uses generic fallback (may be less accurate)
   - With guidance: LLM gets specific examples and constraints (more accurate)
  7. Regenerate golden dataset (auto-discovers new fields via introspection) with an updated version number
  8. Perform an evaluation cycle using the updated golden dataset
  **That's it!** No manual prompt rewriting needed.

  When a metric uses only existing fields:
  1. Add tests for the metric
  2. Implement the metric, including registration 
  3. Perform an evaluation cycle (no need to update the golden dataset)

### Detailed Instructions

#### Register the metric

```python
# src/eval/metrics/citation_placement.py
register_metric(
    MetricSpec(
        name="citation_placement",
        compute=_citation_placement_wrapper,
        required_example_fields={"expected_citation_count"},
        required_response_fields={"text"},
    )
)
```

####  Add field to schema

```python
# src/schemas/eval.py
class GoldenExample(BaseModel):
    # ... existing fields ...
    expected_citation_count: int = Field(
        default=0,
        description="Expected number of inline citations"
    )
```

#### Add field guidance

```python
# src/eval/golden/dataset_generation.py - build_field_guidance()
guidance_map = {
    # ... existing guidance ...
    "expected_citation_count": """
**expected_citation_count**: Integer
- How many [source: ...] citations should appear in the response?
- Count citations in the candidate chunks to estimate
- Typical range: 2-5 citations
- Example: 3
""",
}
```

#### Re-generate dataset (remember to bump the version)

Review and update `src/eval/golden/configs/voltaire_examples.json` if needed:
- Ensure the config includes **diverse philosophical topics** that will test keyword matching across different subject areas
- Consider coverage across different debates, historical figures, and key concepts to verify metrics work with varied content
- Add any new question variations or adversarial examples

Notes:
- LLM will automatically populate keyword fields based on chunk content analysis
- Ensures keywords are grounded in actual source texts
- The prompt template auto-discovers the new field and includes it in the LLM judgment.

---

## Testing Strategy
**Follow Test Development Workflow**:  See top of `eval-harness-plan_v3.md`.

```python
# tests/unit/eval/golden/test_dataset_generation.py

def test_discover_required_fields_includes_schema_fields():
    """Required fields include all GoldenExample fields except core."""
    fields = discover_required_fields()
    assert "expected_chunk_ids" in fields
    assert "id" not in fields  # Core field, excluded

def test_discover_required_fields_includes_metric_fields():
    """Required fields include fields needed by registered metrics."""
    fields = discover_required_fields()
    # retrieval metric requires expected_chunk_ids
    assert "expected_chunk_ids" in fields

def test_build_prompt_includes_all_required_fields():
    """Generated prompt includes all auto-discovered fields."""
    prompt = build_prompt(
        question="Test question",
        author="voltaire",
        language="en",
        chunks=[],
        topics=["test"]
    )
    # Check JSON template includes expected_chunk_ids
    assert '"expected_chunk_ids":' in prompt

def test_generate_golden_example_returns_valid_schema(mock_llm):
    """LLM output is validated against GoldenExample schema."""
    # Mock LLM to return valid JSON
    mock_llm.invoke.return_value.content = json.dumps({
        "id": "test_en",
        "question": "Test?",
        "author": "voltaire",
        "language": "en",
        "expected_chunk_ids": ["abc123"],
    })

    example = generate_golden_example_with_llm(
        question="Test?",
        author="voltaire",
        language="en",
        topics=["test"]
    )

    assert isinstance(example, GoldenExample)
    assert example.id == "test_en"
```

---

## Migration Strategy

### Option 1: Replace Existing Datasets
- Generate new v2.0 datasets using LLM-based approach
- Archive v1.x datasets for historical reference
- Use v2.0+ datasets going forward

### Option 2: Dual-Track Approach
- Keep v1.x datasets for regression testing (detect changes from baseline)
- Create v2.x datasets for quality validation (assess absolute quality)
- Run both types of evaluations for comprehensive coverage

**Recommendation**: Start with Option 2 to validate the new approach while maintaining regression tests. Once confident in LLM-generated datasets, transition to Option 1.

---

## Design Decisions

### Why Deterministic LLM (temperature=0)?
- Ensures reproducible judgments across runs
- Makes generated datasets stable and traceable
- Reduces variance in quality assessments

### Why Retrieve More Chunks Than Production (k=15 vs k=5)?
- Gives LLM broader context to judge relevance
- Prevents missing relevant chunks due to retrieval ranking
- LLM selects best 3-7 from larger candidate set (quality over quantity)

### Why Separate Guidance for Each Field?
- Makes LLM output more accurate for complex fields
- Provides examples to guide LLM toward correct format
- Easy to extend as new field types are added

### Why Auto-Discovery Instead of Hardcoded Fields?
- Eliminates maintenance burden when adding metrics
- Ensures prompt always includes all required fields
- Prevents schema drift between prompt and validation

---

## Troubleshooting

### LLM Returns Invalid JSON
- **Cause**: LLM adds markdown formatting (```json blocks) or explanations
- **Fix**: Update prompt to emphasize "Return ONLY the JSON object"
- **Workaround**: Parse with regex to extract JSON from markdown blocks

### LLM Selects Too Many/Few Chunks
- **Cause**: Guidance unclear about quantity expectations
- **Fix**: Strengthen guidance with specific ranges (3-7 chunks)
- **Alternative**: Post-process to filter out marginal chunks

### Fields Missing from Generated Examples
- **Cause**: Field not in guidance_map or schema introspection failed
- **Fix**: Add field to guidance_map with clear examples
- **Debug**: Print `discover_required_fields()` output to verify detection

### Generated Dataset Fails Validation
- **Cause**: LLM output doesn't match GoldenExample schema constraints
- **Fix**: Review validation errors, update prompt guidance for that field
- **Debug**: Print raw LLM output to see what it's generating

---

## Future Enhancements

1. **Iterative Refinement**: If LLM output fails validation, automatically retry with error messages as additional context
2. **Multi-Model Consensus**: Generate examples with multiple LLMs (mistral, llama3, etc.) and merge judgments
3. **Human-in-the-Loop**: Allow manual review/editing of generated examples before finalizing dataset
4. **Automated Expansion**: Given a base set of questions, automatically generate variations (different phrasings, languages, difficulty levels)
5. **Quality Metrics for Generators**: Track LLM agreement rates, validation success rates, and human approval rates to optimize prompt templates
