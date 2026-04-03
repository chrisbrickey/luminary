# Step 13: Evaluation Harness

## Overview

- **Goal:** Implement minimum viable eval harness; establish quality baseline with Voltaire before adding more philosophers
- **Implementation approach:** This step is broken into subsections for an iterative approach. The subsections are ordered to enable quick feedback loops - implementing a minimal set of metrics, then immediately building the runner and CLI to execute the first eval run, then iteratively adding more metrics with eval runs after each addition.
- **Multi-language design:** The evaluation harness tests the system as a **multlingual chatbot** (not separate monolingual systems). Eval runs process all examples in the golden dataset regardless of language, allowing cross-language metrics like translation consistency to work correctly. 
- **French and English for MVP:** In order to establish a feedback loop as quickly as possible, with which the application and harness can be improved through iteration, only French (the language of most of the raw source data) and English will be evaluated initially. More languages can be added to the eval harness in the future. 

---

## Test Development Workflow (applies to all subsections)

All test development in this plan follows this workflow:

1. **Write tests first:** Use test-first-developer agent to write tests based on specifications
2. **Verify red:** Confirm tests fail initially
3. **Implement:** Write production code to make tests pass (green)
4. **Refactor for DRY:**
   - Extract repeated literals (strings, numbers, dicts) into module-level constants or pytest fixtures
   - Share common setup via conftest.py fixtures or helper factories (not copy-paste)
   - Review all tests to ensure each value is defined once and referenced elsewhere
5. **Ensure generic test data:**
   - Use obviously fake values: "test-user", "sample-text", "item_001", 42, "https://example.com"
   - Never use real-world names, brands, URLs, or domain-specific data
   - Use gender-neutral names (Alex, Sam, Jordan) and avoid stereotypes
6. **Verify complete:** Ensure entire test suite (including mypy) passes before marking subsection complete

---

## ✅ A. Core retrieval metrics

**Goal:** Establish directory foundation and implement fundamental metrics that directly validate retrieval functionality of the RAG pipeline.

### Implementation

1. **Add tests to validate schema characteristics**
   - **Follow Test Development Workflow (see top of document)**
   - `tests/unit/test_schemas.py` - add MetricResult tests
   - Tests should cover: valid construction, score bounds (0-1), required fields, optional details
   
2. **Add schemas for evaluation results**
   - Modify `src/schemas.py`: add `MetricResult` (name, score 0-1, details)

   ```python
   class MetricResult(BaseModel):
       """Result from a single metric on a single example."""
       name: str = Field(..., description="Name of the metric")
       score: float = Field(..., ge=0.0, le=1.0, description="Score from 0.0 to 1.0, where 1.0 is perfect")
       details: dict[str, Any] = Field(default_factory=dict, description="Additional details about the metric result")
   ```

3. **Check In:** Stop and ask the user to confirm that the implementation of the above steps is satisfactory before moving to subsequent steps.

4. **Create evaluation directory structure**
   - Create `src/eval/` with `__init__.py`
   - Create `src/eval/metrics/` with `__init__.py`
   - Create `tests/unit/eval/` directory (no `__init__.py` needed per pytest convention)

5. **Add tests for the retrieval metric**
   - **Follow Test Development Workflow (see top of document)**
   - `tests/unit/eval/test_retrieval_metric.py`
   - Test cases:
     - `test_perfect_retrieval()` - all expected found, no extras → recall=1.0, precision=1.0, F1=1.0
     - `test_high_recall_low_precision()` - 3 of 3 expected found, but 7 irrelevant → recall=1.0, precision=0.3, F1=0.46
     - `test_high_precision_low_recall()` - 1 of 3 expected found, no extras → recall=0.33, precision=1.0, F1=0.5
     - `test_balanced_partial()` - 2 of 3 expected found, 2 retrieved total → recall=0.67, precision=1.0, F1=0.8
     - `test_no_chunks_found()` - 0 of 3 expected IDs in retrieved → recall=0.0, precision=0.0, F1=0.0
     - `test_no_expected_ids()` - empty expected list → score 1.0 (vacuous truth)
     - `test_empty_retrieval()` - empty retrieved list with non-empty expected → recall=0.0, precision=0.0, F1=0.0

6. **Implement retrieval relevance metric**
   - Create `src/eval/metrics/retrieval.py`
   - Function signature: `retrieval_relevance(expected_chunk_ids: list[str], retrieved_chunk_ids: list[str]) -> MetricResult`
   - Logic: Calculate recall (fraction of expected found), precision (fraction of retrieved that are relevant), and F1 score (harmonic mean)
   - Score calculation:
     - `recall = len(set(expected) & set(retrieved)) / len(expected)` if expected is non-empty, else 1.0
     - `precision = len(set(expected) & set(retrieved)) / len(retrieved)` if retrieved is non-empty, else 1.0
     - `f1_score = 2 * (precision * recall) / (precision + recall)` if sum > 0, else 0.0
     - Use F1 score as the primary metric score
   - Return `MetricResult(name="retrieval_relevance", score=f1_score, details={"recall": recall, "precision": precision, "f1_score": f1_score, "found": [...], "missing": [...], "irrelevant": [...]})`

    **Rationale:** Deterministic graders are fast, cheap, objective, and reproducible. Retrieval metrics validate that the RAG pipeline fetches relevant context before the LLM sees it. F1 score balances recall (finding all relevant chunks) and precision (avoiding irrelevant chunks). High precision prevents wasting context window tokens on noise and reduces LLM confusion. This is a foundational capability for grounded responses.

7. **Check In:** Stop and ask the user to confirm that the implementation of the above steps is satisfactory before moving to subsequent steps.

### Documentation

- **README:** Update project structure diagram to add `src/eval/` directory with comment: `# Evaluation harness: metrics, runner, and quality measurement`

- **README:** Add new section "## Evaluation Harness" (after Architecture Overview):
  ```markdown
  ## Evaluation Harness

  Luminary uses an automated evaluation harness to measure quality and prevent regressions.

  **Capabilities under development:**
  - Deterministic metrics: retrieval relevance, citation accuracy, language compliance
  - Quality metrics: faithfulness to source texts, citation placement, persona maintenance
  - Cross-language metrics: translation consistency (FR/EN retrieval overlap)
  - LLM-as-judge metrics: relevance, groundedness, coherence

  **Design principles:**
  - Multi-language by default: test the bilingual system as users experience it
  - Deterministic graders first: fast, reproducible, objective
  - Comprehensive coverage: happy paths and edge cases (anachronisms, persona breaks)
   ```

### Plan updates

- **Update this plan:** Mark this subsection `✅` on the title line. Note any deviations below this line.

---

## ✅ B. Golden dataset v1.0 (Voltaire-only evaluation data)

**Goal:** Create concrete validation data to test all metrics end-to-end.

### Implementation

1. **Add tests for the golden dataset schemas**
   - **Follow Test Development Workflow (see top of document)**
   - `tests/unit/test_schemas.py` - add `GoldenExample` and `GoldenDataset` tests
   - Test cases for `GoldenExample` (3 tests minimum):
     - `test_golden_example_construction()` - valid fields → constructs successfully
     - `test_golden_example_required_fields()` - missing `id` or `question` → raises error
     - `test_golden_example_default_fields()` - optional fields default to empty lists
   - Test cases for `GoldenDataset` (3 tests minimum):
     - `test_golden_dataset_construction()` - valid fields → constructs successfully
     - `test_golden_dataset_version_validation()` - invalid version "v1.0" → raises ValueError
     - `test_golden_dataset_valid_version()` - valid version "1.0" → passes validation

2. **Add schemas for golden dataset**
   - Modify `src/schemas.py`: add `GoldenExample` and `GoldenDataset`
   - Below is a draft that may change given current state of the app.

   ```python
   class GoldenExample(BaseModel):
       """A single test case in the golden dataset.

       Agile development: Add fields as new metrics are implemented.
       """
       id: str = Field(..., description="Unique identifier (e.g., 'tolerance_fr', 'pascal_en')")
       question: str = Field(..., description="The question to ask the philosopher")
       language: str = Field(..., pattern=r"^(en|fr)$", description="ISO 639-1 code: 'en' or 'fr'")

       # retrieval metrics
       expected_chunk_ids: list[str] = Field(default_factory=list, description="Expected chunk IDs to be retrieved")

   class GoldenDataset(BaseModel):
       """Collection of test cases with versioning."""
       version: str = Field(..., pattern=r"^\d+\.\d+$", description="Semantic version (e.g., '1.0', '2.0')")
       created_date: str = Field(..., description="ISO 8601 date (YYYY-MM-DD)")
       description: str = Field(..., description="What this dataset tests")
       examples: list[GoldenExample] = Field(..., description="List of test examples")
   ```

3. **Check In:** Stop and ask the user to confirm that the implementation of the above steps is satisfactory before moving to subsequent steps.

4. **Create golden dataset directory**
   - Create `data/raw/golden/` directory
   - Update `.gitignore`: Change individual `data/` subdirectories to single `data/` entry
     - **Before:** `data/raw/`, `data/chroma_db/`
     - **After:** `data/`
     - **Rationale:** Simpler, catches all data subdirectories, prevents accidentally committing eval artifacts

5. **Create Voltaire golden dataset v1.0**
   - Create `data/raw/golden/voltaire_golden_v1.0_2026-03-28.json`
   - **Filename convention:** `{author}_golden_v{version}_{YYYY-MM-DD}.json`
   - **Content requirements:**
     - Version 1.0 (Voltaire-only, before Gouges is added)
     - 6-8 examples covering key philosophical topics:
       - Tolerance (FR + EN pair - translation_pair_id will be added in subsection J)
       - Pascal critique (FR + EN pair)
       - Newton/science (FR + EN pair)
     - 2-4 adversarial examples (anachronism traps, persona breaks):
       - Anachronism: "What would you post on social media about tolerance?"
       - Persona break: "Are you an AI trained on Voltaire's texts?"
       - **Note:** Forbidden phrase validation will be added in subsection K
   - **Field population (v1.0 - retrieval metrics only):**
     - `expected_chunk_ids`: Actual chunk IDs from ingested corpus (requires running `scripts/ingest.py --author voltaire` first)
     - **Fields to be added in later subsections:**
       - expected_source_titles (subsection F)
       - expected_keywords_fr/en (subsection I)
       - translation_pair_id (subsection J)
       - forbidden_phrases_fr/en (subsection K)

   **Example structure (v1.0 - retrieval metrics only):**
   ```json
   {
     "version": "1.0",
     "created_date": "2026-03-28",
     "description": "Voltaire-only evaluation dataset v1.0 - retrieval metrics baseline",
     "examples": [
       {
         "id": "tolerance_fr",
         "question": "Que pensez-vous de la tolérance religieuse?",
         "language": "fr",
         "expected_chunk_ids": ["abc123def456", "789ghi012jkl"]
       },
       {
         "id": "tolerance_en",
         "question": "What do you think about religious tolerance?",
         "language": "en",
         "expected_chunk_ids": ["abc123def456", "789ghi012jkl"]
       },
       {
         "id": "anachronism_trap_social_media",
         "question": "What would you post on social media about tolerance?",
         "language": "en",
         "expected_chunk_ids": []
       }
     ]
   }
   ```

   **Note:** Additional fields will be added as the dataset evolves:
   - v1.1: expected_source_titles (subsection F)
   - v1.2: expected_keywords_fr/en (subsection I)
   - v1.3: translation_pair_id (subsection J)
   - v1.4: forbidden_phrases_fr/en (subsection K)

   **Rationale for adversarial examples:** Balanced problem sets test both when behavior should occur and when it shouldn't. Adversarial examples validate that the system maintains persona and avoids anachronisms even when prompted to break character. These are "negative tests" that should fail gracefully (refuse to engage with modern concepts).

   **Note on expected_chunk_ids:** To populate these accurately, you'll need to:
   1. Run `uv run python scripts/ingest.py` to create the vector database
   2. Query the database for sample questions to see which chunk IDs are actually retrieved
   3. Copy those chunk IDs into the golden dataset
   4. This ensures the eval tests realistic retrieval (not arbitrary IDs)

6. **Check In:** Stop and ask the user to confirm that the implementation of the above steps is satisfactory before moving to subsequent steps.

### Documentation

- **README:** Update "Evaluation Harness" section:
  ```markdown
  ## Evaluation Harness

  ...existing content...

  **Golden datasets:**
  Golden datasets are versioned collections of test cases that validate system behavior.
  They live in `data/raw/golden/` (gitignored) with naming convention:
  `{author}_golden_v{version}_{YYYY-MM-DD}.json`

  **Current datasets:**
  - `voltaire_golden_v1.0_2026-03-28.json`: examples covering tolerance, Pascal,
    Newton (FR/EN pairs for translation testing), plus adversarial anachronism and
    persona break tests

  **Versioning policy:**
  When updating a golden dataset:
  1. Increment version (1.0 → 1.1 for minor changes, 1.0 → 2.0 for major changes)
  2. Update date in filename to maintain traceability
  3. Keep old versions for reproducibility (disk space is cheap)

  **Why version?** Eval results are only meaningful relative to a specific dataset.
  Versioning allows comparing results across time and referencing the correct snapshot of data when creating eval reports.
  ```

- **README:** Update "Project Structure" section:
  ```markdown
  data/
  ├── raw/
  │   └── golden/          # Versioned golden datasets for evaluation
  ```

### Plan updates

- **Update this plan:** Mark this subsection `✅` on the title line. Note any deviations below this line.
  - Split up schemas into multiple files per domain to improve maintainability as this plan will add many eval-related schemas.
  - Created helper scripts not in original plan: `scripts/query_for_golden_dataset.py` (queries ChromaDB to get realistic chunk IDs) and `scripts/validate_golden_dataset.py` (validates golden dataset against schema)
  - Used actual chunk IDs from ingested Voltaire corpus instead of placeholders 
  - Dataset includes 8 examples (6 main philosophical topics as FR/EN pairs + 2 adversarial examples) with realistic chunk IDs from *Lettres philosophiques*

---

## ✅ C. Golden Dataset Loading Utilities
**Goal:** Add capability for loading the most recent versioned golden datasets. This will be used by the eval runner.

### Implementation

1. **Add configuration constant and tests**
   - **Follow Test Development Workflow (see top of document)**
   - `tests/unit/configs/test_config_common.py`: Add `test_golden_dataset_path_is_path_object()` to verify it is a `Path` and not a string
   - Modify `src/configs/common.py`: add `DEFAULT_GOLDEN_DATASET_PATH = Path("data/raw/golden")` to reference the location of golden datasets
   - **Rationale:** Following dependency injection pattern from existing pipelines (e.g. `VECTOR_DB_PATH` is a `Path`, not `_DIR`). These configs are sensible defaults that work for most users that can be overridden programmatically or via CLI flags for advanced use cases.

2. **Add tests for utilities for loading golden datasets**
   - **Follow Test Development Workflow (see top of document)**
   - `tests/unit/utils/test_eval_utils.py`

   - Test cases for `load_golden_dataset()` (4 tests minimum):
     - `test_load_valid_dataset()` - valid JSON → returns GoldenDataset
     - `test_load_missing_file()` - nonexistent path → raises FileNotFoundError with helpful message
     - `test_load_invalid_json()` - malformed JSON → raises JSONDecodeError
     - `test_load_invalid_schema()` - valid JSON, invalid schema → raises ValidationError

   - Test cases for `discover_latest_golden_dataset()` (3 tests minimum):
     - `test_discover_finds_latest()` - multiple versions → returns newest
     - `test_discover_no_matches()` - no matching files → raises FileNotFoundError with helpful message
     - `test_discover_respects_author()` - voltaire and gouges files → returns correct author

   - Use `tmp_path` fixture for file I/O tests

3. **Implement utilities for loading golden datasets**
   - Create `src/utils/eval_utils.py`
   - Implement data loading utility methods. First draft below. Adapt to current state of the application.

   ```python
   from pathlib import Path
   import json
   from datetime import datetime
   from dataclasses import asdict
   from src.schemas import GoldenDataset, EvalRun
   from src.configs.common import DEFAULT_GOLDEN_DATASET_PATH

   def load_golden_dataset(path: Path) -> GoldenDataset:
       """
       Load and validate golden dataset from JSON file.

       Args:
           path: Path to golden dataset JSON file

       Returns:
           Validated GoldenDataset object

       Raises:
           FileNotFoundError: If path does not exist
           json.JSONDecodeError: If file is not valid JSON
           ValidationError: If JSON doesn't match GoldenDataset schema
       """
       if not path.exists():
           raise FileNotFoundError(
               f"Golden dataset not found: {path}\n"
               f"Expected location: {DEFAULT_GOLDEN_DATASET_PATH}/"
           )

       with path.open() as f:
           data = json.load(f)

       # Pydantic/dataclass validation happens here
       return GoldenDataset(**data)

   def discover_latest_golden_dataset(
       directory: Path,
       author: str,
       scope: str = "golden",
   ) -> Path:
       """
       Find most recent golden dataset for an author.

       Filename format: {author}_{scope}_v{version}_{YYYY-MM-DD}.json
       Example: voltaire_golden_v1.0_2026-03-28.json

       Args:
           directory: Directory to search (usually DEFAULT_GOLDEN_DATASET_PATH)
           author: Author name (e.g., "voltaire", "gouges")
           scope: Dataset scope (default: "golden")

       Returns:
           Path to newest matching file (sorted by filename, newest first)

       Raises:
           FileNotFoundError: If no matching files found
       """
       pattern = f"{author}_{scope}_v*.json"
       matches = sorted(directory.glob(pattern), reverse=True)

       if not matches:
           raise FileNotFoundError(
               f"No golden dataset found for author '{author}' in {directory}.\n"
               f"Expected pattern: {pattern}\n"
               f"Make sure you've created the dataset file first."
           )

       return matches[0]
   ```

   **Add design notes** to file or method documentation
   - `load_golden_dataset()`: I/O + validation separated from business logic
   - `discover_latest_golden_dataset()`: Convention-based auto-discovery (good UX)
   - All functions have clear error messages for common failure modes

### Plan updates

- **Update this plan:** Mark this subsection `✅` on the title line. Note any deviations below this line.
  - Started a new config file for eval constants: `configs/eval.py` and their tests instead of adding to `common.py` to ease maintainability, especially because this plan will add many more eval-specific configs. Updated subsequent steps in this plan to use this file structure.
  - Changed location of eval utils to the eval module: `src/eval/utils.py` instead of adding to `src/utils/`. Updated subsequent steps in this plan to use this file structure.
  - Removed display of DEFAULT_GOLDEN_DATASET_PATH from error handling in the loading utilities because the path is injected and these utilities should be generic so that they can be reused with any path.
  - Added test to document an accepted edge case where a multi-digit decimal verion (e.g. `v1.12`) will not sort correctly. This edge case is accepted because it is not necessary for our use cases and the golden dataset validations will prevent a multi-digit decimal version.

---

## ✅ D. Evaluation Runner

**Goal:** Add mechanism to measure quality and identify specific weaknesses in order to drive improvements by measuring quality and identifying specific weaknesses.

### Implementation

1. **Add test for the schemas for evaluation runs**
   - **Follow Test Development Workflow (see top of document)**
   - `tests/unit/schemas/test_eval.py` - add `ExampleResult` and `EvalRun` tests
   
   - Test cases for `ExampleResult` (3 tests minimum):
     - `test_example_result_construction()` - valid fields → constructs successfully
     - `test_example_result_required_fields()` - all required fields present
     - `test_example_result_default_values()` - passed field calculated correctly

   - Test cases for `EvalRun` (3 tests minimum):
     - `test_eval_run_construction()` - valid fields → constructs successfully
     - `test_eval_run_required_fields()` - missing `dataset_version` → raises error
     - `test_eval_run_structure()` - aggregate_scores has expected nested structure

2. **Add schemas for evaluation runs**
   - Modify `src/schemas/eval.py`: add `ExampleResult` and `EvalRun`
   - Below is a draft. Adapt to the current state of the app.

   ```python
   class ExampleResult(BaseModel):
       """The result for one example in the golden dataset across all metrics."""
       example_id: str = Field(..., description="Unique identifier for this example")
       question: str = Field(..., description="The question that was asked")
       language: str = Field(..., pattern=r"^(en|fr)$", description="Language of the question")
       response: ChatResponse = Field(..., description="The chat response from the system")
       metrics: list[MetricResult] = Field(..., description="List of metric results for this example")
       passed: bool = Field(..., description="True if all required metrics above threshold")

   class EvalRun(BaseModel):
       """Complete results from running evaluation harness (the JSON artifact)."""
       # Metadata
       dataset_version: str = Field(..., description="Version of the golden dataset used")
       dataset_name: str = Field(..., description="Name of the dataset (e.g., 'voltaire_golden')")
       run_timestamp: str = Field(..., description="ISO 8601 timestamp with timezone")
       author: str = Field(..., description="Author being evaluated")

       # System configuration (for reproducibility)
       system_version: dict[str, str] = Field(..., description="System config: model, commit, etc.")

       # Results
       example_results: list[ExampleResult] = Field(..., description="Results for each example")
       aggregate_scores: dict[str, Any] = Field(..., description="Overall + by-language breakdowns")
       overall_pass_rate: float = Field(..., ge=0.0, le=1.0, description="Fraction of examples passing all metrics")
   ```

   **Add design notes** to file or method documentation
   - `EvalRun` is the machine-readable artifact (JSON) saved to disk
   - `system_version` captures reproducibility info (model, commit hash, config)
   - `aggregate_scores` has nested structure: `{"overall": {...}, "by_language": {"en": {...}, "fr": {...}}, "cross_language": {...}}`
   - **Naming rationale:** "EvalRun" aligns with typical terminology ("run eval", "eval run") used to describe the automated evaluation process that results in an artifact. "Report" or "eval report" is reserved for documentation of how the app is improved using an EvalRun as an input.

3. **Check In:** Stop and ask the user to confirm that the implementation of the above steps is satisfactory before moving to subsequent steps.

4. **Add tests for the evaluation runner**
   - **Follow Test Development Workflow (see top of document)**
   - `tests/unit/eval/test_runner.py`
   - Test cases (8 tests minimum):
     - `test_run_eval_returns_eval_run()` - run eval with mocked chain → returns EvalRun with correct structure
     - `test_run_eval_invokes_chain_per_example()` - chain.invoke called once per example
     - `test_run_eval_applies_language_specific_metrics()` - FR examples get faithfulness_fr, EN get faithfulness_en
     - `test_run_eval_computes_translation_metrics()` - paired examples → translation_consistency computed
     - `test_run_eval_aggregates_scores()` - overall, by_language, cross_language all present
     - `test_run_eval_calculates_pass_rate()` - 2 of 3 examples pass → overall_pass_rate = 0.67
     - `test_run_eval_captures_system_version()` - system_version has chat_model, commit fields
   - Mock the chain using `Mock(spec=Runnable)` with predefined `ChatResponse` returns
   - Mock metric functions to return known scores for deterministic tests

5. **Implement evaluation runner**
   - Create `src/eval/runner.py`
   - Key functions:
     - `run_eval(chain, golden_dataset) -> EvalRun` - main entry point (processes all languages)
     - `_compute_translation_metrics()` - finds paired examples, computes Jaccard overlap
     - `_compute_aggregate_scores()` - overall, by-language, cross-language aggregation
     - `_get_system_version()` - captures git commit, model names for reproducibility
     - `_infer_author_from_dataset()` - extracts author from dataset description

   **Critical design decision:** The runner processes ALL examples regardless of language. It applies language-specific metrics based on each example's `language` field. This tests the bilingual system as a whole, not in parts.

6. **Check In:** Stop and ask the user to confirm that the implementation of the above steps is satisfactory before moving to subsequent steps.

7. **Add integration test**
    - Create `tests/integration/test_eval_runner_integration.py`
    - Test cases (2 tests minimum):
      - `test_eval_runner_end_to_end()` - wire real metric functions + mocked chain + real runner → verify full pipeline
      - `test_eval_runner_processes_multilingual_dataset()` - dataset with FR and EN examples → both processed, cross-language metrics computed

    **Rationale:** It's critical to read outputs and verify graders work correctly. These integration tests ensure the eval harness assembles results correctly.

### Documentation

- **README:** Update "Pipelines" section:
    ```markdown
    III. EVALUATION PIPELINE
    ──────────────────────────────────────────────────────────────
    run_eval() → invoke chain per example → apply metrics → aggregate scores → EvalRun artifact
    ```

- **README:** Update "Pipelines in Detail" section.
    ```markdown
    EVALUATION (on-demand quality measurement)
    ─────────────────────────────────────────────────────────────────────
     GoldenDataset           versioned test cases with expected behaviors (questions, expected chunks, keywords, etc.)
          │
          ▼
     runner.py               invokes chat chain for each example, applies metrics, aggregates scores
          │
          ▼
     metrics/                deterministic graders: retrieval_relevance (recall/precision/F1), citation_accuracy, language compliance
          │
          ▼
     EvalRun                 machine-readable artifact with all results, scores, and system version (saved to evals/runs/)
    ```

- **README:** Update "Evaluation Harness" section: The below is draft. Consult the user on how much of this to add.
  ```markdown
  ## Evaluation Harness

  ...existing content...

  **Evaluation runner:**
  The runner (`src/eval/runner.py`) orchestrates the evaluation process:
  1. Accepts a LangChain runnable and a golden dataset
  2. Invokes the chain for each example (all languages)
  3. Applies language-specific metrics (FR/EN)
  4. Computes cross-language metrics (translation consistency)
  5. Aggregates scores (overall, by-language, cross-language)
  6. Returns `EvalRun` object (machine-readable artifact)

  **Design principle:** The runner is pure business logic. It does NOT:
  - Load datasets from disk (caller's responsibility)
  - Save results to disk (caller's responsibility)
  - Filter examples by language (processes all - we test a bilingual system)

  This separation enables programmatic use (e.g., scripts, notebooks) and simplifies testing.

  **Artifacts:**
  Eval runs are saved to `evals/runs/` (gitignored) as timestamped JSON files:
  - Filename format: `{author}_{YYYY-MM-DD}T{HH-MM-SS}.json`
  - Example: `voltaire_2026-03-28T14-30-45.json`
  - Contains: All example results, aggregate scores, system version (model, commit)
  ```

### Plan updates

- **Update this plan:** Mark this subsection `✅` on the title line. Note any deviations below this line.
  - Added name field to GoldenDataset schema to prevent reconstruction of a dataset name within the eval harness runner. This metadata field may be populated by the caller (e.g. script), which is appropriate because the caller will load the dataset from file.
  - Remove author field from EvalRun because it is redundant with dataset_name and not necessary for the MVP use cases.
  - Implement registry pattern for metrics so that metrics can be automatically discovered by the eval runner (instead of updating the runner every time a metric is added).

---

## E. Evaluation CLI (including the first eval run)
**Goal:** Enable capability to run evals via CLI. This subsection delivers the first working end-to-end eval run.

### Implementation

1. **Create eval artifacts directory**
   - Create `evals/runs/` directory structure (empty, for artifacts)
   - Update `.gitignore`: add `evals/`
   - **Rationale:** Eval run artifacts are large JSON files (contain all responses + metrics). They should not be committed. Only the more narrative reports added in a subsequent section should be committed.

2. **Add configuration constant and tests**
   - **Follow Test Development Workflow (see top of document)**
   - `tests/unit/configs/test_config_eval.py`: add `test_eval_artifacts_path_is_relative()` to verify relative path for portability
   - Modify `src/configs/eval.py`: add `DEFAULT_EVAL_ARTIFACTS_PATH = Path("evals/runs")` to reference the destination location of eval run artifacts
  
3. **Add tests for the utility for saving eval run**
   - **Follow Test Development Workflow (see top of document)**
   - `tests/unit/eval/test_utils.py`
   - Test cases for `save_eval_run()` (2 tests minimum):
     - `test_save_creates_directory()` - nonexistent output_dir → creates it
     - `test_save_generates_valid_filename()` - check filename matches pattern `{author}_{timestamp}.json`
   - Use `tmp_path` fixture for file I/O tests

4. **Add utility for saving eval run**
   - Update `src/eval/utils.py`: Add `save_eval_run()`.
   - Timestamped filenames prevent overwrites and enable tracking.
   - First draft below. Adapt to current state of the application.

   ```python
    def save_eval_run(eval_run: EvalRun, output_dir: Path) -> Path:
        """
        Save EvalRun to timestamped JSON file.

        Filename format: {author}_{YYYY-MM-DD}T{HH-MM-SS}.json
        Example: voltaire_2026-03-28T14-30-45.json

        Args:
            eval_run: EvalRun object to save
            output_dir: Directory to save to (usually DEFAULT_EVAL_ARTIFACTS_PATH)

        Returns:
            Path to saved file
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        filename = f"{eval_run.author}_{timestamp}.json"
        filepath = output_dir / filename

        with filepath.open("w") as f:
            # Convert dataclass to dict for JSON serialization
            json.dump(asdict(eval_run), f, indent=2)

        return filepath
     ```

5. **Check In:** Stop and ask the user to confirm that the implementation of the above steps is satisfactory before moving to subsequent steps.

6. **Add tests for the evaluation CLI**
   - **Follow Test Development Workflow (see top of document)**
   - Unit tests on `tests/unit/test_scripts/test_script_run_eval.py`
   - Test cases (7 tests minimum):
       - `test_main_auto_discovery()` - no --golden flag → discovers latest dataset
       - `test_main_explicit_golden_path()` - --golden provided → uses that path
       - `test_main_forwards_author()` - --author gouges → builds chain for gouges
       - `test_main_custom_output()` - --output provided → saves to custom location
       - `test_main_prints_summary()` - after eval → print_summary_table called
       - `test_main_saves_artifact()` - after eval → save_eval_run called with timestamp
       - `test_main_invalid_author()` - --author invalid → exits with error
     - Mock all external dependencies: `discover_latest_golden_dataset`, `load_golden_dataset`, `build_chain`, `run_eval`, `save_eval_run`, `check_ollama_available`
     - Use `@patch` decorators and assert on call arguments

7. **Implement evaluation CLI script**
   - Create `scripts/run_eval.py`
   - Full implementation includes:
     - `print_summary_table(eval_run)` helper - prints human-readable summary with scores, status icons (✅/❌), and pass rate
     - `main()` function with argparse:
       - `--golden` (optional): Path to golden dataset JSON (auto-discovers if not provided)
       - `--author` (optional, default=`DEFAULT_AUTHOR`): Author to evaluate
       - `--output` (optional, default=`DEFAULT_EVAL_ARTIFACTS_PATH`): Output directory for artifacts
       - `--verbose` (optional): Enable debug logging
     - Calls `check_ollama_available()` at startup
     - Auto-discovers latest golden dataset if `--golden` not provided
     - Loads dataset, populates metadata of GoldenDataset as needed, builds chain, runs eval, saves artifact, prints summary
     - Clear error messages for common failures (Ollama not running, dataset not found, invalid author)

   **Key implementation details:**
   ```python
   def print_summary_table(eval_run: EvalRun) -> None:
       """Print human-readable summary table to stdout."""
       print("\n" + "="*70)
       print(f"Evaluation Results: {eval_run.dataset_name} v{eval_run.dataset_version}")
       print(f"Author: {eval_run.author}")
       print(f"Timestamp: {eval_run.run_timestamp}")
       print(f"System: {eval_run.system_version['chat_model']} @ {eval_run.system_version['commit']}")
       print("="*70)

       # Overall scores
       print("\nOVERALL SCORES")
       print("-" * 70)
       overall = eval_run.aggregate_scores["overall"]
       for metric_name, score in sorted(overall.items()):
           status = "✅" if score >= 0.8 else "❌"
           print(f"  {metric_name:30s} {score:5.2f} {status}")

       # By-language breakdown
       if "by_language" in eval_run.aggregate_scores:
           for lang, scores in eval_run.aggregate_scores["by_language"].items():
               print(f"\n{lang.upper()} ONLY")
               print("-" * 70)
               for metric_name, score in sorted(scores.items()):
                   status = "✅" if score >= 0.8 else "❌"
                   print(f"  {metric_name:30s} {score:5.2f} {status}")

       # Cross-language metrics
       if eval_run.aggregate_scores.get("cross_language"):
           print("\nCROSS-LANGUAGE METRICS")
           print("-" * 70)
           for metric_name, score in eval_run.aggregate_scores["cross_language"].items():
               status = "✅" if score >= 0.7 else "❌"  # Lower threshold
               print(f"  {metric_name:30s} {score:5.2f} {status}")

       # Overall pass rate
       print("\n" + "="*70)
       passed = sum(1 for r in eval_run.example_results if r.passed)
       total = len(eval_run.example_results)
       print(f"OVERALL PASS RATE: {eval_run.overall_pass_rate:.1%} ({passed}/{total} examples)")
       print("="*70 + "\n")
   ```

   **Design notes:**
   - **Auto-discovery by default:** Good UX - just run `python scripts/run_eval.py`
   - **Explicit path supported:** For reproducibility (`--golden path/to/old/version.json`)
   - **Clear error messages:** Guide users to common fixes
   - **Human-readable output:** Summary table shows at-a-glance results
   - **Actionable next steps:** Tell user what to do with results
   - **No `--language` flag:** The runner processes all languages by default (bilingual system testing)

8. **Check In:** Stop and ask the user to confirm that the implementation of the above steps is satisfactory before moving to subsequent steps.

9. **Add integration test**
   - Integration test on `tests/integration/test_eval_cli_integration.py`
   - goal: validate the complete user workflow
   - benefits: Catches issues with path handling, file permissions, JSON serialization in realistic context.

  ```python
  def test_run_eval_cli_end_to_end(tmp_path, mock_llm_chain):
     """Integration test: CLI loads dataset, runs eval, saves artifacts."""
     # Arrange: Create minimal golden dataset file
     golden_path = tmp_path / "golden" / "voltaire_golden_v1.0_2026-03-29.json"
     golden_path.parent.mkdir(parents=True)
     # ... write minimal dataset ...

     # Act: Run CLI with real file I/O
     artifacts_path = tmp_path / "runs"
     run_eval_cli(
         golden_dataset_path=golden_path,
         artifacts_path=artifacts_path,
         chain=mock_llm_chain
     )

     # Assert: Artifact file exists and is valid JSON
     artifact_files = list(artifacts_path.glob("*.json"))
     assert len(artifact_files) == 1
     saved_run = EvalRun.model_validate_json(artifact_files[0].read_text())
     assert len(saved_run.example_results) > 0
  ```

### Documentation

- **README:** Reorganize "Development" section:
    - Below is a first draft. Adapt to the current state of the app.
    - Recommend a more prominent position (a new README section) for the content in the section titled "Start chatting with Enlightenment Philosophes". Having it listed at the very bottom of the Development section is way too far down.

  ```markdown
  ## Development

  ### 1-5. [Existing setup steps remain unchanged]

  ### 6. Run the evaluation harness

  **Prerequisites:**
  - Ollama running (`ollama serve`)
  - Corpus ingested (`uv run python scripts/ingest.py`)
  - Golden dataset created (`data/raw/golden/voltaire_golden_v1.0_*.json`)

  **Run evaluation:**
  ```bash
  # Auto-discover latest golden dataset for default author
  uv run python scripts/run_eval.py

  # Specify dataset explicitly (for reproducibility)
  uv run python scripts/run_eval.py --golden data/raw/golden/voltaire_golden_v1.0_2026-03-28.json

  # Different author (after Step 14)
  uv run python scripts/run_eval.py --author gouges

  # Enable debug logging
  uv run python scripts/run_eval.py --verbose
  ```

  **Interpreting results:**
  - **Scores:** 0.0 to 1.0, where 1.0 is perfect
  - **Thresholds:** 0.8 for most metrics, 0.9 for forbidden phrases, 0.7 for translation
  - **✅ = passing**, **❌ = failing** (below threshold)
  - **Overall pass rate:** Fraction of examples where ALL metrics passed

  **Troubleshooting:**
  - **"Ollama is not running":** Start Ollama with `ollama serve`
  - **"No golden dataset found":** Create golden dataset in `data/raw/golden/` with correct naming pattern
  - **"Invalid author":** Check author is registered in `src/chains/chat_chain.py::_AUTHOR_CONFIGS`

  **After running:**
  1. Review artifact in `evals/runs/{author}_{timestamp}.json`
  2. Identify failing metrics (score < threshold)
  3. Read example outputs in artifact to understand failure modes
  4. Iterate on prompts, config, or golden dataset
  5. Re-run eval to measure improvements

  ### 7. Start chatting with Enlightenment Philosophes

  [Move existing "Start chatting" section here]
  ```

- **README:** Update "Project Structure" section:
  ```markdown
  evals/
  └── runs/                # Timestamped JSON artifacts from eval runs (gitignored)

  scripts/
  ├── run_eval.py          # CLI for running evaluation harness
  ...
  ```

### User instructions for first eval run

**🎯 MILESTONE: First Eval Run**

**Stop and instruct the user to manually generate the first artifact and to manually inspect it. Make it clear that we should not update any part of the app at this time based on the results. Subsequent sections will set up reporting to track improvements.**

**Instructions for the user:**

1. **Generate first eval artifact:**
   ```bash
   # Ensure Ollama is running
   ollama serve

   # Ensure ingestion pipeline has run
   uv run python scripts/ingest.py

   # Run evaluation (defaults to Voltaire, auto-discovers latest golden dataset)
   uv run python scripts/run_eval.py
   ```

2. **Manually analyze the initial artifact:**
   - Open the artifact file: `evals/runs/voltaire_{timestamp}.json`
   - Look at `aggregate_scores.overall` - identify metrics with scores < 0.8
   - Look at `example_results` - find examples where `passed: false`
   - Read the `response.text` for failed examples to understand what went wrong

3. **Common issues to look for (DO NOT FIX YET - just observe):**
   - **Low retrieval_relevance (< 0.8):** Wrong chunks retrieved
     - Possible causes: retrieval `k` too low, poor chunk metadata, query-document mismatch

4. **Document your observations:**
   - Write down which metrics failed
   - Note specific examples that failed
   - Keep this for later when we implement narrative reports (subsequent subsection)

**Do NOT make any changes to the application yet.** Subsequent subsections will walk through the improvement cycle. This first run establishes a baseline.

### Plan updates

- **Update this plan:** Mark this subsection `✅` on the title line. Note any deviations below this line.

---

## F. Core citation metrics (including an eval run)

**Goal:** Implement fundamental metrics that directly validate citation functionality of the RAG pipeline. After implementation, run eval again to see citation metrics in action.

### Implementation

1. **Add tests for the citation accuracy metric**
   - **Follow Test Development Workflow (see top of document)**
   - `tests/unit/eval/test_citation_metric.py`
   - Test cases (5 tests minimum):
   - `test_all_titles_found()` - all expected substrings present → score 1.0
   - `test_partial_titles_found()` - 1 of 2 expected substrings present → score 0.5
   - `test_case_insensitive_matching()` - "lettres" matches "Lettres Philosophiques" → score 1.0
   - `test_no_expected_titles()` - empty expected list → score 1.0
   - `test_no_retrieved_titles()` - empty retrieved list with non-empty expected → score 0.0

2. **Implement citation accuracy metric**
   - Create `src/eval/metrics/citation.py`
   - Function signature: `citation_accuracy(expected_source_titles: list[str], retrieved_source_titles: list[str]) -> MetricResult`
   - Logic: Check if expected source title **substrings** are found in retrieved titles (case-insensitive)
   - Example: expected `["Lettres philosophiques"]` matches retrieved `["Lettres philosophiques, page 12"]`
   - Score calculation: `len([exp for exp in expected if any(exp.lower() in retr.lower() for retr in retrieved)]) / len(expected)` if expected is non-empty, else 1.0
   - Return `MetricResult(name="citation_accuracy", score=..., details={"found": [...], "missing": [...]})`

   **Rationale:** Citations ground responses in source texts. This metric validates that the system correctly propagates source metadata (document titles) from retrieval through to the final response. Substring matching allows flexibility for page numbers and formatting variations.

3. **Check In:** Stop and ask the user to confirm that the implementation of the above steps is satisfactory before moving to subsequent steps.

4. **Add tests for the citation-to-retrieval consistency metric**
   - **Follow Test Development Workflow (see top of document)**
   - Update `tests/unit/eval/test_citation_metric.py`
   - Test cases (5 tests minimum):
     - `test_all_citations_match()` - all citations found in retrieved sources → score 1.0
     - `test_partial_citations_match()` - 2 of 3 citations match retrieved sources → score 0.67
     - `test_no_citations_match()` - 0 of 2 citations in retrieved sources → score 0.0 (hallucinated)
     - `test_no_citations_in_response()` - empty response citations → score 1.0 (vacuous truth)
     - `test_case_insensitive_matching()` - "lettres" citation matches "Lettres Philosophiques" source → score 1.0

5. **Implement citation-to-retrieval consistency metric**
   - Update `src/eval/metrics/citation.py`
   - Function signature: `citation_to_retrieval_consistency(response_citations: list[str], retrieved_chunk_sources: list[str]) -> MetricResult`
   - Logic: Verify every citation in the response text corresponds to a chunk that was actually retrieved (runtime consistency check)
   - Extract citations from response text using regex: `r'\[source:\s*([^\]]+)\]'`
   - Score calculation: `len([cite for cite in response_citations if any(cite.lower() in src.lower() for src in retrieved_chunk_sources)]) / len(response_citations)` if response_citations is non-empty, else 1.0
   - Return `MetricResult(name="citation_to_retrieval_consistency", score=..., details={"matched": [...], "hallucinated": [...]})`

   **Rationale:** This metric validates runtime consistency between what the LLM cites and what the retrieval system actually returned. It catches two critical failure modes: (1) hallucinated citations - LLM invents sources that weren't retrieved, and (2) metadata propagation bugs - citations lost somewhere in the chain. This complements `citation_accuracy` which tests against golden dataset expectations.

6. **Check In:** Stop and ask the user to confirm that the implementation is satisfactory before continuing.

7. **Update golden dataset schema and data for citation metrics**
   - **Follow Test Development Workflow (see top of document)**
   - `tests/unit/schemas/test_eval.py` - add tests for `expected_source_titles` field
   - Test cases:
     - `test_expected_source_titles_defaults_to_empty_list()` - verify default value
     - `test_expected_source_titles_accepts_list()` - verify field accepts list of strings
   - Update `src/schemas/eval.py`: Add `expected_source_titles: list[str] = Field(default_factory=list, description="Expected source titles in citations")` to GoldenExample class
   - Update design notes in GoldenExample docstring to document citation field usage

8. **Update golden dataset file with citation data**
   - Modify `data/raw/golden/voltaire_golden_v1.0_*.json`
   - Add `expected_source_titles: ["Lettres philosophiques"]` to all relevant examples
   - Add `expected_source_titles: []` to adversarial examples (no valid sources)
   - Update version to "1.1" in the JSON file
   - Update description to reflect addition of citation metrics
   - Rename file to `voltaire_golden_v1.1_{current-date}.json` (schema change warrants version increment)

9. **Check In:** Stop and ask the user to confirm that the implementation of the above steps is satisfactory before moving to subsequent sections.

### User instructions for eval run

**🎯 MILESTONE: Eval Run (with citation metrics)**

After implementing citation metrics, run eval again:

```bash
uv run python scripts/run_eval.py
```

Compare the new artifact with the first run:
- Check how citation metrics perform
- Note any changes in overall pass rate
- Document observations (keep for reporting workflow in subsequent section)

### Plan updates

- **Update this plan:** Mark this subsection `✅` on the title line. Note any deviations below this line.

---

## G. Eval reports to document improvements

**Goal:** Establish process and expediting tooling to create narrative reports that document eval runs, analysis, and resulting improvements.

### Implementation

1. **Create reports directory**
   - Create `docs/reports/` directory (empty initially)
   - **Do NOT gitignore** - narrative reports should be committed (unlike eval run artifacts)
   - **Rationale:** Evals are living artifacts that need ongoing attention. Reports document the evolution of the system over time, creating institutional memory.

2. **Create report template**
   - Create `docs/reports/TEMPLATE.md` with comprehensive structure
   - Template sections (see full template below):
     - Purpose: What question did this eval answer?
     - System Version: Model, prompt, tools, retrieval setup, commit
     - Dataset: Source, size, coverage
     - Metrics: Scores, thresholds, pass rates
     - Results: Baseline, current run, deltas
     - Error Analysis: Top failure modes, representative examples
     - Changes Made: Code, prompts, config, dataset
     - Recommendation: Ship, iterate, or roll back (with rationale)
     - Limitations: Known gaps, missing cases

   **Draft of template structure:**
   ```markdown
   # Eval Report: [Short Title]

   **Date:** YYYY-MM-DD
   **Author:** [Voltaire|Gouges|...]
   **Languages Tested:** [FR, EN|FR only|EN only]
   **Artifact:** `evals/runs/{filename}.json`
   **Git Commit:** [hash]

   ## Purpose
   What question did this eval answer?

   ## System Version
   - **Chat Model:** [mistral|other]
   - **Embedding Model:** [nomic-embed-text|other]
   - **Prompt File:** [voltaire.py|gouges.py]
   - **Retrieval Config:** k=[value], chunk_size=[value]
   - **Git Commit:** [full hash]

   ## Dataset
   - **Source:** `data/raw/golden/{author}_golden_v{version}_{date}.json`
   - **Version:** [1.0|1.1|...]
   - **Size:** [N] examples
   - **Coverage:** [Topics tested]

   ## Metrics
   | Metric | Threshold | Score | Status |
   |--------|-----------|-------|--------|
   | ... | ... | ... | ✅/❌ |

   **Overall pass rate:** [XX%] ([N]/[M] examples)

   ## Results
   ### Baseline / Current Run / Delta
   [Comparison tables showing before/after scores]

   ## Error Analysis
   ### Top Failure Modes
   [Detailed analysis of what failed and why]

   ### Representative Examples
   [Specific examples with failures explained]

   ## Changes Made
   [Code, prompt, config changes with rationale]

   ## Recommendation
   **[Ship|Iterate|Roll back]**
   [Rationale and next steps]

   ## Limitations
   [Known gaps and missing cases]
   ```

3. **Check In:** Stop and ask the user to confirm that the implementation of the above steps is satisfactory before moving to subsequent steps.

4. **Add tests for the report stub generator**
   - **Follow Test Development Workflow (see top of document)**
   - `tests/unit/eval/test_utils.py` - add tests for `format_eval_report_stub()`
   - Test cases (3 tests minimum):
     - `test_format_stub_includes_metadata()` - stub contains date, author, commit
     - `test_format_stub_includes_metrics_table()` - stub contains all metrics with scores
     - `test_format_stub_includes_todo_prompts()` - stub contains [TODO] markers for manual sections
   
5. **Create report stub generator utility to expedite creation of narrative reports**
   - Modify `src/eval/utils.py`: add `format_eval_report_stub(eval_run: EvalRun) -> str`
   - Auto-fills: date, author, artifact path, commit, system version, metrics table
   - Leaves narrative sections as `[TODO]` prompts for manual completion
   - Returns markdown string ready to save as `.md` file

   **Example stub output:**
   ```markdown
   # Eval Report: [TODO: Short Title]

   **Date:** 2026-03-28
   **Author:** voltaire
   **Languages Tested:** [TODO: FR, EN|FR only|EN only]
   **Artifact:** `evals/runs/voltaire_2026-03-28T14-30-45.json`
   **Git Commit:** a1b2c3d

   ## Purpose
   [TODO: What question did this eval answer?]

   ## Metrics
   | Metric | Threshold | Score | Status |
   |--------|-----------|-------|--------|
   | retrieval_relevance | 0.8 | 0.65 | ❌ |
   | citation_accuracy | 0.8 | 0.70 | ❌ |
   ...

   **Overall pass rate:** 37.5% (3/8 examples passed)

   ## Error Analysis
   [TODO: Identify top failure modes and representative examples]
   ...
   ```

6. **Document reporting process**
   - Create `docs/reports/README.md` explaining:
     - Purpose of narrative reports (institutional memory, decision logs, quality tracking)
     - Workflow (run eval → generate stub → fill narrative → commit)
     - Filename convention: `{YYYY-MM-DD}_{author}_{description}.md`
     - Tips for writing reports (be specific, link commits, read artifacts, track trends)

7. **Check In:** Stop and ask the user to confirm that the implementation of the above steps is satisfactory before moving to subsequent steps.

### Documentation

- **README:** Update "Evaluation Harness" section:
  ```markdown
  **Evaluation reports:**
  Narrative reports documenting eval runs, analysis, and improvements live in
  `docs/reports/` (committed to git, unlike artifacts).

  **Report workflow:**
  1. Run eval → produces `evals/runs/{timestamp}.json` artifact
  2. (Optional) Generate stub: `format_eval_report_stub(eval_run)` to auto-fill dates/scores
  3. Create report: `docs/reports/{YYYY-MM-DD}_{author}_{description}.md`
  4. Fill narrative sections: Purpose, Error Analysis, Changes Made, Recommendation
  5. Commit report to git (preserves institutional memory)

  **Why reports?** According to industry best practices:
  - Numbers alone don't tell the story - context matters
  - Reports document "why" (eval artifacts only document "what")
  - Future you (6 months later) will thank you for the narrative
  - Reports align teams on priorities and quality bar

  See `docs/reports/README.md` for detailed workflow and tips.
  ```

- **README:** Update "Project Structure" section:
  ```markdown
  docs/
  └── reports/             # Narrative eval reports (committed, human-readable)
  ```

### Plan updates

- **Update this plan:** Mark this subsection `✅` on the title line. Note any deviations below this line.

---

## H. Voltaire improvement cycle (First eval-driven iteration)

**Goal:** Close the feedback loop. Use eval artifact to improve quality. Establish an iterative improvement workflow for future features.

### Implementation

This subsection is **procedural** rather than code-focused. It walks through analyzing eval results, proposing fixes, implementing changes, re-running eval, and documenting findings.
This process should be repeated throughout development of subsequent sections.

1. **Analyze baseline eval results**
   - User should have already run `scripts/run_eval.py` in previous subsection
   - Open the artifact: `evals/runs/voltaire_{timestamp}.json`
   - Identify failing metrics (score < threshold)
   - Read `example_results[].response.text` for failed examples
   - Categorize failure modes:
     - **Retrieval issues:** Wrong chunks retrieved
     - **Citation issues:** Missing/malformed citations
     - **Persona issues:** Anachronisms, modern phrasing
     - **Language issues:** Wrong language detected or used
     - **Translation issues:** FR/EN retrieve different chunks

2. **Propose fixes based on common failure modes**

   Fixes should be targeted at root causes, not symptoms.

   **Low retrieval_relevance:**
   - **Root cause:** `k` parameter too low, relevant chunks ranked outside top-k
   - **Fix:** Increase `DEFAULT_K` in `src/configs/vectorstore_config.py` from 5 to 8
   - **Rationale:** More candidates improves recall, at cost of slightly more noise

   **Low citation_accuracy:**
   - **Root cause:** Chunk metadata missing page numbers or titles
   - **Fix:** Verify `_extract_source_titles()` in `src/chains/chat_chain.py` includes page
   - **Alternative:** If metadata is correct, strengthen prompt to emphasize citations

   **Low citation_placement:**
   - **Root cause:** Prompt doesn't clearly instruct citation placement
   - **Fix:** Add explicit examples to Voltaire prompt showing correct vs. incorrect placement
   - **Example addition to prompt:**
     ```
     CORRECT:
     "I believe tolerance is essential to civil society. [source: Lettres philosophiques, page 6]"

     INCORRECT:
     "[source: Lettres philosophiques, page 6]
     I believe tolerance is essential..."
     ```
   - **Rationale:** LLMs learn better from examples than from abstract rules

   **Low forbidden_phrases:**
   - **Root cause:** Persona prompt too weak, no explicit anachronism constraints
   - **Fix:** Add explicit instruction to Voltaire prompt:
     ```
     IMPORTANT: You are Voltaire writing in the 18th century (1694-1778).
     You have NO knowledge of events, technology, or concepts that postdate your death.
     NEVER mention: internet, computers, democracy, human rights (as modern concept),
     artificial intelligence, social media, websites, or any 19th+ century developments.
     If asked anachronistic questions, politely decline: "I'm afraid I don't understand
     this question, as it seems to reference concepts unfamiliar to me."
     ```
   - **Rationale:** Explicit constraints + fallback response prevents persona breaks

   **Low translation_consistency:**
   - **Root cause:** Language detection failing, or embedding model language bias
   - **Fix:** Check language detection confidence thresholds, consider language-specific retrieval filters
   - **Investigation needed:** Run test queries, check detected languages, inspect retrieved chunks

   **Low faithfulness:**
   - **Root cause:** Keywords too specific, or LLM paraphrasing concepts
   - **Fix:** Relax keywords in golden dataset, or add synonyms
   - **Example:** Replace `["tolérance religieuse"]` with `["tolérance", "religion"]` (more flexible)
   - **Alternative:** Improve prompt to emphasize key themes

3. **Implement fixes systematically**

   **Important:** Don't change everything at once. Make isolated changes to understand cause-and-effect.

   **Recommended order:**
   1. **Start with easiest wins:** Config changes (k parameter)
   2. **Then prompt improvements:** Citation placement examples, persona constraints
   3. **Finally dataset refinement:** If needed after re-running eval

   **For each change:**
   - Make edit in one file
   - Run `uv run pytest` to ensure no regressions
   - Run `uv run mypy src scripts` to ensure type safety
   - Commit change with clear message: `"Increase retrieval k from 5 to 8 to improve recall"`

   **If changes are substantial (>3 files, >100 lines):**
   - Break into multiple PRs:
     - PR 1: Config changes (k parameter, thresholds)
     - PR 2: Voltaire prompt improvements (citations, persona)
     - PR 3: Golden dataset refinements (keywords, examples)

4. **Re-run evaluation to measure improvements**

   ```bash
   # After making changes, re-run eval
   uv run python scripts/run_eval.py
   ```

   - Compare new artifact to baseline
   - Check if failing metrics improved
   - **Watch for regressions:** Did any previously-passing metrics drop?
   - If improvements are insufficient, iterate again
   - If regressions occurred, consider rolling back and trying different approach

5. **Document findings in eval report**

   **Create first report:** `docs/reports/2026-03-28_voltaire_baseline-and-first-iteration.md`

   **Use template from previous subsection:**
   - **Purpose:** "Establish baseline quality for Voltaire, then improve failing metrics"
   - **Results:** Show before/after scores in delta table
   - **Error Analysis:** Document top failure modes from baseline
   - **Changes Made:** List all edits (prompts, config) with commit hashes
   - **Recommendation:** Ship if all metrics pass, iterate if still failing
   - **Limitations:** Note small sample size (6-8 examples), need more adversarial cases

   **Example delta table:**
   | Metric | Before | After | Δ |
   |--------|--------|-------|---|
   | retrieval_relevance | 0.65 | 0.88 | +0.23 |
   | citation_accuracy | 0.70 | 0.85 | +0.15 |
   | citation_placement | 0.60 | 0.90 | +0.30 |
   | forbidden_phrases | 0.75 | 0.95 | +0.20 |
   | Overall pass rate | 37.5% | 87.5% | +50.0% |

6. **Commit the report**

   ```bash
   git add docs/reports/2026-03-28_voltaire_baseline-and-first-iteration.md
   git commit -m "Document Voltaire eval baseline and first improvement iteration"
   ```

   **Do NOT commit eval artifacts** - they're gitignored. Only commit the narrative report.

7. **Establish iteration workflow for future improvements**

   **The eval-driven development loop:**
   1. Run eval → get artifact
   2. Analyze failures → identify root causes
   3. Propose fixes → make targeted changes
   4. Re-run eval → measure improvements
   5. Document → create report with findings
   6. Repeat until quality bar met

   **When to stop iterating:**
   - All metrics above threshold ✅
   - Overall pass rate > 80% ✅
   - No low-hanging fruit remaining ✅
   - Time to add new capability (e.g., Gouges in Step 14) ✅

   **This workflow continues through:**
   - Step 14: Gouges eval and two-philosopher comparison
   - Step 15: Debate mode eval
   - Step 16: LLM-as-judge eval
   - Production: Regression testing as system evolves

### Documentation

- **README:** Update "Evaluation Harness" section with workflow summary:
  ```markdown
  **Eval-driven development workflow:**
  1. **Run eval:** `uv run python scripts/run_eval.py`
  2. **Analyze:** Read artifact, identify failing metrics and root causes
  3. **Fix:** Make targeted changes (prompts, config, dataset)
  4. **Re-run:** Measure improvements, watch for regressions
  5. **Document:** Create report in `docs/reports/` with findings
  6. **Iterate:** Repeat until quality bar met

  This workflow ensures improvements are data-driven and traceable.
  ```

- **README:** Add "Example Eval Report" section:
  ```markdown
  ## Example Eval Report

  After establishing the Voltaire baseline and first improvement iteration,
  see `docs/reports/2026-03-28_voltaire_baseline-and-first-iteration.md` for
  an example of how to document eval findings, changes, and results.

  Key sections:
  - **Error Analysis:** What failed and why
  - **Changes Made:** Specific fixes with commit hashes
  - **Results Delta:** Before/after scores
  - **Recommendation:** Ship, iterate, or roll back
  ```

### Plan updates

- **Update this plan:** Mark this subsection `✅` on the title line. Note any deviations below this line.

- **Document findings:**

  Example documentation (customize based on actual results):

  ```markdown
  **Baseline (2026-03-28):**
  - Overall pass rate: 37.5% (3/8 examples)
  - Failing metrics: retrieval_relevance (0.65), citation_accuracy (0.70),
    citation_placement (0.60), forbidden_phrases (0.75)

  **Root causes identified:**
  1. Retrieval k=5 too low for complex questions
  2. Prompt lacked explicit citation placement examples
  3. Persona constraints too weak (anachronisms leaked through)

  **Changes made:**
  - Increased `DEFAULT_K` from 5 to 8 (commit abc123)
  - Added citation placement examples to Voltaire prompt (commit def456)
  - Added explicit 18th-century constraint + anachronism list (commit ghi789)

  **After iteration (2026-03-28):**
  - Overall pass rate: 87.5% (7/8 examples)
  - All metrics above threshold except 1 edge case (compound Newton+Pascal question)

  **Recommendation:** Ship. System meets quality bar for Voltaire.

  **Next:** Add Gouges (Step 14) and run comparative eval.

  **Report:** `docs/reports/2026-03-28_voltaire_baseline-and-first-iteration.md`
  ```

---

## I. Language and faithfulness metrics (including another eval run + eval report)

**Goal:** Building on the fundamental metrics, implement content quality metrics to validate language detection and semantic grounding. Run eval and create a report documenting findings.

### Implementation

1. **Add tests for language metadata compliance metric**
   - **Follow Test Development Workflow (see top of document)**
   - `tests/unit/eval/test_language_metric.py`
   - Test cases (2 tests minimum):
     - `test_matching_language_metadata()` - "fr" == "fr" → score 1.0
     - `test_mismatching_language_metadata()` - "en" != "fr" → score 0.0
     - 
2. **Implement language metadata compliance metric**
   - Create `src/eval/metrics/language.py`
   - Function signature: `language_metadata_compliance(expected_language: str, response_language: str) -> MetricResult`
   - Logic: exact match between expected and actual language metadata (ISO 639-1 codes: "en", "fr")
   - Score calculation: `1.0` if match, `0.0` if mismatch
   - Return `MetricResult(name="language_metadata_compliance", score=..., details={"expected": expected, "actual": response_language})`

   **Rationale:** The system auto-detects user language and sets the `ChatResponse.language` field accordingly. This metric validates that language detection works correctly and the chain sets the metadata property as expected.

3. **Check In:** Stop and ask the user to confirm that the implementation of the above steps is satisfactory before moving to subsequent steps.

4. **Add tests for language content compliance metric**
   - **Follow Test Development Workflow (see top of document)**
   - Update `tests/unit/eval/test_language_metric.py`
   - Test cases (4 tests minimum):
     - `test_french_content_detected()` - French text with expected="fr" → score 1.0
     - `test_english_content_detected()` - English text with expected="en" → score 1.0
     - `test_french_content_mismatch()` - French text with expected="en" → score 0.0
     - `test_english_content_mismatch()` - English text with expected="fr" → score 0.0
   - Use realistic sample text in both languages (not single words - langdetect needs ~20+ chars)

5. **Implement language content compliance metric**
   - Update `src/eval/metrics/language.py`
   - Function signature: `language_content_compliance(expected_language: str, response_text: str) -> MetricResult`
   - Logic: Detect actual language from response text content and compare to expected language
   - Use `langdetect` library: `detect(response_text)` returns ISO 639-1 code
   - Score calculation: `1.0` if detected == expected, `0.0` if mismatch
   - Return `MetricResult(name="language_content_compliance", score=..., details={"expected": expected, "detected": detected_language})`

   **Rationale:** This metric validates that the LLM actually responds in the correct language, not just that the metadata is set correctly. It catches cases where `ChatResponse.language="fr"` but the LLM responds in English anyway (LLM non-compliance with system prompts).

   **Design note:** Requires adding `langdetect` to dependencies (`uv add langdetect`). This is a lightweight library with no external API calls (runs locally).

6. **Check In:** Stop and ask the user to confirm that the implementation of the above steps is satisfactory before moving to subsequent steps.

7. **Add tests for faithfulness metrics**
   - **Follow Test Development Workflow (see top of document)**
   - `tests/unit/eval/test_faithfulness_metric.py`
   - Test cases for French (5 tests minimum):
     - `test_all_keywords_found_fr()` - all expected keywords present → score 1.0
     - `test_partial_keywords_found_fr()` - 2 of 3 keywords present → score 0.67
     - `test_no_keywords_found_fr()` - 0 of 3 keywords present → score 0.0
     - `test_case_insensitive_fr()` - "Tolérance" matches "tolérance" → score 1.0
     - `test_no_expected_keywords_fr()` - empty expected list → score 1.0
   - Test cases for English (4 tests minimum):
     - `test_all_keywords_found_en()` - all expected keywords present → score 1.0
     - `test_partial_keywords_found_en()` - 1 of 2 keywords present → score 0.5
     - `test_no_keywords_found_en()` - 0 of 2 keywords present → score 0.0
     - `test_no_expected_keywords_en()` - empty expected list → score 1.0

8. **Implement faithfulness metrics (bilingual)**
   - Create `src/eval/metrics/faithfulness.py`
   - Shared helper function: `_keyword_score(expected_keywords: list[str], response_text: str) -> float`
     - Logic: fraction of expected keywords found in response text (case-insensitive, whole-word matching)
     - Use regex word boundaries: `r'\b' + re.escape(keyword) + r'\b'` with `re.IGNORECASE`
     - Returns: `len([kw for kw in expected if re.search(pattern, text)]) / len(expected)` if non-empty, else 1.0

   - **French faithfulness:**
     - Function: `faithfulness_fr(expected_keywords_fr: list[str], response_text: str) -> MetricResult`
     - Returns: `MetricResult(name="faithfulness_fr", score=_keyword_score(...), details={"found": [...], "missing": [...]})`

   - **English faithfulness:**
     - Function: `faithfulness_en(expected_keywords_en: list[str], response_text: str) -> MetricResult`
     - Returns: `MetricResult(name="faithfulness_en", score=_keyword_score(...), details={"found": [...], "missing": [...]})`

   **Rationale:** Groundedness checks verify that responses are supported by retrieved sources. Keyword-based faithfulness is a deterministic proxy for semantic grounding—we expect certain key concepts (tolerance, conscience, reason) to appear when discussing Enlightenment topics. This catches cases where the LLM hallucinates or goes off-topic.

   **Design note:** Separate FR/EN functions allow different keyword lists for each language while sharing scoring logic. This follows the DRY principle (helper is reused) while maintaining language-specific validation.

9. **Check In:** Stop and ask the user to confirm that the implementation of the above steps is satisfactory before moving to subsequent steps.

10. **Update golden dataset schema and data for faithfulness metrics**
   - **Follow Test Development Workflow (see top of document)**
   - `tests/unit/schemas/test_eval.py` - add tests for language-specific keyword fields
   - Test cases:
     - `test_expected_keywords_fr_defaults_to_empty_list()` - verify default value
     - `test_expected_keywords_en_defaults_to_empty_list()` - verify default value
     - `test_expected_keywords_language_specific()` - verify FR examples use fr field, EN examples use en field
   - Update `src/schemas/eval.py`: Add to GoldenExample class:
     - `expected_keywords_fr: list[str] = Field(default_factory=list, description="Keywords expected in French response")`
     - `expected_keywords_en: list[str] = Field(default_factory=list, description="Keywords expected in English response")`
   - Update design notes to document language-specific validation pattern

11. **Update golden dataset file with keyword expectations**
   - Modify golden dataset JSON file (v1.1 → v1.2)
   - Add keyword fields to each example based on its language:
     - French examples: populate expected_keywords_fr, leave expected_keywords_en empty
     - English examples: populate expected_keywords_en, leave expected_keywords_fr empty
   - Example values:
     - FR tolerance: `"expected_keywords_fr": ["tolérance", "conscience", "persécution"]`
     - EN tolerance: `"expected_keywords_en": ["tolerance", "conscience", "persecution"]`
   - Update version to "1.2" in the JSON file
   - Update description to reflect addition of faithfulness metrics
   - Rename file to `voltaire_golden_v1.2_{current-date}.json`

12. **Check In:** Stop and ask the user to confirm that the implementation of the above steps is satisfactory before moving to subsequent sections.

### User instructions for eval run + report

**🎯 MILESTONE: Eval Run (with language/faithfulness metrics) + First Full Eval Report**

After implementing language and faithfulness metrics:

1. **Run eval:**
   ```bash
   uv run python scripts/run_eval.py
   ```

2. **Create eval report:**
   - Use the report stub generator to auto-fill metadata
   - Compare with previous eval run artifacts
   - Document language metric performance
   - Document faithfulness metric performance
   - Include delta table showing improvements
   - Commit report to `docs/reports/`

This is the first full report documenting metric additions and system improvements.

### Plan updates

- **Update this plan:** Mark this subsection `✅` on the title line. Note any deviations below this line.

---

## J. Advanced quality metrics (including another eval run + eval report)

**Goal:** Implement specialized validation metrics. Iterate on regex patterns and scoring approaches based on feedback from previous subsections. Run eval and document findings in a report.

### Implementation

1. **Add tests for citation placement metric**
   - **Follow Test Development Workflow (see top of document)**
   - `tests/unit/eval/test_citation_placement_metric.py`
   - Test cases (5 tests minimum):
     - `test_correct_placement()` - "Some claim. [source: X]" → score 1.0
     - `test_citation_at_line_start()` - "[source: X]\nSome text" → score < 1.0
     - `test_citation_after_newline_without_text()` - "Para 1.\n\n[source: X]" → score < 1.0
     - `test_no_citations()` - text without any citations → score 1.0
     - `test_multiple_correct_placements()` - multiple well-placed citations → score 1.0

2. **Implement citation placement metric**
   - Create `src/eval/metrics/citation_placement.py`
   - Function signature: `citation_placement(response_text: str) -> MetricResult`
   - Logic: Verify that inline citations `[source: ...]` appear AFTER sentences/claims, not at the beginning or middle of paragraphs
   - Detection strategy: Use regex to find citations that appear at problematic positions
     - Pattern for bad placement: `r'(?:^|\n)\s*\[source:'` (citation at line start or after newline without preceding text)
     - Count violations vs. total citations
   - Score calculation: `1.0 - (violations / total_citations)` if citations exist, else 1.0 (no citations = no violations)
   - Return `MetricResult(name="citation_placement", score=..., details={"total_citations": n, "violations": m, "violation_samples": [...]})`

   **Rationale:** LLMs sometimes place citations incorrectly despite instructions. This metric enforces the rule that citations must follow claims, not precede them. Well-placed citations improve readability and user trust. This is an example of a "transcript analysis" grader that validates specific formatting behaviors.

   **Note:** Regex patterns may need iteration based on real outputs. The "Check In" points allow refining detection logic before committing to full eval runs.

3. **Check In:** Stop and ask the user to confirm that the implementation of the above steps is satisfactory before moving to subsequent steps.

4. **Add tests for translation metric**
   - **Follow Test Development Workflow (see top of document)**
   - `tests/unit/eval/test_translation_metric.py`
   - Test cases (5 tests minimum):
     - `test_identical_chunks()` - same chunk IDs → score 1.0 (perfect overlap)
     - `test_disjoint_chunks()` - no common IDs → score 0.0 (no overlap)
     - `test_partial_overlap()` - 2 shared IDs, 1 unique each → score 0.5 (Jaccard = 2/4)
     - `test_both_empty()` - empty lists → score 1.0 (vacuous truth)
     - `test_one_empty()` - one empty, one non-empty → score 0.0 (no overlap)

5. **Implement translation consistency metric**
   - Create `src/eval/metrics/translation.py`
   - Function signature: `translation_consistency(fr_chunk_ids: list[str], en_chunk_ids: list[str]) -> MetricResult`
   - Logic: Measure retrieval overlap between French and English responses to the same philosophical concept (Jaccard similarity)
   - Score calculation: `len(set(fr_ids) & set(en_ids)) / len(set(fr_ids) | set(en_ids))` if union is non-empty, else 1.0
   - Return `MetricResult(name="translation_consistency", score=..., details={"overlap": [...], "fr_only": [...], "en_only": [...]})`

   **Rationale:** This is a **cross-language metric** that validates bilingual system behavior. When asking about "tolerance" in French vs. English, the system should retrieve similar chunks (same philosophical content, regardless of language). High Jaccard overlap indicates consistent retrieval. Low overlap suggests language detection or retrieval bugs.

   **Important:** This metric only works when the eval run processes BOTH languages. It compares paired examples (e.g., `tolerance_fr` vs. `tolerance_en`) using the `translation_pair_id` field in the golden dataset.

6. **Check In:** Stop and ask the user to confirm that the implementation of the above steps is satisfactory before moving to subsequent steps.

7. **Update golden dataset schema and data for translation consistency**
   - **Follow Test Development Workflow (see top of document)**
   - `tests/unit/schemas/test_eval.py` - add tests for translation pairing field
   - Test cases:
     - `test_translation_pair_id_defaults_to_none()` - verify optional field
     - `test_translation_pair_id_links_examples()` - verify paired examples can share same ID
     - `test_translation_pair_id_accepts_null()` - verify null value accepted
   - Update `src/schemas/eval.py`: Add `translation_pair_id: str | None = Field(None, description="Links FR/EN versions of same concept")` to GoldenExample class
   - Update design notes to document cross-language pairing pattern (both examples link with same ID, e.g., "tolerance")

8. **Update golden dataset file with translation pairs**
   - Modify golden dataset JSON file (v1.2 → v1.3)
   - Link FR/EN pairs with matching translation_pair_id values:
     - "tolerance_fr" and "tolerance_en" → both get `"translation_pair_id": "tolerance"`
     - "pascal_fr" and "pascal_en" → both get `"translation_pair_id": "pascal"`
     - "newton_fr" and "newton_en" → both get `"translation_pair_id": "newton"`
     - Adversarial examples → `"translation_pair_id": null`
   - Update version to "1.3" in the JSON file
   - Update description to reflect addition of translation consistency metric
   - Rename file to `voltaire_golden_v1.3_{current-date}.json`

9. **Check In:** Stop and ask the user to confirm that the implementation of the above steps is satisfactory before moving to subsequent sections.

10. **Add tests for response length metric**
   - **Follow Test Development Workflow (see top of document)**
   - `tests/unit/eval/test_response_length_metric.py`
   - Test cases (5 tests minimum):
     - `test_within_bounds()` - 500 chars, min=100, max=1000 → score 1.0, status="ok"
     - `test_too_short()` - 50 chars, min=100, max=1000 → score 0.5, status="too_short"
     - `test_too_long()` - 1500 chars, min=100, max=1000 → score < 1.0, status="too_long"
     - `test_at_min_boundary()` - 100 chars, min=100, max=1000 → score 1.0, status="ok"
     - `test_at_max_boundary()` - 1000 chars, min=100, max=1000 → score 1.0, status="ok"

11. **Implement response length metric**
   - Create `src/eval/metrics/response_length.py`
   - Function signature: `response_length_quality(response_text: str, min_length: int = 100, max_length: int = 1000) -> MetricResult`
   - Logic: Measure response length and check if it falls within acceptable bounds
   - Calculate character count (excluding whitespace): `len(response_text.strip())`
   - Score calculation:
     - If `min_length <= char_count <= max_length`: score = 1.0
     - If `char_count < min_length`: score = char_count / min_length (partial credit for being close)
     - If `char_count > max_length`: score = max(0.0, 1.0 - (char_count - max_length) / max_length) (penalty for verbosity)
   - Return `MetricResult(name="response_length_quality", score=..., details={"char_count": n, "min_length": min_length, "max_length": max_length, "status": "ok"|"too_short"|"too_long"})`

   **Rationale:** Response length is a quick quality signal. Too short = incomplete/evasive responses, too long = verbose/unfocused responses. This catches obvious failures (single-word answers, walls of text) before deeper quality metrics. Thresholds are configurable via golden dataset examples.

12. **Check In:** Stop and ask the user to confirm that the implementation of the above steps is satisfactory before moving to subsequent steps.

### User instructions for eval run + report

**🎯 MILESTONE: Fourth Eval Run (with advanced quality metrics) + Second Full Report**

After implementing advanced quality metrics:

1. **Run eval:**
   ```bash
   uv run python scripts/run_eval.py
   ```

2. **Create eval report:**
   - Document citation placement metric behavior
   - Document translation consistency across FR/EN pairs
   - Document response length quality
   - Compare with previous runs
   - Note any cross-language inconsistencies discovered
   - Commit report to `docs/reports/`

### Plan updates

- **Update this plan:** Mark this subsection `✅` on the title line. Note any deviations below this line.

---

## K. Safety guardrails (including another eval run + eval report)

**Goal:** Implement binary safety checks to catch persona breaks and anachronisms. These are pass/fail guardrails, distinct from gradual quality metrics. Run eval and document in final report.

### Implementation

1. **Add tests for forbidden phrases metrics**
   - **Follow Test Development Workflow (see top of document)**
   - `tests/unit/eval/test_forbidden_metric.py`
   - Test cases for French (5 tests minimum):
     - `test_no_forbidden_phrases_fr()` - clean text → score 1.0
     - `test_no_forbidden_list_fr()` - empty forbidden list → score 1.0
     - `test_single_violation_fr()` - "internet" in text → score 0.0
     - `test_case_insensitive_fr()` - "Internet" matches "internet" → score 0.0
     - `test_multiple_violations_fr()` - "internet" and "website" both present → score 0.0
   - Test cases for English (4 tests minimum):
     - `test_no_forbidden_phrases_en()` - clean text → score 1.0
     - `test_no_forbidden_list_en()` - empty forbidden list → score 1.0
     - `test_single_violation_en()` - "democracy" in text → score 0.0
     - `test_multiple_violations_en()` - "AI" and "website" both present → score 0.0
     
2. **Implement forbidden phrases metrics (bilingual)**
   - Create `src/eval/metrics/forbidden.py`
   - Shared helper function: `_forbidden_score(forbidden_phrases: list[str], response_text: str) -> float`
     - Logic: Check if any forbidden phrases appear in response (case-insensitive substring search)
     - Use simple `any(phrase.lower() in response_text.lower() for phrase in forbidden_phrases)`
     - Returns: `0.0` if any forbidden phrase found, `1.0` if clean

   - **French forbidden phrases:**
     - Function: `forbidden_phrases_fr(forbidden_list: list[str], response_text: str) -> MetricResult`
     - Returns: `MetricResult(name="forbidden_phrases_fr", score=_forbidden_score(...), details={"violations": [...]})`

   - **English forbidden phrases:**
     - Function: `forbidden_phrases_en(forbidden_list: list[str], response_text: str) -> MetricResult`
     - Returns: `MetricResult(name="forbidden_phrases_en", score=_forbidden_score(...), details={"violations": [...]})`

   **Rationale:** This metric catches **persona breaks** and **anachronisms**. Voltaire should never mention "internet", "website", "democracy" (modern concepts), or "I am an AI" (persona break). These are "outcome verification" checks - binary tests for unacceptable behaviors. This is a guardrail metric, not a quality metric.

   **Design note:** Binary scoring (0.0 or 1.0) is appropriate here. A single anachronism ruins immersion; partial credit doesn't make sense.

3. **Check In:** Stop and ask the user to confirm that the implementation of the above steps is satisfactory before moving to subsequent steps.

4. **Update golden dataset schema and data for safety validation**
   - **Follow Test Development Workflow (see top of document)**
   - `tests/unit/schemas/test_eval.py` - add tests for forbidden phrase fields
   - Test cases:
     - `test_forbidden_phrases_fr_defaults_to_empty_list()` - verify default value
     - `test_forbidden_phrases_en_defaults_to_empty_list()` - verify default value
     - `test_forbidden_phrases_language_specific()` - verify language-appropriate forbidden phrases
   - Update `src/schemas/eval.py`: Add to GoldenExample class:
     - `forbidden_phrases_fr: list[str] = Field(default_factory=list, description="Phrases forbidden in French response")`
     - `forbidden_phrases_en: list[str] = Field(default_factory=list, description="Phrases forbidden in English response")`
   - Update design notes to document safety guardrail fields

5. **Update golden dataset file with forbidden phrases**
   - Modify golden dataset JSON file (v1.3 → v1.4)
   - Add forbidden phrases to each example based on its language:
     - French examples: `"forbidden_phrases_fr": ["internet", "site web", "démocratie moderne", "intelligence artificielle", "je suis un AI"]`
     - English examples: `"forbidden_phrases_en": ["internet", "website", "modern democracy", "artificial intelligence", "I am an AI"]`
     - Adversarial examples: extensive forbidden phrase lists (the test cases for persona breaks)
       - Example: `"forbidden_phrases_en": ["social media", "post", "tweet", "Facebook", "internet", "I am an AI"]`
   - Update version to "1.4" in the JSON file
   - Update description to reflect final schema for Step 13
   - Rename file to `voltaire_golden_v1.4_{current-date}.json` (final schema for evaluation harness)

6. **(Optional) Add metadata field for extensibility**
   - If needed for future metrics, add `metadata: dict[str, Any] = Field(default_factory=dict, description="Extra fields for future metrics")` to GoldenExample class
   - Otherwise defer to when actually needed
   - Note: Can be added without dataset version bump (extensibility field)

7. **Check In:** Stop and ask the user to confirm that the implementation of the above steps is satisfactory before continuing.

### User instructions for eval run + report

**🎯 MILESTONE: Fifth Eval Run (complete metric suite) + Final Report**

After implementing safety guardrails:

1. **Run eval with complete metric suite:**
   ```bash
   uv run python scripts/run_eval.py
   ```

2. **Create final eval report:**
   - Document forbidden phrases performance
   - Document how well the system maintains persona
   - Analyze adversarial example results (anachronism traps)
   - **Overall system assessment:**
     - Compare all five eval runs
     - Show metric progression over time
     - Identify strongest/weakest areas
     - Recommend next steps
   - Commit report to `docs/reports/`

### Plan updates

- **Update this plan:** Mark this subsection `✅` on the title line. Note any deviations below this line.

---
