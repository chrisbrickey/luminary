# Evaluation Harness

## Overview

**Design principles:**
- Traceability: Eval artifacts are linked to versions of golden datasets and system attributes.
- Multi-language by default: Test the multilingual system as users experience it, starting with English and French.
- Iterative approach: Deterministic graders first: fast, reproducible, objective. Add more languages and complex metrics as the eval harness stabilizes.
- Single source of truth: Centralized METRIC_REGISTRY provides the single list of metrics so eval runners and other callers don't need to be updated to detect new metrics.
- Comprehensive coverage: happy paths and edge cases (anachronisms, persona breaks)

**Metrics Categories:**
- Deterministic metrics: retrieval relevance, citation accuracy, language compliance
- Quality metrics: faithfulness to source texts, citation placement, persona maintenance
- Cross-language metrics: translation consistency (FR/EN retrieval overlap)
- LLM-as-judge metrics: relevance, groundedness, coherence

**Golden datasets:**
- Golden datasets are versioned collections of test cases that validate system behavior. Versioning allows comparing results across time and referencing the correct snapshot of data when creating eval reports. 
- They live in `evals/golden/` (gitignored) with naming convention: `{scope}_{authors}_v{version}_{YYYY-MM-DD}.json`

**Eval Artifacts:**
- Eval runs are saved to `evals/runs/` (gitignored) as timestamped JSON files with filename format: `{YYYY-MM-DD}T{HH-MM-SS}.json`.
- Contains all example results, aggregate scores, and system version metadata for traceability

## Run the evaluation harness

**Prerequisites:**
- Ollama running (`ollama serve`)
- Corpus ingested (`uv run python scripts/ingest.py`)
- Golden dataset created (in `evals/golden/` with naming pattern `{scope}_{authors}_v{version}_{YYYY-MM-DD}.json`)

**Run evaluation:**
```
# Auto-discover latest golden dataset for default author
uv run python scripts/run_eval.py

# Specify dataset explicitly (for reproducibility)
uv run python scripts/run_eval.py --golden-path path/to/golden/dataset
```

Optional flags:

| Flag          | Description                          | Default                                             |
|---------------|--------------------------------------|-----------------------------------------------------|
| --golden-path | Path to golden dataset JSON file     | Auto-discovery of latest dataset for default author |
| --output-path | Output directory for eval artifacts  | `evals/runs`                                        |
| --verbose     | Enable debug logging                 | False                                               |


**Interpreting results:**
- **Scores:** 0.0 to 1.0, where 1.0 is perfect
- **Threshold examples:** 0.8 for most metrics
- **✅ = passing**, **❌ = failing** (below threshold)
- **Overall pass rate:** Fraction of examples where ALL metrics passed

## Troubleshooting 

### Missing metrics

There are multiple levels of protection to ensure that eval runs include all metrics, including an automated metrics registry.
If one or more metrics are still excluded from an eval run, you will see a warning in the terminal: `registered metric(s) were NOT applied to any examples`.

Potential causes:
- Field name mismatch: Metric requires fields that don't exist in ChatResponse schema
- Missing data: Golden dataset examples don't have required fields populated
- Language mismatch: Metric only applies to languages not in this dataset

Recommendations:
- Review metric requirements above\n"
- Check ChatResponse schema for correct field names\n"
- Ensure golden dataset has appropriate test data\n"
