# Eval Report [YYYY-MM-DDTHH-MM-SS]
Following eval runs, narrative reports document improvements made to the Luminary app.
[Repeat or delete the `###` subsections as needed.]


## Source Data
The eval run and its golden dataset that were used as the initial input to this report.

- **Eval Run Artifact:** `evals/runs/{filename}.json`
- **Dataset Identifier:** `evals/golden/{filename}.json`


## System Snapshot
- **Commit:** [git-hash]
- **Chat Model:** [mistral|other]
- **Embedding Model:** [bge-m3|other]
- **Retrieval Chunk Count (k):** [value]
- **Retrieval Chunk Size:** [value]
- **Ollama PID:** [value]
- **Ollama Uptime (seconds):** [value]
- **Ollama Version:** [value]
- **Ollama Loaded Models:** [value]


## Eval Run Summary

**Overall pass rate:** [XX%]
**Overall average:** [X.XX]

| Metric Name | Effective Threshold | Score | Status    |
|-------------|---------------------|-------|-----------|
| ...         | ...                 | ... | Pass/Fail |
| ...         | ...                 | ... | Pass/Fail |
| ...         | ...                 | ... | Pass/Fail |


## Issue Analysis
Description of top failure modes (what failed and why), including representative examples.

### [Issue title]
[Analysis of this issue.]


## Changes Made
Listing of changes made (e.g., code, prompts, configs, datasets), including rationales.

### [Change title]
[Description of this change and the rationale for implementing it.]


## Changes Deferred
Listing of changes to be deferred (e.g., known gaps, missing cases), including rationales.

### [Deferred change title]
[Description of this change and the rationale for deferring it.]


## Changes Rejected
Listing of changes that were considered and rejected, including rationales.

### [Rejected change title]
[Description of this change and the rationale for rejecting it.]
