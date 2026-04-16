# Luminary

A RAG application where Enlightenment writers (e.g., Voltaire, Olympe de Gouges) answer questions grounded exclusively in their historical texts with sources cited. 
Available as both a web UI and interactive CLI.

## Interfaces

Luminary provides two interfaces for interacting with the philosophes. Both interfaces support the same core features and personas.

- **Web UI**: Browser-based chat interface with message history, sidebar controls, and visual source citations
- **CLI**: Command-line interactive chat with optional debug output and chunk inspection

## Key Features

- **Grounded citations:** All responses include source references with page numbers in the text response. Additionally, a sources footer shows each unique source only once.
- **Local-first execution:** Runs entirely locally with Ollama—no external API calls, no data sharing, no usage costs.
- **Semantic search over historical corpus:** Uses vector similarity to find relevant passages across all texts, not just keyword matching.
- **Transparent retrieval:** Optional debug mode shows exact text chunks and similarity scores used for each response.
- **Multilingual chat:** Automatically detects the user's language and directs LLM to respond in the same language. _Local LLMs are not deterministic. So adherence to the detected language depends on the training data available to the selected model._ 
- **Localized interfaces:** Chat interface formatting (labels, buttons, loading messages) available in multiple languages and adapts to the detected language of the user.
- **Evaluation harness:** Bespoke, automated evaluation harness to measure response quality and prevent regressions.

## Architecture

### Technology

| Dependency          | Purpose                                |
|---------------------|----------------------------------------|
| python 3.13         | language runtime                       |
| uv                  | package manager                        |
| pydantic            | data validation and schema definitions |
| langchain           | LLM orchestration framework            |
| langdetect          | automatic language detection           |
| chromaDB            | vector store for document embeddings   |
| streamlit           | web UI framework                       |
| pytest + pytest-cov | test runner and coverage               |
| mypy                | static type checking                   |
| autoflake           | unused import detection                |

### Pipelines

Luminary will be organized into four pipelines. The ingestion and query pipelines are complete. 
Evaluation harness and multi-agent debate are under development.

```
I. INGESTION PIPELINE
──────────────────────────────────────────────────────────────
fetch raw data → persist JSON documents → generate overlapping chunks → embed vectors in vector DB

II. QUERY PIPELINE
──────────────────────────────────────────────────────────────
user question → retrieve relevant chunks from vector DB → format context → prompt LLM with agent personas → respond to user

III. EVALUATION PIPELINE
──────────────────────────────────────────────────────────────
load golden dataset -> invoke chat chain → apply all metrics → aggregate scores → persist eval run artifact
```

##### Pipelines in Detail

```
INGESTION (one-time / on-demand via scripts)
─────────────────────────────────────────────────────────────────────
 *_loader.py             fetches source data, strips formatting, returns LangChain Documents
      │
      ▼
 chunker.py              splits Documents into overlapping chunks and adds metadata
      │
      ▼
 chroma.py               embeds chunks and persists in a vector database


QUERY (real-time via user prompt)
─────────────────────────────────────────────────────────────────────
  raw string             input from user in their natural language on which we detect language code
      │
      ▼
 retriever.py            embeds the prompt, performs similarity search on the vector database and retrieves top-k semantically similar chunks
      │
      ▼
 chat_chain.py           orchestrates retrieval, context formatting with labels, and LLM call including persona prompt and language; returns ChatResponse
      │
      ▼
 ChatResponse            validated and structured response in user's language
 
 
EVALUATION (on-demand quality measurement)
─────────────────────────────────────────────────────────────────────
 GoldenDataset            versioned example cases with expected behaviors (e.g., questions, expected chunks, keywords)
      │
      ▼
 runner.py                invokes chat chain for each example, applies metrics, aggregates scores
      │
      ▼
 metrics/                 deterministic graders such as retrieval_relevance, citation_accuracy, language compliance
      │
      ▼
 EvalRun                  machine-readable artifact with all results, scores, and system version (saved to evals/runs/)
```


## Setup

### 1. Clone the repo

```
git clone <repo-url>
cd luminary
```

### 2. Install dependencies

```
# Install core dependencies (CLI scripts)
uv sync

# Install with web UI support (adds Streamlit)
uv sync --extra ui
```

### 3. Set up Ollama

Ollama runs the LLMs locally. It must be installed and running before you start the app.

```
brew install ollama
ollama pull mistral
ollama pull nomic-embed-text
ollama serve
```

Verify it's running at http://localhost:11434 or with:

```
ollama list
```

### 4. Run the ingestion pipeline
Ingestion is automated with scripts. Ingestion components are designed to be idempotent so re-running scripts will update existing data rather than creating duplicates.

The unified `ingest.py` script runs the entire ingestion pipeline:
- **Scrape phase:** Fetches source data from online sources, parses HTML, and saves documents as JSON files to target (at `data/raw/<document_id>/` in local env)
- **Embed phase:** Loads documents from storage, splits them into overlapping chunks with added metadata, embeds the chunks, and stores in a vector database (at `data/chroma_db/` in local env)
_If you only need to run a portion of the pipeline, see the Troubleshooting section for lower-level scripts._

```
# run everything - all parts of ingestion pipeline for all sources and all authors
uv run python scripts/ingest.py

# run all parts of ingestion pipeline for only one author
uv run python scripts/ingest.py --author voltaire
```

**Optional flags:**

| Flag                                                                                                           | Description | Default              |
|-----------------|-------------|----------------------|
| `--author` | Author key to process. | all authors included |
| `--raw-data-path` | Base directory for scraped documents | `data/raw`           |
| `--skip-scrape` | Skip scraping phase and use existing scraped documents | false                |
| `--skip-embed` | Skip embedding phase (only scrape documents)  | false                |
| `--verbose` | Debugging with verbose logging | false                |


## Usage: Debate with Enlightenment Philosophes
Luminary provides two interfaces for interacting with enlightenment personas. 
They both launch interactive chat sessions where you can ask questions and receive grounded responses from the selected writer.

### Example Questions
Ask Voltaire anything grounded in their writings. Here are some questions to get you started:

| English | Français |
| --- | --- |
| In your view, what is the nature of tolerance, and why is it essential for human society? | Selon vous, quelle est la nature de la tolérance, et pourquoi est‑elle essentielle à la vie en société ? |
| How do you reconcile the pursuit of personal happiness with the moral duty to others? | Comment conciliez‑vous la recherche du bonheur personnel avec le devoir moral que nous avons envers autrui ? |
| What is the proper role of the philosopher in challenging injustice? | Quel est, selon vous, le rôle propre du philosophe lorsqu'il s'agit de dénoncer et de combattre l'injustice ? |


### Prerequisites
- Ollama must be running.
- Both models must be pulled: nomic-embed-text (for embedding queries), mistral (for LLM responses).
- Vector database (chromaDB) must be populated by the ingestion script.

### Chat via CLI

```
# chat with default author (Voltaire as of 2026)
uv run python scripts/chat.py

# specify the author
uv run python scripts/chat.py --author voltaire
```

To exit: Type `quit` or press Ctrl+C.

**Optional flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `--author` | Author to query (e.g., `voltaire`) | `voltaire` |
| `--show-chunks` | Display retrieved chunks with IDs and contexts | `False` |
| `--verbose` | Enable verbose logging | `False` |

### Chat via UI

1. Ensure Streamlit is installed. It is an optional requirement that is not included in `uv sync`.

```
uv sync --extra ui
```

2. Launch the web interface for a more visual chat experience.

```
uv run streamlit run chat_ui.py
```

3. Open `http://localhost:8501` in your browser if it does not automatically open.
Use the sidebar to change the author persona or clear the conversation. 
_Changes of author will automatically rebuild the chat chain and clear message history._

## Development

### Run the test suite
```
# Run all tests except those tagged as 'external'. This includes unit tests, integration tests, and linting checks.
uv run pytest

# Run all tests including those tagged as 'external', which make network calls to confirm API contracts.
uv run pytest -m external
```

**Automatically clean up unused variables and imports**
```
uv run autoflake --in-place --recursive --remove-all-unused-imports --remove-unused-variables --ignore-init-module-imports src scripts tests
```

### Run the evaluation harness

See `src/eval/README.md` for detailed instructions and troubleshooting tips.

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
| --threshold   | Override threshold for ALL metrics (0.0-1.0) | Uses default_threshold from each metric |
| --verbose     | Enable debug logging                 | False                                               |


### Project Structure

```
luminary/
├── chat_ui.py               # web ui for chat
├── pyproject.toml           # project metadata, dependencies, and configuration
├── uv.lock                  # locked dependency versions
│
├── data/                    (gitignored)
│   ├── chroma_db/           # ChromaDB vector database
│   └── raw/
│       └── <document_id>/   # scraped documents saved as JSON
│
├── docs/
├── evals/                   (gitignored)
│   ├── golden/              # versioned golden datasets for the evaluation harness
│   └── runs/                # timestamped JSON artifacts from eval runs
│
├── locales/                 # standard user-facing messages that should adapt to detected language
│
├── src/
│   ├── chains/              # RAG chain orchestration with retrieval + LLM
│   ├── configs/             # configurations shared across modules
│   ├── document_loaders/    # fetch and parse data, returning standardised LangChain Documents
│   ├── eval/                # evaluation harness: metrics, runner, and quality measurement
│   ├── i18n/                # localization keys and message loading
│   ├── prompts/             # author-specific persona prompts (e.g., Voltaire, Gouges)
│   ├── schemas/             # shared Pydantic data models to validate data structures
│   ├── utils/               # shared utility functions
│   └── vectorstores/        # ingestion-time storage and query-time retrieval operations
│
├── scripts/                 # CLI entrypoints for ingestion, chat, eval, etc.
│
└── tests/
    ├── unit/                # fast offline tests; all external boundaries mocked
    └── integration/         # tests across modules and services*
```
_*Tests that make http or grpc calls to confirm API contracts are tagged with 'external' annotation._