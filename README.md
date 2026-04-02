# Luminary

A RAG chatbot where Enlightenment philosophers (e.g., Voltaire, Olympe de Gouges) answer questions grounded exclusively in their historical texts with sources cited. 
Available as both a web UI and interactive CLI.

## Interfaces

Luminary provides two interfaces for interacting with the philosophers:

- **Web UI**: Browser-based chat interface with message history, sidebar controls, and visual source citations
- **CLI**: Command-line interactive chat with optional debug output and chunk inspection

Both interfaces support the same core features and philosopher personas.

## Key Features

- **Grounded citations:** All responses include source references with page numbers in the text response. Additionally, a sources footer shows each unique source only once.
- **Local-first execution:** Runs entirely locally with Ollama—no external API calls, no data sharing, no usage costs.
- **Semantic search over historical corpus:** Uses vector similarity to find relevant passages across all texts, not just keyword matching.
- **Transparent retrieval:** Optional debug mode shows exact text chunks and similarity scores used for each response.
- **Multilingual chat:** Automatically detects the user's language and directs LLM to respond in the same language. _Local LLMs are not deterministic. So adherence to the detected language depends on the training data available to the selected model._ 
- **Localized interfaces:** Chat interface formatting (labels, buttons, loading messages) available in multiple supported languages and adapts to the detected language of the user.

### Example Questions
Ask Voltaire anything grounded in their writings. Here are some questions to get you started:

| English | Français |
| --- | --- |
| In your view, what is the nature of tolerance, and why is it essential for human society? | Selon vous, quelle est la nature de la tolérance, et pourquoi est‑elle essentielle à la vie en société ? |
| How do you reconcile the pursuit of personal happiness with the moral duty to others? | Comment conciliez‑vous la recherche du bonheur personnel avec le devoir moral que nous avons envers autrui ? |
| What is the proper role of the philosopher in challenging injustice? | Quel est, selon vous, le rôle propre du philosophe lorsqu'il s'agit de dénoncer et de combattre l'injustice ? |

## Technology

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

## Architecture

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

#### Pipelines in Detail

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

### Project Structure

```
luminary/
├── pyproject.toml           # project metadata, dependencies, and configuration
├── uv.lock                  # locked dependency versions
├── chat_ui.py               # web ui for chat
│
├── data/                    # (gitignored)
│   ├── chroma_db/           # ChromaDB vector database
│   └── raw/
│       ├── golden/          # versioned golden datasets for evaluation
│       └── <document_id>/   # scraped documents saved as JSON
│
├── locales/                 # user-facing messages that should adapt to detected language
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
├── docs/
└── tests/
    ├── unit/                # fast offline tests; all external boundaries mocked
    └── integration/         # tests across modules and services*
```
_*Tests that make http or grpc calls to confirm API contracts are tagged with 'external' annotation._

### Evaluation Harness

Luminary uses a bespoke, automated evaluation harness to measure quality and prevent regressions.

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
- They live in `data/raw/golden/` (gitignored) with naming convention: `{author}_golden_v{version}_{YYYY-MM-DD}.json`

**Eval Artifacts:**
- Eval runs are saved to `evals/runs/` (gitignored) as timestamped JSON files with filename format: `{author}_{YYYY-MM-DD}T{HH-MM-SS}.json`.
- Contains all example results, aggregate scores, and system version metadata for traceability

## Development

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

### 4. Run the test suite

```
# Run all tests except those tagged as 'external'
# This includes unit tests, integration tests, and linting checks (mypy, autoflake)
uv run pytest

# Run all tests including those tagged as 'external', which make network calls to confirm API contracts
uv run pytest -m external
```

#### Run linting checks independently

Run type checks
```
uv run mypy
```

Run checks on unused variables
```
uv run pytest tests/integration/test_autoflake.py

# clean up unused imports automatically
uv run autoflake --in-place --recursive --remove-all-unused-imports --remove-unused-variables --ignore-init-module-imports src scripts tests
```

### 5. Run the ingestion pipeline
Ingestion is automated with scripts. Ingestion components are designed to be idempotent so re-running scripts will update existing data rather than creating duplicates.

The unified `ingest.py` script runs the entire pipeline:
```
# run everything - all parts of ingestion pipeline (scrape + embed) for all sources and all authors
uv run python scripts/ingest.py
```
_If you only need to run a portion of the pipeline, see the Troubleshooting section for lower-level scripts._

**What it does:**
1. **Scrape phase:** Fetches source data from online sources, parses HTML, and saves documents as JSON files to target (at `data/raw/<document_id>/` in local env)
2. **Embed phase:** Loads documents from storage, splits them into overlapping chunks with added metadata, embeds the chunks, and stores in a vector database (at `data/chroma_db/` in local env)

**Options:**
- `--author` (optional): Author key to process. Defaults to all configured authors. Currently available: `voltaire`
- `--raw-data-path` (optional): Base directory for scraped documents (default: `data/raw`)
- `--skip-scrape` (optional): Skip scraping phase and use existing scraped documents
- `--skip-embed` (optional): Skip embedding phase (only scrape documents)

**Examples using options:**
```
# Run all parts of ingestion pipeline for ony one author
uv run python scripts/ingest.py --author voltaire

# Run ony the scraping portion (skip embedding)
uv run python scripts/ingest.py --skip-embed

# Run only the embedding portion (uses existing scraped documents)
uv run python scripts/ingest.py --skip-scrape
```

### 6. Start chatting with Enlightenment Philosophes

#### Prerequisites
- Ollama must be running.
- Both models must be pulled: nomic-embed-text (for embedding queries), mistral (for LLM responses).
- Vector database (chromaDB) must be populated by the ingestion script.


#### Chat via CLI
This script launches an interactive chat session where you can ask questions and receive grounded responses from the selected philosopher.

```
# chat with default author (Voltaire as of 2026)
uv run python scripts/chat.py

# specify the author
uv run python scripts/chat.py --author voltaire

# show raw retrieved chunks for debugging
uv run python scripts/chat.py --show-chunks
```

To exit: Type `quit` or press Ctrl+C.

**Available flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `--author` | Author to query (e.g., `voltaire`) | `voltaire` |
| `--show-chunks` | Display retrieved chunks with IDs and contexts | `False` |
| `--verbose` | Enable verbose logging | `False` |

#### Chat via UI

1. Ensure Streamlit is installed. It is an optional requirement that is not included in `uv sync`.

```
uv sync --extra ui
```

2. Launch the web interface for a more visual chat experience.

```
uv run streamlit run chat_ui.py
```

Open `http://localhost:8501` in your browser.
- Chat interface with message history
- Use the sidebar to change the author persona or clear the conversation.*
- Each response adheres to the personality and perspective's of the selected author.
- Responses display deduplicated sources as a caption.

_*Changes of author will automatically rebuild the chat chain and clear message history._

## Troubleshooting

### Running ingestion steps separately

If you need to run scraping and embedding as separate steps (for debugging or development), use the individual scripts:

**Script 1 of 2 — Scrape:** fetches data from designated 3rd party sources, parses the data, formulates LangChain documents, and persists json files

```
uv run python scripts/scrape_wikisource.py

# scrape only documents tagged with author: Voltaire
uv run python scripts/scrape_wikisource.py --author voltaire
```
_Output: data/raw/voltaire_lettres_philosophiques-1734/page_01.json, page_02.json, ..._

**Options:**
- `--author` (optional): Author key to scrape. Defaults to all configured authors. Currently available: `voltaire`
- `--output-path` (optional): Base directory for saving scraped documents (default: `data/raw`)

**Output location:**
Documents are saved to `data/raw/<document_id>/` as `page_NN.json` files containing:
- `page_content`: The extracted text content
- `metadata`: Document metadata including `document_id`, `document_title`, `author`, `source` URL, and `page_number`

**Script 2 of 2 — Embed and Store:** loads JSON files from disk that were persisted in Step 1,
splits each letter into overlapping chunks,
converts each chunk into a vector using Ollama nomic-embed-text (a small neural network that captures the meaning of text as a list of numbers),
and stores both the vectors and the original text in the vector database (at `data/chroma_db/` in local env).
Once stored, chunks can be retrieved by semantic similarity — the basis for RAG.

```
uv run python scripts/embed_and_store.py

# embed and store only documentes tagged with author: Voltaire
uv run python scripts/embed_and_store.py --author voltaire
```
**Options:**
- `--author` (optional): Author key to process. Defaults to all configured authors. Currently available: `voltaire`
- `--input-path` (optional): Base directory containing scraped documents (default: `data/raw`)

**Output location:**
Embeddings are stored in the ChromaDB vector database (at `data/chroma_db/` in local env) with collection name `philosophes`.