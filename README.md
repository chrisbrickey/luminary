# Luminary

A RAG chatbot where Enlightenment philosophers (e.g., Voltaire, Olympe de Gouges) answer questions grounded exclusively in their historical texts.

## Technology

| Dependency | Purpose                                |
| --- |----------------------------------------|
| Python 3.14 | language runtime                       |
| uv | package manager                        |
| LangChain | LLM orchestration framework            |
| ChromaDB | vector store for document embeddings   |
| Pydantic | data validation and schema definitions |
| pytest + pytest-cov | test runner and coverage               |
| mypy | static type checking                   |


## Project Structure

```
luminary/
├── pyproject.toml           # project metadata, dependencies, and configuration
├── uv.lock                  # locked dependency versions
├── docs/
│
├── data/                    # (gitignored)
│   └── raw/                 # scraped documents saved as JSON, organised by document_id
│
├── src/
│   ├── schemas.py           # shared Pydantic data models to validate data structures
│   │
│   ├── configs/             # configurations shared across modules
│   ├── document_loaders/    # fetch and parse data, returning standardised LangChain Documents
│   └── utils/               # shared utility functions
│
├── scripts/                 # CLI entrypoints for ingestion, chat, eval, etc.
│
└── tests/
    ├── unit/                # fast offline tests; all external boundaries mocked
    └── integration/         # tests across modules and services*
```
_*Tests that make http or grpc calls to confirm API contracts are tagged with 'external' annotation._

## Development

### 1. Clone the repo

```
git clone <repo-url>
cd luminary
```

### 2. Install dependencies

```
uv sync
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
uv run pytest

# Run all tests including those tagged as 'external', which make network calls to confirm API contracts
uv run pytest -m external

# Run type checker only
uv run mypy
```

### 5. Run the ingestion pipeline
Initially, this will be controlled by two, sequential scripts to be run in order. Each is idempotent and will overwrite previous output.
Later, I will endeavor to combine these into one script.

**Script 1 of 2 — Scrape:** fetches data from designated 3rd party sources, parses the data, formulates LangChain documents, and persists json files

```
uv run python scripts/scrape_wikisource.py
```

**Options:**
- `--author` (optional): Author key to scrape. Defaults to all configured authors. Currently available: `voltaire`
- `--output-dir` (optional): Base directory for saving scraped documents (default: `data/raw`)

**Output location:**
Documents are saved to `data/raw/<document_id>/` as `page_NN.json` files containing:
- `page_content`: The extracted text content
- `metadata`: Document metadata including `document_id`, `document_title`, `author`, `source` URL, and `page_number`

**Example using options:**
```
# scrape only documents tagged with author: Voltaire
uv run python scripts/scrape_wikisource.py --author voltaire
```
_Output: data/raw/voltaire_lettres_philosophiques-1734/page_01.json, page_02.json, ..._

**Script 2 of 2 — Embed:** (placeholder)
