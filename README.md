# Luminary

A RAG chatbot where Enlightenment philosophers (e.g., Voltaire, Olympe de Gouges) answer questions grounded exclusively in their historical texts.

## Technology

| Dependency | Purpose                                |
| --- |----------------------------------------|
| Python 3.13 | language runtime                       |
| uv | package manager                        |
| LangChain | LLM orchestration framework            |
| ChromaDB | vector store for document embeddings   |
| Pydantic | data validation and schema definitions |
| pytest + pytest-cov | test runner and coverage               |
| mypy | static type checking                   |

## Architecture

### Overview

Luminary will be organized into four pipelines. The ingestion and query pipelines are complete. 
Evaluation harness and multi-agent debate are under development.

```
I. INGESTION PIPELINE
──────────────────────────────────────────────────────────────
  load    →    persist   →   chunk    →    embed  →  vector DB
(web scrape)  (JSON files) (overlapping)  (vectors)  (persist)


II. QUERY PIPELINE
──────────────────────────────────────────────────────────────
user question → retriever → format context → prompt LLM with agent personas → respond
```

## Pipelines in Detail

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
  raw string             prompt from user in their natural language
      │
      ▼
 retriever.py            embeds prompt, performs similarity search on ChromaDB, retrieves top-k chunks
      │
      ▼
 chat_chain.py           formats retrieved chunks, populates the prompt template, calls an LLM, returns structured response
      │
      ▼
 ChatResponse            validated and structured response
```

### Project Structure

```
luminary/
├── pyproject.toml           # project metadata, dependencies, and configuration
├── uv.lock                  # locked dependency versions
├── docs/
│
├── data/                    # (gitignored)
│   ├── chroma_db/           # ChromaDB vector store
│   └── raw/                 # scraped documents saved as JSON, organised by document_id
│
├── src/
│   ├── schemas.py           # shared Pydantic data models to validate data structures
│   │
│   ├── configs/             # configurations shared across modules
│   ├── chains/              # RAG chain orchestration with retrieval + LLM
│   ├── document_loaders/    # fetch and parse data, returning standardised LangChain Documents
│   ├── prompts/             # author-specific persona prompts (e.g., Voltaire, Gouges)
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
Ingestion is automated with scripts. Ingestion components are designed to be idempotent so re-running scripts will update existing data rather than creating duplicates.

The unified `ingest.py` script runs the entire pipeline:
```
# run everything - all parts of ingestion pipeline (scrape + embed) for all sources and all authors
uv run python scripts/ingest.py
```
_If you only need to run a portion of the pipeline, see the next section for lower-level scripts._

**What it does:**
1. **Scrape phase:** Fetches source data from online sources, parses HTML, and saves documents as JSON files to `data/raw/<document_id>/`
2. **Embed phase:** Loads documents from storage, splits them into overlapping chunks with added metadata, embeds the chunks, and stores in a vector database at `data/chroma_db/`

**Options:**
- `--author` (optional): Author key to process. Defaults to all configured authors. Currently available: `voltaire`
- `--raw-dir` (optional): Base directory for scraped documents (default: `data/raw`)
- `--db` (optional): ChromaDB persist directory (default: `data/chroma_db`)
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
- `--output-dir` (optional): Base directory for saving scraped documents (default: `data/raw`)

**Output location:**
Documents are saved to `data/raw/<document_id>/` as `page_NN.json` files containing:
- `page_content`: The extracted text content
- `metadata`: Document metadata including `document_id`, `document_title`, `author`, `source` URL, and `page_number`

**Script 2 of 2 — Embed and Store:** loads JSON files from disk that were persisted in Step 1,
splits each letter into overlapping chunks,
converts each chunk into a vector using Ollama nomic-embed-text (a small neural network that captures the meaning of text as a list of numbers),
and stores both the vectors and the original text in ChromaDB at data/chroma_db/.
Once stored, chunks can be retrieved by semantic similarity — the basis for RAG.

```
uv run python scripts/embed_and_store.py

# embed and store only documentes tagged with author: Voltaire
uv run python scripts/embed_and_store.py --author voltaire
```
**Options:**
- `--author` (optional): Author key to process. Defaults to all configured authors. Currently available: `voltaire`
- `--input-dir` (optional): Base directory containing scraped documents (default: `data/raw`)
- `--db` (optional): ChromaDB persist directory (default: `data/chroma_db`)

**Output location:**
Embeddings are stored in the ChromaDB vector database at `data/chroma_db/` with collection name "philosophes".
