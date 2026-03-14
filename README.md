# Luminary

A RAG chatbot where Enlightenment philosophers (e.g., Voltaire, Olympe de Gouges) answer questions grounded exclusively in their historical texts with sources cited.

## Chat Features

- **Grounded citations:** All responses include source references with page numbers.
- **Deduplicated sources:** The sources footer shows each unique source only once.
- **Bilingual support and language detection:** The app automatically detects the language of your question (French/English) and responds in the same language. English responses include translations of cited French passages.

### Example Questions
Ask Voltaire anything grounded in his writings. Here are some questions to get you started:

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
 retriever.py            embeds the prompt, performs similarity search on ChromaDB, retrieves top-k semantically similar chunks from vector db
      │
      ▼
 chat_chain.py           orchestrates retrieval, context formatting with labels, and LLM call including persona prompt; returns ChatResponse
      │
      ▼
 ChatResponse            validated and structured response
```

### Project Structure

```
luminary/
├── pyproject.toml           # project metadata, dependencies, and configuration
├── uv.lock                  # locked dependency versions
├── chat_ui.py               # web ui for chat
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
├── docs/
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

### 6. Start chatting with Enlightenment Philosophes

#### Prerequisites
- Ollama must be running.
- Both models must be pulled: nomic-embed-text (for embedding queries), mistral (for LLM responses).
- ChromaDB must be populated by the ingestion script.


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
| `--db` | Path to ChromaDB directory | `data/chroma_db` |
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
- Use the sidebar to change the author persona or database location.*
- Each response adheres to the personality and perspective's of the selected author.
- Responses display deduplicated sources as a caption.

_*Changes to the database path or philosopher will automatically rebuild the chat chain and clear message history._