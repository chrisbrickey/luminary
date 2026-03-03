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
│
├── docs/
│
├── src/
│   ├── schemas.py           # shared Pydantic data models to validate data structures
│   │
│   ├── configs/             # configurations shared across modules
│   ├── document_loaders/    # fetch and parse data, returning standardised LangChain Documents
│   └── utils/               # shared utility functions
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
uv run mypy src
```