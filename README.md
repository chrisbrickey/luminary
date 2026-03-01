# Luminary

A RAG chatbot where Enlightenment philosophers (e.g., Voltaire, Olympe de Gouges) answer questions grounded exclusively in their historical texts.

## Technology

| Dependency | Purpose |
| --- | --- |
| Python 3.14 | Language runtime |
| uv | Package manager |
| LangChain | LLM orchestration framework |
| LangChain-Ollama | Ollama LLM and embeddings integration |
| LangChain-Community | Community integrations (document loaders, etc.) |
| LangChain-Chroma | ChromaDB vector store integration |
| ChromaDB | Vector store for document embeddings |
| Pydantic | Data validation and schema definitions |
| pytest + pytest-cov | Test runner and coverage |
| mypy | Static type checking |


## Project Structure

```
luminary/
├── pyproject.toml           # project metadata, dependencies, and configuration
├── uv.lock                  # locked dependency versions
├── docs/
│ 
├── src/
│   └── utils/               # shared functions 
│ 
└──tests/
```

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