# Plan: Luminary — Enlightenment Philosopher RAG Chatbot

## Context
Luminary is a RAG chatbot where Enlightenment philosophers (e.g., Voltaire, Olympe de Gouges) answer questions grounded in their historical texts. 
It will feature an ingestion pipeline (initially integrated with Wikisource), retrieval-augmented chat with per-philosopher persona prompts, bilingual support (French/English), a multi-philosopher debate mode with LangChain agents, a deterministic + LLM-as-judge evaluation harness, and deployment to Heroku with configurable LLM/embedding providers. 
This plan covers the complete system from project scaffolding through production deployment. This plan is intended to be read by coding agents. It is intentionally not optimized for human readers.

## Hosting target: Heroku

All design decisions should keep Heroku deployment in mind:

1. **ChromaDB storage** — Heroku has an ephemeral filesystem. Database path configured via `CHROMA_DB_PATH` env var (implemented 2026-03-19). Step 17 introduces `ChromaDBConfig` with `EMBEDDED` (local dev) and `SERVER` (production) modes via env vars.
2. **LLM + embeddings** — Heroku cannot run Ollama. Step 18 adds provider abstraction: `LLM_PROVIDER` and `EMBEDDING_PROVIDER` env vars to swap between Ollama (local) and hosted APIs (Anthropic, OpenAI).
3. **Port binding** — `Procfile` binds Streamlit web UI to `$PORT`.

**Design rule:** prefer injectable dependencies (LLM, embeddings, retriever) over hardcoded defaults so local and hosted implementations can be swapped via configuration without touching business logic.

## Python version downgrade

**Note:** The project was initially scaffolded with Python 3.14 (Step 1), but was downgraded to Python 3.13 between Steps 4 and 5 due to a ChromaDB compatibility issue. ChromaDB's internal Pydantic V1 functionality is not compatible with Python 3.14, causing the error: "Core Pydantic V1 functionality isn't compatible with Python 3.14". All references to Python 3.14 in Step 1 are historical; the project now uses Python 3.13.

## Database path configuration (2026-03-19)

**Change:** Removed user-facing database path overrides from all scripts and UI. Database location is now configured solely via environment variable.

- **Before:** Scripts had `--db` CLI flags, UI had "Database Path" input, functions accepted `persist_dir` parameters
- **After:** All removed. `DEFAULT_DB_PATH` in `src/configs/common.py` reads from `CHROMA_DB_PATH` environment variable (defaults to "data/chroma_db")
- **Rationale:** Simplifies API by treating database location as infrastructure concern rather than user-facing parameter. Don't anticipate needing more than one vector database per environment or sharding.
- **Impact:** Affects Steps 5–10, 15–17; see "DB path override removal (2026-03-19)" deviation notes in affected steps

## Prerequisites (manual)
- Install Ollama: https://ollama.ai
- Pull the LLM model: `ollama pull mistral`
- Pull the embedding model: `ollama pull nomic-embed-text`
- Start Ollama: `ollama serve` (scripts validate this at startup via `ollama_health.py`)

## Test fixture convention
Test fixtures use **broad Enlightenment topics** (e.g., "la tolérance religieuse", "la liberté de conscience", "les Lumières", "la raison", "les droits naturels") rather than names of specific religious groups or political figures. This keeps content thematically appropriate without anchoring tests to any single letter or tract.

## Default test commands
- Unit tests (fast, no network): `uv run pytest`
- External tests (real HTTP/gRPC): `uv run pytest -m external`
- Type checker: `uv run mypy src scripts`

---

## ✅ Step 1: Project scaffolding
- `uv init --python 3.14` to specify Python 3.14
- Create only the structure needed for this step's deliverables:
  ```
  src/__init__.py
  src/utils/__init__.py       # utilities package
  tests/unit/utils/
  tests/integration/
  ```
  Subsequent steps add their own directories when first needed.
- Add dependencies: `langchain`, `langchain-ollama`, `langchain-community`, `langchain-chroma`, `chromadb`, `pydantic`
- Add dev dependencies: `pytest`, `pytest-cov`, `mypy`, `types-requests`
- Configure `pyproject.toml`:
  ```
  [tool.pytest.ini_options]
  addopts = "-m 'not external'"
  markers = [
      "external: marks tests that make real network/service calls (deselect with -m 'not external')",
      "integration: marks integration tests that wire multiple internal modules",
  ]

  [tool.mypy]
  python_version = "3.14"
  warn_return_any = true
  warn_unused_configs = true
  disallow_untyped_defs = true
  check_untyped_defs = true
  mypy_path = "src"
  packages = ["src"]
  ```
- Update `.gitignore`: `__pycache__/`, `.venv/`, `.idea/`
- Create `src/utils/ollama_health.py`: `check_ollama_available(base_url="http://localhost:11434") -> None` — sends GET to `/api/tags`; raises `RuntimeError("Ollama is not running. Start it with: ollama serve")` on connection failure; called at the top of all CLI scripts that need Ollama
- **Test:** `tests/unit/utils/test_ollama_health.py` — mock `urllib.request.urlopen`: assert raises `RuntimeError` with expected message on `ConnectionRefusedError`; assert returns `None` on normal response
- **Test:** `tests/integration/test_mypy.py` — run `mypy src scripts` via subprocess; assert exit code 0
- **Verify:** `uv run python -c "import langchain; print('ok')"`
- **README:** Add Technology table (all initial dependencies + purposes); add Setup section (clone, `uv sync`, Ollama install + model pulls); add project structure diagram
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.

## ✅ Step 2: Pydantic schemas
- Create `src/schemas.py` with:
  - `ChunkInfo`: `chunk_id` (SHA256 of `document_id:chunk_index`, truncated to 12 chars), `document_id`, `document_title: str | None = None`, `author: str | None = None` (lowercase), `chunk_index`, `source` (+ `extra="allow"` for domain-specific fields)
  - `ChatResponse`: `text`, `retrieved_passage_ids: list[str]`, `retrieved_contexts: list[str]`, `retrieved_source_titles: list[str]`, `language` (ISO 639-1: `^[a-z]{2}$`)
- **Test:** `tests/unit/test_schemas.py` — validate construction, required fields, type enforcement, extra fields on ChunkInfo, language regex validation, page_number optional field
- **README:** No user-facing command; update project structure diagram to add `src/schemas.py`
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.

## ✅ Step 3: Corpus ingestion — loader
- Create new directories with `__init__.py`: `src/configs/`, `src/document_loaders/`; create `scripts/`
- Update `pyproject.toml` mypy config: change `packages = ["src"]` to `packages = ["src", "scripts"]`
- Update `tests/integration/test_mypy.py`: add `"scripts"` to the mypy command
- Update `.gitignore`: add `data/raw/`
- Create `src/configs/loader_configs.py`:
  - `DEFAULT_DB_PATH = Path("data/chroma_db")` — simplified to relative path (not resolved)
  - `WikisourceCollectionConfig` (Pydantic): `document_id`, `document_title`, `author` (lowercase), `page_title_template` (with `{n}` placeholder), `total_pages: int | None` (auto-discovered if None), `api_url` (default: `https://fr.wikisource.org/w/api.php`)
  - `LETTRES_PHILOSOPHIQUES_CONFIG`: pre-built config for Voltaire's Lettres philosophiques (renamed from VOLTAIRE_LETTRES_CONFIG)
  - `INGEST_CONFIGS: dict[str, WikisourceCollectionConfig]` — registry mapping author key → config; initially `{"voltaire": LETTRES_PHILOSOPHIQUES_CONFIG}`
- Create `src/document_loaders/wikisource_loader.py`:
  - `WikisourceLoader`: class-based loader with `load() -> list[Document]`; fetches HTML via Wikisource API; parses with `_WikisourceHTMLExtractor` (HTMLParser subclass, skip-depth tracking to strip ws-noexport, reference, reflist, script, style)
  - Metadata per Document: `document_id`, `document_title`, `author`, `source` (canonical URL), `page_number`
  - Polite delay (`delay: float = 1.0`) between requests
  - **Retry with exponential backoff** (max 3 retries, base 2s) for transient HTTP errors (429, 5xx, connection timeouts) — improvement over rag-chat-1 which had no retry logic
- Create `src/utils/io.py`:
  - `save_documents_to_disk(docs, directory) -> list[Path]`: saves each Document as `page_NN.json`
  - ~~`load_documents_from_disk(directory) -> list[Document]`~~ — deferred to Step 7 (not needed until combined ingestion script)
- Create `scripts/scrape_wikisource.py` — CLI entrypoint: loads config, calls loader, saves to `data/raw/<document_id>/`; does NOT call `check_ollama_available()` (no Ollama dependency for scraping)
- **Test:** `tests/unit/test_script_scrape_wikisource.py` — mock all external dependencies; test successful scraping, invalid author, no documents loaded, loader exception, save exception, custom output directory, output path construction, default scraping all authors (9 tests)
- **Test:** `tests/unit/test_loader_configs.py` — validate LETTRES_PHILOSOPHIQUES_CONFIG fields, `INGEST_CONFIGS` has `"voltaire"` key, DEFAULT_DB_PATH correctness, custom API URL (6 tests)
- **Test:** `tests/unit/document_loaders/test_wikisource_loader.py` — comprehensive tests for HTML parser and loader with mocked HTTP calls; test retry on 429/5xx, network errors, max retries, non-transient errors, auto-discovery, empty responses, metadata construction (15 tests for parser and loader combined)
- **Test:** `tests/unit/utils/test_io.py` — test save operations: directory creation, error handling, Unicode preservation, padding (6 tests for save only; load tests deferred to Step 7)
- **README:** Added Usage > Ingestion section with `scripts/scrape_wikisource.py` command, options table, output location details, and example; updated project structure diagram to add `scripts/` and clarify directories
- **Deviations from plan:**
  - `DEFAULT_DB_PATH` simplified to relative path `Path("data/chroma_db")` instead of absolute resolved path; later moved to `src/configs/common.py` in 2026-03-08 refactoring (see Step 6 deviations)
  - Config name changed from `VOLTAIRE_LETTRES_CONFIG` to `LETTRES_PHILOSOPHIQUES_CONFIG`
  - `--author` flag changed from required to optional (defaults to all configured authors)
  - Enhanced logging with user-friendly progress messages and simplified format to show real-time scraping progress
  - `load_documents_from_disk()` and associated tests deferred to Step 7 (not needed until combined ingestion script)
  - Test organization: created `tests/unit/document_loaders/conftest.py` with shared fixtures and helpers following DRY principles
  - Added integration test `tests/integration/document_loaders/test_wikisource_loader_integration.py` for external API contract testing
  - Refactored `scripts/scrape_wikisource.py` with `scrape_author()` helper function for cleaner code organization

## ✅ Step 4: Corpus ingestion — chunking
- Create `src/utils/chunker.py`
- Use LangChain `RecursiveCharacterTextSplitter` (chunk_size=1200 chars, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""] for French prose)
- Attach chunk metadata: `chunk_id` (SHA256 of `document_id:chunk_index`[:12]), `chunk_index` (0-indexed per document); preserve all source metadata
- **Validate chunks via ChunkInfo** — construct a `ChunkInfo` from each chunk's metadata before returning, ensuring schema compliance at chunking time (improvement over rag-chat-1 where ChunkInfo was defined but never enforced)
- **Test:** `tests/unit/utils/test_chunker.py` — given a known Document, assert chunks are within size bounds, metadata is correct, no text is lost, ChunkInfo validation passes; test empty document edge case
- **README:** No user-facing command; update pipeline diagram to show chunking stage
- **Deviations from plan:**
  - Added 14 comprehensive tests covering all edge cases including deterministic chunk ID generation, multi-document processing, custom parameters, and validation failures
  - Added pipeline stages overview to README showing Scrape → Chunk → Embed flow
  - Used `langchain_text_splitters` import instead of `langchain.text_splitter`

## ✅ Step 5: Embedding + ChromaDB storage
- Create new directories: `src/vectorstores/` with `__init__.py`; `tests/unit/vectorstores/`
- Update `.gitignore`: add `data/chroma_db/`
- Create `tests/conftest.py`: `FakeEmbeddings(Embeddings)` fixture returning constant 8-dim vectors; disable ChromaDB telemetry via `os.environ.setdefault` before import
- Create `src/vectorstores/chroma.py` (renamed from `embed_and_store.py` for better module naming)
- `embed_and_store(chunks, collection_name="philosophes", embeddings=None) -> Chroma`:
  - Uses `DEFAULT_DB_PATH` from config (no user-facing persist_dir parameter)
  - Uses `Chroma.from_documents()` with **`ids=` set to each chunk's `chunk_id`** — idempotent upserts on re-run (improvement over rag-chat-1 which omitted `ids=`, causing duplicates)
  - Accepts injectable embeddings for testing; defaults to `OllamaEmbeddings(model="nomic-embed-text")`
  - Refactored with private helper functions `_get_embeddings_instance()` and `_extract_and_validate_chunk_ids()` following SRP
- Create `scripts/embed_and_store.py` — CLI entrypoint: calls `check_ollama_available()`, loads chunks from disk, embeds and stores (no `--db` flag)
- **Test:** `tests/unit/vectorstores/test_chroma.py` (renamed from `test_embed_and_store.py`) — ingest 2-3 fixture chunks with FakeEmbeddings; verify retrievable by similarity search; verify re-running with same IDs does not create duplicates
- **README:** Add ingestion step 2 (`scripts/embed_and_store.py`) with command, output location (`data/chroma_db/` or `$CHROMA_DB_PATH`); update project structure diagram to add `src/vectorstores/`, `data/chroma_db/`; update pipeline diagram to reference `chroma.py`
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.
- **Deviations from plan:**
  - Implemented with Python 3.13 instead of 3.14 (see "Python version downgrade" section above)
  - Module renamed from `embed_and_store.py` to `chroma.py` to eliminate module/function name redundancy and allow room for future ChromaDB utilities
  - Refactored `embed_and_store()` function with private helper functions `_get_embeddings_instance()` and `_extract_and_validate_chunk_ids()` for better separation of concerns
  - Test file renamed from `test_embed_and_store.py` to `test_chroma.py` to match module name
  - Script tests moved to `tests/unit/test_scripts/` directory for better organization
  - Added `load_documents_from_disk()` function to `src/utils/io.py` (originally planned for Step 7) with comprehensive tests for load operations
  - Made `--author` flag optional in `scripts/embed_and_store.py` (defaults to all authors), consistent with `scripts/scrape_wikisource.py`
  - Test suite includes 6 comprehensive tests for vectorstore module + 12 tests for script + 7 tests for I/O operations (vs. suggested "2-3 fixture chunks"): basic functionality, idempotent upserts, custom collection name, missing chunk_id validation, complete metadata preservation, and all CLI argument combinations
  - Fixed chunk ID generation to include `page_number` in hash (`document_id:page_number:chunk_index`) to prevent duplicate IDs across pages within the same document
  - Added ChromaDB telemetry disable and Python 3.14 compatibility workaround to `tests/conftest.py` (the workaround is no longer needed with Python 3.13 but kept for documentation)
  - **Configuration refactoring (2026-03-08):** `chroma.py` and `scripts/embed_and_store.py` updated to use shared constants from `src/configs/vectorstore_config.py` (see Step 6 deviations for full details)
  - **DB path override removal (2026-03-19):** Removed `persist_dir` parameter from `embed_and_store()` function and `--db` flag from script. Database location now configured solely via `CHROMA_DB_PATH` environment variable (defaults to "data/chroma_db"). Simplifies API by treating database location as infrastructure concern rather than user-facing parameter.

## ✅ Step 6: Retrieval chain
- Create `src/vectorstores/retriever.py`
- `build_retriever(collection_name="philosophes", embeddings=None, k=5, author=None) -> VectorStoreRetriever`
  - Uses `DEFAULT_DB_PATH` from config (no user-facing persist_dir parameter)
- Wrap ChromaDB as LangChain retriever; apply author filter at retriever level via `search_kwargs={"filter": {"author": author}}` (ChromaDB metadata filter, not post-retrieval)
- **Test:** `tests/unit/vectorstores/test_retriever.py` — with small fixture DB, query and assert relevant chunks returned with intact metadata; test author filter; test without filter returns all
- **Test:** `tests/integration/test_retriever.py` — wire real ChromaDB + FakeEmbeddings end-to-end; embed fixtures then retrieve; verify round-trip
- **README:** No user-facing command; update pipeline diagram to show retrieval stage
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.
- **Deviations from plan:**
  - Added 7 comprehensive unit tests covering: basic retrieval, author filtering for different authors, retrieval without filter, k parameter limiting, custom collection name
  - Added 3 integration tests covering: full end-to-end round-trip with metadata preservation, multi-author filtering across full pipeline, and persistence across separate sessions
  - All tests pass; total of 10 tests vs. suggested "with small fixture DB" approach
  - Type checking passes via mypy integration test
  - **Configuration refactoring (2026-03-08):** Extracted shared constants to improve maintainability and observability:
    - Created `src/configs/common.py` with `DEFAULT_DB_PATH` constant (moved from `loader_configs.py`)
    - Created `src/configs/vectorstore_config.py` with `COLLECTION_NAME`, `EMBEDDING_MODEL`, and `DEFAULT_K` constants
    - Updated `retriever.py` to use constants as defaults and added logging before opening ChromaDB collection (logger.info with collection name, path, k, and author filter)
    - Updated `chroma.py` to use shared constants instead of hardcoded literals
    - Updated `scripts/embed_and_store.py` to import from new config modules
    - Added comprehensive test coverage: `tests/unit/test_config_common.py` and `tests/unit/test_config_vectorstore.py`
    - Updated existing tests to import constants from new locations
    - Benefits: DRY principle (constants defined once), improved observability (logging), easier maintenance (change values in one place)
  - **Chunk ID isolation (2026-03-11):** Added integration test `test_rag_chain_does_not_expose_chunk_ids_to_llm` in `tests/integration/test_rag_pipeline.py` to verify chunk IDs are preserved in metadata but never sent to the LLM in the full pipeline. Integration test count increased from 5 to 6; all 184 tests pass.
  - **DB path override removal (2026-03-19):** Removed `persist_dir` parameter from `build_retriever()` function. Database location now configured solely via `CHROMA_DB_PATH` environment variable (defaults to "data/chroma_db"). Simplifies API by treating database location as infrastructure concern.

## ✅ Step 7: Combined ingestion script
- **Goal:** Create a single `scripts/ingest.py` that can be used instead of running scrape + embed scripts separately
- Keep the two original scripts and their tests so that the two parts of ingestion can be run separately as lower level alternatives
- Create `scripts/ingest.py`: `ingest_author(author, raw_dir, skip_scrape, skip_embed) -> None` function + `main()` with argparse
  - Flags: `--author` (optional, defaults to all registered in `INGEST_CONFIGS`), `--skip-scrape`, `--skip-embed`, `--raw-dir` (no `--db` flag)
  - Calls existing `scrape_wikisource.py` and `embed_and_store.py` scripts via `subprocess.run()` (maintains script separation)
  - Calls `check_ollama_available()` unless `--skip-embed` (only embedding needs Ollama)
- **Test:** `tests/unit/test_scripts/test_script_ingest.py` — mock all external deps; test default (scrape+embed), skip-scrape, skip-embed, all-authors, single-author, invalid-author, both-skip-flags error, custom directories, ollama availability checks, file-not-found errors, general exceptions (17 tests total covering main() and ingest_author())
- **README:** Moved two-step ingestion instructions to new ## Troubleshooting section at bottom. In section 5 (Run the ingestion pipeline), added unified `scripts/ingest.py` as primary command with comprehensive documentation of all flags and usage examples
- **Deviations from plan:**
  - `ingest_author()` signature uses positional parameters `author, raw_dir, skip_scrape, skip_embed` (no `db_dir` parameter) — directly takes author string and looks up config internally for cleaner interface
  - Added `--raw-dir` flag for consistency with individual scripts (not in original plan)
  - **Implementation approach:** Uses `subprocess.run()` to call the existing `scrape_wikisource.py` and `embed_and_store.py` scripts rather than importing their functions — this maintains complete separation between scripts and avoids the need to make scripts a Python package
  - `ingest_author()` returns `None` instead of `tuple[int, int]` since subprocess calls don't return counts
  - Final summary simplified (no database location shown)
  - Added validation to prevent both `--skip-scrape` and `--skip-embed` flags being used together
  - Added `subprocess.CalledProcessError` handling for script execution failures
  - Added 17 comprehensive tests vs. suggested 6 test scenarios, covering both main() and ingest_author() functions — tests mock `subprocess.run()` and verify correct command arguments
  - Test organization follows existing pattern in `tests/unit/test_scripts/`
  - **DB path override removal (2026-03-19):** Removed `db_dir` parameter from `ingest_author()` function signature and `--db` flag from script. Database location now configured solely via `CHROMA_DB_PATH` environment variable.

## ✅ Step 8: Voltaire prompt + chat chain
- Create new directories with `__init__.py`: `src/prompts/`, `src/chains/`; create `tests/unit/chains/`
- Create `src/prompts/voltaire.py` — Voltaire system prompt (irony/wit, mandatory inline citations, responds in `{language}`); export `build_voltaire_prompt() -> ChatPromptTemplate`
  - **Bilingual from the start** — prompt includes `{language}` placeholder (improvement over rag-chat-1 which had to retrofit this later)
- Create `src/chains/chat_chain.py`:
  - `DEFAULT_AUTHOR = "voltaire"` — canonical author key
  - `_AUTHOR_CONFIGS: dict[str, AuthorConfig]` — extensible registry (AuthorConfig is a dataclass with `prompt_factory` and `exit_message` fields)
  - `build_chain(author=DEFAULT_AUTHOR, retriever=None, llm=None, prompt=None) -> Runnable[dict, ChatResponse]` — unified API supporting both production use (with defaults) and testing use (with mocks); uses `DEFAULT_DB_PATH` from config (no user-facing persist_dir parameter); chain accepts dict input `{"question": str, "language": str}` via `.invoke()`
  - Internal `_run(question, language) -> ChatResponse`: retrieves top-k docs, formats context with `[source: {title}, page {page_number}]` labels (page-specific from the start), invokes LLM with language parameter, returns structured ChatResponse
  - Helpers: `_format_docs_with_titles()`, `_extract_chunk_ids()`, `_extract_source_titles()` (combines document_title + page_number; fallback: title-only → source URL → "unknown")
- **Test:** `tests/unit/chains/test_chat_chain.py` — mock LLM and retriever; assert chain wires retrieval + prompt correctly; test `_extract_source_titles()` with title+page, title-only, fallback to source URL, missing metadata; test empty retrieval; test unknown author raises ValueError; test language detection integration
- **Test:** `tests/integration/test_chat_chain_integration.py` — wire small fixture ChromaDB (FakeEmbeddings) → real retriever → real prompt → FakeChatModel; assert full chain returns ChatResponse with correct retrieved_source_titles and non-empty text
- **README:** Add Architecture Overview section with full pipeline diagram (Ingestion, Query, Debate, Eval); update project structure diagram to add `src/prompts/`, `src/chains/`
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.
- **Deviations from plan:**
  - Created `FakeChatModel` in integration tests (similar to `FakeEmbeddings`) for testing without real LLM calls
  - Language detection support added to chain API — chain accepts `language` parameter via `.invoke({"question": str, "language": str})`, but actual detection logic implemented later in Step 12 at call site (CLI/UI), not inside chain
  - Comprehensive test coverage: 21 unit tests covering all helper functions, chain wiring, error cases, and author registry; 2 integration tests covering full end-to-end pipeline with real ChromaDB and retriever
  - All tests pass; mypy type checking passes
  - **Post-implementation refactoring:**
    - Moved `DEFAULT_AUTHOR = "voltaire"` from `chat_chain.py` to `src/configs/common.py` for consistency with Step 6 config pattern; added 2 tests; all future CLI scripts (Steps 9, 15) will import this constant for their argparse defaults
    - **Unified API:** Merged `build_default_chain()` and `build_chat_chain()` into single `build_chain()` method. This method takes all parameters with sensible defaults, supporting both production use (`build_chain()` or `build_chain(author="gouges")`) and testing use (`build_chain(retriever=mock, llm=mock, prompt=mock)`). Eliminates API confusion and reduces code duplication. Updated all 25 test call sites. Added 1 new test for injection mode. All 154 tests pass. Single public API is clearer and more maintainable.
    - **Chunk ID isolation (2026-03-11):** Removed chunk IDs from LLM-facing context and prompt instructions. Chunk IDs are internal metadata for debugging (visible only with `--show-chunks` flag) and should never be sent to the LLM or appear in responses. Changes: (1) Updated `src/prompts/voltaire.py` to remove `| chunk_id: xxx` from citation format instruction, (2) Updated `src/chains/chat_chain.py::_format_docs_with_titles()` to format context as `[source: {title}, page {page}]` without chunk IDs, (3) Added unit test `test_prompt_does_not_instruct_chunk_id_citations` to verify prompt doesn't instruct chunk ID citations, (4) Updated existing `test_context_formatting` to assert chunk IDs are NOT in LLM input, (5) Added integration test `test_rag_chain_does_not_expose_chunk_ids_to_llm` to verify end-to-end pipeline isolation. All 184 tests pass.
    - **Prompt refinement (2026-03-15, commit e372784):** Strengthened LLM prompting for better instruction clarity and citation control:
      1. Rewrote Voltaire prompt from French to English — LLMs follow English instructions more reliably
      2. Enhanced prompt with explicit formatting examples showing correct vs. incorrect citation placement (citations must appear at end of sentences, never at beginning of paragraphs)
      3. Created dedicated test file `tests/unit/prompts/test_voltaire.py` with comprehensive prompt validation tests
      4. Refactored and reorganized tests in `tests/unit/chains/test_chat_chain.py` for better clarity and maintainability
    - **Personalized exit messages (2026-03-15, commit 8d9cc29):** Extended author configuration to include UI-related metadata:
      1. Changed `AuthorConfig` from tuple `(prompt_factory, language)` to frozen dataclass with fields: `prompt_factory: Callable[[], ChatPromptTemplate]` and `exit_message: str`
      2. Added personalized exit messages displayed when user exits interaction (e.g., "Je vous embrasse - V" for Voltaire)
      3. CLI exit message display: shown when user exits via `quit`, Ctrl+C, or EOFError (3 additional tests in `test_script_chat.py`)
      4. Web UI exit message display: shown via toast notification when user clicks "Clear conversation" button (2 additional tests in `test_chat_ui.py`)
      5. This change establishes pattern for extending author config with non-LLM metadata (UI strings, personality traits, etc.) while keeping prompt logic separate
    - **DB path override removal (2026-03-19):** Removed `persist_dir` parameter from `build_chain()` function. Database location now configured solely via `CHROMA_DB_PATH` environment variable (defaults to "data/chroma_db"). Simplifies API by treating database location as infrastructure concern.

## ✅ Step 9: Chat CLI script
- Create `scripts/chat.py` — interactive CLI chat loop; flags: `--author`, `--show-chunks`, `--verbose` (no `--db` flag); calls `check_ollama_available()`; prints deduplicated "Sources:" footer (with page numbers); exits on `quit`, Ctrl+C, EOFError
- **Important:** Import and use `DEFAULT_AUTHOR` from `src.configs.authors` as the default value for `--author` argparse argument
- **Test:** `tests/unit/test_scripts/test_script_chat.py` — mock `build_chain` and `input()`; assert CLI flags forwarded, chain invoked per question, source footer always printed, chunks only with `--show-chunks`, all exit paths
- **README:** Add CLI chat section: `scripts/chat.py` command, all flags table; add Example Usage section with French and English chat examples
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.
- **Deviations from plan:**
  - `DEFAULT_AUTHOR` is imported from `src.configs.authors` (not `src.configs.common` as stated in plan) — this matches the actual codebase structure from Step 8
  - Created comprehensive test suite with 28 tests (vs. suggested basic mocking) covering:
    - Helper functions: `deduplicate_sources()`, `format_sources_footer()`, `format_chunks_output()` (7 tests)
    - Interactive chat loop: `run_interactive_chat()` (14 tests) — all flags, exit paths, error handling, multiple questions
    - Main function: `main()` (7 tests) — argument parsing and forwarding
  - All tests use proper mocking via `@patch` decorators and assert on behavior, not implementation
  - Logging tests use `caplog` fixture instead of `capsys` for proper log capture
  - Script includes helper functions for formatting output (following SRP)
  - Enhanced error handling with try/except around chain invocation to allow continuation after errors
  - Verbose logging includes debug messages for chain invocation and initialization steps
  - All 178 tests pass (including existing tests from previous steps)
  - **Chunk ID isolation (2026-03-11):** Added test `test_chunk_ids_not_in_response_output` to verify that chunk IDs never appear in the main chat output (they should only appear in the `--show-chunks` section). This test ensures chunk IDs remain internal debugging metadata and don't leak into user-facing responses. Test count increased from 28 to 29; all 184 tests pass.
  - **DB path override removal (2026-03-19):** Removed `--db` flag and `db_path` parameter from `run_interactive_chat()` function. Database location now configured solely via `CHROMA_DB_PATH` environment variable. Simplifies CLI interface.

## ✅ Step 10: Web UI
- Add dependency: `uv add --optional ui streamlit`
- Create `chat_ui.py` — chat interface with:
  - Text input for user questions
  - Response text display with deduplicated sources caption (page-specific)
  - Sidebar: author selector (no DB path input)
- **Test:** `tests/unit/test_chat_ui.py` — mock Streamlit API; assert title renders, session_state initialised, question forwarded to chain, messages appended, sources caption shown, early return on no input, st.error on ValueError
- **Verify:** `uv run streamlit run chat_ui.py` — manual browser testing
- **README:** Add Streamlit UI section: launch command, URL (`http://localhost:8501`), sidebar config description; update project structure diagram to add `chat_ui.py` at root
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.
- **Deviations from plan:**
  - Created comprehensive test suite with 24 tests (vs. suggested basic mocking) covering:
    - Helper functions: `deduplicate_sources()`, `format_sources_caption()` (5 tests)
    - Session state initialization: empty state, existing state (2 tests)
    - Chain rebuilding: first time, author changed, no change, error handling (8 tests)
    - Main UI function: rendering, sidebar, message display, user input processing, error handling (9 tests)
  - Added `SessionStateMock` custom test fixture to support both dictionary-style and attribute-style access for Streamlit's session_state
  - All tests use proper mocking via `@patch` decorators with context managers for `st.chat_message`, `st.spinner`, and `st.sidebar`
  - Enhanced UI features:
    - Automatic chain rebuild when configuration changes (author only)
    - Message history cleared when switching authors
    - Loading spinner with "Reflecting..." message during response generation
    - Graceful error handling with user-friendly messages for Ollama availability, configuration errors, and chain invocation failures
    - Page configuration with custom title and icon
    - Helpful sidebar caption explaining RAG functionality
  - README updated with comprehensive documentation: Technology table (added Streamlit), project structure (added chat_ui.py), and detailed "Chat via Web UI" section
  - All 208 tests pass (including new chat_ui tests); mypy type checking passes
  - **DB path override removal (2026-03-19):** Removed "Database Path" text input from sidebar and removed `db_path` parameter from `rebuild_chain_if_needed()` function. Database location now configured solely via `CHROMA_DB_PATH` environment variable. Simplifies UI by removing infrastructure concern from user interface.

## ✅ Step 11: User interface localization (i18n)
- **Goal:** Provide localized user-facing strings for chat interfaces (CLI and web UI) with language-appropriate messages
- Add dependencies: `pyyaml`, `types-pyyaml` (dev dependency for type checking)
- Create `locales/` directory at project root with YAML files for each supported language:
  - `locales/en.yaml` — English strings
  - `locales/fr.yaml` — French strings
  - Structure: nested YAML with categories (chat, status, sources, errors) containing message templates with `{placeholders}` for interpolation
- Create `src/i18n/` package with:
  - `messages.py` — core localization module:
    - `SUPPORTED_LANGUAGES: frozenset[str]` — explicit set of available locales (`"en"`, `"fr"`)
    - `load_messages(language) -> dict[str, Any]` — loads and caches YAML files; falls back to `DEFAULT_RESPONSE_LANGUAGE` if language unsupported
    - `get_message(key, language, **kwargs) -> str` — retrieves localized string with optional interpolation; automatically capitalizes author names
    - `clear_cache() -> None` — testing utility for cache invalidation
    - Private helper `_get_nested_value()` for dot-notation key traversal
  - `keys.py` — type-safe message key constants (e.g., `CHAT_CHATTING_WITH = "chat.chatting_with"`) to prevent typos and enable IDE autocomplete
  - `key_registry.py` — dynamic registry that collects all keys from `keys.py` into `ALL_REQUIRED_KEYS: frozenset[str]` for validation purposes
  - `__init__.py` — exports `get_message` and `clear_cache` as public API
- Create `src/utils/formatting.py` — unified formatting utilities:
  - `deduplicate_sources(response) -> list[str]` — preserves order of first appearance
  - `format_sources(response, language) -> str` — markdown-formatted source citations using localized labels; works for both CLI (readable plaintext) and web UI (rendered markdown)
- Modify `scripts/chat.py` and `chat_ui.py`:
  - Import localization functions from `src.i18n` and `src.utils.formatting`
  - Replace hardcoded English strings with `get_message()` calls using constants from `src.i18n.keys`
  - Use `format_sources()` for consistent source citation formatting
  - Messages adapt to detected response language for better user experience
- **Test:** `tests/unit/i18n/test_messages.py` — comprehensive coverage (241 lines / 42 tests):
  - Locale file loading with caching
  - Fallback to default language for unsupported locales
  - Nested key traversal with dot notation
  - String interpolation with and without author capitalization
  - Error handling for missing files, invalid keys, malformed YAML, non-string values
  - Cache clearing functionality
  - Integration test validating all registered keys exist in all locale files
- **Test:** `tests/unit/utils/test_formatting.py` — comprehensive coverage (144 lines / 20 tests):
  - Source deduplication while preserving order
  - Markdown formatting for both languages
  - Edge cases: empty sources, single source, multiple sources, duplicate sources
- **Test:** Update `tests/unit/test_scripts/test_script_chat.py` and `tests/unit/test_chat_ui.py` — refactor to use localization (removed ~150 lines of duplicated formatting logic replaced by shared utilities)
- **Test:** Add `tests/conftest.py` fixture: `reset_i18n_cache` for clean test isolation
- **README:** Update project structure diagram to add `locales/` directory and `src/i18n/` package
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.
- **Deviations from plan:**
  - Localization system was not in original plan; added as enhancement after completing Step 10 (Web UI)
  - Design benefits:
    - **DRY principle:** Eliminates ~150 lines of duplicated string literals and formatting logic across CLI and UI modules
    - **Type safety:** Constants in `keys.py` prevent typos and provide IDE autocomplete
    - **Extensibility:** Adding a new language requires only a new YAML file; no code changes
    - **Testability:** Shared formatting utilities have single test suite instead of duplicated tests
    - **User experience:** Messages adapt to detected response language for native feel
  - Implementation approach uses YAML instead of gettext/po files for simplicity and better support for nested message structures
  - Key registry uses dynamic introspection to auto-discover all defined keys, reducing maintenance burden
  - Comprehensive test coverage: 62 tests added (42 for i18n module, 20 for formatting utilities)
  - All 270+ tests pass; mypy type checking passes

## ✅ Step 12: Language detection utility
- **Goal:** Standalone language detection module, decoupled from the chain (chain already has `{language}` placeholder from Step 8)
- Add dependency: `langdetect` (add via `uv add langdetect`)
- Create `src/utils/language.py`: `detect_language(text, default="fr", min_length=15) -> str` using `langdetect.detect_langs()` — returns default if text shorter than `min_length` or top language confidence < 0.7; graceful fallback on exception
- Wire into CLI/UI: Both `scripts/chat.py` and `chat_ui.py` call `detect_language(question)` before invoking chain; pass detected language to `chain.invoke({"question": ..., "language": ...})`; chain sets `ChatResponse.language` from input
- **Test:** `tests/unit/utils/test_language.py` — French/English detection, empty/whitespace → default, short text → default, low-confidence → default, exception → default
- **Test:** `tests/unit/chains/test_chat_chain.py` — add tests verifying chain accepts language parameter and passes it to prompt correctly
- **README:** Update CLI and UI sections to note auto-detected response language; add note about English responses translating French passages
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.
- **Deviations from plan:**
  - Implementation uses `confidence` parameter (default 0.7) instead of `default` parameter — `detect_language(text, min_length=15, confidence=0.7)` reads default from `DEFAULT_RESPONSE_LANGUAGE` constant for consistency with config system
  - Chain API changes: Added `language` parameter to `chain.invoke(question, language=...)` for explicit language override; detection happens at call site (CLI/UI), not inside `_run()`. Removed `detect_user_language` boolean parameter. This design:
    - Separates detection (caller's concern) from chain logic (LLM invocation)
    - Makes testing simpler — inject desired language directly
    - Allows override for use cases where language is known
  - Comprehensive test coverage: 18 unit tests (vs. suggested 6) covering all edge cases: French, English, Spanish, Swahili, empty/whitespace, short text, custom min_length, custom confidence, low/high confidence thresholds, detection failures, exceptions
  - Added 5 integration tests (`tests/integration/test_language_detection_integration.py`) to verify language flows through full chain with FakeChatModel: French detected, English detected, Italian (not in SUPPORTED_LANGUAGES) detected, config overrides detection, detection failure fallback
  - Added 3 external tests (`tests/external/test_language_detection_external.py`) with real Ollama LLM to verify end-to-end behavior: French question → French response, English question → English response, Portuguese question → Portuguese response (demonstrates support for languages beyond SUPPORTED_LANGUAGES)
  - All 290 tests pass (18 unit + 5 integration + 3 external + existing tests); mypy type checking passes
  - README updated with language detection documentation in Architecture section and CLI/UI usage notes
  - Created `tests/external/` directory for external-only test files (separate from integration tests which don't make real network calls)

## Step 13: Evaluation harness — schemas + deterministic metrics (Voltaire-only)
- **Goal:** Eval data models and all pure-function deterministic metrics; establish quality baseline with Voltaire before adding more philosophers
- Create `src/eval/` and `src/eval/metrics/` directories (with `__init__.py` in each) and `tests/unit/eval/` directory
- Modify `src/schemas.py`: add `GoldenExample` (with `question_fr`, `question_en`, `expected_chunk_ids`, `expected_source_title_substrings`, `expected_language`, `expected_keywords_fr`, `expected_keywords_en`, `forbidden_keywords_fr`, `forbidden_keywords_en` — all lists default `[]`); `GoldenDataset` (with `version: str`, `examples: list[GoldenExample]`); `MetricResult` (name, score 0-1, details); `EvalResult` (question, metrics list, fr_response, en_response optional); `EvalReport` (dataset_version, results, per-metric averages)
- Create `src/eval/metrics/retrieval.py`: fraction of expected_chunk_ids found in retrieved
- Create `src/eval/metrics/faithfulness.py`: shared `_keyword_score()` helper; `faithfulness()` uses `expected_keywords_fr`; `faithfulness_en()` uses `expected_keywords_en`
- Create `src/eval/metrics/citation.py`: expected source title substrings found in retrieved_source_titles
- Create `src/eval/metrics/citation_placement.py`: verifies that inline citations `[source: ...]` appear AFTER sentences/claims (not at the beginning or middle of paragraphs) — uses regex to detect citations that appear at start of lines or immediately after newlines without preceding text; penalizes misplaced citations
- Create `src/eval/metrics/language.py`: response.language == expected_language (now meaningful with Step 12 language detection working)
- Create `src/eval/metrics/translation.py`: Jaccard overlap of chunk IDs between FR and EN responses (retrieval proxy)
- Create `src/eval/metrics/forbidden.py`: shared `_forbidden_score()` helper; `forbidden_phrases()` / `forbidden_phrases_en()` — catches persona breaks, anachronisms
- Create data directories: `data/eval/`, `data/eval/reports/`; update `.gitignore` to add `data/eval/reports/`
- Create `data/eval/golden_dataset.json` **v1.0 — Voltaire-only**: 5-6 examples covering tolerance, Pascal critique, Newton; 2 adversarial traps (anachronism, persona break); all include expected_keywords, forbidden_keywords for FR and EN
- **Test:** `tests/unit/eval/test_retrieval_metric.py` — all/partial/none found, no expected IDs, empty retrieval (5 tests)
- **Test:** `tests/unit/eval/test_faithfulness_metric.py` — FR: all/partial/none/case-insensitive/no expected; EN: all/partial/no expected (9 tests)
- **Test:** `tests/unit/eval/test_citation_metric.py` — all/partial substrings, case insensitive, no expected, no titles (5 tests)
- **Test:** `tests/unit/eval/test_citation_placement_metric.py` — correct placement (end of sentence), citation at line start, citation after newline without text, no citations, multiple correct placements (5 tests)
- **Test:** `tests/unit/eval/test_language_metric.py` — matching and mismatching language (2 tests)
- **Test:** `tests/unit/eval/test_translation_metric.py` — identical/disjoint/partial/both empty/one empty (5 tests)
- **Test:** `tests/unit/eval/test_forbidden_metric.py` — FR: no forbidden/none triggered/triggered/case-insensitive/multi; EN: none/triggered (9 tests)
- **Test:** `tests/unit/test_schemas.py` — add GoldenExample, GoldenDataset, MetricResult, EvalResult, EvalReport construction and validation
- **README:** Update project structure diagram to add `src/eval/`. Update Architecture Overview to show EVALUATION PIPELINE.
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.

## Step 14: Gouges corpus + persona + eval expansion
- **Goal:** Add second philosopher with her own texts, prompt, and registry entry; expand eval dataset to include Gouges examples
- Create `src/prompts/gouges.py` — Gouges system prompt (her voice, mandatory citations from her texts, responds in `{language}`, passionate advocacy for women's rights); export `build_gouges_prompt() -> ChatPromptTemplate`
- Register `"gouges": AuthorConfig(prompt_factory=build_gouges_prompt, exit_message=<personalized message>)` in `_AUTHOR_CONFIGS` in `chat_chain.py`
- Add `GOUGES_DECLARATION_CONFIG` to `src/configs/loader_configs.py`
- Register `INGEST_CONFIGS["gouges"] = GOUGES_DECLARATION_CONFIG` in `loader_configs.py`
- **Expand golden dataset:** Update `data/eval/golden_dataset.json` to **v2.0** — add 3-4 Gouges examples covering women's rights, social contracts; ensure both Voltaire and Gouges examples present
- **Test:** `tests/unit/test_prompts_gouges.py` — assert prompt template has correct placeholders (`{context}`, `{question}`, `{language}`); assert `"gouges"` registered in `_AUTHOR_CONFIGS`
- **Test:** `tests/unit/chains/test_chat_chain.py` — add test for `author="gouges"` in `build_default_chain` (mock LLM); assert it selects Gouges prompt
- **Test:** `tests/unit/test_loader_configs.py` — add tests: validate GOUGES config fields, `INGEST_CONFIGS["gouges"]` points to correct config
- **README:** Update Technology table if new deps; update CLI flags docs to show `gouges` as valid `--author` value; note that `scripts/ingest.py --author gouges` populates her corpus
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.

## Step 15: Philosopher agents + debate
- **Goal:** LangChain agents that can decide when and how to retrieve; debate orchestrator runs multiple agents against the same question
- **Note:** Debate responses will be evaluated as part of Step 16 (eval runner + CLI); no golden dataset changes needed in this step
- Add DEBATE PIPELINE to README.md Architecture Overview:
```
DEBATE PIPELINE (planned)
──────────────────────────────────────────────────────────────
question → [philosopher_agent(voltaire), philosopher_agent(gouges)] →
  list[DebateResponse]
```

### Background: chains vs. agents
Steps 1–9 implement a RAG chain: a fixed pipeline (retrieve → format → prompt → LLM → respond). A LangChain agent lets the LLM decide which tools to call, in what order, and whether to iterate — enabling richer behaviour like follow-up retrieval and self-critique.

### Implementation
- Create `src/agents/` directory (with `src/agents/__init__.py`) and `tests/unit/agents/` directory
- Add `DebateResponse` to `src/schemas.py`: `author`, `text`, `retrieved_passage_ids`, `retrieved_source_titles`, `language`; add corresponding tests to `tests/unit/test_schemas.py`
- Create `src/agents/philosopher_agent.py`:
  - `build_philosopher_agent(author, llm=None) -> AgentExecutor` (uses `DEFAULT_DB_PATH` from config, no persist_dir parameter)
  - Wraps the author's ChromaDB retriever as a LangChain `Tool`
  - Uses ReAct-style agent loop; `max_iterations=5`
  - Returns `DebateResponse`
- Create `src/agents/debate_orchestrator.py`:
  - `run_debate(question, authors, llm=None) -> list[DebateResponse]` (uses `DEFAULT_DB_PATH` from config, no persist_dir parameter)
  - Instantiates one `philosopher_agent` per author
  - Runs **sequentially** (avoids thread-safety issues with shared ChromaDB)
  - Does **not** synthesize or adjudicate
- Create `scripts/debate.py` — CLI: `--authors voltaire gouges`, `--show-chunks` (no `--db` flag); prints each philosopher's response under their name header + shared Sources footer
- **Note:** `--authors` accepts multiple values; consider using all registered authors from `_AUTHOR_CONFIGS` as default (or a sensible subset)
- **Test:** `tests/unit/agents/test_philosopher_agent.py` — mock LLM and retriever tool; assert agent calls retrieval, constructs DebateResponse, handles empty retrieval
- **Test:** `tests/unit/agents/test_debate_orchestrator.py` — mock both agents; assert each called once with same question, responses in declared author order
- **Test:** `tests/unit/test_script_debate.py` — mock `run_debate` and `input()`; assert flags forwarded, headers per philosopher, source footer, `--show-chunks` behaviour
- **README:** Add debate CLI section: command with `--authors`, flags table, example question; add debate row to Example Usage table; update project structure to include `src/agents/`
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.

## Step 16: Evaluation harness — judge + runner + CLI
- **Goal:** LLM-as-judge, eval runner orchestration, and CLI for comprehensive evaluation
- **Note:** Golden dataset already exists from Steps 13 (v1.0 Voltaire) and 14 (v2.0 + Gouges); this step wires it into a runner + CLI
- Create `src/eval/judge.py`: `run_llm_judge(question, response_text, contexts, llm) -> list[MetricResult]` — scores relevance, groundedness, coherence on 0-1 scale; parses structured LLM output; clamps to [0,1]; handles unparseable output gracefully
- Create `src/eval/runner.py`:
  - `load_golden_dataset(path) -> GoldenDataset` (validates version field)
  - `run_eval(chain_fr, golden_dataset, chain_en=None, use_llm_judge=False, judge_llm=None) -> EvalReport`
  - `chain_en` optional; when absent, EN metrics skipped
  - FR metrics: retrieval_relevance, faithfulness, citation_accuracy, language_compliance, forbidden_phrases
  - EN metrics (when chain_en provided): language_compliance_en, faithfulness_en, forbidden_phrases_en, translation_drift
  - LLM judge on FR response; also on EN when provided
  - Aggregates per-metric averages
  - Uses deterministic metrics from Step 13
- Create `scripts/run_eval.py`: CLI with `--golden`, `--author`, `--output-dir`, `--llm-judge` (no `--db` flag); prints summary table + saves timestamped JSON to `data/eval/reports/`; uses `DEFAULT_DB_PATH` from config
- **Important:** Import and use `DEFAULT_AUTHOR` from `src.configs.authors` as the default value for `--author` argparse argument (consistency with chat CLI and chain)
- **Test:** `tests/unit/eval/test_judge.py` — valid scores, clamped bounds, unparseable output, missing dimension, LLM invocation (5 tests)
- **Test:** `tests/unit/eval/test_runner.py` — FR-only, with EN chain, LLM judge FR-only, LLM judge both chains, averages, chain invocation, load/validate golden dataset (9 tests)
- **Test:** `tests/unit/test_script_run_eval.py` — mock runner; assert CLI flags forwarded, summary printed, report saved
- **README:** Add Evaluation section: plain-language explanation of the five harness steps; split metrics into deterministic vs. LLM-judge; score ranges and troubleshooting guide; update project structure diagram to add `data/eval/`
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.

## Step 17: ChromaDB server mode
- **Goal:** Support both embedded (local) and server (HTTP) ChromaDB modes via env vars
- **Note:** Basic environment-based path configuration already implemented (2026-03-19): `DEFAULT_DB_PATH` in `src/configs/common.py` reads from `CHROMA_DB_PATH` env var (defaults to "data/chroma_db"). This step extends that to support server mode.
- Create `src/configs/db_config.py`:
  - `ChromaMode` enum: `EMBEDDED`, `SERVER`
  - `ChromaDBConfig` Pydantic model: `mode`, `persist_dir` (required for embedded), `host`, `port`
  - `chromadb_config_from_env() -> ChromaDBConfig`: reads `CHROMA_MODE`, `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_DB_PATH` (uses `DEFAULT_DB_PATH` for embedded mode)
- Create `src/vectorstores/chroma_client.py`: `build_chroma_vectorstore(config, collection_name, embeddings) -> Chroma`
  - Embedded: `Chroma(persist_directory=DEFAULT_DB_PATH, ...)` (or from config)
  - Server: `Chroma(client=chromadb.HttpClient(host, port), ...)`
- Modify `src/vectorstores/chroma.py` and `src/vectorstores/retriever.py`: detect mode from env and use appropriate client factory
- Create `tests/unit/configs/` if not already exists
- **Test:** `tests/unit/configs/test_db_config.py` — default embedded mode, server mode, validation, env var parsing
- **Test:** `tests/unit/vectorstores/test_chroma_client.py` — embedded uses persist_directory, server uses HttpClient (mock HttpClient)
- **README:** Add ChromaDB configuration section: env vars table (`CHROMA_MODE`, `CHROMA_DB_PATH`, `CHROMA_HOST`, `CHROMA_PORT`), instructions for `chroma run`, verification command
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.

## Step 18: Heroku deployment
- **Goal:** Make the app deployable to Heroku with configurable LLM/embedding providers
- Create `Procfile`: `web: streamlit run chat_ui.py --server.port=$PORT --server.address=0.0.0.0`
- Create `runtime.txt`: pin Python 3.14
- Create `src/configs/provider_config.py`:
  - `LLMProvider` enum: `OLLAMA`, `ANTHROPIC`, `OPENAI`
  - `EmbeddingProvider` enum: `OLLAMA`, `OPENAI`
  - `build_llm_from_env() -> BaseChatModel`: reads `LLM_PROVIDER` env var; returns appropriate LangChain chat model
  - `build_embeddings_from_env() -> Embeddings`: reads `EMBEDDING_PROVIDER` env var
  - Add `langchain-anthropic` and `langchain-openai` as optional dependency groups
- Modify `src/chains/chat_chain.py`: `build_default_chain` uses `build_llm_from_env()` and `build_embeddings_from_env()` when env vars are set; falls back to Ollama defaults otherwise
- **Test:** existing unit tests require no changes (they mock LLM and embeddings); add `tests/unit/configs/test_provider_config.py` — mock env vars, assert correct provider classes returned, assert fallback to Ollama
- **README:** Add "Deploying to Heroku" section: required env vars, `heroku config:set` commands, note to pre-populate ChromaDB; update project structure diagram to add `Procfile`, `runtime.txt` at root
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.

---

## Implementation Order
1. **Step 1** (scaffolding) — everything depends on project structure and dependencies
2. **Step 2** (schemas) — data models used by all subsequent steps
3. **Step 3** (loader) — first stage of ingestion pipeline; configs used throughout
4. **Step 4** (chunking) — second stage; depends on loader's Document output
5. **Step 5** (embed + store) — third stage; depends on chunks
6. **Step 6** (retriever) — query-time counterpart to embed; depends on stored data
7. **Step 7** (combined ingest) — convenience script wrapping the ingestion pipeline; depends on all ingestion pieces (Steps 3–5)
8. **Step 8** (Voltaire prompt + chat chain) — wires retriever + LLM; core feature; bilingual prompt from the start
9. **Step 9** (chat CLI script) — interactive CLI over the chain; separated for reviewability
10. **Step 10** (Streamlit UI) — visual interface over the chain
11. **Step 11** (i18n localization) — user-facing strings for CLI and UI; eliminates duplicated formatting logic; establishes extensible localization pattern
12. **Step 12** (language detection) — standalone utility wired into chain; makes eval metrics meaningful
13. **Step 13** (eval schemas + metrics) — establishes quality baseline with Voltaire-only golden dataset (v1.0); deterministic metrics ready for use
14. **Step 14** (Gouges corpus + eval expansion) — second philosopher + expand golden dataset to v2.0 with Gouges examples
15. **Step 15** (debate agents) — multi-philosopher debate mode; depends on both philosophers registered
16. **Step 16** (eval runner + CLI) — orchestration and CLI for running evals; adds LLM-as-judge; uses metrics from Step 13 and dataset from Steps 13-14
17. **Step 17** (ChromaDB server mode) — isolated infrastructure; prerequisite for Heroku
18. **Step 18** (Heroku deployment) — requires server mode + provider abstraction

---

## Verification (end-to-end)
1. `uv run pytest` — all unit + integration tests pass (external excluded)
2. `uv run mypy src scripts` — type checker clean
3. `uv run python scripts/ingest.py --author voltaire` — scrape + embed Voltaire
4. `uv run python scripts/chat.py --show-chunks` → ask "Que pensez-vous de la tolérance?" → Voltaire-style French response with page-specific citations
5. `uv run python scripts/chat.py` → ask "What do you think about tolerance?" → English response with language detection
6. `uv run python scripts/run_eval.py --golden data/eval/golden_dataset.json --author voltaire` → eval report for Voltaire baseline (after Step 13)
7. `uv run python scripts/ingest.py --author gouges` — scrape + embed Gouges
8. `uv run python scripts/chat.py --author gouges` → ask about rights → Gouges-style response
9. `uv run python scripts/debate.py --authors voltaire gouges` → ask about rights → two distinct grounded responses (after Step 15)
10. `uv run python scripts/run_eval.py --golden data/eval/golden_dataset.json --llm-judge` → full eval report with both authors + judge (after Step 16)
11. `uv run streamlit run chat_ui.py` → manual browser verification
12. Set `CHROMA_MODE=server`, run `chroma run`, verify chat works over HTTP

## Project structure
```
luminary/
├── src/
│   ├── schemas.py
│   ├── configs/
│   ├── document_loaders/
│   ├── utils/
│   ├── vectorstores/
│   ├── prompts/
│   ├── chains/
│   ├── agents/
│   ├── i18n/
│   └── eval/
│       └── metrics/
├── scripts/
├── data/
│   ├── raw/
│   ├── chroma_db/
│   └── eval/
│       └── reports/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── external/
├── locales/
├── docs/
└── chat_ui.py
```
