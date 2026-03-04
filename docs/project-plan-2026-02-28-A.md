# Plan: Luminary — Enlightenment Philosopher RAG Chatbot

## Context
Luminary is a RAG chatbot where Enlightenment philosophers (e.g., Voltaire, Olympe de Gouges) answer questions grounded in their historical texts. 
It will feature an ingestion pipeline (initially integrated with Wikisource), retrieval-augmented chat with per-philosopher persona prompts, bilingual support (French/English), a multi-philosopher debate mode with LangChain agents, a deterministic + LLM-as-judge evaluation harness, and deployment to Heroku with configurable LLM/embedding providers. 
This plan covers the complete system from project scaffolding through production deployment.

## Hosting target: Heroku

All design decisions should keep Heroku deployment in mind:

1. **ChromaDB storage** — Heroku has an ephemeral filesystem. Step 16 introduces `ChromaDBConfig` with `EMBEDDED` (local dev) and `SERVER` (production) modes via env vars.
2. **LLM + embeddings** — Heroku cannot run Ollama. Step 17 adds provider abstraction: `LLM_PROVIDER` and `EMBEDDING_PROVIDER` env vars to swap between Ollama (local) and hosted APIs (Anthropic, OpenAI).
3. **Port binding** — `Procfile` binds Streamlit web UI to `$PORT`.

**Design rule:** prefer injectable dependencies (LLM, embeddings, retriever) over hardcoded defaults so local and hosted implementations can be swapped via configuration without touching business logic.

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
  - `DEFAULT_DB_PATH` simplified to relative path `Path("data/chroma_db")` instead of absolute resolved path
  - Config name changed from `VOLTAIRE_LETTRES_CONFIG` to `LETTRES_PHILOSOPHIQUES_CONFIG`
  - `--author` flag changed from required to optional (defaults to all configured authors)
  - Enhanced logging with user-friendly progress messages and simplified format to show real-time scraping progress
  - `load_documents_from_disk()` and associated tests deferred to Step 7 (not needed until combined ingestion script)
  - Test organization: created `tests/unit/document_loaders/conftest.py` with shared fixtures and helpers following DRY principles
  - Added integration test `tests/integration/document_loaders/test_wikisource_loader_integration.py` for external API contract testing
  - Refactored `scripts/scrape_wikisource.py` with `scrape_author()` helper function for cleaner code organization

## Step 4: Corpus ingestion — chunking
- Create `src/utils/chunker.py`
- Use LangChain `RecursiveCharacterTextSplitter` (chunk_size=1200 chars, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""] for French prose)
- Attach chunk metadata: `chunk_id` (SHA256 of `document_id:chunk_index`[:12]), `chunk_index` (0-indexed per document); preserve all source metadata
- **Validate chunks via ChunkInfo** — construct a `ChunkInfo` from each chunk's metadata before returning, ensuring schema compliance at chunking time (improvement over rag-chat-1 where ChunkInfo was defined but never enforced)
- **Test:** `tests/unit/utils/test_chunker.py` — given a known Document, assert chunks are within size bounds, metadata is correct, no text is lost, ChunkInfo validation passes; test empty document edge case
- **README:** No user-facing command; update pipeline diagram to show chunking stage
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.

## Step 5: Embedding + ChromaDB storage
- Create new directories: `src/vectorstores/` with `__init__.py`; `tests/unit/vectorstores/`
- Update `.gitignore`: add `data/chroma_db/`
- Create `tests/conftest.py`: `FakeEmbeddings(Embeddings)` fixture returning constant 8-dim vectors; disable ChromaDB telemetry via `os.environ.setdefault` before import
- Create `src/vectorstores/embed_and_store.py`
- `embed_and_store(chunks, persist_dir, collection_name="philosophes", embeddings=None) -> Chroma`:
  - Uses `Chroma.from_documents()` with **`ids=` set to each chunk's `chunk_id`** — idempotent upserts on re-run (improvement over rag-chat-1 which omitted `ids=`, causing duplicates)
  - Accepts injectable embeddings for testing; defaults to `OllamaEmbeddings(model="nomic-embed-text")`
- Create `scripts/embed_and_store.py` — CLI entrypoint: calls `check_ollama_available()`, loads chunks from disk, embeds and stores
- **Test:** `tests/unit/vectorstores/test_embed_and_store.py` — ingest 2-3 fixture chunks with FakeEmbeddings; verify retrievable by similarity search; verify re-running with same IDs does not create duplicates
- **README:** Add ingestion step 2 (`scripts/embed_and_store.py`) with command, output location (`data/chroma_db/`); update project structure diagram to add `src/vectorstores/`, `data/chroma_db/`
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.

## Step 6: Retrieval chain
- Create `src/vectorstores/retriever.py`
- `build_retriever(persist_dir, collection_name="philosophes", embeddings=None, k=5, author=None) -> VectorStoreRetriever`
- Wrap ChromaDB as LangChain retriever; apply author filter at retriever level via `search_kwargs={"filter": {"author": author}}` (ChromaDB metadata filter, not post-retrieval)
- **Test:** `tests/unit/vectorstores/test_retriever.py` — with small fixture DB, query and assert relevant chunks returned with intact metadata; test author filter; test without filter returns all
- **Test:** `tests/integration/test_retriever.py` — wire real ChromaDB + FakeEmbeddings end-to-end; embed fixtures then retrieve; verify round-trip
- **README:** No user-facing command; update pipeline diagram to show retrieval stage
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.

## Step 7: Combined ingestion script
- **Goal:** Single `scripts/ingest.py` replaces running scrape + embed separately; moved here because the full ingestion pipeline (Steps 3–5) is now complete
- **Note:** Add `load_documents_from_disk()` function to `src/utils/io.py` (deferred from Step 3) with corresponding tests for load operations and round-trip testing
- Create `scripts/ingest.py`: `ingest_author(config, raw_dir, db_dir, skip_scrape, skip_embed)` function + `main()` with argparse
  - Flags: `--author` (optional, defaults to all registered in `INGEST_CONFIGS`), `--skip-scrape`, `--skip-embed`, `--db`
  - Calls existing `WikisourceLoader`, `save_documents_to_disk`, `load_documents_from_disk`, `chunk_documents`, `embed_and_store`
  - Calls `check_ollama_available()` unless `--skip-embed` (only embedding needs Ollama)
- **Test:** `tests/unit/test_script_ingest.py` — mock all external deps; test default (scrape+embed), skip-scrape, skip-embed, all-authors, single-author, invalid-author
- **README:** Replace two-step ingestion instructions with `scripts/ingest.py` as primary command; document all flags; keep individual scripts as lower-level alternatives
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.

## Step 8: Voltaire prompt + chat chain
- Create new directories with `__init__.py`: `src/prompts/`, `src/chains/`; create `tests/unit/chains/`
- Create `src/prompts/voltaire.py` — Voltaire system prompt (irony/wit, mandatory inline citations, responds in `{language}`, translates cited French passages when responding in English); export `RESPONSE_LANGUAGE = "fr"` and `build_voltaire_prompt() -> ChatPromptTemplate`
  - **Bilingual from the start** — prompt includes `{language}` placeholder (improvement over rag-chat-1 which had to retrofit this later)
- Create `src/chains/chat_chain.py`:
  - `DEFAULT_AUTHOR = "voltaire"` — canonical author key
  - `_AUTHOR_CONFIGS: dict[str, tuple[Callable[[], ChatPromptTemplate], str]]` — extensible registry
  - `build_default_chain(persist_dir, author=DEFAULT_AUTHOR) -> Runnable[str, ChatResponse]` — production entry point; validates author against registry; uses `ChatOllama(model="mistral")` as default LLM
  - `build_chat_chain(retriever, llm=None, prompt=None, default_language="fr", detect_user_language=True) -> Runnable[str, ChatResponse]` — injectable helper
  - Internal `_run(question) -> ChatResponse`: retrieves top-k docs, formats context with `[source: {title}, page {page_number} | chunk_id: {id}]` labels (page-specific from the start), detects question language, invokes LLM, returns structured ChatResponse
  - Helpers: `_format_docs_with_titles()`, `_extract_chunk_ids()`, `_extract_source_titles()` (combines document_title + page_number; fallback: title-only → source URL → "unknown")
- **Test:** `tests/unit/chains/test_chat_chain.py` — mock LLM and retriever; assert chain wires retrieval + prompt correctly; test `_extract_source_titles()` with title+page, title-only, fallback to source URL, missing metadata; test empty retrieval; test unknown author raises ValueError; test language detection integration
- **Test:** `tests/integration/test_chat_chain_integration.py` — wire small fixture ChromaDB (FakeEmbeddings) → real retriever → real prompt → FakeChatModel; assert full chain returns ChatResponse with correct retrieved_source_titles and non-empty text
- **README:** Add Architecture Overview section with full pipeline diagram (Ingestion, Query, Debate, Eval); update project structure diagram to add `src/prompts/`, `src/chains/`
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.

## Step 9: Chat CLI script
- Create `scripts/chat.py` — interactive CLI chat loop; flags: `--db`, `--author`, `--show-chunks`, `--verbose`; calls `check_ollama_available()`; prints deduplicated "Sources:" footer (with page numbers); exits on `quit`, Ctrl+C, EOFError
- **Test:** `tests/unit/test_script_chat.py` — mock `build_default_chain` and `input()`; assert CLI flags forwarded, chain invoked per question, source footer always printed, chunks only with `--show-chunks`, all exit paths
- **README:** Add CLI chat section: `scripts/chat.py` command, all flags table; add Example Usage section with French and English chat examples
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.

## Step 10: Web UI
- Add dependency: `uv add --optional ui streamlit`
- Create `chat_ui.py` — chat interface with:
  - Text input for user questions
  - Response text display with deduplicated sources caption (page-specific)
  - Sidebar: DB path, author selector
- **Test:** `tests/unit/test_chat_ui.py` — mock Streamlit API; assert title renders, session_state initialised, question forwarded to chain, messages appended, sources caption shown, early return on no input, st.error on ValueError
- **Verify:** `uv run streamlit run chat_ui.py` — manual browser testing
- **README:** Add Streamlit UI section: launch command, URL (`http://localhost:8501`), sidebar config description; update project structure diagram to add `chat_ui.py` at root
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.

## Step 11: Language detection utility
- **Goal:** Standalone language detection module, decoupled from the chain (chain already has `{language}` placeholder from Step 8)
- Add dependency: `langdetect` (add via `uv add langdetect`)
- Create `src/utils/language.py`: `detect_language(text, default="fr", min_length=15) -> str` using `langdetect.detect_langs()` — returns default if text shorter than `min_length` or top language confidence < 0.7; graceful fallback on exception
- Wire into `src/chains/chat_chain.py`: `_run()` calls `detect_language(question)` when `detect_user_language=True`; passes detected language to prompt and sets `ChatResponse.language`
- **Test:** `tests/unit/utils/test_language.py` — French/English detection, empty/whitespace → default, short text → default, low-confidence → default, exception → default
- **Test:** `tests/unit/chains/test_chat_chain.py` — add tests: mock `detect_language`, verify English detection, detection disabled, language passed to prompt
- **README:** Update CLI and UI sections to note auto-detected response language; add note about English responses translating French passages
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.

## Step 12: Gouges corpus + persona
- **Goal:** Add second philosopher with her own texts, prompt, and registry entry
- Create `src/prompts/gouges.py` — Gouges system prompt (her voice, mandatory citations from her texts, responds in `{language}`, passionate advocacy for women's rights); export `build_gouges_prompt() -> ChatPromptTemplate`
- Register `"gouges": (build_gouges_prompt, "fr")` in `_AUTHOR_CONFIGS` in `chat_chain.py`
- Add `GOUGES_DECLARATION_CONFIG` to `src/configs/loader_configs.py`
- Register `INGEST_CONFIGS["gouges"] = GOUGES_DECLARATION_CONFIG` in `loader_configs.py`
- **Test:** `tests/unit/test_prompts_gouges.py` — assert prompt template has correct placeholders (`{context}`, `{question}`, `{language}`); assert `"gouges"` registered in `_AUTHOR_CONFIGS`
- **Test:** `tests/unit/chains/test_chat_chain.py` — add test for `author="gouges"` in `build_default_chain` (mock LLM); assert it selects Gouges prompt
- **Test:** `tests/unit/test_loader_configs.py` — add tests: validate GOUGES config fields, `INGEST_CONFIGS["gouges"]` points to correct config
- **README:** Update Technology table if new deps; update CLI flags docs to show `gouges` as valid `--author` value; note that `scripts/ingest.py --author gouges` populates her corpus
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.

## Step 13: Philosopher agents + debate
- **Goal:** LangChain agents that can decide when and how to retrieve; debate orchestrator runs multiple agents against the same question

### Background: chains vs. agents
Steps 1–9 implement a RAG chain: a fixed pipeline (retrieve → format → prompt → LLM → respond). A LangChain agent lets the LLM decide which tools to call, in what order, and whether to iterate — enabling richer behaviour like follow-up retrieval and self-critique.

### Implementation
- Create `src/agents/` directory (with `src/agents/__init__.py`) and `tests/unit/agents/` directory
- Add `DebateResponse` to `src/schemas.py`: `author`, `text`, `retrieved_passage_ids`, `retrieved_source_titles`, `language`; add corresponding tests to `tests/unit/test_schemas.py`
- Create `src/agents/philosopher_agent.py`:
  - `build_philosopher_agent(author, persist_dir, llm=None) -> AgentExecutor`
  - Wraps the author's ChromaDB retriever as a LangChain `Tool`
  - Uses ReAct-style agent loop; `max_iterations=5`
  - Returns `DebateResponse`
- Create `src/agents/debate_orchestrator.py`:
  - `run_debate(question, authors, persist_dir, llm=None) -> list[DebateResponse]`
  - Instantiates one `philosopher_agent` per author
  - Runs **sequentially** (avoids thread-safety issues with shared ChromaDB)
  - Does **not** synthesize or adjudicate
- Create `scripts/debate.py` — CLI: `--authors voltaire gouges`, `--show-chunks`, `--db`; prints each philosopher's response under their name header + shared Sources footer
- **Test:** `tests/unit/agents/test_philosopher_agent.py` — mock LLM and retriever tool; assert agent calls retrieval, constructs DebateResponse, handles empty retrieval
- **Test:** `tests/unit/agents/test_debate_orchestrator.py` — mock both agents; assert each called once with same question, responses in declared author order
- **Test:** `tests/unit/test_script_debate.py` — mock `run_debate` and `input()`; assert flags forwarded, headers per philosopher, source footer, `--show-chunks` behaviour
- **README:** Add debate CLI section: command with `--authors`, flags table, example question; add debate row to Example Usage table; update project structure to include `src/agents/`
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.

## Step 14: Evaluation harness — schemas + deterministic metrics
- **Goal:** Eval data models and all pure-function deterministic metrics
- Create `src/eval/` and `src/eval/metrics/` directories (with `__init__.py` in each) and `tests/unit/eval/` directory
- Modify `src/schemas.py`: add `GoldenExample` (with `question_fr`, `question_en`, `expected_chunk_ids`, `expected_source_title_substrings`, `expected_language`, `expected_keywords_fr`, `expected_keywords_en`, `forbidden_keywords_fr`, `forbidden_keywords_en` — all lists default `[]`); `GoldenDataset` (with `version: str`, `examples: list[GoldenExample]`); `MetricResult` (name, score 0-1, details); `EvalResult` (question, metrics list, fr_response, en_response optional); `EvalReport` (dataset_version, results, per-metric averages)
- Create `src/eval/metrics/retrieval.py`: fraction of expected_chunk_ids found in retrieved
- Create `src/eval/metrics/faithfulness.py`: shared `_keyword_score()` helper; `faithfulness()` uses `expected_keywords_fr`; `faithfulness_en()` uses `expected_keywords_en`
- Create `src/eval/metrics/citation.py`: expected source title substrings found in retrieved_source_titles
- Create `src/eval/metrics/language.py`: response.language == expected_language (meaningful post-Step 11 when language is detected, not hardcoded)
- Create `src/eval/metrics/translation.py`: Jaccard overlap of chunk IDs between FR and EN responses (retrieval proxy)
- Create `src/eval/metrics/forbidden.py`: shared `_forbidden_score()` helper; `forbidden_phrases()` / `forbidden_phrases_en()` — catches persona breaks, anachronisms
- **Test:** `tests/unit/eval/test_retrieval_metric.py` — all/partial/none found, no expected IDs, empty retrieval (5 tests)
- **Test:** `tests/unit/eval/test_faithfulness_metric.py` — FR: all/partial/none/case-insensitive/no expected; EN: all/partial/no expected (9 tests)
- **Test:** `tests/unit/eval/test_citation_metric.py` — all/partial substrings, case insensitive, no expected, no titles (5 tests)
- **Test:** `tests/unit/eval/test_language_metric.py` — matching and mismatching language (2 tests)
- **Test:** `tests/unit/eval/test_translation_metric.py` — identical/disjoint/partial/both empty/one empty (5 tests)
- **Test:** `tests/unit/eval/test_forbidden_metric.py` — FR: no forbidden/none triggered/triggered/case-insensitive/multi; EN: none/triggered (9 tests)
- **Test:** `tests/unit/test_schemas.py` — add GoldenExample, GoldenDataset, MetricResult, EvalResult, EvalReport construction and validation
- **README:** No user-facing command yet; update project structure diagram to add `src/eval/`
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.

## Step 15: Evaluation harness — judge + runner + CLI
- **Goal:** LLM-as-judge, eval runner orchestration, golden dataset, and CLI
- Create `src/eval/judge.py`: `run_llm_judge(question, response_text, contexts, llm) -> list[MetricResult]` — scores relevance, groundedness, coherence on 0-1 scale; parses structured LLM output; clamps to [0,1]; handles unparseable output gracefully
- Create `src/eval/runner.py`:
  - `load_golden_dataset(path) -> GoldenDataset` (validates version field)
  - `run_eval(chain_fr, golden_dataset, chain_en=None, use_llm_judge=False, judge_llm=None) -> EvalReport`
  - `chain_en` optional; when absent, EN metrics skipped
  - FR metrics: retrieval_relevance, faithfulness, citation_accuracy, language_compliance, forbidden_phrases
  - EN metrics (when chain_en provided): language_compliance_en, faithfulness_en, forbidden_phrases_en, translation_drift
  - LLM judge on FR response; also on EN when provided
  - Aggregates per-metric averages
- Create data directories: `data/eval/`, `data/eval/reports/`; update `.gitignore` to add `data/eval/reports/`
- Create `data/eval/golden_dataset.json` v2.1: 8 examples covering Quakers, Pascal critique, Bourse de Londres, Newton vs Descartes, inoculation; 3 adversarial traps (anachronism, persona trap); all include expected_keywords, forbidden_keywords for FR and EN
- Create `scripts/run_eval.py`: CLI with `--db`, `--golden`, `--author`, `--output-dir`, `--llm-judge`; prints summary table + saves timestamped JSON to `data/eval/reports/`
- **Test:** `tests/unit/eval/test_judge.py` — valid scores, clamped bounds, unparseable output, missing dimension, LLM invocation (5 tests)
- **Test:** `tests/unit/eval/test_runner.py` — FR-only, with EN chain, LLM judge FR-only, LLM judge both chains, averages, chain invocation, load/validate golden dataset (9 tests)
- **Test:** `tests/unit/test_script_run_eval.py` — mock runner; assert CLI flags forwarded, summary printed, report saved
- **README:** Add Evaluation section: plain-language explanation of the five harness steps; split metrics into deterministic vs. LLM-judge; score ranges and troubleshooting guide; update project structure diagram to add `data/eval/`
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.

## Step 16: ChromaDB server mode
- **Goal:** Support both embedded (local) and server (HTTP) ChromaDB modes via env vars
- Create `src/configs/db_config.py`:
  - `ChromaMode` enum: `EMBEDDED`, `SERVER`
  - `ChromaDBConfig` Pydantic model: `mode`, `persist_dir` (required for embedded), `host`, `port`
  - `chromadb_config_from_env(default_persist_dir) -> ChromaDBConfig`: reads `CHROMA_MODE`, `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_PERSIST_DIR`
- Create `src/vectorstores/chroma_client.py`: `build_chroma_vectorstore(config, collection_name, embeddings) -> Chroma`
  - Embedded: `Chroma(persist_directory=..., ...)`
  - Server: `Chroma(client=chromadb.HttpClient(host, port), ...)`
- Modify `src/vectorstores/embed_and_store.py` and `src/vectorstores/retriever.py`: add optional `db_config: ChromaDBConfig | None` param; when provided, use factory
- Create `tests/unit/configs/`
- **Test:** `tests/unit/configs/test_db_config.py` — default mode, validation, env var parsing
- **Test:** `tests/unit/vectorstores/test_chroma_client.py` — embedded uses persist_directory, server uses HttpClient (mock HttpClient)
- **README:** Add ChromaDB configuration section: env vars table, instructions for `chroma run`, verification command
- **Update this plan:** After implementing, mark step `✅`, note deviations, update project structure.

## Step 17: Heroku deployment
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
11. **Step 11** (language detection) — standalone utility wired into chain; makes eval metrics meaningful
12. **Step 12** (Gouges corpus) — second philosopher; depends on stable chain + registry
13. **Step 13** (debate agents) — depends on both philosophers registered
14. **Step 14** (eval metrics) — pure functions; depends on stable schemas
15. **Step 15** (eval runner + CLI) — orchestration; depends on metrics + chain
16. **Step 16** (ChromaDB server mode) — isolated infrastructure; prerequisite for Heroku
17. **Step 17** (Heroku deployment) — requires server mode + provider abstraction

---

## Verification (end-to-end)
1. `uv run pytest` — all unit + integration tests pass (external excluded)
2. `uv run mypy src scripts` — type checker clean
3. `uv run python scripts/ingest.py --author voltaire` — scrape + embed Voltaire
4. `uv run python scripts/chat.py --show-chunks` → ask "Que pensez-vous de la tolérance?" → Voltaire-style French response with page-specific citations
5. `uv run python scripts/chat.py` → ask "What do you think about tolerance?" → English response
6. `uv run python scripts/ingest.py --author gouges` — scrape + embed Gouges
7. `uv run python scripts/debate.py --authors voltaire gouges` → ask about rights → two distinct grounded responses
8. `uv run python scripts/run_eval.py --golden data/eval/golden_dataset.json` → eval report generated
9. `uv run streamlit run chat_ui.py` → manual browser verification
10. Set `CHROMA_MODE=server`, run `chroma run`, verify chat works over HTTP

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
│   └── integration/
├── docs/
└── chat_ui.py
```
