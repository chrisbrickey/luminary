# Ingestion Pipeline Details
The top-level `README.md` contains the most pertinent information for understanding and running the ingestion pipeline.
This file includes supplementary information.

## Run ingestion scripts separately

This section contains information on the individual scripts for scraping and embedding that are both used by the single, unified command in the Setup section of top-level `README.md`.

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
- `--verbose` (optional): Debugging with verbose logging

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
- `--verbose` (optional): Debugging with verbose logging

**Output location:**
Embeddings are stored in the ChromaDB vector database (at `data/chroma_db/` in local env) with collection name `philosophes`.