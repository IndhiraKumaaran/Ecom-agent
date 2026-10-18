# E-Commerce Shopping Assistant

A modular FastAPI foundation for an agentic product recommendation and shopping assistant. It provides a versioned FastAPI API, SQLite persistence, fictional laptop catalog data, and a grounded product FAQ assistant backed by Chroma. Shopping recommendation orchestration remains a separate workflow phase.

## Requirements

- Python 3.12 (available in the provided Codespace)
- `pip`
- On Ubuntu Codespaces, the `python3.12-venv` package

## Setup

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
cp .env.example .env
```

The default configuration is `LLM_MODE=mock` and `FAQ_EMBEDDING_PROVIDER=local_hash`; no API key, model download, or network access is needed to start, ingest, or test. The local hash embedder is deterministic and intended for offline demonstrations, not production semantic quality. The `.env` file is ignored by Git.

## Run

```bash
source .venv/bin/activate
uvicorn ecom_assistant.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. OpenAPI documentation is at `/docs`; health checks are available at `/api/v1/health` and `/health`. On startup, SQLite tables and the catalog are initialized, then catalog specifications and `data/faq_documents.json` are reconciled into the persistent Chroma index. The default database is `data/ecom_assistant.db`; the default vector store is `data/chroma`. Generated database and vector-store files are not tracked.

## Product FAQ RAG

Ask a catalog-grounded question:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/faq/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What processor and RAM are listed?","product_id":"LAP-AST-014"}'
```

The response includes `status` (`grounded` or `insufficient_evidence`), an answer, source references, and retrieved snippets with product/document metadata. An unknown product ID returns 404; a Chroma or embedding-provider failure returns 503. Insufficient evidence returns an explicit catalog-insufficiency message with no cited sources. No cart is changed by this endpoint.

Product documents are built from SQLite catalog descriptions and specifications. FAQ and warranty records are supplied in `data/faq_documents.json`; all warranty terms there are explicitly fictional demonstration content, not manufacturer claims. Re-ingestion uses stable IDs and content hashes to update changed documents, skips unchanged embeddings, and removes stale documents managed by this application. To refresh the index manually after changing source data:

```bash
python -m ecom_assistant.rag.ingest
```

Chroma persists locally under `FAQ_PERSIST_DIRECTORY` (default `data/chroma`). Set `FAQ_EMBEDDING_PROVIDER=openai` and `FAQ_EMBEDDING_MODEL=text-embedding-3-small` to use OpenAI embeddings; this requires a valid `OPENAI_API_KEY`, network access, and may incur API charges. This setting is independent of `LLM_MODE` and `FAQ_CHAT_MODEL`. To enable ChatOpenAI phrasing, set `LLM_MODE=openai`, `FAQ_CHAT_MODEL=gpt-4o-mini`, and `OPENAI_API_KEY`. The API key is required for either OpenAI provider. The live answer prompt treats user questions and retrieved documents as untrusted data; generated text is checked against evidence terms and falls back to the exact retrieved source text if unsupported. OpenAI calls may send the question and retrieved evidence to the configured provider.

Using `local_hash` requires no model files. OpenAI embeddings are hosted and require no local model download. To use a local trained embedding model instead, install and configure a compatible local LangChain embeddings provider and download its model explicitly; the local hash provider remains the offline default.

The product catalog, stock quantities, and delivery estimates are fictional sample data for development and demonstration only. Prices are in INR. This phase does not create orders or process payments.

## Test

```bash
source .venv/bin/activate
pytest
```

## Layout

```text
src/ecom_assistant/
  api/            HTTP routes and versioned routers
  db/             SQLAlchemy models, SQLite connection, initialization
  repositories/   Persistence boundary for future features
  schemas/        Pydantic API contracts
  services/       Business-service boundary and catalog import
  tools/          LangChain tool boundary (later phase)
  workflow/       LangGraph workflow boundary (later phase)
  rag/            FAQ documents, embeddings, ingestion, and retrieval
  config.py       Validated environment configuration
  logging_config.py
  main.py         FastAPI application and lifespan
data/catalog.csv  Fictional seed catalog
data/faq_documents.json  Fictional FAQ and warranty source records
tests/            Startup, health, database, and seeding checks
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `APP_NAME` | `E-Commerce Shopping Assistant` | FastAPI application title |
| `APP_ENV` | `development` | Environment label |
| `LLM_MODE` | `mock` | `mock` for offline operation; `openai` requires a key |
| `OPENAI_API_KEY` | unset | Secret for future real-LLM integrations; never returned by the API |
| `DATABASE_URL` | `sqlite:///./data/ecom_assistant.db` | SQLite database URL |
| `LOG_LEVEL` | `INFO` | Python logging level |

Only SQLite URLs are accepted in this phase. `DATABASE_URL` and other settings can be overridden in the environment or `.env`.