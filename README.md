# E-Commerce Shopping Assistant

A Python 3.12 FastAPI shopping assistant with deterministic SQLite catalog services, an explicit LangGraph recommendation workflow, and a catalog-grounded Chroma FAQ assistant. Product, price, stock, delivery, and warranty examples are fictional demonstration data, not live market information. The application does not place orders or process payments.

## Current Scope

Implemented: validated configuration, repeatable SQLite catalog seeding, deterministic product search/comparison/inventory/budget APIs, mock or configured ChatOpenAI requirement extraction, four read-only LangChain catalog tools, a compiled multi-stage LangGraph recommendation workflow with deterministic hard-constraint ranking, and FAQ RAG.

Implemented: persisted simulated carts with explicit two-step mutation proposals, atomic confirmation, replay protection, expiry/stale-state checks, and stock validation. Not implemented: payment, real orders, or external purchases. Recommendation does not add items to a cart.

## Architecture

```text
POST /shopping/recommend -> compiled LangGraph
  requirement extraction -> product search -> comparison
    -> inventory validation -> budget validation
      -> deterministic ranking / recovery
            |                    |
       LangChain tools      CatalogService -> SQLite

FAQ API -> FAQService -> LangChain embeddings -> Chroma
                       `-> optional ChatOpenAI FAQ phrasing

Cart API -> CartService -> SQLite transaction
  proposal (no mutation) -> explicit proposal-ID confirmation -> revalidate -> apply once

POST /shopping/chat -> deterministic ConversationWorkflow
  intent detection -> one existing capability -> safe answer + operational trace
  structured follow-up references -> expiring SQLite conversation context
```

The graph owns typed state and records each stage. Tools open and close their own request-independent SQLAlchemy sessions and delegate to the same `CatalogService` used by product APIs. Tools are read-only. Recommendation scores and hard-constraint decisions are deterministic Python; no model can change stored price, stock, or strict-match rules.

In `LLM_MODE=mock`, a limited deterministic phrase parser is used and responses clearly report `extraction_mode: "mock"`; no LLM is called. In `LLM_MODE=openai`, the configured ChatOpenAI model returns a structured requirement schema. Extracted hard values are checked against explicit detectable text; extraction errors return `503` and do not silently fall back to mock parsing. Model-generated explanation is not used for ranking or factual product fields.

The FAQ service builds product specification documents from SQLite and loads fictional FAQ/warranty documents from JSON. It stores product ID, document type, source, stable document ID and a content hash in Chroma metadata. Re-ingestion upserts changed documents, skips unchanged content, and removes stale documents managed by this application. Product-scoped retrieval includes matching product documents and general FAQ documents, not other products' records.

Offline responses quote retrieved text. In OpenAI chat mode, generated output is accepted only when it is an exact contiguous excerpt from retrieved evidence; otherwise the service returns exact evidence and adds a warning. Questions and retrieved documents are untrusted data in the model prompt. This conservative lexical guard is not a formal proof of entailment; review is still required before production use.

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

The default configuration is `LLM_MODE=mock` and `FAQ_EMBEDDING_PROVIDER=local_hash`; no API key, model download, or network access is needed to start or test. The local hash embedder is deterministic and intended for offline demonstrations, not production semantic quality. The `.env` file is ignored by Git.

## Run

```bash
source .venv/bin/activate
uvicorn ecom_assistant.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. OpenAPI documentation is at `/docs`; health checks are available at `/api/v1/health` and `/health`. On startup, SQLite tables and catalog rows are initialized, then catalog specifications and `data/faq_documents.json` are reconciled into Chroma. The default database is `data/ecom_assistant.db`; the default vector store is `data/chroma`. Generated database and vector-store files are not tracked.

## Product FAQ RAG

Ask a catalog-grounded question:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/faq/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What processor and RAM are listed?","product_id":"LAP-AST-014"}'
```

The response includes `status` (`grounded` or `insufficient_evidence`), an answer, source references, retrieved snippets with product/document metadata, and warnings. An unknown product ID returns 404; a Chroma or embedding-provider failure returns 503; malformed input returns 422. Insufficient evidence returns an explicit catalog-insufficiency message with no cited sources. No cart is changed by this endpoint.

Product documents are built from SQLite catalog descriptions and specifications. FAQ and warranty records are supplied in `data/faq_documents.json`; all warranty terms there are explicitly fictional demonstration content, not manufacturer claims. Re-ingestion uses stable IDs and content hashes to update changed documents, skips unchanged embeddings, and removes stale documents managed by this application. To refresh the index manually after changing source data:

```bash
python -m ecom_assistant.rag.ingest
```

Chroma persists locally under `FAQ_PERSIST_DIRECTORY` (default `data/chroma`). Set `FAQ_EMBEDDING_PROVIDER=openai` and `FAQ_EMBEDDING_MODEL=text-embedding-3-small` to use OpenAI embeddings; this requires a valid `OPENAI_API_KEY`, network access, and may incur API charges. This setting is independent of `LLM_MODE` and `FAQ_CHAT_MODEL`. To enable ChatOpenAI phrasing, set `LLM_MODE=openai`, `FAQ_CHAT_MODEL=gpt-4o-mini`, and `OPENAI_API_KEY`. The API key is required for either OpenAI provider. Generated wording must be an exact contiguous excerpt from retrieved evidence; otherwise the service returns exact source text and a warning. OpenAI calls may send the question and retrieved evidence to the configured provider.

Using `local_hash` requires no model files. OpenAI embeddings are hosted and require no local model download. To use a local trained embedding model instead, install and configure a compatible local LangChain embeddings provider and download its model explicitly; the local hash provider remains the offline default.

The product catalog, stock quantities, delivery estimates, and supplied warranty entries are fictional sample data for development and demonstration only. Prices are in INR. The FAQ endpoint does not create orders or process payments.

## Simulated Cart

Cart mutations require two requests. Proposing an operation stores a pending proposal but leaves the cart unchanged. The proposal records a UUID, operation, product ID, quantity, server-catalog unit price, projected total, cart revision, relevant inventory snapshot, context fingerprint, and expiration time. The client must explicitly post `{"confirm":true}` to the confirmation route containing that exact cart ID and proposal ID. Confirmation atomically consumes only a pending, unexpired proposal, rechecks the catalog price, inventory, and cart context, then applies the exact operation once. A replay returns `already_confirmed`; expired proposals return `410`; stale or rejected proposals return `409`. Neither endpoint creates an order or reserves stock in the catalog.

Example flow:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/cart
```

Use the returned `cart_id` to propose an add (replace `CART_ID` with that ID):

```bash
curl -X POST http://127.0.0.1:8000/api/v1/cart/CART_ID/proposals \
  -H 'Content-Type: application/json' \
  -d '{"operation":"add","product_id":"LAP-AST-017","quantity":1}'
```

The response contains a `proposal_id` and calculated `projected_total_inr`. The cart remains unchanged until the client explicitly confirms that same ID:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/cart/CART_ID/confirmations/PROPOSAL_ID \
  -H 'Content-Type: application/json' \
  -d '{"confirm":true}'
```

`POST /api/v1/cart/{cart_id}/proposals` accepts `add`, `update`, or `remove`. Add/update require a quantity from 1 to 50; remove must omit quantity. Price is always read from the catalog; client-supplied price fields are rejected. Current sample stock is rechecked both when proposing and confirming. Confirmed cart quantities across other carts are counted as simulated allocations to avoid exceeding sample stock; catalog stock itself is never decremented. Missing inventory is `unknown`, not zero stock.

| Method | Path | Behavior |
|---|---|---|
| `POST` | `/api/v1/cart` | Create an empty simulated cart |
| `GET` | `/api/v1/cart/{cart_id}` | Read items and server-calculated totals |
| `POST` | `/api/v1/cart/{cart_id}/proposals` | Persist a validated pending mutation |
| `POST` | `/api/v1/cart/{cart_id}/confirmations/{proposal_id}` | Explicitly confirm one exact proposal |

Cart IDs are not user-authenticated or isolated; this is a single-service demonstration, not secure multi-user storage. Proposal state and cart items are persisted in SQLite. The default proposal lifetime is 300 seconds, configurable with `CART_PROPOSAL_TTL_SECONDS` (30-3600). SQLite serializes writers; production multi-process deployments should use a database with robust row-level locking before treating stock allocations as authoritative.

Two additional fictional records, `LAP-AST-017` and `LAP-SOL-015`, were added without changing the original eight seed records. Their sample prices are ₹58,990 and ₹57,990; both have 16 GB RAM and estimated battery life above 10 hours. They make the budget/RAM search example demonstrable, but are not live listings or a recommendation.

## Product Search, Comparison, Inventory, and Budget

List products with filters and pagination:

```bash
curl 'http://127.0.0.1:8000/api/v1/products?max_price_inr=60000&min_ram_gb=16&in_stock=true&sort_by=price_asc'
```

This sample query returns the two new fictional rows. Search also accepts `query`, `brand`, `category`, `min_price_inr`, `min_storage_gb`, `limit`, `offset`, and `sort_by` (`relevance`, `price_asc`, `price_desc`, `battery_desc`, or `name`). `in_stock=false` selects products with an explicit zero-stock record; products with missing inventory are reported as unknown and excluded. Empty results return `200` with an empty `products` list. Invalid ranges or unsupported sort fields return `422`.

| Method | Path | Behavior |
|---|---|---|
| `GET` | `/api/v1/products` | Search/filter and paginate catalog products |
| `GET` | `/api/v1/products/{product_id}` | Product details and missing-specification indicators |
| `POST` | `/api/v1/products/compare` | Compare 2-8 unique catalog IDs in request order |
| `GET` | `/api/v1/inventory/{product_id}` | Sample stock status and delivery estimate |
| `POST` | `/api/v1/budget/validate` | Calculate catalog unit price × quantity against budget |

Compare request:

```json
{"product_ids": ["LAP-AST-017", "LAP-SOL-015"]}
```

Budget request:

```json
{"product_id": "LAP-AST-017", "budget_inr": "60000.00", "quantity": 1}
```

The budget service does not apply discounts. Prices, stock and delivery are database-backed sample values. Missing product IDs return `404`; incomplete text specifications are returned as null with `specifications_complete=false` and a list of missing fields.

## Shopping Recommendations

The workflow extracts requirements, searches the selected catalog category, compares returned IDs, checks inventory, validates budget, then ranks only strict matches. Products that miss explicit hard constraints appear separately under `alternatives` with the exact relaxed constraints; they are never labeled strict recommendations. The deterministic mock parser recognizes common categories, numeric budget/RAM/storage/battery phrases, common CPU names, in-stock wording, and a small list of preference phrases. Unrecognized or missing product-category requests ask for clarification. Structured `preferences` override only fields explicitly supplied in that object.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/shopping/recommend \
  -H 'Content-Type: application/json' \
  -d '{"customer_request":"laptop under ₹60,000 with at least 16 GB RAM and good battery life"}'
```

The two strict matches in the seeded fictional catalog are `LAP-AST-017` (₹58,990, 16 GB RAM) and `LAP-SOL-015` (₹57,990, 16 GB RAM). Higher-priced or lower-RAM products are returned only as alternatives with violations listed. Amounts and inventory are read from SQLite sample rows. The response includes extracted requirements, candidate IDs, product/comparison details, inventory, budget results, score factors, reasons/trade-offs, warnings, errors, execution metadata, and stage trace. This endpoint never creates a cart.

## Conversational Shopping API

`POST /api/v1/shopping/chat` routes one message to the existing catalog, recommendation, FAQ, or cart services. Intent routing is deterministic in both mock and OpenAI modes; OpenAI mode is used only by the existing requirement extractor inside recommendations. Supported actions are product search, recommendation, comparison, inventory, budget validation, FAQ, cart view, cart proposal, cart confirmation, clarification, and unknown-intent recovery. Each graph turn has bounded routing: intent detection plus one capability handler. The returned trace contains operational event names only, not prompts or private chain-of-thought.

Routing is phrase-based, not arbitrary-paraphrase language understanding. The compact evaluation tests cover the documented common phrasings and recovery cases; an unsupported paraphrase may be routed to `unknown` or clarification. “Which is cheaper?” compares prior products; “show cheaper options” searches below prior candidates without dropping stored hard constraints.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/shopping/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Find laptops under ₹60,000 with at least 16 GB RAM"}'
```

Run the complete mock/offline walkthrough without touching the default database or Chroma files:

```bash
python scripts/demo_conversation.py
```

The script creates a temporary SQLite database and Chroma directory, then demonstrates search, recommendation, comparison, a sourced FAQ, no-results recovery, a pending cart proposal with an unchanged cart, and explicit proposal confirmation. It makes no paid API calls and deletes the temporary files when it exits.

The response contains `session_id`, `intent`, `status`, `answer`, structured product/comparison/recommendation/inventory/budget/FAQ/cart/proposal results where relevant, clarification/recovery messages, and an operational trace. For example, a search response returns actual catalog `products`; a comparison request naming two product IDs returns ordered `comparison`; a FAQ response returns the existing RAG `sources`; no-result recovery returns `status: no_results` and suggestions.

Follow-ups pass the returned `session_id`, for example `{"message":"Compare those two","session_id":"<returned UUID>"}`. SQLite persists only the last product IDs, structured requirements, active cart ID, last intent, and a proposal reference; it does not store a transcript. Context expires after `CONVERSATION_CONTEXT_TTL_SECONDS` (default 3600; 60-86400). Expired context cannot resolve pronouns; complete new requests still run. New stated budget/specification values replace the old values and are revalidated.

Cart actions require a `cart_id`. A chat add/update/remove creates only a pending proposal and returns its exact `proposal_id`; the cart stays unchanged. A conversational confirmation requires an explicit confirm/approve message, the matching cart ID, and exact proposal ID (via request body or an unambiguous single pending proposal reference). Generic “okay”, negated confirmation, an ambiguous product, or multiple pending proposals cannot mutate the cart. The dedicated explicit cart confirmation endpoint remains available as documented above.

## Test

```bash
source .venv/bin/activate
pytest
```

## Layout

```text
data/catalog.csv  Fictional seed catalog
data/faq_documents.json  Fictional FAQ and warranty source records
tests/            Startup, health, database, and seeding checks
.
├── data/
│   ├── catalog.csv                 # fictional laptops and sample inventory
│   └── faq_documents.json          # fictional FAQ and warranty source records
├── src/ecom_assistant/
│   ├── api/                        # FastAPI routes and dependencies
│   ├── db/                         # SQLAlchemy models, sessions, initialization
│   ├── rag/                        # document conversion, embeddings, Chroma service
│   ├── repositories/               # persistence boundary for future services
│   ├── schemas/                    # Pydantic API contracts
│   ├── services/                   # CSV import and deterministic catalog operations
│   ├── services/cart.py            # transactional simulated cart lifecycle
│   ├── tools/catalog.py            # four typed, read-only LangChain catalog tools
│   ├── workflow/requirements.py    # mock/live Pydantic requirement extraction
│   ├── workflow/graph.py           # typed state, nodes, routing, and ranking
│   ├── workflow/conversation.py    # deterministic multi-capability conversation graph
│   ├── schemas/conversation.py     # validated chat API contracts
│   ├── config.py
│   ├── logging_config.py
│   └── main.py
├── tests/                          # startup, catalog, workflow, cart, and FAQ tests
├── scripts/demo_conversation.py    # disposable mock-mode end-to-end walkthrough
├── .env.example
├── pyproject.toml
└── requirements.txt
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `APP_NAME` | `E-Commerce Shopping Assistant` | FastAPI application title |
| `APP_ENV` | `development` | Environment label |
| `LLM_MODE` | `mock` | `mock` for offline operation; `openai` requires a key |
| `OPENAI_API_KEY` | unset | Secret required by configured OpenAI modes; never returned by the API |
| `DATABASE_URL` | `sqlite:///./data/ecom_assistant.db` | SQLite database URL |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `FAQ_SOURCE_PATH` | `data/faq_documents.json` | Fictional FAQ source JSON |
| `FAQ_PERSIST_DIRECTORY` | `data/chroma` | Persistent Chroma directory |
| `FAQ_COLLECTION_NAME` | `product_faq` | Chroma collection |
| `FAQ_EMBEDDING_PROVIDER` | `local_hash` | `local_hash` or `openai`, independent of chat mode |
| `FAQ_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model when selected |
| `FAQ_CHAT_MODEL` | `gpt-4o-mini` | ChatOpenAI model when `LLM_MODE=openai` |
| `CART_PROPOSAL_TTL_SECONDS` | `300` | Pending cart proposal lifetime (30-3600 seconds) |
| `CONVERSATION_CONTEXT_TTL_SECONDS` | `3600` | Persisted structured chat context lifetime (60-86400 seconds) |

`LLM_MODE=openai` enables ChatOpenAI requirement extraction using `FAQ_CHAT_MODEL` and requires `OPENAI_API_KEY`. It performs network calls and may incur provider charges. OpenAI mode is not required for offline catalog or recommendation tests. Live model behavior is tested with injected fake runnables; the test suite does not call a paid external service.

Only SQLite URLs are accepted. `DATABASE_URL` and other settings can be overridden in the environment or `.env`.

## API Reference

The API is served at `http://127.0.0.1:8000`; Swagger UI is at `/docs` and OpenAPI JSON is at `/openapi.json`.

| Method | Path | Behavior |
|---|---|---|
| `GET` | `/api/v1/health` | Database-backed readiness and configured mode |
| `GET` | `/health` | Convenience health endpoint |
| `GET` | `/api/v1/products` | Search/filter product catalog |
| `GET` | `/api/v1/products/{product_id}` | Product details |
| `POST` | `/api/v1/products/compare` | Multi-product comparison |
| `GET` | `/api/v1/inventory/{product_id}` | Sample inventory and delivery |
| `POST` | `/api/v1/budget/validate` | Deterministic budget check |
| `POST` | `/api/v1/shopping/recommend` | Stateful LangGraph recommendation workflow |
| `POST` | `/api/v1/shopping/chat` | Intent-routed conversational assistant over existing capabilities |
| `POST` | `/api/v1/cart` | Create simulated cart |
| `GET` | `/api/v1/cart/{cart_id}` | Read simulated cart and totals |
| `POST` | `/api/v1/cart/{cart_id}/proposals` | Propose add/update/remove; does not mutate |
| `POST` | `/api/v1/cart/{cart_id}/confirmations/{proposal_id}` | Explicitly confirm exact pending proposal |
| `POST` | `/api/v1/faq/ask` | Ask using retrieved catalog/FAQ evidence |

The curl request above is a reproducible FAQ example. A real offline run returned a grounded response in this shape (snippet shortened; Chroma distance may vary between dependency/index versions):

```json
{
  "status": "grounded",
  "answer": "Mock/offline answer from retrieved catalog evidence: Product: AsterBook Air 14 ...",
  "sources": [
    {
      "document_id": "seed-faq:warranty-lap-ast-014",
      "product_id": "LAP-AST-014",
      "document_type": "warranty",
      "source": "fictional_faq_documents.json",
      "distance": 0.86516
    }
  ],
  "snippets": [
    {
      "document_id": "seed-faq:warranty-lap-ast-014",
      "product_id": "LAP-AST-014",
      "document_type": "warranty",
      "source": "fictional_faq_documents.json",
      "distance": 0.86516,
      "content": "Product: AsterBook Air 14 ... Fictional sample FAQ only ..."
    }
  ],
  "warnings": []
}
```

An insufficient-evidence response has this shape:

```json
{
  "status": "insufficient_evidence",
  "answer": "The catalog does not contain enough information to answer that question.",
  "sources": [],
  "snippets": [],
  "warnings": []
}
```

For grounded answers, `sources` and `snippets` identify the actual retrieved documents. Retrieval distance and result ordering depend on the persistent vector index, so they are intentionally not represented as fixed example values here.

## Test and Import Checks

```bash
source .venv/bin/activate
pytest
python -c "from ecom_assistant.main import app; print(app.title)"
```

Tests use temporary SQLite files and isolated Chroma directories. The default suite does not require API keys, paid APIs, or downloaded embedding models.

## Known Limitations

- Mock extraction supports a deliberately limited phrase grammar; it may ask for clarification on paraphrases it does not recognize. Use configured OpenAI mode for broader extraction and inspect extracted fields and the trace.
- Live OpenAI extraction is not exercised against a paid external service by automated tests. Timeout, model errors, invalid results, and unsupported hard-constraint output are tested with injected fakes.
- Recommendation ranking is deterministic and based on fictional catalog fields; preference scoring is transparent but intentionally simple. The graph limits candidates to 100.
- Cart endpoints are not authenticated and do not isolate carts by user. Do not expose them as a multi-user service without adding ownership/authentication.
- Conversation context is persistent but minimal and unauthenticated; session IDs are not user ownership or access control.
- Intent routing is deterministic and phrase-based. Complex paraphrases may return `unknown` or clarification rather than being classified by an LLM.
- Inventory values are fictional and confirmation does not reserve or decrement catalog stock; aggregate cart contents are only checked as simulated allocations.
- Recommendation does not mutate the cart, and cart confirmation does not place orders.
- Sample product prices, inventory, delivery estimates, and warranty entries are fictional, not live market data or manufacturer claims.
- `local_hash` retrieval is keyword-feature based and has lower semantic quality than a trained embedding model.
- Exact-excerpt validation limits live-model phrasing and is not a general-purpose semantic entailment system.
- No payment processing, purchase, or external order placement is implemented.