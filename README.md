# RAG Platform 


A fully open-source Retrieval-Augmented Generation (RAG) system: upload a PDF, ask questions about it, get answers grounded in its content.

## How it works

```
PDF upload -> ingestion -> embeddings -> vector store -> retrieval -> LLM -> streamed answer
```

- **`ingestion/`** — loads a PDF, splits it into sentences, and groups them into semantically coherent chunks.
- **`retrieval/`** — embeds chunks/queries locally (`sentence-transformers`) and stores/searches them in Qdrant.
- **`generation/`** — builds a prompt from retrieved chunks and streams an answer from an LLM (Groq).

Each layer is isolated from the others — see `CLAUDE.md` for the engineering rules behind that separation.

## Stack

- Python 3.11+, FastAPI, Poetry
- Qdrant (vector store, via Docker or local on-disk mode)
- `sentence-transformers` (local embeddings)
- Groq (hosted LLM inference)

## Running it

```bash
poetry install
docker-compose up -d          # starts Qdrant
poetry run fastapi dev main.py
```

Create a `.env` file (see `.env.example`) with your `GROQ_API_KEY`.

## API

- `POST /ingest` — upload a PDF, get back its semantic chunks (also persisted to the vector store)
- `POST /query` — ask a question, get a streamed, context-grounded answer
- `GET /health` — liveness check

Interactive API docs are available at `/docs` once the server is running.

## Tests

```bash
poetry run pytest
poetry run mypy .
poetry run ruff check .
```
