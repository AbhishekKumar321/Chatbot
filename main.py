"""FastAPI application entrypoint for the RAG platform."""

import logging
import os
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from generation.guardrails import GuardrailViolation, validate_query
from generation.llm_client import GroqClient, LlmClientError
from generation.prompt_builder import build_prompt
from ingestion.pipeline import IngestionError, ingest_pdf
from ingestion.semantic_chunker import Chunk
from retrieval.embedding_model import embed
from retrieval.vector_store import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="RAG Platform")

_vector_store = VectorStore(
    embedding_fn=embed,
    url=os.environ.get("QDRANT_URL"),
    path=None if os.environ.get("QDRANT_URL") else "./.qdrant_data",
)
_llm_client = GroqClient()


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    query: str


@app.get("/health")
async def health() -> dict[str, str]:
    """Report service liveness.

    Returns:
        A status payload indicating the service is running.
    """
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile) -> list[Chunk]:
    """Ingest an uploaded PDF: chunk it semantically and persist it for retrieval.

    Args:
        file: The uploaded PDF file.

    Returns:
        The list of semantic chunks extracted from and stored for the document.

    Raises:
        HTTPException: With status 400 if ingestion or storage fails.
    """
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / (file.filename or "upload.pdf")
            tmp_path.write_bytes(await file.read())
            chunks = await ingest_pdf(str(tmp_path), embedding_fn=embed)
        await _vector_store.upsert(chunks)
        return chunks
    except (IngestionError, VectorStoreError) as exc:
        logger.exception("Ingestion request failed for %s", file.filename)
        raise HTTPException(status_code=400, detail="Failed to ingest document") from exc


@app.post("/query")
async def query(request: QueryRequest) -> StreamingResponse:
    """Answer a question using retrieved context, streaming the LLM response.

    Args:
        request: The user's query.

    Returns:
        A streaming plain-text response with the generated answer.

    Raises:
        HTTPException: With status 400 for invalid queries or if retrieval
            fails; the LLM itself is streamed so mid-stream failures are
            surfaced in-band rather than as an HTTP error.
    """
    try:
        validated_query = validate_query(request.query)
        chunks = await _vector_store.search(validated_query)
    except GuardrailViolation as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except VectorStoreError as exc:
        logger.exception("Retrieval failed for query: %s", request.query)
        raise HTTPException(status_code=400, detail="Failed to retrieve context") from exc

    prompt = build_prompt(validated_query, chunks)

    async def stream() -> AsyncIterator[str]:
        try:
            async for token in _llm_client.generate_stream(prompt):
                yield token
        except LlmClientError as exc:
            logger.exception("Generation failed for query: %s", request.query)
            yield f"\n[error: {exc}]"

    return StreamingResponse(stream(), media_type="text/plain")
