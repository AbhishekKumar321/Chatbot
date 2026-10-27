"""Async wrapper around a Qdrant collection for storing and searching chunks."""

import asyncio
import logging
import uuid
from collections.abc import Callable, Coroutine
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from ingestion.semantic_chunker import Chunk

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 384
_COLLECTION_NAME = "document_chunks"

EmbeddingFn = Callable[[list[str]], Coroutine[Any, Any, list[list[float]]]]


class VectorStoreError(Exception):
    """Raised when the vector store cannot be reached or an operation fails."""


class VectorStore:
    """Persists and retrieves document chunks via a Qdrant collection.

    Connects to a local Qdrant instance by default (per CLAUDE.md's Docker
    stack); pass `location=":memory:"` for tests or environments without a
    running Qdrant server.
    """

    def __init__(
        self,
        embedding_fn: EmbeddingFn,
        url: str | None = None,
        location: str | None = None,
        path: str | None = None,
        collection_name: str = _COLLECTION_NAME,
    ) -> None:
        """Initialize the vector store client and ensure its collection exists.

        Exactly one of `url`, `location`, or `path` should be given:

        Args:
            embedding_fn: Async function mapping text to embedding vectors,
                used both when upserting chunks and when embedding queries.
            url: URL of a running Qdrant server (matches docker-compose.yml).
            location: Use ":memory:" for an ephemeral in-process store
                (tests only).
            path: Local directory for on-disk persistent storage without a
                server (dev without Docker).
            collection_name: Name of the Qdrant collection to use.

        Raises:
            VectorStoreError: If the Qdrant server is unreachable.
        """
        self._embedding_fn = embedding_fn
        self._collection_name = collection_name
        try:
            if path:
                self._client = QdrantClient(path=path)
            elif location:
                self._client = QdrantClient(location=location)
            else:
                self._client = QdrantClient(url=url or "http://localhost:6333")
            self._ensure_collection()
        except Exception as exc:
            logger.exception("Failed to connect to Qdrant")
            raise VectorStoreError("Vector store unreachable") from exc

    def _ensure_collection(self) -> None:
        try:
            self._client.get_collection(self._collection_name)
        except (UnexpectedResponse, ValueError):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=_EMBEDDING_DIM, distance=Distance.COSINE),
            )

    async def upsert(self, chunks: list[Chunk]) -> None:
        """Embed and persist a batch of chunks.

        Args:
            chunks: Semantic chunks to store.

        Raises:
            VectorStoreError: If embedding or the upsert call fails.
        """
        if not chunks:
            return
        try:
            vectors = await self._embedding_fn([chunk.text for chunk in chunks])
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=chunk.model_dump(),
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
            await asyncio.to_thread(
                self._client.upsert, collection_name=self._collection_name, points=points
            )
        except Exception as exc:
            logger.exception("Failed to upsert %d chunks", len(chunks))
            raise VectorStoreError("Failed to persist chunks") from exc

    async def search(self, query: str, top_k: int = 5) -> list[Chunk]:
        """Find the most relevant stored chunks for a query.

        Args:
            query: The user's natural-language query.
            top_k: Maximum number of chunks to return.

        Returns:
            Up to `top_k` chunks ordered by relevance, most relevant first.

        Raises:
            VectorStoreError: If embedding the query or the search call fails.
        """
        try:
            [query_vector] = await self._embedding_fn([query])
            results = await asyncio.to_thread(
                self._client.query_points,
                collection_name=self._collection_name,
                query=query_vector,
                limit=top_k,
            )
            return [Chunk(**point.payload) for point in results.points if point.payload]
        except Exception as exc:
            logger.exception("Search failed for query: %s", query)
            raise VectorStoreError("Failed to search chunks") from exc
