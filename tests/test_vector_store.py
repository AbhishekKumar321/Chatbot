"""Unit tests for retrieval.vector_store, using an in-memory Qdrant instance."""

import pytest

from ingestion.semantic_chunker import Chunk
from retrieval.vector_store import VectorStore

_VECTORS: dict[str, list[float]] = {
    "Cats are small mammals.": [1.0, 0.0] + [0.0] * 382,
    "Stock markets fell sharply.": [0.0, 1.0] + [0.0] * 382,
}


async def _fake_embedder(texts: list[str]) -> list[list[float]]:
    return [_VECTORS[t] for t in texts]


def _make_store() -> VectorStore:
    return VectorStore(
        embedding_fn=_fake_embedder,
        location=":memory:",
        collection_name="test_chunks",
    )


@pytest.mark.asyncio
async def test_upsert_and_search_returns_closest_chunk() -> None:
    store = _make_store()
    chunks = [
        Chunk(
            text="Cats are small mammals.", start_sentence_idx=0, end_sentence_idx=1, token_count=4
        ),
        Chunk(
            text="Stock markets fell sharply.",
            start_sentence_idx=1,
            end_sentence_idx=2,
            token_count=4,
        ),
    ]
    await store.upsert(chunks)

    results = await store.search("Cats are small mammals.", top_k=1)

    assert len(results) == 1
    assert results[0].text == "Cats are small mammals."


@pytest.mark.asyncio
async def test_upsert_empty_list_is_a_noop() -> None:
    store = _make_store()
    await store.upsert([])
