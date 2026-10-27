"""Unit tests for ingestion.semantic_chunker, using an injected fake embedder."""

import pytest

from ingestion.semantic_chunker import Chunk, EmbeddingFn, SemanticChunker


def _fake_embedder(vectors: dict[str, list[float]]) -> EmbeddingFn:
    async def embed(sentences: list[str]) -> list[list[float]]:
        return [vectors[s] for s in sentences]

    return embed


@pytest.mark.asyncio
async def test_empty_input_returns_empty_list() -> None:
    chunker = SemanticChunker(embedding_fn=_fake_embedder({}))
    assert await chunker.chunk([]) == []


@pytest.mark.asyncio
async def test_single_sentence_returns_one_chunk() -> None:
    chunker = SemanticChunker(embedding_fn=_fake_embedder({"Hello.": [1.0, 0.0]}))
    result = await chunker.chunk(["Hello."])
    assert result == [Chunk(text="Hello.", start_sentence_idx=0, end_sentence_idx=1, token_count=1)]


@pytest.mark.asyncio
async def test_splits_at_semantic_breakpoint() -> None:
    sentences = ["Cats are mammals.", "Cats purr softly.", "Stock prices fell today."]
    vectors = {
        "Cats are mammals.": [1.0, 0.0],
        "Cats purr softly.": [0.95, 0.05],
        "Stock prices fell today.": [0.0, 1.0],
    }
    chunker = SemanticChunker(
        embedding_fn=_fake_embedder(vectors),
        breakpoint_percentile=50.0,
        min_chunk_size=1,
    )
    result = await chunker.chunk(sentences)

    assert len(result) == 2
    assert result[0].text == "Cats are mammals. Cats purr softly."
    assert result[1].text == "Stock prices fell today."


@pytest.mark.asyncio
async def test_max_chunk_size_forces_split() -> None:
    sentences = [f"Sentence {i}." for i in range(5)]
    vectors = {s: [1.0, 0.0] for s in sentences}
    chunker = SemanticChunker(
        embedding_fn=_fake_embedder(vectors),
        max_chunk_size=2,
        min_chunk_size=1,
    )
    result = await chunker.chunk(sentences)

    assert [c.end_sentence_idx - c.start_sentence_idx for c in result] == [2, 2, 1]


@pytest.mark.asyncio
async def test_small_trailing_chunk_is_merged() -> None:
    sentences = ["A.", "B.", "C."]
    vectors = {"A.": [1.0, 0.0], "B.": [1.0, 0.0], "C.": [0.0, 1.0]}
    chunker = SemanticChunker(
        embedding_fn=_fake_embedder(vectors),
        breakpoint_percentile=1.0,
        min_chunk_size=2,
    )
    result = await chunker.chunk(sentences)

    assert len(result) == 1
    assert result[0].text == "A. B. C."


def test_invalid_percentile_raises() -> None:
    with pytest.raises(ValueError):
        SemanticChunker(embedding_fn=_fake_embedder({}), breakpoint_percentile=150.0)


def test_invalid_chunk_size_bounds_raises() -> None:
    with pytest.raises(ValueError):
        SemanticChunker(embedding_fn=_fake_embedder({}), min_chunk_size=5, max_chunk_size=2)
