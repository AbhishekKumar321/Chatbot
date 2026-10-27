"""Embedding-based semantic chunking for document ingestion."""

import logging
from collections.abc import Awaitable, Callable

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)

EmbeddingFn = Callable[[list[str]], Awaitable[list[list[float]]]]


class Chunk(BaseModel):
    """A contiguous group of sentences forming one semantic chunk."""

    text: str
    start_sentence_idx: int
    end_sentence_idx: int
    token_count: int


class SemanticChunker:
    """Groups sentences into chunks based on embedding similarity breakpoints.

    Sentences are embedded and compared to their neighbors; a chunk boundary
    is inserted wherever the semantic distance between adjacent sentences
    exceeds a percentile-based dynamic threshold, so the split adapts to each
    document's own writing style instead of using one fixed cutoff.
    """

    def __init__(
        self,
        embedding_fn: EmbeddingFn,
        breakpoint_percentile: float = 95.0,
        min_chunk_size: int = 2,
        max_chunk_size: int = 20,
    ) -> None:
        """Initialize the chunker.

        Args:
            embedding_fn: Async function mapping a list of sentences to a
                list of embedding vectors. Injected so this module stays
                decoupled from any specific embedding model implementation.
            breakpoint_percentile: Percentile of sentence-to-sentence distance
                above which a chunk boundary is inserted (0-100).
            min_chunk_size: Minimum number of sentences per chunk; smaller
                trailing groups are merged into the previous chunk.
            max_chunk_size: Maximum number of sentences per chunk; longer
                runs are force-split even without a semantic breakpoint.

        Raises:
            ValueError: If breakpoint_percentile is out of range, or chunk
                size bounds are invalid.
        """
        if not 0.0 <= breakpoint_percentile <= 100.0:
            raise ValueError("breakpoint_percentile must be between 0 and 100")
        if min_chunk_size < 1 or max_chunk_size < min_chunk_size:
            raise ValueError("max_chunk_size must be >= min_chunk_size >= 1")

        self._embedding_fn = embedding_fn
        self._breakpoint_percentile = breakpoint_percentile
        self._min_chunk_size = min_chunk_size
        self._max_chunk_size = max_chunk_size

    async def chunk(self, sentences: list[str]) -> list[Chunk]:
        """Group sentences into semantically coherent chunks.

        Args:
            sentences: Ordered sentences from a single document.

        Returns:
            A list of Chunk objects covering all input sentences in order.
            Returns an empty list if given no sentences.
        """
        if not sentences:
            return []
        if len(sentences) == 1:
            return [self._make_chunk(sentences, 0, 1)]

        embeddings = await self._embedding_fn(sentences)
        distances = self._pairwise_distances(embeddings)
        boundaries = self._find_boundaries(distances)
        return self._build_chunks(sentences, boundaries)

    @staticmethod
    def _pairwise_distances(embeddings: list[list[float]]) -> list[float]:
        vectors = np.array(embeddings)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / np.clip(norms, a_min=1e-10, a_max=None)
        similarities = np.sum(normalized[:-1] * normalized[1:], axis=1)
        return [float(d) for d in (1.0 - similarities)]

    def _find_boundaries(self, distances: list[float]) -> set[int]:
        """Return sentence indices after which a chunk boundary falls."""
        if not distances:
            return set()
        threshold = float(np.percentile(distances, self._breakpoint_percentile))
        return {i for i, dist in enumerate(distances) if dist > threshold}

    def _build_chunks(self, sentences: list[str], boundaries: set[int]) -> list[Chunk]:
        chunks: list[Chunk] = []
        start = 0
        for i in range(len(sentences)):
            at_semantic_boundary = i in boundaries
            at_max_size = (i - start + 1) >= self._max_chunk_size
            is_last = i == len(sentences) - 1
            if is_last or at_max_size or at_semantic_boundary:
                chunks.append(self._make_chunk(sentences, start, i + 1))
                start = i + 1
        return self._merge_small_chunks(chunks)

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        if len(chunks) <= 1:
            return chunks

        merged: list[Chunk] = [chunks[0]]
        for chunk in chunks[1:]:
            previous = merged[-1]
            chunk_size = chunk.end_sentence_idx - chunk.start_sentence_idx
            previous_size = previous.end_sentence_idx - previous.start_sentence_idx
            if chunk_size < self._min_chunk_size or previous_size < self._min_chunk_size:
                merged[-1] = Chunk(
                    text=f"{previous.text} {chunk.text}".strip(),
                    start_sentence_idx=previous.start_sentence_idx,
                    end_sentence_idx=chunk.end_sentence_idx,
                    token_count=previous.token_count + chunk.token_count,
                )
            else:
                merged.append(chunk)
        return merged

    @staticmethod
    def _make_chunk(sentences: list[str], start: int, end: int) -> Chunk:
        text = " ".join(sentences[start:end])
        return Chunk(
            text=text,
            start_sentence_idx=start,
            end_sentence_idx=end,
            token_count=len(text.split()),
        )
