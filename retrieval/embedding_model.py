"""Local BGE embedding model, loaded once and reused across requests."""

import asyncio
import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class EmbeddingModelError(Exception):
    """Raised when the embedding model fails to load or run inference."""


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    logger.info("Loading embedding model %s", _MODEL_NAME)
    return SentenceTransformer(_MODEL_NAME)


async def embed(sentences: list[str]) -> list[list[float]]:
    """Embed a batch of sentences using the local BGE model.

    Args:
        sentences: Sentences to embed.

    Returns:
        A list of embedding vectors, one per input sentence, in order.

    Raises:
        EmbeddingModelError: If the model fails to load or run inference.
    """
    if not sentences:
        return []
    try:
        return await asyncio.to_thread(_embed_sync, sentences)
    except Exception as exc:
        logger.exception("Embedding inference failed for batch of size %d", len(sentences))
        raise EmbeddingModelError("Failed to generate embeddings") from exc


def _embed_sync(sentences: list[str]) -> list[list[float]]:
    model = _get_model()
    vectors = model.encode(sentences, normalize_embeddings=True)
    return [[float(x) for x in vector] for vector in vectors]
