"""End-to-end PDF ingestion orchestration: load, split, chunk."""

import logging

from ingestion.pdf_loader import PdfLoadError, load_pdf
from ingestion.semantic_chunker import Chunk, EmbeddingFn, SemanticChunker
from ingestion.sentence_splitter import split_into_sentences

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """Raised when the ingestion pipeline fails to process a document."""


async def ingest_pdf(file_path: str, embedding_fn: EmbeddingFn) -> list[Chunk]:
    """Load a PDF, split it into sentences, and group it into semantic chunks.

    Args:
        file_path: Path to the PDF file on disk.
        embedding_fn: Async function mapping sentences to embedding vectors,
            supplied by the retrieval layer.

    Returns:
        A list of semantically coherent Chunk objects spanning the document.

    Raises:
        IngestionError: If the PDF cannot be loaded or chunking fails.
    """
    try:
        pages = await load_pdf(file_path)
    except PdfLoadError as exc:
        raise IngestionError(f"Ingestion failed while loading: {file_path}") from exc

    sentences = [sentence for page in pages for sentence in split_into_sentences(page)]
    if not sentences:
        logger.warning("No extractable text found in %s", file_path)
        return []

    try:
        chunker = SemanticChunker(embedding_fn=embedding_fn)
        return await chunker.chunk(sentences)
    except Exception as exc:
        logger.exception("Semantic chunking failed for %s", file_path)
        raise IngestionError(f"Ingestion failed while chunking: {file_path}") from exc
