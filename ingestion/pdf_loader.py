"""PDF text extraction utilities for the ingestion pipeline."""

import asyncio
import logging

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)


class PdfLoadError(Exception):
    """Raised when a PDF file cannot be read or parsed."""


async def load_pdf(file_path: str) -> list[str]:
    """Extract raw text content from a PDF, one entry per page.

    Args:
        file_path: Path to the PDF file on disk.

    Returns:
        A list of strings, each containing the extracted text of one page.

    Raises:
        PdfLoadError: If the file is missing, unreadable, or not a valid PDF.
    """
    try:
        return await asyncio.to_thread(_read_pages, file_path)
    except (PdfReadError, FileNotFoundError, OSError) as exc:
        logger.exception("Failed to load PDF at %s", file_path)
        raise PdfLoadError(f"Could not load PDF: {file_path}") from exc


def _read_pages(file_path: str) -> list[str]:
    reader = PdfReader(file_path)
    return [page.extract_text() or "" for page in reader.pages]
