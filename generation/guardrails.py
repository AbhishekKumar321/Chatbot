"""Lightweight input/content guardrails for the generation layer.

These are basic sanitization and validation checks, not a complete defense
against prompt injection or abuse — they reduce obvious failure modes
(empty/oversized queries, control characters, instruction-like content
smuggled in via retrieved documents) rather than guaranteeing safety.
"""

import re

_MAX_QUERY_LENGTH = 1000
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class GuardrailViolation(Exception):
    """Raised when input fails a guardrail check."""


def validate_query(query: str) -> str:
    """Validate and normalize a user query before it reaches the LLM.

    Args:
        query: The raw user-supplied query string.

    Returns:
        The query with surrounding whitespace stripped.

    Raises:
        GuardrailViolation: If the query is empty or exceeds the maximum
            allowed length.
    """
    stripped = query.strip()
    if not stripped:
        raise GuardrailViolation("Query must not be empty")
    if len(stripped) > _MAX_QUERY_LENGTH:
        raise GuardrailViolation(f"Query must not exceed {_MAX_QUERY_LENGTH} characters")
    return stripped


def sanitize_context(text: str) -> str:
    """Strip control characters from retrieved document text before use in a prompt.

    Args:
        text: Raw chunk text retrieved from the vector store.

    Returns:
        The text with control characters removed.
    """
    return _CONTROL_CHARS.sub("", text)
