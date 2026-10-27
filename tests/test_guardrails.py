"""Unit tests for generation.guardrails."""

import pytest

from generation.guardrails import GuardrailViolation, sanitize_context, validate_query


def test_validate_query_strips_whitespace() -> None:
    assert validate_query("  hello  ") == "hello"


def test_validate_query_rejects_empty() -> None:
    with pytest.raises(GuardrailViolation):
        validate_query("   ")


def test_validate_query_rejects_too_long() -> None:
    with pytest.raises(GuardrailViolation):
        validate_query("a" * 1001)


def test_sanitize_context_strips_control_characters() -> None:
    assert sanitize_context("hello\x00world") == "helloworld"
