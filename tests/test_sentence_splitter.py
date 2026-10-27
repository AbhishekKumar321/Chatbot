"""Unit tests for ingestion.sentence_splitter."""

from ingestion.sentence_splitter import split_into_sentences


def test_empty_text_returns_empty_list() -> None:
    assert split_into_sentences("") == []
    assert split_into_sentences("   ") == []


def test_splits_multiple_sentences() -> None:
    result = split_into_sentences("This is one sentence. This is another one.")
    assert result == ["This is one sentence.", "This is another one."]
