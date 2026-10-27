"""Unit tests for generation.prompt_builder."""

from generation.prompt_builder import build_prompt
from ingestion.semantic_chunker import Chunk


def test_build_prompt_includes_context_and_query() -> None:
    chunks = [
        Chunk(text="The sky is blue.", start_sentence_idx=0, end_sentence_idx=1, token_count=4)
    ]
    prompt = build_prompt("What color is the sky?", chunks)

    assert "The sky is blue." in prompt
    assert "What color is the sky?" in prompt


def test_build_prompt_handles_no_chunks() -> None:
    prompt = build_prompt("Anything?", [])
    assert "(no relevant context found)" in prompt
