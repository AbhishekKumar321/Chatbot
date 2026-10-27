"""Sentence segmentation utilities used prior to semantic chunking."""

from syntok import segmenter


def split_into_sentences(text: str) -> list[str]:
    """Split raw text into a flat list of sentences.

    Args:
        text: Raw text content, typically a single PDF page.

    Returns:
        A list of sentence strings with surrounding whitespace stripped.
        Empty input yields an empty list.
    """
    if not text.strip():
        return []

    sentences: list[str] = []
    for paragraph in segmenter.process(text):
        for sentence in paragraph:
            joined = "".join(token.spacing + token.value for token in sentence).strip()
            if joined:
                sentences.append(joined)
    return sentences
