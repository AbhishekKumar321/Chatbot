"""Builds grounded prompts from retrieved context and a user query."""

from generation.guardrails import sanitize_context
from ingestion.semantic_chunker import Chunk

_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not contained in the context, say you don't know — do not guess.

Context:
{context}

Question: {query}

Answer:"""


def build_prompt(query: str, chunks: list[Chunk]) -> str:
    """Construct a grounded prompt from a query and retrieved chunks.

    Args:
        query: The validated user query.
        chunks: Chunks retrieved from the vector store, most relevant first.

    Returns:
        A prompt string instructing the model to answer only from context.
    """
    context = "\n\n".join(sanitize_context(chunk.text) for chunk in chunks)
    return _PROMPT_TEMPLATE.format(context=context or "(no relevant context found)", query=query)
