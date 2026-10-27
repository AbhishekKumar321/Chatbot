"""Async client for Groq's hosted chat completions API."""

import json
import logging
import os
from collections.abc import AsyncIterator

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
_DEFAULT_MODEL = "openai/gpt-oss-120b"
_TIMEOUT_SECONDS = 60.0


class LlmClientError(Exception):
    """Raised when the LLM API is unreachable or returns an error."""


class GroqClient:
    """Thin async wrapper over Groq's OpenAI-compatible chat completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        model: str = _DEFAULT_MODEL,
        transport: httpx.AsyncBaseTransport | None = None,
        ca_bundle: str | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            api_key: Groq API key. Falls back to the ``GROQ_API_KEY``
                environment variable when not provided.
            base_url: Base URL of the Groq API.
            model: Name of the hosted model to use for generation.
            transport: Optional custom transport, used to inject a mock
                transport in tests instead of making real HTTP calls.
            ca_bundle: Path to a CA bundle for TLS verification. Falls back
                to the ``REQUESTS_CA_BUNDLE`` environment variable, needed
                when running behind a TLS-inspecting corporate proxy.

        Raises:
            LlmClientError: If no API key is provided or configured.
        """
        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise LlmClientError("GROQ_API_KEY is not set")
        self._api_key = resolved_key
        self._base_url = base_url
        self._model = model
        self._transport = transport
        self._ca_bundle = ca_bundle or os.environ.get("REQUESTS_CA_BUNDLE")

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream a completion for the given prompt, token chunk by chunk.

        Args:
            prompt: The fully-constructed prompt to send to the model.

        Yields:
            Successive text chunks as they are produced by the model.

        Raises:
            LlmClientError: If the Groq API is unreachable, returns a
                non-success response, or an invalid key/rate limit error.
        """
        try:
            async with (
                httpx.AsyncClient(
                    timeout=_TIMEOUT_SECONDS,
                    transport=self._transport,
                    verify=self._ca_bundle or True,
                ) as client,
                client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={
                        "model": self._model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": True,
                    },
                ) as response,
            ):
                if response.status_code == 401:
                    logger.error("Groq API rejected the configured API key")
                    raise LlmClientError("LLM provider authentication failed")
                if response.status_code == 429:
                    logger.error("Groq API rate limit exceeded")
                    raise LlmClientError("LLM provider rate limit exceeded")
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[len("data: ") :]
                    if data == "[DONE]":
                        break
                    payload = json.loads(data)
                    delta = payload["choices"][0]["delta"]
                    chunk = delta.get("content", "")
                    if chunk:
                        yield chunk
        except httpx.HTTPError as exc:
            logger.exception("Groq generation request failed")
            raise LlmClientError("Failed to generate a response from the LLM") from exc
