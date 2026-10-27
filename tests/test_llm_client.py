"""Unit tests for generation.llm_client, using a mocked HTTP transport."""

import httpx
import pytest

from generation.llm_client import GroqClient, LlmClientError


def _mock_transport(response_lines: list[bytes], status_code: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, stream=httpx.ByteStream(b"\n".join(response_lines)))

    return httpx.MockTransport(handler)


def _sse_chunk(content: str) -> bytes:
    payload = f'{{"choices": [{{"delta": {{"content": "{content}"}}}}]}}'
    return f"data: {payload}".encode()


@pytest.mark.asyncio
async def test_generate_stream_yields_tokens() -> None:
    lines = [_sse_chunk("Hello"), _sse_chunk(" world"), b"data: [DONE]"]
    client = GroqClient(api_key="test-key", transport=_mock_transport(lines))

    tokens = [token async for token in client.generate_stream("hi")]

    assert tokens == ["Hello", " world"]


@pytest.mark.asyncio
async def test_generate_stream_raises_on_http_error() -> None:
    client = GroqClient(api_key="test-key", transport=_mock_transport([b"error"], status_code=500))

    with pytest.raises(LlmClientError):
        async for _ in client.generate_stream("hi"):
            pass


@pytest.mark.asyncio
async def test_generate_stream_raises_on_auth_error() -> None:
    client = GroqClient(
        api_key="test-key", transport=_mock_transport([b"unauthorized"], status_code=401)
    )

    with pytest.raises(LlmClientError):
        async for _ in client.generate_stream("hi"):
            pass


@pytest.mark.asyncio
async def test_generate_stream_raises_on_rate_limit() -> None:
    client = GroqClient(
        api_key="test-key", transport=_mock_transport([b"rate limited"], status_code=429)
    )

    with pytest.raises(LlmClientError):
        async for _ in client.generate_stream("hi"):
            pass


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(LlmClientError):
        GroqClient()
