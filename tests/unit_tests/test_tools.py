"""Behaviour tests with the HTTP layer mocked: payloads, defaults and errors."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException

from langchain_sofya import (
    SofyaAPIWrapper,
    SofyaExtract,
    SofyaFetch,
    SofyaResearch,
    SofyaSearch,
    __version__,
)


class _Recorder:
    """Swaps httpx's transport for one that records requests and replies."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, status: int = 200, body: Any = None):
        self.requests: list[httpx.Request] = []
        self.body = {"ok": True, "credits_used": 1} if body is None else body
        self.status = status
        transport = httpx.MockTransport(self._handle)
        real_client, real_async = httpx.Client, httpx.AsyncClient

        def client(*a: Any, **kw: Any) -> httpx.Client:
            return real_client(*a, transport=transport, **kw)

        def async_client(*a: Any, **kw: Any) -> httpx.AsyncClient:
            return real_async(*a, transport=transport, **kw)

        monkeypatch.setattr(httpx, "Client", client)
        monkeypatch.setattr(httpx, "AsyncClient", async_client)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self.body, str):
            return httpx.Response(self.status, text=self.body)
        return httpx.Response(self.status, json=self.body)

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]

    @property
    def last_json(self) -> dict[str, Any]:
        return json.loads(self.last.content)


def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOFYA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SOFYA_API_KEY"):
        SofyaSearch()


def test_key_from_env_and_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOFYA_API_KEY", "sofya_live_env")
    monkeypatch.setenv("SOFYA_BASE_URL", "http://localhost:9999/")
    w = SofyaAPIWrapper()
    assert w.api_key.get_secret_value() == "sofya_live_env"
    assert w._url("/v1/search") == "http://localhost:9999/v1/search"


def test_key_is_secret() -> None:
    tool = SofyaSearch(api_key="sofya_live_secret")
    assert "sofya_live_secret" not in repr(tool)


def test_search_payload_headers_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _Recorder(monkeypatch)
    tool = SofyaSearch(api_key="k", max_results=5, include_answer=True, topic="news")
    out = tool.invoke({"query": "q", "max_results": 3, "include_domains": ["github.com"]})
    assert out == {"ok": True, "credits_used": 1}
    assert str(rec.last.url) == "https://sofya.co/v1/search"
    assert rec.last.headers["authorization"] == "Bearer k"
    assert rec.last.headers["user-agent"] == f"langchain-sofya/{__version__}"
    # the call's own value wins, tool defaults fill the rest, unset keys are omitted
    assert rec.last_json == {
        "query": "q",
        "max_results": 3,
        "include_answer": True,
        "include_domains": ["github.com"],
        "topic": "news",
    }


def test_search_minimal_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _Recorder(monkeypatch)
    SofyaSearch(api_key="k").invoke({"query": "q"})
    assert rec.last_json == {"query": "q"}


def test_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _Recorder(monkeypatch)
    SofyaFetch(api_key="k").invoke({"urls": ["https://a.test", "https://b.test"]})
    assert str(rec.last.url).endswith("/v1/fetch")
    assert rec.last_json == {"urls": ["https://a.test", "https://b.test"]}
    SofyaFetch(api_key="k", include_raw_html=True).invoke({"urls": ["https://a.test"]})
    assert rec.last_json == {"urls": ["https://a.test"], "include_raw_html": True}


def test_fetch_rejects_too_many_urls() -> None:
    tool = SofyaFetch(api_key="k")
    with pytest.raises(Exception):
        tool.invoke({"urls": [f"https://x{i}.test" for i in range(11)]})


def test_extract(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _Recorder(monkeypatch)
    SofyaExtract(api_key="k").invoke({"url": "https://a.test", "prompt": "p"})
    assert str(rec.last.url).endswith("/v1/extract")
    assert rec.last_json == {"url": "https://a.test", "prompt": "p"}


def test_research(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _Recorder(monkeypatch)
    SofyaResearch(api_key="k", max_sources=10).invoke({"query": "q", "topic": "news"})
    assert str(rec.last.url).endswith("/v1/research")
    assert rec.last_json == {"query": "q", "topic": "news", "max_sources": 10}


async def test_async(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _Recorder(monkeypatch)
    out = await SofyaSearch(api_key="k").ainvoke({"query": "q"})
    assert out["ok"] is True
    assert rec.last_json == {"query": "q"}


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, {"detail": "Invalid API key"}, "missing or invalid"),
        (402, {"detail": "Insufficient credits"}, "out of credits"),
        (429, {"detail": "Too many requests"}, "rate limit"),
        (422, {"detail": [{"msg": "bad"}]}, "rejected the request"),
        (500, "boom", "HTTP 500"),
    ],
)
def test_errors_become_tool_messages(
    monkeypatch: pytest.MonkeyPatch, status: int, body: Any, expected: str
) -> None:
    _Recorder(monkeypatch, status=status, body=body)
    tool = SofyaSearch(api_key="k")
    with pytest.raises(ToolException, match=expected):
        tool._run(query="q")
    # Through invoke with a tool call, the error is returned to the model.
    msg = tool.invoke({"name": tool.name, "args": {"query": "q"}, "id": "1", "type": "tool_call"})
    assert isinstance(msg, ToolMessage)
    assert expected in str(msg.content)


def test_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_client = httpx.Client

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(
        httpx, "Client", lambda *a, **kw: real_client(*a, transport=httpx.MockTransport(boom), **kw)
    )
    with pytest.raises(ToolException, match="request failed"):
        SofyaSearch(api_key="k")._run(query="q")
