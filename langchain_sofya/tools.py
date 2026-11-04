"""LangChain tools for Sofya: web search, fetch, extract and research."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator

from langchain_sofya._client import SofyaAPIWrapper

_FRESHNESS = (
    'Only results from this period: "day", "week", "month", "year", '
    'or a date range "YYYY-MM-DD:YYYY-MM-DD".'
)
_TOPIC = '"general" for web search, "news" for news articles.'


class _SofyaTool(BaseTool):
    """Shared setup: an API wrapper built from api_key / SOFYA_API_KEY."""

    api_wrapper: SofyaAPIWrapper = Field(default=None)  # type: ignore[assignment]
    handle_tool_error: bool = True

    @model_validator(mode="before")
    @classmethod
    def _build_wrapper(cls, values: Any) -> Any:
        if isinstance(values, dict) and values.get("api_wrapper") is None:
            values["api_wrapper"] = SofyaAPIWrapper(
                **{k: values.pop(k) for k in ("api_key", "base_url", "timeout") if k in values}
            )
        return values

    def _payload(self, given: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
        """Call arguments win; unset ones fall back to the tool's own settings."""
        out: dict[str, Any] = {}
        for k in keys:
            v = given.get(k)
            if v is None:
                v = getattr(self, k, None)
            if v is not None:
                out[k] = v
        return out


# --------------------------------------------------------------------- search


class SofyaSearchInput(BaseModel):
    """Input for SofyaSearch."""

    query: str = Field(description="What to search the web for.")
    search_depth: Literal["basic", "snippets"] | None = Field(
        default=None,
        description='"basic" returns the text of each result page (2 credits); '
        '"snippets" returns short snippets only (1 credit).',
    )
    max_results: int | None = Field(
        default=None, ge=1, le=20, description="How many results to return, 1-20."
    )
    include_answer: bool | None = Field(
        default=None,
        description="Add a short AI answer written from the results (+2 credits).",
    )
    include_domains: list[str] | None = Field(
        default=None,
        description='Only return results from these domains, e.g. ["github.com"]. Up to 10.',
    )
    exclude_domains: list[str] | None = Field(
        default=None,
        description='Leave out results from these domains, e.g. ["pinterest.com"]. Up to 10.',
    )
    topic: Literal["general", "news"] | None = Field(default=None, description=_TOPIC)
    freshness: str | None = Field(default=None, description=_FRESHNESS)


_SEARCH_KEYS = (
    "query",
    "search_depth",
    "max_results",
    "include_answer",
    "include_domains",
    "exclude_domains",
    "topic",
    "freshness",
)


class SofyaSearch(_SofyaTool):  # type: ignore[override]
    """Search the web with Sofya.

    Returns ranked results with the text of each page (or short snippets), and
    optionally a short AI answer.

    Setup:
        .. code-block:: bash

            pip install -U langchain-sofya
            export SOFYA_API_KEY="sofya_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_sofya import SofyaSearch

            tool = SofyaSearch(max_results=5, include_answer=True)

    Invoke:
        .. code-block:: python

            tool.invoke({"query": "latest stable Python release"})
    """

    name: str = "sofya_search"
    description: str = (
        "Search the web for current information. Returns ranked results with "
        "title, URL and page text, and optionally a short AI answer. Input is a "
        "search query; you can filter by domain, topic (general or news) and "
        "freshness."
    )
    args_schema: type[BaseModel] = SofyaSearchInput

    search_depth: Literal["basic", "snippets"] | None = None
    """Default depth when the call does not set one (API default: basic)."""
    max_results: int | None = None
    """Default number of results, 1-20 (API default: 10)."""
    include_answer: bool | None = None
    """Add a short AI answer by default."""
    include_domains: list[str] | None = None
    """Domains to restrict results to by default."""
    exclude_domains: list[str] | None = None
    """Domains to leave out by default."""
    topic: Literal["general", "news"] | None = None
    """Default topic."""
    freshness: str | None = None
    """Default freshness filter."""

    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = self._payload({"query": query, **kwargs}, _SEARCH_KEYS)
        return self.api_wrapper.post("/v1/search", payload)

    async def _arun(
        self,
        query: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = self._payload({"query": query, **kwargs}, _SEARCH_KEYS)
        return await self.api_wrapper.apost("/v1/search", payload)


# ---------------------------------------------------------------------- fetch


class SofyaFetchInput(BaseModel):
    """Input for SofyaFetch."""

    urls: list[str] = Field(
        min_length=1,
        max_length=10,
        description="URLs to read, up to 10. Web pages and documents (PDF, Word, "
        "Excel, PowerPoint and more).",
    )


class SofyaFetch(_SofyaTool):  # type: ignore[override]
    """Read web pages and documents as clean markdown with Sofya.

    Handles HTML pages and documents such as PDF, DOCX, XLSX and PPTX. Costs
    1 credit per URL; URLs that fail are not charged.

    Instantiate:
        .. code-block:: python

            from langchain_sofya import SofyaFetch

            tool = SofyaFetch()

    Invoke:
        .. code-block:: python

            tool.invoke({"urls": ["https://example.com"]})
    """

    name: str = "sofya_fetch"
    description: str = (
        "Read one or more web pages or documents (PDF, Word, Excel, PowerPoint) "
        "and return their text as clean markdown. Input is a list of up to 10 URLs."
    )
    args_schema: type[BaseModel] = SofyaFetchInput

    include_raw_html: bool = False
    """Also return each page's raw HTML."""

    def _run(
        self,
        urls: list[str],
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"urls": urls}
        if self.include_raw_html:
            payload["include_raw_html"] = True
        return self.api_wrapper.post("/v1/fetch", payload)

    async def _arun(
        self,
        urls: list[str],
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"urls": urls}
        if self.include_raw_html:
            payload["include_raw_html"] = True
        return await self.api_wrapper.apost("/v1/fetch", payload)


# -------------------------------------------------------------------- extract


class SofyaExtractInput(BaseModel):
    """Input for SofyaExtract."""

    url: str = Field(description="The page or document to read.")
    prompt: str = Field(
        description='What to pull out of it, e.g. "list every pricing tier with its price".'
    )


class SofyaExtract(_SofyaTool):  # type: ignore[override]
    """Answer a question about one page with Sofya.

    Sofya reads the page and an AI model answers the prompt from its text.
    Costs 10 credits (1 when the page has too little text to answer).

    Instantiate:
        .. code-block:: python

            from langchain_sofya import SofyaExtract

            tool = SofyaExtract()

    Invoke:
        .. code-block:: python

            tool.invoke({
                "url": "https://en.wikipedia.org/wiki/Eiffel_Tower",
                "prompt": "How tall is it and when did it open?",
            })
    """

    name: str = "sofya_extract"
    description: str = (
        "Read one web page or document and return just the information asked "
        "for. Input is a URL and a prompt describing what to extract."
    )
    args_schema: type[BaseModel] = SofyaExtractInput

    def _run(
        self,
        url: str,
        prompt: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict[str, Any]:
        return self.api_wrapper.post("/v1/extract", {"url": url, "prompt": prompt})

    async def _arun(
        self,
        url: str,
        prompt: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> dict[str, Any]:
        return await self.api_wrapper.apost("/v1/extract", {"url": url, "prompt": prompt})


# ------------------------------------------------------------------- research


class SofyaResearchInput(BaseModel):
    """Input for SofyaResearch."""

    query: str = Field(description="The research question or topic.")
    topic: Literal["general", "news"] | None = Field(default=None, description=_TOPIC)
    freshness: str | None = Field(default=None, description=_FRESHNESS)
    max_sources: int | None = Field(
        default=None, ge=5, le=30, description="Most sources to read, 5-30."
    )


_RESEARCH_KEYS = ("query", "topic", "freshness", "max_sources")


class SofyaResearch(_SofyaTool):  # type: ignore[override]
    """Research a question across many sources with Sofya.

    Splits the question into several searches, reads the sources and writes a
    report with citations. Costs 50 credits and usually takes 20-60 seconds.

    Instantiate:
        .. code-block:: python

            from langchain_sofya import SofyaResearch

            tool = SofyaResearch(max_sources=15)

    Invoke:
        .. code-block:: python

            tool.invoke({"query": "How do heat pumps perform in cold climates?"})
    """

    name: str = "sofya_research"
    description: str = (
        "Research an open-ended question across many web sources and return a "
        "written report with citations and the source list. Input is the "
        "question."
    )
    args_schema: type[BaseModel] = SofyaResearchInput

    topic: Literal["general", "news"] | None = None
    """Default topic."""
    freshness: str | None = None
    """Default freshness filter."""
    max_sources: int | None = None
    """Default number of sources, 5-30 (API default: 20)."""

    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = self._payload({"query": query, **kwargs}, _RESEARCH_KEYS)
        return self.api_wrapper.post("/v1/research", payload)

    async def _arun(
        self,
        query: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = self._payload({"query": query, **kwargs}, _RESEARCH_KEYS)
        return await self.api_wrapper.apost("/v1/research", payload)
