"""LangChain standard unit tests for each Sofya tool."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_sofya import SofyaExtract, SofyaFetch, SofyaResearch, SofyaSearch

_KEY = "sofya_live_test"


class TestSofyaSearchUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return SofyaSearch

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": _KEY, "max_results": 5}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "latest stable Python release", "include_answer": True}

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return {"SOFYA_API_KEY": "sofya_live_env"}, {}, {"name": "sofya_search"}


class TestSofyaFetchUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return SofyaFetch

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": _KEY}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"urls": ["https://example.com"]}

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return {"SOFYA_API_KEY": "sofya_live_env"}, {}, {"name": "sofya_fetch"}


class TestSofyaExtractUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return SofyaExtract

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": _KEY}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {
            "url": "https://en.wikipedia.org/wiki/Eiffel_Tower",
            "prompt": "How tall is it and when did it open?",
        }

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return {"SOFYA_API_KEY": "sofya_live_env"}, {}, {"name": "sofya_extract"}


class TestSofyaResearchUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return SofyaResearch

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": _KEY, "max_sources": 10}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "How do heat pumps perform in cold climates?"}

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return {"SOFYA_API_KEY": "sofya_live_env"}, {}, {"name": "sofya_research"}
