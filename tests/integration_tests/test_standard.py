"""LangChain standard integration tests. Need SOFYA_API_KEY; they spend credits.

Research is left out by default (50 credits a call); set SOFYA_TEST_RESEARCH=1
to include it.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_sofya import SofyaExtract, SofyaFetch, SofyaResearch, SofyaSearch

pytestmark = pytest.mark.skipif(not os.environ.get("SOFYA_API_KEY"), reason="SOFYA_API_KEY not set")


class TestSofyaSearchIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return SofyaSearch

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"max_results": 3, "search_depth": "snippets"}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "Python programming language"}


class TestSofyaFetchIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return SofyaFetch

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"urls": ["https://example.com"]}


class TestSofyaExtractIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return SofyaExtract

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"url": "https://example.com", "prompt": "What is the page's heading?"}


@pytest.mark.skipif(not os.environ.get("SOFYA_TEST_RESEARCH"), reason="set SOFYA_TEST_RESEARCH=1")
class TestSofyaResearchIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return SofyaResearch

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"max_sources": 5}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "What is the capital of Australia and why was it chosen?"}
