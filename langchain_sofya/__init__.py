"""Sofya integration for LangChain: web search, fetch, extract and research."""

from langchain_sofya._client import SofyaAPIWrapper
from langchain_sofya._version import __version__
from langchain_sofya.tools import (
    SofyaExtract,
    SofyaFetch,
    SofyaResearch,
    SofyaSearch,
)

__all__ = [
    "SofyaAPIWrapper",
    "SofyaExtract",
    "SofyaFetch",
    "SofyaResearch",
    "SofyaSearch",
    "__version__",
]
