"""HTTP client for the Sofya REST API."""

from __future__ import annotations

import os
from typing import Any

import httpx
from langchain_core.tools import ToolException
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from langchain_sofya._version import __version__

DEFAULT_BASE_URL = "https://sofya.co"
# Research and AI answers can take up to a minute; leave headroom.
DEFAULT_TIMEOUT = 120.0


class SofyaAPIWrapper(BaseModel):
    """Calls the Sofya API (https://sofya.co/docs).

    The API key is read from the ``SOFYA_API_KEY`` environment variable unless
    passed as ``api_key``. Create one at https://sofya.co/dashboard.
    """

    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr = Field(default=SecretStr(""))
    base_url: str = Field(default=DEFAULT_BASE_URL)
    timeout: float = Field(default=DEFAULT_TIMEOUT)

    @model_validator(mode="before")
    @classmethod
    def _from_env(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        if not values.get("api_key"):
            key = os.environ.get("SOFYA_API_KEY", "")
            if not key:
                raise ValueError(
                    "Sofya API key not found. Pass api_key or set the "
                    "SOFYA_API_KEY environment variable. Get a key at "
                    "https://sofya.co/dashboard"
                )
            values["api_key"] = key
        if not values.get("base_url"):
            values["base_url"] = os.environ.get("SOFYA_BASE_URL") or DEFAULT_BASE_URL
        return values

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "User-Agent": f"langchain-sofya/{__version__}",
        }

    def _url(self, path: str) -> str:
        return self.base_url.rstrip("/") + path

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST a JSON body and return the JSON response."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(self._url(path), json=payload, headers=self._headers())
        except httpx.HTTPError as exc:
            raise ToolException(f"Sofya request failed: {exc}") from exc
        return _handle(resp)

    async def apost(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Async version of :meth:`post`."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self._url(path), json=payload, headers=self._headers())
        except httpx.HTTPError as exc:
            raise ToolException(f"Sofya request failed: {exc}") from exc
        return _handle(resp)


def _handle(resp: httpx.Response) -> dict[str, Any]:
    if resp.is_success:
        data = resp.json()
        if not isinstance(data, dict):
            raise ToolException("Sofya returned an unexpected response.")
        return data
    raise ToolException(_error_message(resp))


def _error_message(resp: httpx.Response) -> str:
    detail: Any = None
    try:
        body = resp.json()
        detail = body.get("detail") if isinstance(body, dict) else body
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("error") or detail
    except ValueError:
        detail = resp.text[:500] or None

    status = resp.status_code
    if status == 401:
        prefix = "Sofya API key is missing or invalid."
    elif status == 402:
        prefix = "The Sofya account is out of credits. Top up at https://sofya.co/dashboard."
    elif status == 429:
        prefix = "Sofya rate limit reached; retry shortly."
    elif status == 422:
        prefix = "Sofya rejected the request parameters."
    else:
        prefix = f"Sofya API error (HTTP {status})."
    if detail and not isinstance(detail, str):
        detail = str(detail)
    return f"{prefix} {detail}" if detail and detail not in prefix else prefix
