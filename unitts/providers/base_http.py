"""Shared HTTP provider helpers."""

from __future__ import annotations

import logging
from typing import Any

import requests

from unitts.core.provider import BaseProvider
from unitts.core.schemas import TTSRequest

logger = logging.getLogger(__name__)


class HttpProvider(BaseProvider):
    """Base class for REST-backed providers."""

    timeout: float = 120.0

    def _check_response(self, response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            hint = {
                401: "Check your API key and its provider.",
                403: "Check your API key permissions and access to this model or voice.",
                404: "Check the provider's voice ID, model, and endpoint.",
                429: "Check your quota or billing; retry later if rate limited.",
            }.get(response.status_code, "Check the provider response below.")
            raise RuntimeError(
                f"{self.name} request failed (HTTP {response.status_code}). {hint} {response.text[:500]}"
            ) from exc

    def _warnings_for(self, request: TTSRequest) -> list[str]:
        return [f"{field} is not supported by provider '{self.name}' and was ignored" for field in self.capabilities.unsupported_request_fields(request)]

    def _post_audio(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any] | None = None,
        data: bytes | str | None = None,
    ) -> bytes:
        response = requests.post(url, headers=headers, json=json, data=data, timeout=self.timeout)
        self._check_response(response)
        return response.content

    def _get_json(self, url: str, *, headers: dict[str, str]) -> Any:
        response = requests.get(url, headers=headers, timeout=self.timeout)
        self._check_response(response)
        return response.json()
