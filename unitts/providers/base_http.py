"""Shared HTTP provider helpers."""

from __future__ import annotations

import logging
from typing import Any

import requests

from unitts.core.provider import BaseProvider
from unitts.core.schemas import TTSRequest, TTSResponse

logger = logging.getLogger(__name__)


class HttpProvider(BaseProvider):
    """Base class for REST-backed providers."""

    timeout: float = 120.0

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
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            message = response.text[:500]
            raise RuntimeError(f"{self.name} request failed: {response.status_code} {message}") from exc
        return response.content

    def _get_json(self, url: str, *, headers: dict[str, str]) -> Any:
        response = requests.get(url, headers=headers, timeout=self.timeout)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"{self.name} request failed: {response.status_code} {response.text[:500]}") from exc
        return response.json()
