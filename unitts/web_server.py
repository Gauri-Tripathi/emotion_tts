"""Small local web interface for UniTTS.

The server intentionally uses only the Python standard library. Provider modules
remain lazy and a model is loaded only by a synthesis request.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from unitts import TTSRequest, UniTTS
from unitts.core.registry import ProviderRegistry

WEB_ROOT = Path(__file__).with_name("web")
SYNTHESIS_LOCK = threading.Lock()
MAX_BODY_BYTES = 32 * 1024 * 1024


def provider_payloads() -> list[dict[str, Any]]:
    """Return the registry metadata needed by the browser without loading models."""
    providers = []
    for name, provider_cls in sorted(ProviderRegistry.all().items()):
        providers.append({"name": name, "capabilities": provider_cls.capabilities.model_dump(mode="json")})
    return providers


def _decode_reference(reference: dict[str, Any] | None) -> Path | None:
    if not reference or not reference.get("data"):
        return None
    name = Path(str(reference.get("name", "reference.wav")))
    suffix = name.suffix if name.suffix and len(name.suffix) <= 12 else ".wav"
    try:
        raw = base64.b64decode(str(reference["data"]), validate=True)
    except ValueError as exc:
        raise ValueError("Reference audio is not valid base64 data") from exc
    if not raw:
        raise ValueError("Reference audio is empty")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(raw)
        return Path(handle.name)


def synthesize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one request and return browser-safe audio metadata."""
    text = str(payload.get("text", "")).strip()
    if not text:
        raise ValueError("Text is required")
    provider = str(payload.get("provider", "")).strip()
    if not provider:
        raise ValueError("Provider is required")

    reference = _decode_reference(payload.get("reference_audio"))
    try:
        request = TTSRequest(
            text=text,
            language=payload.get("language") or "en",
            voice=payload.get("voice") or None,
            speed=payload.get("speed") or 1.0,
            reference_audio=[reference] if reference else [],
            reference_text=payload.get("reference_text") or None,
        )
        options = {"model": payload["model"]} if payload.get("model") else {}
        with UniTTS(provider=provider, device=payload.get("device", "auto"), **options) as tts:
            response = tts.synthesize(request)
        return {
            "audio": base64.b64encode(response.audio).decode("ascii"),
            "sample_rate": response.sample_rate,
            "warnings": response.warnings,
            "metadata": response.metadata,
        }
    finally:
        if reference:
            reference.unlink(missing_ok=True)


class UniTTSRequestHandler(BaseHTTPRequestHandler):
    """Serve the UI and its small local JSON API."""

    server_version = "UniTTSWeb/0.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/providers":
            self._send_json(HTTPStatus.OK, {"providers": provider_payloads()})
            return
        relative = "index.html" if self.path in {"/", "/index.html"} else self.path.lstrip("/")
        target = (WEB_ROOT / relative).resolve()
        if WEB_ROOT not in target.parents and target != WEB_ROOT:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        if not target.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        self._send_bytes(HTTPStatus.OK, target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/synthesize":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValueError("Request body must be between 1 byte and 32 MB")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        if not SYNTHESIS_LOCK.acquire(blocking=False):
            self._send_json(HTTPStatus.CONFLICT, {"error": "A synthesis request is already running"})
            return
        try:
            self._send_json(HTTPStatus.OK, synthesize_payload(payload))
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        finally:
            SYNTHESIS_LOCK.release()

    def log_message(self, _format: str, *_args: Any) -> None:
        """Keep the local server quiet unless a request fails in the browser."""

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        self._send_bytes(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _send_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local UniTTS web interface")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), UniTTSRequestHandler)
    print(f"UniTTS web UI: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
