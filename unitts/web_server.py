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
from urllib.parse import parse_qs, urlparse

from unitts import TTSRequest, UniTTS
from unitts.core.registry import ProviderRegistry
from unitts.core.config import load_env_file, provider_options
from unitts.core.schemas import OutputFormat

WEB_ROOT = Path(__file__).with_name("web")
SYNTHESIS_LOCK = threading.Lock()
MAX_BODY_BYTES = 32 * 1024 * 1024


def public_error(exc: Exception) -> str:
    """Show actionable service failures without forwarding credential-bearing bodies."""
    status = getattr(exc, "status_code", None)
    cause_response = getattr(exc.__cause__, "response", None)
    if cause_response is not None:
        status = cause_response.status_code
    messages = {
        401: "Authentication failed. Check this provider's API key in your environment file.",
        402: "This provider requires payment. Check your plan and available credits.",
        403: "Access denied. Check permissions for this model and voice.",
        404: "Model or voice not found. Check the selected voice and model.",
        429: "Quota or rate limit reached. Check your credits and try again later.",
    }
    if status:
        return messages.get(status, f"The provider returned HTTP {status}. Try again later.")
    if isinstance(exc, (ValueError, ImportError, RuntimeError, OSError)):
        return str(exc)
    return "Generation failed. Check the provider setup and try again."


class InferenceSession:
    """Keep one provider warm; serialize access to its mutable model state."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tts: UniTTS | None = None
        self._key: tuple[str, str, str | None] | None = None

    def synthesize(self, provider: str, device: str, model: str | None, request: TTSRequest):
        key = (provider, device, model)
        with self._lock:
            if self._key != key or self._tts is None:
                self.close()
                options = {"model": model} if model else {}
                self._tts = UniTTS(provider=provider, device=device, **options)
                self._key = key
            try:
                return self._tts.synthesize(request)
            except Exception:
                # A failed inference may leave a model unusable. Recreate it next time.
                self.close()
                raise

    def close(self) -> None:
        with self._lock:
            tts, self._tts = self._tts, None
            self._key = None
            if tts is not None:
                tts.cleanup()


INFERENCE_SESSION = InferenceSession()


def provider_payloads() -> list[dict[str, Any]]:
    """Return the registry metadata needed by the browser without loading models."""
    providers = []
    seen = set()
    for name, provider_cls in sorted(ProviderRegistry.all().items()):
        if provider_cls in seen:
            continue
        seen.add(provider_cls)
        canonical = provider_cls.name
        options = provider_options(canonical, {})
        configured = bool(options.get("api_key")) and (canonical != "azure" or bool(options.get("region")))
        providers.append({"name": canonical, "configured": configured,
                          "capabilities": provider_cls.capabilities.model_dump(mode="json")})
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
    capabilities = ProviderRegistry.get(provider).capabilities
    formats = capabilities.output_formats
    output_format = payload.get("output_format") or ("wav" if "wav" in formats else formats[0])
    if output_format not in formats:
        raise ValueError(f"{provider} supports: {', '.join(formats)}")
    device = payload.get("device", "auto")
    if capabilities.requires_gpu and device == "cpu":
        raise ValueError(f"{provider} requires a GPU. Choose a CPU provider or CUDA.")

    reference = _decode_reference(payload.get("reference_audio"))
    try:
        request = TTSRequest(
            text=text,
            language=payload.get("language") or "en",
            voice=payload.get("voice") or None,
            speed=payload.get("speed", 1.0),
            output_format=OutputFormat(output_format),
            reference_audio=[reference] if reference else [],
            reference_text=payload.get("reference_text") or None,
        )
        response = INFERENCE_SESSION.synthesize(
            provider, device, payload.get("model") or None, request,
        )
        return {
            "audio": base64.b64encode(response.audio).decode("ascii"),
            "sample_rate": response.sample_rate,
            "warnings": response.warnings,
            "metadata": response.metadata,
            "output_format": response.output_format.value,
            "duration_seconds": response.duration_seconds,
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
        if urlparse(self.path).path == "/api/voices":
            try:
                provider = parse_qs(urlparse(self.path).query).get("provider", [""])[0]
                # Listing voices does not load a synthesis model.
                with UniTTS(provider=provider) as engine:
                    voices = engine.list_voices()
                self._send_json(HTTPStatus.OK, {"voices": voices})
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": public_error(exc)})
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
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": public_error(exc)})
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
    parser.add_argument("--env-file", type=Path, help="Load local API credentials from this file")
    args = parser.parse_args()
    if args.env_file:
        load_env_file(args.env_file)
    server = ThreadingHTTPServer((args.host, args.port), UniTTSRequestHandler)
    print(f"UniTTS web UI: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        INFERENCE_SESSION.close()


if __name__ == "__main__":
    main()
