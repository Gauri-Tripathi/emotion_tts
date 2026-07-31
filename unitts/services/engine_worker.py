"""Versioned JSON-lines control transport for one engine per OS process.

Large audio currently uses base64 in the response. The envelope deliberately keeps
transport descriptors open for a later shared-memory or gRPC implementation.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from typing import Any

from unitts.contracts import PROTOCOL_VERSION, InferenceContext, LoadContext, ProtocolVersionError, SynthesisRequest
from unitts.runtime.worker import EngineWorker
from unitts.services.api import build_registry


async def serve() -> None:
    engine_id = os.environ.get("UNITTS_ENGINE_ID", "dummy")
    replica_id = os.environ.get("UNITTS_REPLICA_ID", f"{engine_id}-0")
    registry = build_registry()
    worker = EngineWorker(registry.create(engine_id))
    await worker.load(LoadContext(replica_id=replica_id, device=os.environ.get("UNITTS_DEVICE", "cpu")))
    while line := await asyncio.to_thread(sys.stdin.readline):
        try:
            envelope = json.loads(line)
            if envelope.get("protocol_version") != PROTOCOL_VERSION:
                raise ProtocolVersionError(f"Unsupported protocol: {envelope.get('protocol_version')}")
            operation = envelope.get("operation")
            if operation == "generate":
                request = SynthesisRequest.model_validate(envelope["request"])
                result = await worker.generate(request, InferenceContext(request_id=request.request_id, tenant_id=envelope.get("tenant_id", "local")))
                payload: dict[str, Any] = result.model_dump(exclude={"data"}, mode="json")
                payload["audio_base64"] = base64.b64encode(result.data).decode("ascii")
            elif operation == "health":
                payload = (await worker.health()).model_dump(mode="json")
            elif operation == "unload":
                await worker.unload()
                payload = {"unloaded": True}
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            response = {"protocol_version": PROTOCOL_VERSION, "ok": True, "result": payload}
        except Exception as exc:
            error = exc.as_dict() if hasattr(exc, "as_dict") else {"code": "worker_error", "message": str(exc)}
            response = {"protocol_version": PROTOCOL_VERSION, "ok": False, "error": error}
        print(json.dumps(response), flush=True)


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
