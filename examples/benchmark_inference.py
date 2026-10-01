"""Measure the local UI's cold and warm inference path without starting a server."""

import argparse
import json
from time import perf_counter

from unitts.web_server import INFERENCE_SESSION, synthesize_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="piper")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model")
    parser.add_argument("--text", default="Hello. This is UniTTS running on my CPU.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--reload", action="store_true", help="Unload before each request to compare loading overhead.")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    payload = {"provider": args.provider, "device": args.device, "model": args.model, "text": args.text}
    try:
        for index in range(args.runs):
            if args.reload:
                INFERENCE_SESSION.close()
            start = perf_counter()
            result = synthesize_payload(payload)
            print(json.dumps({
                "run": index + 1,
                "state": "cold" if index == 0 or args.reload else "warm",
                "elapsed_seconds": perf_counter() - start,
                "synthesis_seconds": result["metadata"].get("synthesis_seconds"),
                "real_time_factor": result["metadata"].get("real_time_factor"),
            }), flush=True)
    finally:
        INFERENCE_SESSION.close()


if __name__ == "__main__":
    main()
