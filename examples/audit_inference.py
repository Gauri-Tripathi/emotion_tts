"""Run real CPU inference checks and opt-in hosted API smoke tests.

Run: python examples/audit_inference.py [--live-api]
Artifacts and a machine-readable report are written beneath outputs/provider-audit.
Credentials are read through UniTTS configuration; their values are never printed.
"""

import argparse
import io
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf

from unitts import UniTTS
from unitts.core.config import provider_options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-api", action="store_true")
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    if os.environ.get("OPENAI_API_KEY"):
        os.environ.setdefault("UNITTS_OPENAI_API_KEY", os.environ["OPENAI_API_KEY"])
    output = Path("outputs/provider-audit")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    report = output / ("api-results.json" if args.api_only else "results.json")

    def run(engine, label, text, **kwargs):
        start = perf_counter()
        row = {"provider": engine.provider_name, "case": label}
        try:
            response = engine.synthesize(text, **kwargs)
            samples, rate = sf.read(io.BytesIO(response.audio))
            assert samples.size > 0, "empty audio"
            assert np.isfinite(samples).all(), "non-finite audio"
            peak = float(np.max(np.abs(samples)))
            if kwargs.get("volume") == 0:
                assert peak == 0, "volume=0 did not mute output"
            else:
                assert peak > 0, "silent audio"
            artifact = output / f"{engine.provider_name}-{label}.{response.output_format.value}"
            artifact.write_bytes(response.audio)
            row.update(status="pass", seconds=len(samples) / rate, sample_rate=rate,
                       peak=peak, metadata_sample_rate=response.sample_rate,
                       metadata_duration=response.duration_seconds, artifact=str(artifact))
        except Exception as exc:
            # Avoid exposing service response bodies or credentials in persisted reports.
            row.update(status="fail", error_type=type(exc).__name__)
            code = getattr(exc, "code", None)
            if isinstance(code, str) and code.replace("_", "").isalnum():
                row["error_code"] = code
            if isinstance(exc, AssertionError):
                row["reason"] = str(exc)
        row["elapsed_seconds"] = round(perf_counter() - start, 3)
        rows.append(row)
        print(json.dumps(row), flush=True)
        report.write_text(json.dumps(rows, indent=2))

    text = "Hello. This is a CPU inference test of UniTTS."
    for name in (() if args.api_only else ("espeak", "piper", "kokoro")):
        with UniTTS(provider=name, device="cpu") as engine:
            run(engine, "default", text)
            if name == "kokoro":
                engine.set_provider(name, lang_code="a")
            run(engine, "baseline", text)
            run(engine, "repeat", text)
            run(engine, "fast", text, speed=1.5)
            run(engine, "slow", text, speed=0.75)
            run(engine, "numbers", "The total is 42 dollars and 50 cents. Call 12345.")
            run(engine, "unicode", "Café, naïve, résumé. Hello — welcome!")
            run(engine, "multiline", text + "\n" + "This second paragraph must also be spoken. " * 6)
            if name in ("espeak", "piper"):
                run(engine, "mute", text, volume=0)
            if name == "piper":
                run(engine, "alternate-voice", text, voice="en_US-lessac-high")

        cases = {row["case"]: row for row in rows if row["provider"] == name and row.get("status") == "pass"}
        for label, passed in (
            ("speed-order", all(key in cases for key in ("fast", "baseline", "slow")) and cases["fast"]["seconds"] < cases["baseline"]["seconds"] < cases["slow"]["seconds"]),
            ("multiline-completeness", all(key in cases for key in ("multiline", "baseline")) and cases["multiline"]["seconds"] > 2 * cases["baseline"]["seconds"]),
        ):
            row = {"provider": name, "case": label, "status": "pass" if passed else "fail"}
            rows.append(row)
            print(json.dumps(row), flush=True)

    for name in ("openai", "elevenlabs", "fish", "smallest", "azure"):
        options = provider_options(name, {})
        configured = bool(options.get("api_key")) and (name != "azure" or bool(options.get("region")))
        if args.live_api and configured:
            with UniTTS(provider=name) as engine:
                for fmt in engine.provider.capabilities.output_formats:
                    run(engine, "live-" + fmt, "Hello from UniTTS.", output_format=fmt)
        else:
            row = {"provider": name, "status": "blocked", "reason": "missing credentials" if not configured else "use --live-api"}
            rows.append(row)
            print(json.dumps(row), flush=True)
    report.write_text(json.dumps(rows, indent=2))
    return int(any(row["status"] == "fail" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
