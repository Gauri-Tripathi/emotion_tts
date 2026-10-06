"""Real GPU inference audit, intended for an isolated model environment.

python examples/benchmark_gpu_inference.py --provider qwen3-tts --runs 3
Use --help for reference-audio, model, and extended-test options.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import io
import json
import platform
import random
import sys
from pathlib import Path
from time import perf_counter

PACKAGES = {"qwen3-tts": ("qwen_tts", "qwen-tts"), "f5-tts": ("f5_tts", "f5-tts"),
            "xtts-v2": ("TTS", "coqui-tts"), "dia": ("dia", "nari-tts")}


def primary_model(provider):
    """Find the actual neural module, including wrappers without .to()."""
    if provider.name == "qwen3-tts":
        return getattr(provider._model, "model", None)
    if provider.name == "f5-tts":
        return getattr(provider._infer, "ema_model", None)
    if provider.name == "xtts-v2":
        return getattr(getattr(provider._model, "synthesizer", None), "tts_model", None)
    if provider.name == "dia":
        return getattr(provider._model, "model", None)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=PACKAGES, default="qwen3-tts")
    parser.add_argument("--model")
    parser.add_argument("--voice")
    parser.add_argument("--language", default="en")
    parser.add_argument("--text", default="Hello. This is UniTTS running on a GPU in Google Colab.")
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--reference-text")
    parser.add_argument("--voice-description")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16", help="Precision for Qwen/Dia; F5 and XTTS use upstream defaults")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--extended", action="store_true", help="Also test numbers, Unicode and longer text")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/colab-gpu"))
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.provider in {"f5-tts", "xtts-v2"} and not args.reference_audio:
        parser.error("This audit requires --reference-audio for F5 and XTTS")
    if args.reference_audio and not args.reference_audio.is_file():
        parser.error("Reference audio does not exist")
    if args.reference_audio and args.provider in {"f5-tts", "qwen3-tts"} and not args.reference_text:
        parser.error("Provide an accurate --reference-text to avoid transcription dependencies")

    root = args.output_dir / args.provider
    root.mkdir(parents=True, exist_ok=True)
    report = {"provider": args.provider, "model_override": args.model, "python": platform.python_version(),
              "requested_precision": args.dtype if args.provider in {"qwen3-tts", "dia"} else "upstream-default",
              "language": args.language, "reference_present": bool(args.reference_audio), "seed": 1234, "runs": []}
    def save():
        (root / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    for module in ("torch", "numpy", "soundfile", PACKAGES[args.provider][0]):
        if importlib.util.find_spec(module) is None:
            report.update(status="blocked", reason=f"Missing package: {module}")
            save()
            print(json.dumps(report))
            return 2

    import numpy as np
    import soundfile as sf
    import torch
    from unitts import TTSRequest, UniTTS

    if not torch.cuda.is_available():
        report.update(status="blocked", reason="CUDA unavailable; no CPU fallback was attempted")
        save()
        print(json.dumps(report))
        return 2
    if args.provider in {"qwen3-tts", "dia"} and args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        report.update(status="blocked", reason="This GPU does not support the requested bfloat16 precision; use float16")
        save()
        print(json.dumps(report))
        return 2
    report.update(gpu=torch.cuda.get_device_name(0), total_vram_gb=torch.cuda.get_device_properties(0).total_memory / 2**30,
                  torch=torch.__version__, cuda=torch.version.cuda)
    report["versions"] = {dist: importlib.metadata.version(dist) for dist in ("torch", "torchaudio", PACKAGES[args.provider][1])}
    (root / "packages.txt").write_text("\n".join(sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions())), encoding="utf-8")
    options = {"model": args.model} if args.model else {}
    if args.provider == "qwen3-tts":
        options.update(dtype=args.dtype, attn_implementation="sdpa")
    elif args.provider == "dia":
        options.update(dtype=args.dtype)
    text = args.text
    if args.provider == "dia" and not text.startswith("[S1]"):
        text = "[S1] Welcome to our speech studio. [S2] We are testing dialogue on a GPU. [S1] Let's listen to the result."
    cases = [(f"run-{index + 1}", text) for index in range(args.runs)]
    if args.extended:
        for label, script in [("numbers", "The meeting starts at 10:30, and the total is 42 dollars."),
                              ("unicode", "Welcome to the café. What a lovely day!"),
                              ("longer", "This is the first sentence. The next sentence must also be spoken. We want a complete recording from beginning to end.")]:
            if args.provider == "dia":
                script = "[S1] " + script + " [S2] That sounds good. [S1] Thank you for listening."
            cases.append((label, script))
    try:
        with UniTTS(provider=args.provider, device="cuda", **options) as engine:
            for index, (label, script) in enumerate(cases):
                random.seed(1234)
                np.random.seed(1234)
                torch.manual_seed(1234)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = perf_counter()
                row = {"case": label, "state": "cold" if index == 0 else "warm"}
                try:
                    request = TTSRequest(text=script, language=args.language, voice=args.voice,
                                         reference_audio=[args.reference_audio] if args.reference_audio else [],
                                         reference_text=args.reference_text, voice_description=args.voice_description,
                                         sample_rate=None, seed=1234)
                    response = engine.synthesize(request)
                    torch.cuda.synchronize()
                    elapsed = perf_counter() - started
                    data, rate = sf.read(io.BytesIO(response.audio), always_2d=True)
                    if not data.size or not np.isfinite(data).all() or not np.any(data):
                        raise RuntimeError("Empty, silent or non-finite audio output")
                    duration = len(data) / rate
                    model = primary_model(engine.provider)
                    devices = sorted({str(parameter.device) for parameter in model.parameters()}) if model is not None else []
                    verified = any(device.startswith("cuda") for device in devices)
                    artifact = root / f"{label}.wav"
                    artifact.write_bytes(response.audio)
                    row.update(status="pass" if verified else "unverified", elapsed_seconds=elapsed, audio_seconds=duration,
                               real_time_factor=elapsed / duration, sample_rate=rate, metadata_sample_rate=response.sample_rate,
                               peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30, model_devices=devices,
                               gpu_verified=verified, warnings=response.warnings, artifact=str(artifact))
                    if not verified:
                        row["reason"] = "Could not verify a primary model tensor on CUDA"
                except Exception as exc:
                    row.update(status="fail", error_type=type(exc).__name__, message=str(exc)[:600])
                report["runs"].append(row)
                save()
                print(json.dumps(row), flush=True)
                if row["status"] != "pass":
                    break  # Do not repeatedly load a broken or memory-exhausted model.
    except Exception as exc:
        report.update(status="fail", error_type=type(exc).__name__, message=str(exc)[:600])
        save()
        print(json.dumps(report), flush=True)
        return 1
    report["status"] = "pass" if all(row["status"] == "pass" for row in report["runs"]) else "fail"
    save()
    return int(report["status"] != "pass")


if __name__ == "__main__":
    sys.exit(main())
