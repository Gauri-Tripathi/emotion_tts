# UniTTS

Turn text into speech through a local model or a hosted API using one Python interface.

**Start here: [Getting started](docs/getting-started.md)** — Windows setup,
CPU speech, API keys, working commands, and troubleshooting.

For repeated generation, see [Inference performance](docs/inference.md): keep
models loaded, run the web UI in WSL, and benchmark cold versus warm requests.

## First local speech (Windows PowerShell)

Requires Python 3.11 or newer. Run these commands from this repository:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[cpu]"
$env:UNITTS_PIPER_MODEL_CACHE = "$PWD/.unitts-cache/piper"
.\.venv\Scripts\python.exe -m unitts.cli synthesize "Hello from UniTTS" --provider piper -o hello.wav
```

The first run downloads a voice and needs internet access. Subsequent runs use the
cached model on your CPU. No GPU or API key is required. Select
`.venv/Scripts/python.exe` as your IDE interpreter too.

For hosted speech, follow the [API walkthrough](docs/getting-started.md#hosted-speech-with-elevenlabs).

## Where to work in the code

The CLI and local web UI use this path:

```text
text → UniTTS (core/engine.py) → provider (providers/) → audio bytes/file
              ↑
       core/config.py: YAML + environment variables + constructor options
```

Start with that path when debugging CPU or API speech. `unitts/core/config.py`
only loads provider settings; it does not install models or generate speech.
The separate `runtime/`, `engine_sdk/`, and `services/` directories contain an
in-progress runtime migration described below. A registered provider or model
manifest does not establish that its dependencies are installed or that it has
passed a real model test.

## Advanced runtime migration

UniTTS is an extensible generative speech runtime for serving heterogeneous TTS
architectures through one stable contract. Core owns validation, planning,
admission, worker lifecycle, device placement, streaming, and observability
boundaries. Engine plugins own model code, batching, weights, and heavy dependencies.

The repository is migrating incrementally from its original provider facade. The
existing `UniTTS(provider=...)`, CLI, and local web UI remain available. New code
uses `SynthesisRequest`, `SpeechEngine`, and `UniTTSRuntime`.

## Current status

- `dummy` is ready and provides deterministic generation and ordered streaming.
- The request → plan → scheduler → worker → engine → result/stream vertical slice works locally.
- Qwen3-TTS is migrated behind the lifecycle but remains experimental pending an isolated GPU smoke test.
- Ten requested families have truthful manifests. A manifest is not an implementation.

## Local development

Python 3.11 or newer is required.

```bash
python -m pip install -e ".[dev,service]"
python -m pytest
unitts-api
```

The API listens on `http://127.0.0.1:8000`. Try the dummy engine:

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "content-type: application/json" \
  -d '{"model":"dummy","input":"Hello from UniTTS"}' --output hello.wav
```

Run one versioned JSON-lines engine worker process:

```bash
$env:UNITTS_ENGINE_ID="dummy"
python -m unitts.services.engine_worker
```

For Qwen, create the environment from `engines/qwen3-tts`, install this repository,
set `UNITTS_ENGINE_ID=qwen3-tts` and `UNITTS_DEVICE=cuda`, then run the same worker.
Normal CI never downloads model weights.

Production uses one container and OS process per replica, with one model instance
and CUDA context. Containers share the host NVIDIA driver; verify driver, CUDA,
compute capability, code license, and weight license for every exact checkpoint.
F5 and XTTS weights default to research/non-commercial, and IndexTTS commercial
use requires explicit authorization.

See [architecture](docs/architecture.md), [engine development](docs/engine-development.md),
and [model support](docs/model-support.md).
