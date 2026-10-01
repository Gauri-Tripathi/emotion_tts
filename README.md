# UniTTS

UniTTS provides one Python interface for local and hosted text-to-speech systems.
It includes a Python SDK, command-line interface, local web UI, and an experimental
service runtime for isolated speech engines.

> **Project status:** pre-release. Piper is the recommended local CPU path. Other
> providers require their own credentials, packages, model weights, and hardware.

## Highlights

- Common request and response models across providers
- Lazy loading for optional CPU, GPU, and API dependencies
- Provider capability discovery from the CLI and web UI
- Reusable model sessions for repeated local inference
- Isolated engine manifests and lifecycle contracts for service deployments

## Quick start

UniTTS requires Python 3.11 or newer. From the repository root in WSL or Linux:

```bash
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install -e ".[cpu]"

export UNITTS_PIPER_MODEL_CACHE="$PWD/.unitts-cache/piper"
unitts synthesize \
  "Hello from UniTTS" \
  --provider piper \
  --voice en_US-lessac-medium \
  --output hello.wav
```

The first Piper request downloads the selected voice. Later requests reuse the
cached model and run locally on CPU.

## Python API

```python
from unitts import UniTTS

with UniTTS(
    provider="piper",
    device="cpu",
    model_cache=".unitts-cache/piper",
) as tts:
    result = tts.synthesize_to_file(
        "Hello from UniTTS",
        "hello.wav",
        voice="en_US-lessac-medium",
    )

print(result.metadata)
```

## Commands

```bash
# Inspect registered providers and capabilities.
unitts providers

# List voices exposed by a provider.
unitts voices --provider piper

# Start the local UI at http://localhost:8765.
unitts-web
```

The web UI retains the selected model between requests, reducing repeated inference
startup cost. The `unitts-api` command starts the experimental engine runtime; its
`dummy` engine generates test tones rather than speech.

## Development

```bash
python -m pip install -e ".[dev,service]"
python -m pytest -q
```

GPU engines should use separate environments because their PyTorch, CUDA, and model
package requirements can conflict. Normal tests do not download model weights.

## Project layout

- `unitts/core` and `unitts/providers`: synchronous SDK, CLI, and web UI
- `unitts/contracts` and `unitts/engine_sdk`: engine interfaces and wire models
- `unitts/runtime` and `unitts/services`: scheduling, workers, streaming, and API
- `engines`: isolated engine manifests and packaging metadata

## License

MIT
