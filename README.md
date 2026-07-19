# UniTTS

UniTTS is a Python package that puts multiple text-to-speech backends behind one API.
It is designed around provider isolation: a basic install stays lightweight, API
providers do not import GPU libraries, and local GPU providers are loaded only when
you choose them.

Current status: package foundation with passing unit tests. Piper/eSpeak/API/GPU
providers are wired through real integration paths, but every external provider still
needs live validation with its current service or model package before a production
release.

## Features

- One Python API: `UniTTS(provider="...").synthesize(...)`
- One CLI: `unitts synthesize "text" --provider piper -o out.wav`
- Pydantic v2 request and response schemas
- Provider registry with capability metadata
- Lazy provider imports for optional CPU/GPU/API dependencies
- Graceful degradation warnings for unsupported request fields
- Config loading from constructor kwargs, environment variables, and
  `~/.unitts/config.yaml`
- CPU, API, and open-weight provider families

## Install

From this repository:

```bash
python -m pip install -e ".[dev]"
```

Install optional provider groups as needed:

```bash
python -m pip install -e ".[cpu]"
python -m pip install -e ".[api]"
python -m pip install -e ".[gpu]"
python -m pip install -e ".[all]"
```

`.[gpu]` installs only UniTTS's shared GPU and audio dependencies. Install each
model provider separately; do not assume it includes every model package. The
[GPU provider guide](docs/gpu-providers.md) gives isolated, provider-specific
installation commands and Python examples.

When published to PyPI, the equivalent commands will be:

```bash
pip install unitts
pip install "unitts[cpu]"
pip install "unitts[api]"
pip install "unitts[gpu]"
```

## Quick Start

Python:

```python
from unitts import UniTTS

with UniTTS(provider="piper") as tts:
    response = tts.synthesize_to_file(
        "Hello from UniTTS.",
        "hello.wav",
        voice="en_US-lessac-medium",
    )

print(response.warnings)
```

CLI:

```bash
python -m unitts.cli providers
python -m unitts.cli synthesize "Hello from UniTTS" --provider piper --voice en_US-lessac-medium -o hello.wav
```

After installing the package, the console command is also available:

```bash
unitts providers
unitts synthesize "Hello from UniTTS" --provider piper -o hello.wav
```

## Local Web UI

Start the local interface without loading a model:

```bash
unitts-web
```

Open `http://127.0.0.1:8765`. Model work starts only when **Synthesize** is
pressed. For a remote Slurm node, use your usual SSH port forwarding rather than
binding the server to a public interface.

## Providers

| Provider | Type | Cloning | Emotion/style | Streaming | Status |
| --- | --- | --- | --- | --- | --- |
| Piper | CPU | No | No | No | Implemented, needs local dependency/model validation |
| eSpeak-NG | CPU | No | Prosody controls | No | Implemented through subprocess |
| Kokoro | CPU/open-weight | No | No | No | Integration wrapper |
| F5-TTS | GPU/open-weight | Yes | No | No | Integration wrapper |
| Dia | GPU/open-weight | Dialogue prompts | Non-verbal tags | No | Integration wrapper |
| Qwen3-TTS | GPU/open-weight | Yes | Description control | Yes | Configurable integration wrapper |
| CosyVoice | GPU/open-weight | Yes | Style/emotion | Yes | FunAudioLLM integration wrapper |
| Fish Speech | GPU/self-hosted | Yes | Text emotion markers | No | Connects to the local Fish Speech API |
| XTTS v2 | GPU/open-weight | Yes | No | No | Coqui TTS integration wrapper |
| OpenAI | API | No | No | Yes | Implemented through official SDK |
| ElevenLabs | API | Yes | Voice settings | Yes | REST integration |
| Fish Audio | API | Yes | No | Yes | REST integration |
| Smallest AI | API | No | No | Yes | REST integration |
| Azure | API | No | SSML styles | No | REST/SSML integration |

Parler large and StyleTTS 2 are not registered as built-ins. XTTS v2 was added as
an optional provider because it remains useful for voice cloning workflows, but it
is loaded lazily through the Coqui TTS package.

## Configuration

Configuration priority:

1. Constructor kwargs
2. Environment variables
3. `~/.unitts/config.yaml`
4. Provider defaults

Create the config file:

```bash
unitts config init
```

Example environment variables:

```bash
UNITTS_OPENAI_API_KEY=...
UNITTS_ELEVENLABS_API_KEY=...
UNITTS_AZURE_API_KEY=...
UNITTS_AZURE_REGION=eastus
```

## Documentation

- [Installation](docs/installation.md)
- [GPU Providers](docs/gpu-providers.md)
- [Python API](docs/python-api.md)
- [CLI](docs/cli.md)
- [Providers](docs/providers.md)
- [Configuration](docs/configuration.md)
- [Testing](docs/testing.md)
- [Production Roadmap](docs/roadmap.md)
- [Contributing](docs/contributing.md)

## Testing

Use the Python environment where the package is installed:

```bash
python -m pytest
```

Do not use plain `pytest` if it points to a different Python installation. In that
case dependencies such as `pydantic` may not be available even though the active
project environment works.

Current local result reported from the conda environment:

```text
4 passed
```

## Production Readiness

This repository is not ready for PyPI production release yet. The next milestones are:

- Validate every provider against current vendor/model APIs
- Add mocked API tests for every REST provider
- Add integration tests behind `pytest.mark.integration`
- Add CI for Python 3.10, 3.11, and 3.12
- Add linting, formatting, and type checking
- Add real audio conversion/resampling/normalization coverage
- Build and verify wheels through TestPyPI before publishing
