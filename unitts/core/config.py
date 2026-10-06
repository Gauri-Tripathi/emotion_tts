"""Configuration loading for UniTTS."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path.home() / ".unitts" / "config.yaml"


def load_env_file(path: Path) -> None:
    """Load an explicit local credential file without replacing exported values."""
    from dotenv import load_dotenv

    if not path.is_file():
        raise ValueError(f"Environment file not found: {path}")
    load_dotenv(path, override=False)
    aliases = {
        "UNITTS_OPENAI_API_KEY": ("OPENAI_API_KEY",),
        "UNITTS_ELEVENLABS_API_KEY": ("ELEVENLABS_API_KEY", "ELEVEN_LAB_KEY"),
        "UNITTS_FISH_API_KEY": ("FISH_API_KEY",),
        "UNITTS_SMALLEST_API_KEY": ("SMALLEST_API_KEY",),
        "UNITTS_AZURE_API_KEY": ("AZURE_SPEECH_KEY",),
        "UNITTS_AZURE_REGION": ("AZURE_SPEECH_REGION",),
    }
    for target, sources in aliases.items():
        for source in sources:
            if os.environ.get(source):
                os.environ.setdefault(target, os.environ[source])
                break


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load YAML config with environment variable interpolation."""

    path = config_path or CONFIG_PATH
    if not path.exists():
        return {}
    raw = os.path.expandvars(path.read_text(encoding="utf-8"))
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def provider_options(provider: str, constructor_options: dict[str, Any]) -> dict[str, Any]:
    """Merge config using constructor > env vars > config file > defaults."""

    config = load_config()
    defaults = config.get("defaults", {}) if isinstance(config.get("defaults", {}), dict) else {}
    providers = config.get("providers", {}) if isinstance(config.get("providers", {}), dict) else {}
    specific = providers.get(provider, {}) if isinstance(providers.get(provider, {}), dict) else {}
    merged: dict[str, Any] = {**defaults, **specific}
    prefix = f"UNITTS_{provider.upper().replace('-', '_')}_"
    for key, value in os.environ.items():
        if key.startswith(prefix):
            merged[key[len(prefix) :].lower()] = value
    common_key = f"UNITTS_{provider.upper().replace('-', '_')}_API_KEY"
    if common_key in os.environ:
        merged["api_key"] = os.environ[common_key]
    merged.update({key: value for key, value in constructor_options.items() if value is not None})
    return merged


def init_config(path: Path | None = None) -> Path:
    """Create a default config file if one does not exist."""

    target = path or CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text(yaml.safe_dump({
            "defaults": {"model_cache": (Path.home() / ".unitts" / "models").as_posix()},
            "providers": {},
        }), encoding="utf-8")
    return target
