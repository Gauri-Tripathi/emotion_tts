"""Click command-line interface for UniTTS."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import click

from unitts.core.config import init_config
from unitts.core.registry import ProviderRegistry
from unitts.core.schemas import OutputFormat, TTSRequest
from unitts.utils.downloader import hf_download
from unitts import UniTTS


@click.group()
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def main(verbose: bool = False) -> None:
    """Universal Text-to-Speech command line tool."""

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


@main.command()
@click.argument("text", required=False)
@click.option("--file", "-f", "input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Read text from a file.")
@click.option("--provider", "-p", default="piper", show_default=True, help="Provider name.")
@click.option("--voice", help="Voice id or model voice name.")
@click.option("--language", help="Language code.")
@click.option("--speed", type=float, help="Speech speed multiplier.")
@click.option("--pitch", type=float, help="Pitch in semitones where supported.")
@click.option("--volume", type=float, help="Volume multiplier where supported.")
@click.option("--format", "output_format", type=click.Choice([item.value for item in OutputFormat]), default="wav", show_default=True)
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), default=Path("output.wav"), show_default=True)
def synthesize(
    text: str | None,
    input_file: Path | None,
    provider: str,
    voice: str | None,
    language: str | None,
    speed: float | None,
    pitch: float | None,
    volume: float | None,
    output_format: str,
    output: Path,
) -> None:
    """Synthesize text to an audio file."""

    if input_file:
        body = input_file.read_text(encoding="utf-8")
    elif text:
        body = text
    else:
        body = click.get_text_stream("stdin").read()
    if not body.strip():
        raise click.UsageError("Provide text, --file, or stdin input")

    request = TTSRequest(
        text=body.strip(),
        voice=voice,
        language=language,
        speed=speed,
        pitch=pitch,
        volume=volume,
        output_format=OutputFormat(output_format),
    )
    with UniTTS(provider=provider) as engine:
        response = engine.synthesize_to_file(request, output)
    for warning in response.warnings:
        click.echo(f"warning: {warning}", err=True)
    click.echo(str(output))


@main.command()
@click.option("--provider", "-p", required=True, help="Provider name.")
def voices(provider: str) -> None:
    """List voices for a provider."""

    with UniTTS(provider=provider) as engine:
        click.echo(json.dumps(engine.list_voices(), indent=2))


@main.command("providers")
def providers_command() -> None:
    """List registered providers and capabilities."""

    rows = []
    for name, provider_cls in sorted(ProviderRegistry.all().items()):
        cap = provider_cls.capabilities
        rows.append((name, cap.provider_type.value, cap.voice_cloning, cap.emotion, cap.streaming, ",".join(cap.output_formats)))
    click.echo("name\ttype\tcloning\temotion\tstreaming\tformats")
    for row in rows:
        click.echo("\t".join(str(item) for item in row))


@main.group()
def config() -> None:
    """Manage UniTTS configuration."""


@config.command("init")
def config_init() -> None:
    """Create ~/.unitts/config.yaml."""

    click.echo(str(init_config()))


@main.group()
def models() -> None:
    """Manage cached model files."""


@models.command("list")
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path), default=Path.home() / ".unitts" / "models", show_default=True)
def models_list(cache_dir: Path) -> None:
    """List files in the UniTTS model cache."""

    if not cache_dir.exists():
        click.echo("No model cache found.")
        return
    for path in sorted(item for item in cache_dir.rglob("*") if item.is_file()):
        click.echo(str(path.relative_to(cache_dir)))


@models.command("download")
@click.argument("repo_id")
@click.argument("filename")
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path), default=Path.home() / ".unitts" / "models", show_default=True)
def models_download(repo_id: str, filename: str, cache_dir: Path) -> None:
    """Download a model file from Hugging Face Hub."""

    click.echo(str(hf_download(repo_id, filename, cache_dir=cache_dir)))


@models.command("clear")
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path), default=Path.home() / ".unitts" / "models", show_default=True)
def models_clear(cache_dir: Path) -> None:
    """Clear the UniTTS model cache."""

    if not cache_dir.exists():
        click.echo("No model cache found.")
        return
    click.confirm(f"Delete cached model files under {cache_dir}?", abort=True)
    shutil.rmtree(cache_dir)
    click.echo("Model cache cleared.")


if __name__ == "__main__":
    main()
