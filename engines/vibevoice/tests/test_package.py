from pathlib import Path

import yaml

from unitts.contracts import EngineManifest


def test_manifest_is_valid() -> None:
    path = Path(__file__).parents[1] / "manifest.yaml"
    EngineManifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
