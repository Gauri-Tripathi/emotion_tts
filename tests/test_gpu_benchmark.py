"""GPU audits must never report CPU inference as a successful GPU test."""

import json
import sys
from types import SimpleNamespace

import pytest

from examples.benchmark_gpu_inference import main, primary_model
from examples.package_colab_source import build_bundle


@pytest.mark.parametrize("name,attributes", [
    ("qwen3-tts", {"_model": SimpleNamespace(model="neural-module")}),
    ("f5-tts", {"_infer": SimpleNamespace(ema_model="neural-module")}),
    ("xtts-v2", {"_model": SimpleNamespace(synthesizer=SimpleNamespace(tts_model="neural-module"))}),
    ("dia", {"_model": SimpleNamespace(model="neural-module")}),
])
def test_primary_model_unwraps_provider(name, attributes):
    assert primary_model(SimpleNamespace(name=name, **attributes)) == "neural-module"


def test_gpu_audit_blocks_cpu_fallback(monkeypatch, tmp_path):
    import examples.benchmark_gpu_inference as audit

    monkeypatch.setattr(audit.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace())
    monkeypatch.setattr(sys, "argv", ["audit", "--output-dir", str(tmp_path)])
    assert main() == 2
    report = json.loads((tmp_path / "qwen3-tts/results.json").read_text())
    assert report["status"] == "blocked"
    assert report["runs"] == []
    assert "no CPU fallback" in report["reason"]


def test_colab_source_bundle_excludes_private_data(tmp_path):
    from zipfile import ZipFile

    for name in ["pyproject.toml", "README.md", "examples/benchmark_gpu_inference.py", "unitts/__init__.py",
                 "unitts/web/app.js", "unitts/.env", ".env", "local-docs/USER_GUIDE.md", "unitts/__pycache__/cache.py"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    bundle = build_bundle(tmp_path, tmp_path / "outputs/colab/source.zip")
    with ZipFile(bundle) as archive:
        assert set(archive.namelist()) == {"pyproject.toml", "README.md", "examples/benchmark_gpu_inference.py",
                                         "unitts/__init__.py", "unitts/web/app.js", "SOURCE_MANIFEST.json"}
        manifest = json.loads(archive.read("SOURCE_MANIFEST.json"))
        assert len(manifest["files"]) == 5


def test_dia_precision_option_reaches_loader(monkeypatch):
    from unittest.mock import Mock

    from unitts import TTSRequest
    from unitts.providers.open_weights import DiaProvider
    import unitts.providers.open_weights as providers

    load = Mock(return_value=SimpleNamespace(generate=lambda *a, **kw: [0.1]))
    monkeypatch.setitem(sys.modules, "dia.model", SimpleNamespace(Dia=SimpleNamespace(from_pretrained=load)))
    monkeypatch.setattr(providers, "_write_float_wav", lambda *args: None)
    DiaProvider(device="cuda", dtype="float16").synthesize(TTSRequest(text="[S1] Hello"))
    assert load.call_args.kwargs == {"device": "cuda", "compute_dtype": "float16"}
