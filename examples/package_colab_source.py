"""Package current UniTTS source for Colab, excluding credentials and local data.

Run from the repository: python examples/package_colab_source.py
Only Python source, bundled web assets, package metadata, README and the GPU
audit runner are included. No Git operations are performed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def build_bundle(root: Path, output: Path) -> Path:
    paths = [root / "pyproject.toml", root / "README.md", root / "examples/benchmark_gpu_inference.py"]
    license_file = root / "LICENSE"
    if license_file.is_file():
        paths.append(license_file)
    paths.extend(path for path in (root / "unitts").rglob("*") if path.is_file()
                 and path.suffix in {".py", ".html", ".css", ".js"}
                 and not any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(root).parts))
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"kind": "current-local-source", "files": {}}
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            name = path.relative_to(root).as_posix()
            payload = path.read_bytes()
            manifest["files"][name] = hashlib.sha256(payload).hexdigest()
            archive.writestr(name, payload)
        archive.writestr("SOURCE_MANIFEST.json", json.dumps(manifest, indent=2))
    output.with_suffix(".sha256").write_text(hashlib.sha256(output.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(build_bundle(root, root / "outputs/colab/unitts-colab-source.zip"))
