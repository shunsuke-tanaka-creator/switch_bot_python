"""Small reproducibility manifest, not an optimizer-resume checkpoint."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess


def package_versions():
    packages = {}
    for name in ("numpy", "torch", "sentence-transformers", "fastapi"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return packages


def experiment_id(source, settings):
    root = Path(__file__).resolve().parent
    code = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        code.update(path.name.encode())
        code.update(path.read_bytes())
    value = dict(source=source, settings=settings, code=code.hexdigest(), packages=package_versions(), python=platform.python_version())
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def run_metadata(seed, epochs, settings, splits):
    packages = package_versions()
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL, text=True, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    return dict(seed=seed, epochs=epochs, settings=settings, splits=splits,
                python=platform.python_version(), platform=platform.platform(),
                packages=packages, git_sha=revision, git_dirty=dirty,
                checkpoint_purpose="inference; retrain from seed (not optimizer resume)")
