"""Reproducibility helpers: seeding, environment logging, hashing and run folders."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import platform
import random
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

_PACKAGES = ("numpy", "pandas", "scipy", "torch", "implicit", "PyYAML", "scikit-learn")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) and request deterministic kernels."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Embedding backward on CUDA has no deterministic kernel; warn instead of failing.
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # torch is optional for data-only commands
        pass


def environment_info() -> dict[str, Any]:
    """Collect interpreter, package, OS and hardware information for the run log."""
    info: dict[str, Any] = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": {},
    }
    for name in _PACKAGES:
        try:
            info["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            info["packages"][name] = None
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        info["cuda_available"] = False
    return info


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Return the SHA-256 digest of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_arrays(*arrays: np.ndarray) -> str:
    """Return a SHA-256 digest over the raw bytes, dtypes and shapes of NumPy arrays."""
    h = hashlib.sha256()
    for arr in arrays:
        a = np.ascontiguousarray(arr)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def make_run_dir(runs_dir: str | Path, slug: str) -> Path:
    """Create ``runs/<YYYY-MM-DD>-<HHMMSS>-<slug>``; names sort chronologically as text.

    A collision within the same second gets a zero-padded suffix (``-02``, ``-03`` ...).
    """
    now = _dt.datetime.now()
    base = Path(runs_dir) / f"{now:%Y-%m-%d}-{now:%H%M%S}-{slug}"
    path, n = base, 2
    while path.exists():
        path = Path(f"{base}-{n:02d}")
        n += 1
    path.mkdir(parents=True)
    return path


def code_sha256() -> str:
    """Hash of the package source files, recorded with every result."""
    h = hashlib.sha256()
    pkg = Path(__file__).resolve().parent
    for f in sorted(pkg.glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def write_manifest(run_dir: Path, command: str, extra: dict[str, Any]) -> dict[str, Any]:
    """Write ``manifest.json`` identifying the run (used to select runs instead of folder names)."""
    import uuid

    manifest = {"run_id": uuid.uuid4().hex, "command": command, "run_dir": str(run_dir),
                "created_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "code_sha256": code_sha256()}
    manifest.update(extra)
    save_json(manifest, Path(run_dir) / "manifest.json")
    return manifest


def save_json(obj: Any, path: str | Path) -> None:
    """Write JSON with stable key order; NumPy scalars are converted to Python types."""

    def _default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Not JSON serialisable: {type(o)}")

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=_default, ensure_ascii=False)


def get_logger(run_dir: Path | None = None, name: str = "steam_nffmtl") -> logging.Logger:
    """Return a logger that writes to stdout and, if given, ``run_dir/run.log``."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if run_dir is not None:
        fileh = logging.FileHandler(Path(run_dir) / "run.log", encoding="utf-8")
        fileh.setFormatter(fmt)
        logger.addHandler(fileh)
    logger.propagate = False
    return logger
