"""Checksums, atomic machine-readable state and durable progress logging."""

import hashlib
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

LOGGER = logging.getLogger("powerco")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value: object) -> None:
    """Atomic replacement is used only for owned run-state/output files."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, default=str, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def configure_logging(path: Path) -> None:
    """Write the same progress messages to console and this run's log file."""
    for handler in LOGGER.handlers[:]:
        handler.close()
        LOGGER.removeHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    for handler in [logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")]:
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def require(condition: bool, message: str) -> None:
    """Do not use assert for external input/integrity checks (python -O removes it)."""
    if not condition:
        raise ValueError(message)
