"""Pequeños helpers compartidos."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def utc_now() -> str:
    """Devuelve la marca temporal ISO-8601 en UTC."""
    return datetime.now(timezone.utc).isoformat()


def configure_logging(level: str | None = None) -> None:
    """Configura un logger de consola legible una vez por proceso."""
    logging.basicConfig(
        level=getattr(logging, (level or os.getenv("LOG_LEVEL", "INFO")).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def git_sha() -> str | None:
    """Devuelve el SHA del commit actual cuando el proyecto está en un repositorio git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _json_default(value: Any) -> Any:
    """Convierte los tipos escalares de numpy/pandas a tipos Python compatibles con JSON."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Escribe JSON formateado, creando los directorios padre cuando sea necesario."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)


def read_json(path: Path) -> dict[str, Any]:
    """Lee un documento JSON desde el disco."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}. Run the previous pipeline stages first."
        )
    return json.loads(path.read_text(encoding="utf-8"))
