"""Filesystem locations shared by the worker, the model registry and the scripts."""

from __future__ import annotations

import os
from pathlib import Path

CATALOG_DIR_ENV = "WIGTN_AGENT_CATALOG_DIR"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def catalog_dir() -> Path:
    """Catalog root (``catalog/`` next to the project by default)."""
    return Path(os.environ.get(CATALOG_DIR_ENV, PROJECT_ROOT / "catalog"))
