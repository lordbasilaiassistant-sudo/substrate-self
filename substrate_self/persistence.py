"""Substrate persistence.

Default location: ~/.substrate-self/substrate.json
Override with SUBSTRATE_PATH env var or pass `path=` explicitly.

Atomic writes via tmp + rename — never leaves a half-written file even
if the process is killed mid-save.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from substrate_self.core import Substrate


def default_path() -> Path:
    override = os.environ.get("SUBSTRATE_PATH")
    if override:
        return Path(override)
    home = Path.home()
    return home / ".substrate-self" / "substrate.json"


def load(path: Path | None = None) -> Substrate:
    """Load the substrate from disk; create a fresh one if absent."""
    p = path or default_path()
    if not p.exists():
        return Substrate()
    raw = p.read_text(encoding="utf-8")
    return Substrate.model_validate_json(raw)


def save(substrate: Substrate, path: Path | None = None) -> Path:
    """Atomic save: write to .tmp, then rename. Creates parents."""
    p = path or default_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(substrate.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(p)
    return p
