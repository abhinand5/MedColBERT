"""Config loading from YAML files.

Loads configuration files and merges them with base defaults.
Config files are the source of truth for all parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a single YAML file and return its contents as a dictionary.

    Args:
        path: Path to a .yaml or .yml file.

    Returns:
        Parsed YAML content as a dict.

    Raises:
        FileNotFoundError: If the path does not exist.
        yaml.YAMLError: If the file cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_configs(*paths: Path | str) -> dict[str, Any]:
    """Load and merge multiple YAML configs in order.

    Later configs override keys from earlier ones. Returns an empty dict
    if no paths are provided.
    """
    merged: dict[str, Any] = {}
    for p in paths:
        cfg = load_yaml(p)
        _deep_merge(merged, cfg)
    return merged


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> None:
    """Merge overlay into base in-place. Nested dicts are merged recursively."""
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def resolve_path(root: Path, value: Any) -> Any:
    """If value is a relative path string, resolve it against root.

    Handles nested dicts and lists recursively.
    """
    if isinstance(value, str) and not value.startswith(("http://", "https://")):
        candidate = root / value
        # Only resolve if it looks like a path that could exist
        if any(
            ext in value for ext in (".yaml", ".json", ".parquet", ".csv", "/data/")
        ) or "/" in value:
            return str(candidate)
    if isinstance(value, dict):
        return {k: resolve_path(root, v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_path(root, v) for v in value]
    return value
