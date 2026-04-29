"""Configuration loader — single source of truth for all settings."""

from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
_config_cache: dict[str, Any] | None = None


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load and cache the YAML configuration."""
    global _config_cache
    if _config_cache is not None and path is None:
        return _config_cache
    config_path = path or _CONFIG_PATH
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if path is None:
        _config_cache = config
    return config


def get_seed() -> int:
    """Get the global random seed."""
    return load_config()["seed"]
