"""Configuration for deterministic quality scoring."""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

DEFAULT_QUALITY_CONFIG = files("skillevaluator.config").joinpath("quality-default.yaml")


class QualityConfigError(ValueError):
    """Raised when a quality configuration is invalid."""


def load_quality_config(path: Path | None = None) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8") if path else DEFAULT_QUALITY_CONFIG.read_text(encoding="utf-8")
    try:
        raw = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise QualityConfigError(f"{path or 'packaged quality-default.yaml'}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("limitations", {}), dict):
        raise QualityConfigError("quality config: top-level and limitations must be mappings")
    limitations = raw["limitations"]
    required = limitations.get("required", True)
    heading = limitations.get("heading", "## Limitations")
    if not isinstance(required, bool) or not isinstance(heading, str) or not heading.strip():
        raise QualityConfigError("quality config: limitations.required must be boolean and heading non-empty")
    return {"limitations": {"required": required, "heading": heading}}
