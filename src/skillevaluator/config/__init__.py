# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration loaders for SkillEvaluator patterns."""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent


def load_pii_patterns() -> dict:
    """Load PII detection patterns from YAML config."""
    config_path = CONFIG_DIR / "pii_patterns.yaml"
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_license_config() -> dict:
    """Load license compliance configuration from YAML config.

    Returns:
        dict containing:
        - allowed_licenses: List of SPDX identifiers for permissive licenses
        - blocked_licenses: List of SPDX identifiers for restrictive licenses
        - license_patterns: Dict mapping license IDs to detection patterns
        - proprietary_indicators: List of strings indicating proprietary content
        - spdx_detection: Config for SPDX header scanning
        - license_file_names: List of standard license file names
    """
    config_path = CONFIG_DIR / "license_config.yaml"
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_unicode_smuggle_patterns() -> dict:
    """Load Unicode smuggling detection patterns from YAML config."""
    config_path = CONFIG_DIR / "unicode_smuggle_patterns.yaml"
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


__all__ = ["load_license_config", "load_pii_patterns", "load_unicode_smuggle_patterns"]
