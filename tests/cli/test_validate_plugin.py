# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 1 dispatch tests for the plugin content type (run_validation)."""

from pathlib import Path

from skillevaluator.constants import CONTENT_TYPE_PLUGIN
from skillevaluator.tier1.commands import run_validation

_VALID_MANIFEST = """
name: my-bundle
author:
  email: dev@example.com
skills:
  refs:
    - "github::example-org/example-repo::skills::build-infra"
"""


def _make_plugin(tmp_path: Path) -> Path:
    (tmp_path / "agent_plugin.yaml").write_text(_VALID_MANIFEST)
    return tmp_path


def test_run_validation_plugin_uses_plugin_schema_validator(tmp_path: Path):
    results = run_validation(_make_plugin(tmp_path), checks="schema", content_type=CONTENT_TYPE_PLUGIN)
    names = [r.validator_name or "" for r in results]
    assert any("Plugin Schema" in n for n in names)
    assert all(r.passed for r in results)


def test_run_validation_plugin_skips_quality_and_lint(tmp_path: Path):
    """quality/lint are skill-only and must be skipped for plugins."""
    results = run_validation(
        _make_plugin(tmp_path),
        checks="schema,quality,lint",
        content_type=CONTENT_TYPE_PLUGIN,
    )
    names = [r.validator_name or "" for r in results]
    assert any("Plugin Schema" in n for n in names)
    assert not any("Quality" in n for n in names)
    assert not any("Lint" in n or "Script" in n for n in names)


def test_run_validation_plugin_does_not_require_catalog_path(tmp_path: Path):
    """Plugin validation must succeed without any catalog (no dedup dependency)."""
    results = run_validation(_make_plugin(tmp_path), checks="schema", content_type=CONTENT_TYPE_PLUGIN)
    assert results
    assert all(r.passed for r in results)
