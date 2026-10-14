# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 1 dispatch tests for the plugin content type (run_validation)."""

from pathlib import Path

import pytest

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


_SKILL_CHECK_VALIDATORS = {"QUALITY", "SCRIPT_LINT", "Semantic Version Validation"}


def _bundled_skill(plugin: Path, name: str, *, version: str | None = None) -> Path:
    skill = plugin / "skills" / name
    skill.mkdir(parents=True)
    version_line = f'  version: "{version}"\n' if version else ""
    (skill / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: Bundled {name} skill used to exercise per-skill Tier 1 checks.\n"
        "metadata:\n"
        "  author: Test Author <test@example.com>\n"
        f"{version_line}"
        "---\n\n"
        f"# {name}\n\n## Instructions\nFollow the request.\n\n## Examples\nRun the example.\n",
        encoding="utf-8",
    )
    return skill


def test_run_validation_plugin_without_bundled_skills_skips_skill_checks(tmp_path: Path):
    """version/quality/lint have nothing to score when a plugin bundles no skills."""
    results = run_validation(
        _make_plugin(tmp_path),
        checks="schema,version,quality,lint",
        content_type=CONTENT_TYPE_PLUGIN,
    )
    names = [r.validator_name or "" for r in results]
    assert any("Plugin Schema" in n for n in names)
    assert not _SKILL_CHECK_VALIDATORS & set(names)


def test_run_validation_plugin_runs_skill_checks_per_bundled_skill(tmp_path: Path):
    """version/quality/lint run once per bundled skill and attribute findings to it."""
    (tmp_path / "plugin").mkdir()
    plugin = _make_plugin(tmp_path / "plugin")
    alpha = _bundled_skill(plugin, "alpha")
    (alpha / "scripts").mkdir()
    (alpha / "scripts" / "flat.py").write_text("print('flat')\n", encoding="utf-8")
    _bundled_skill(plugin, "beta", version="1.0")
    # Root-owned plugin content is not a skill: it must not be scored or
    # linted as one (and so cannot double-report with the per-skill pass).
    (plugin / "scripts").mkdir()
    (plugin / "scripts" / "root_flat.py").write_text("print('root')\n", encoding="utf-8")

    results = run_validation(plugin, checks="version,quality,lint", content_type=CONTENT_TYPE_PLUGIN)
    by_name = {r.validator_name: r for r in results}

    assert set(by_name) == _SKILL_CHECK_VALIDATORS

    lint = by_name["SCRIPT_LINT"]
    assert lint.findings
    assert {f.file_path.split("] ")[0] for f in lint.findings} == {"[alpha"}
    assert "flat_script" in {f.check_name for f in lint.findings}
    assert not any("root_flat.py" in f.file_path for r in results for f in r.findings)

    version = by_name["Semantic Version Validation"]
    assert [f.file_path.split("] ")[0] for f in version.findings] == ["[beta"]
    assert all(f.check_name == "version_semver" for f in version.findings)

    quality = by_name["QUALITY"]
    scored = sorted(entry["skill_name"] for entry in quality.metadata["quality_scores_all"])
    assert scored == ["alpha", "beta"]
    assert all(f.file_path.startswith(("[alpha] ", "[beta] ")) for f in quality.findings)
    # Each finding is reported exactly once.
    keys = [(f.check_name, f.file_path, f.message) for r in results for f in r.findings]
    assert len(keys) == len(set(keys))


def test_run_validation_plugin_does_not_require_catalog_path(tmp_path: Path):
    """Plugin validation must succeed without any catalog (no dedup dependency)."""
    results = run_validation(_make_plugin(tmp_path), checks="schema", content_type=CONTENT_TYPE_PLUGIN)
    assert results
    assert all(r.passed for r in results)


def test_plugin_bundle_security_runs_when_schema_is_not_selected(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    _make_plugin(plugin)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (plugin / "skills").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    results = run_validation(plugin, checks="quality", content_type=CONTENT_TYPE_PLUGIN)

    assert len(results) == 1
    assert not results[0].passed
    assert results[0].metadata["security_failure"] is True
