# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from skillevaluator.deduplication.plugin.intra_plugin_validator import IntraPluginValidator
from skillevaluator.deduplication.plugin.ref_utils import find_duplicate_refs, normalize_ref
from skillevaluator.models.result import Severity
from skillevaluator.tier2.commands import run_plugin_dedup_scan, run_plugin_skill_context_dedup


def test_public_selector_and_canonical_forms_normalize_together() -> None:
    selector = {"source": "github", "repo": "Example/Repo.git", "path": "skills/deploy/helper"}
    assert normalize_ref(selector) == "github::example/repo::skills::deploy/helper"
    groups = find_duplicate_refs([selector, "GitHub::Example/Repo.git::Skills::deploy/helper"])
    assert [group.canonical_id for group in groups] == ["github::example/repo::skills::deploy/helper"]


def test_duplicate_refs_are_medium_and_advisory(tmp_path: Path) -> None:
    (tmp_path / "agent_plugin.yaml").write_text(
        """
name: public-plugin
author: {email: dev@example.com}
skills:
  refs:
    - github::example/repo::skills::demo
    - {source: github, repo: example/repo, path: skills/demo}
""",
        encoding="utf-8",
    )
    result = IntraPluginValidator().validate(tmp_path)
    assert result.passed
    assert len(result.findings) == 1
    assert result.findings[0].severity == Severity.MEDIUM
    assert result.metadata["advisory_tier2"] is True


def test_invalid_manifest_is_an_optional_skip(tmp_path: Path) -> None:
    (tmp_path / "agent_plugin.yaml").write_text("name: [unterminated", encoding="utf-8")
    result = IntraPluginValidator().validate(tmp_path)
    assert result.passed
    assert result.metadata["execution_status"] == "skipped"
    assert result.metadata["optional"] is True


def test_symlinked_manifest_outside_plugin_is_a_security_failure(tmp_path: Path) -> None:
    outside = tmp_path / "outside.yaml"
    outside.write_text(
        "name: outside\nauthor: {email: dev@example.com}\nskills:\n  refs: [github::example/repo::skills::a]\n",
        encoding="utf-8",
    )
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    try:
        (plugin / "agent_plugin.yaml").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    result = IntraPluginValidator().validate(plugin)

    assert not result.passed
    assert result.metadata["execution_status"] == "failed"
    assert result.metadata["security_failure"] is True
    assert result.metadata["optional"] is False

    scan_results = run_plugin_dedup_scan(plugin, run_context=False)
    assert not scan_results[0].passed
    assert scan_results[0].findings[0].severity == Severity.HIGH
    assert scan_results[0].metadata["execution_status"] == "failed"


def test_public_plugin_scan_never_requires_remote_catalog(tmp_path: Path) -> None:
    (tmp_path / "agent_plugin.yaml").write_text(
        "name: p\nauthor: {email: a@example.com}\nskills:\n  refs: [github::example/repo::skills::a]\n",
        encoding="utf-8",
    )
    results = run_plugin_dedup_scan(tmp_path, run_context=False)
    assert len(results) == 2
    assert all(result.passed for result in results)
    assert all(result.metadata.get("advisory_tier2") for result in results)
    assert results[1].metadata["execution_status"] == "skipped"


def test_plugin_context_scan_rejects_linked_skills_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    try:
        (plugin / "skills").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    [result] = run_plugin_skill_context_dedup(plugin)

    assert not result.passed
    assert result.metadata["security_failure"] is True
    assert result.findings[0].severity == Severity.HIGH

    scan_results = run_plugin_dedup_scan(plugin, run_context=False)
    assert any(result.metadata.get("security_failure") for result in scan_results)
    assert any(not result.passed for result in scan_results)


def test_plugin_context_scan_skips_before_provider_work_above_skill_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.constants import MAX_PLUGIN_DEDUP_SKILLS

    plugin = tmp_path / "plugin"
    skills = plugin / "skills"
    skills.mkdir(parents=True)
    discovered = [skills / f"skill-{index}" for index in range(MAX_PLUGIN_DEDUP_SKILLS + 1)]
    monkeypatch.setattr("skillevaluator.utils.helpers.find_bundled_plugin_skills", lambda _root: discovered)

    [result] = run_plugin_skill_context_dedup(plugin)

    assert result.passed
    assert result.metadata["work_limit_exceeded"] is True
    assert result.metadata["actual_skills"] == MAX_PLUGIN_DEDUP_SKILLS + 1
