# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from skillevaluator.deduplication.plugin.intra_plugin_validator import IntraPluginValidator
from skillevaluator.deduplication.plugin.ref_utils import find_duplicate_refs, normalize_ref
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.reporting.cli import CLIReporter
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
    assert result.metadata["skipped"] is True


def test_every_plugin_tier2_skip_is_reported_as_skipped(tmp_path: Path) -> None:
    (tmp_path / "agent_plugin.yaml").write_text("name: [unterminated", encoding="utf-8")

    results = run_plugin_dedup_scan(tmp_path, run_context=False)

    # Check A (unparseable manifest), C-intra (no provider), C-inter and B (no catalog).
    assert [result.metadata["skipped"] for result in results] == [True, True, True, True]
    assert all(result.metadata["skip_reason"] in result.warnings for result in results)
    rendered = CLIReporter().render_all(results[:2])
    assert rendered.count("Skipped (see warnings)") == 2
    assert "OK" not in rendered


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
    # Check A, C-intra, and the local-catalog C-inter and B (skipped without a catalog).
    assert len(results) == 4
    assert all(result.passed for result in results)
    assert all(result.metadata.get("advisory_tier2") for result in results)
    assert all(result.metadata["execution_status"] == "skipped" for result in results[1:])


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


def test_hard_linked_bundled_skill_file_is_a_blocking_security_failure(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    skill = plugin / "skills" / "foo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: foo\ndescription: d\n---\n# Foo\n", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("# Outside notes\n", encoding="utf-8")
    try:
        os.link(outside, skill / "ref.md")
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {exc}")

    [result] = run_plugin_skill_context_dedup(plugin)

    assert not result.passed
    assert result.metadata["security_failure"] is True
    assert result.metadata["execution_status"] == "failed"
    assert result.metadata["optional"] is False
    assert [finding.check_name for finding in result.findings] == ["unsafe_hardlink"]
    assert result.findings[0].severity == Severity.CRITICAL


def test_plugin_context_scan_keeps_a_raised_unsafe_path_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier2 import commands
    from skillevaluator.utils.secure_fs import SecurePathError

    plugin = tmp_path / "plugin"
    skill = plugin / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: d\n---\n# Demo\n", encoding="utf-8")

    class UnsafePathValidator:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def validate(self, _skill_dir: Path) -> ValidationResult:
            raise SecurePathError("path_identity_changed", "Path changed while reading: SKILL.md")

    monkeypatch.setattr(commands, "IntraSkillValidator", UnsafePathValidator)

    [result] = run_plugin_skill_context_dedup(plugin)

    assert not result.passed
    assert result.metadata["security_failure"] is True
    assert result.metadata["execution_status"] == "failed"
    assert [(finding.check_name, finding.severity) for finding in result.findings] == [
        ("unsafe_plugin_filesystem", Severity.HIGH)
    ]


def test_plugin_context_scan_caps_findings_once_and_keeps_plain_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier2 import commands

    plugin = tmp_path / "plugin"
    skill = plugin / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: d\n---\n# Demo\n", encoding="utf-8")
    skill_result = ValidationResult(validator_name="Context Deduplication")
    skill_result.add_finding(Finding("DUPLICATE", Severity.HIGH, "duplicate", "Repeated instructions", "SKILL.md"))
    skill_result.add_warning("Optional provider note")
    skill_result.add_error("LLM analysis did not complete for 1 of 2 content clusters")

    class FixedResultValidator:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def validate(self, _skill_dir: Path) -> ValidationResult:
            return skill_result

    monkeypatch.setattr(commands, "IntraSkillValidator", FixedResultValidator)

    [result] = run_plugin_skill_context_dedup(plugin)

    assert result.passed
    assert result.metadata["advisory_tier2"] is True
    assert [finding.severity for finding in result.findings] == [Severity.MEDIUM]
    assert result.errors == []
    assert result.warnings == [
        "[demo] [DUPLICATE-MEDIUM] Repeated instructions in SKILL.md",
        "[demo] Optional provider note",
        "[demo] LLM analysis did not complete for 1 of 2 content clusters",
    ]
    assert result.summary.warnings == 3
    assert result.summary.errors == 0
    assert result.summary.high_count == 0
    assert result.summary.medium_count == 1


def test_plugin_context_scan_caps_single_skill_llm_budget_at_cluster_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.constants import CONTENT_DEDUP_MAX_LLM_CLUSTERS, MAX_PLUGIN_DEDUP_LLM_CALLS
    from skillevaluator.deduplication.intra_skill import intra_skill_validator

    assert MAX_PLUGIN_DEDUP_LLM_CALLS > CONTENT_DEDUP_MAX_LLM_CLUSTERS
    plugin = tmp_path / "plugin"
    skill = plugin / "skills" / "only"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: only\ndescription: d\n---\n## Section A\n" + "a" * 200 + "\n## Section B\n" + "b" * 200,
        encoding="utf-8",
    )
    embedding_client = MagicMock()
    embedding_client.return_value.embed.return_value = [[1.0, 0.0], [0.0, 1.0]]
    llm_client = MagicMock()
    monkeypatch.setattr(intra_skill_validator, "EmbeddingClient", embedding_client)
    monkeypatch.setattr(intra_skill_validator, "LLMClient", llm_client)

    [result] = run_plugin_skill_context_dedup(plugin)

    assert result.passed
    assert result.findings == []
    assert result.metadata["max_llm_calls"] == CONTENT_DEDUP_MAX_LLM_CLUSTERS
    embedding_client.return_value.embed.assert_called_once()
    llm_client.assert_not_called()


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
    assert result.metadata["skipped"] is True
    assert result.warnings == [result.metadata["skip_reason"]]


def test_skill_level_finding_points_at_the_bundled_skill_in_every_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Tier 2 finding about a whole bundled skill points at skills/foo, not skills/foo/foo."""
    from click.testing import CliRunner

    from skillevaluator.cli import cli
    from skillevaluator.deduplication.intra_skill import intra_skill_validator

    plugin = tmp_path / "demo-plugin"
    (plugin / ".claude-plugin").mkdir(parents=True)
    (plugin / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": "demo-plugin", "version": "1.0.0", "description": "Demo plugin."}), encoding="utf-8"
    )
    skill = plugin / "skills" / "foo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: foo\ndescription: Use when you need foo.\n---\n\n## Section A\n"
        + "alpha " * 60
        + "\n\n## Section B\n"
        + "bravo " * 60,
        encoding="utf-8",
    )
    embedding_client = MagicMock()
    embedding_client.return_value.embed.return_value = [[1.0, 0.0], [0.0, 1.0]]
    monkeypatch.setattr(intra_skill_validator, "EmbeddingClient", embedding_client)
    # Two chunks of two-dimensional vectors are 2 scalar comparisons, over a limit of 1.
    monkeypatch.setattr(intra_skill_validator, "CONTENT_DEDUP_MAX_SCALAR_COMPARISONS", 1)
    monkeypatch.setenv("SKILL_EVAL_EMBEDDING_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    out = tmp_path / "reports"

    CliRunner().invoke(
        cli,
        ["validate", str(plugin), "--tiers", "1,2", "--checks", "schema", "-r", "json,sarif,markdown", "-o", str(out)],
        catch_exceptions=False,
    )

    report = json.loads(next(path for path in out.glob("*.json") if not path.name.endswith(".sarif.json")).read_text())
    [finding] = [
        finding
        for result in report["results"]
        for finding in result["findings"]
        if finding["check_name"] == "scalar_comparison_limit"
    ]
    assert finding["file_path"] == "[foo] ."
    sarif = json.loads(next(out.glob("*.sarif.json")).read_text())
    [located] = [
        item for item in sarif["runs"][0]["results"] if item["properties"]["checkName"] == "scalar_comparison_limit"
    ]
    assert located["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == "skills/foo"
    assert located["properties"]["pluginComponent"]["path"] == "skills/foo"
    [markdown] = [path for path in out.glob("*.md") if path.name != "BENCHMARK.md"]
    assert "<code>[foo] .</code>" in markdown.read_text()
