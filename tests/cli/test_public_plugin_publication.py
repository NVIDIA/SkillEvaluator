# SPDX-License-Identifier: Apache-2.0
"""Completed plugin checks certify their actual source and applicable lineup."""

import json
from pathlib import Path

import pytest

from skillevaluator.models.result import ValidationResult
from skillevaluator.reporting.base import (
    assess_publication,
    missing_tier1_publication_checks,
    result_has_execution_evidence,
)
from skillevaluator.tier1.commands import run_validation
from skillevaluator.utils.secure_fs import SecurePathError
from skillevaluator.utils.tool_runner import ToolResult, Tools
from skillevaluator.validators.security import SecurityValidator

CORE_CHECKS = "schema,security,pii,license,code-integrity,unicode"


@pytest.fixture
def clean_scanners(monkeypatch):
    def skillspector(*_args, **_kwargs):
        result = ValidationResult()
        result.add_success("skillspector", "Scanner completed without findings")
        return result

    monkeypatch.setattr(SecurityValidator, "_run_skillspector", skillspector)
    monkeypatch.setattr(Tools.gitleaks, "_path", "/fixture/gitleaks")
    monkeypatch.setattr(
        Tools.gitleaks,
        "run",
        lambda *_args, **_kwargs: ToolResult(success=True, stdout="[]", stderr="", exit_code=0),
    )


def plugin(root: Path, *, with_member: bool) -> Path:
    manifest = root / ".claude-plugin/plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"name": "demo", "version": "1.0.0", "description": "Demo plugin"}), encoding="utf-8"
    )
    license_text = (Path(__file__).parents[2] / "LICENSE").read_text(encoding="utf-8")
    (root / "LICENSE").write_text(license_text, encoding="utf-8")
    if with_member:
        member = root / "skills/simple"
        member.mkdir(parents=True)
        fixture = (Path(__file__).parents[1] / "fixtures/skills/simple/SKILL.md").read_text(encoding="utf-8")
        (member / "SKILL.md").write_text(
            fixture.replace("name: simple", "name: simple\nlicense: Apache-2.0"), encoding="utf-8"
        )
        (member / "LICENSE").write_text(license_text, encoding="utf-8")
    return root


def assessment(results):
    for result in results:
        result.metadata["benchmark_policy"] = {"tier2_required": False, "tier3_required": False}
    return assess_publication(results, expected_skill_name="demo")


@pytest.mark.parametrize("with_member", [False, True])
def test_complete_plugin_lineup_has_publication_execution_evidence(tmp_path, clean_scanners, with_member):
    root = plugin(tmp_path / "demo", with_member=with_member)
    results = run_validation(root, content_type="plugin", min_score=0, on_check=lambda _check: None)

    assert all(result.passed and not result.is_incomplete for result in results)
    assert all(result_has_execution_evidence(result) for result in results)
    assert assessment(results).status == "pass"


def test_core_only_run_cannot_certify_discovered_member_checks(tmp_path, clean_scanners):
    root = plugin(tmp_path / "demo", with_member=True)
    results = run_validation(root, checks=CORE_CHECKS, content_type="plugin", on_check=lambda _check: None)

    assert all(result.passed and not result.is_incomplete for result in results)
    assert missing_tier1_publication_checks(results) == ("lint", "quality", "version")
    assert assessment(results).status == "incomplete"


@pytest.mark.parametrize("mutation", ["unscoped-quality", "different-parent"])
def test_member_quality_does_not_hide_other_target_identity_conflicts(tmp_path, clean_scanners, mutation):
    root = plugin(tmp_path / "demo", with_member=True)
    results = run_validation(root, content_type="plugin", min_score=0, on_check=lambda _check: None)
    quality = next(result for result in results if result.metadata.get("plugin_check") == "quality")
    assert quality.metadata["quality_scores"]["skill_name"] == "simple"
    if mutation == "unscoped-quality":
        quality.metadata.pop("plugin_member")
    else:
        quality.metadata["publication_target"]["skill_name"] = "different-parent"

    assert assessment(results).status == "incomplete"
    assert "Validation results contain conflicting target skill identities." in assessment(results).reasons


@pytest.mark.parametrize("manifest", [None, "{broken"])
def test_missing_or_malformed_manifest_cannot_claim_complete_schema(tmp_path, manifest):
    root = plugin(tmp_path / "demo", with_member=False)
    target = root / ".claude-plugin/plugin.json"
    target.unlink() if manifest is None else target.write_text(manifest, encoding="utf-8")
    results = run_validation(root, checks="schema", content_type="plugin", on_check=lambda _check: None)

    assert not results[0].passed
    assert not any(detail.check_name == "plugin_schema" for detail in results[0].success_details)
    assert assessment(results).status != "pass"


def test_failed_snapshot_does_not_claim_completed_tree_coverage(tmp_path, clean_scanners, monkeypatch):
    root = plugin(tmp_path / "demo", with_member=True)

    def changed(*_args, **_kwargs):
        raise SecurePathError("source_changed", "Source changed while staging")

    monkeypatch.setattr("skillevaluator.tier1.plugin_checks._snapshot", changed)
    results = run_validation(root, checks=CORE_CHECKS, content_type="plugin", on_check=lambda _check: None)
    coverage = next(result for result in results if result.validator_name == "Plugin Tree Coverage")

    assert coverage.is_incomplete
    assert not coverage.success_details
    assert assessment(results).status != "pass"
