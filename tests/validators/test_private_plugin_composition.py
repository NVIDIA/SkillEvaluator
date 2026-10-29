# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import json
from pathlib import Path

from skillevaluator.tier1.plugin_checks import PrivatePluginChecks, run_plugin_validation


def bundle(root: Path):
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin/plugin.json").write_text(json.dumps({"name": "demo"}))
    for name in ("a", "b"):
        skill = root / "skills" / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Demo skill\nmetadata:\n  author: Demo <demo@example.com>\n  version: invalid\n---\n# Demo\n"
        )
    return root


def test_whole_tree_and_each_skill_checks_keep_coverage(tmp_path):
    root = bundle(tmp_path)
    (root / "README.md").write_text("[missing](absent.md)")
    (root / "skills/a/help.md").write_text("[missing](absent-too.md)")
    checks = PrivatePluginChecks(checks=frozenset({"schema", "hygiene", "version", "quality", "lint"}))
    results = run_plugin_validation(root, options=checks)
    hygiene = next(r for r in results if r.validator_name == "Code Integrity & Hygiene")
    assert sum("absent.md" in error for error in hygiene.errors) == 1
    assert sum("absent-too.md" in error for error in hygiene.errors) == 1
    version = [r for r in results if r.validator_name == "Semantic Version Validation"]
    assert {r.metadata["plugin_member"] for r in version} == {"skills/a", "skills/b"}
    assert all(not r.passed for r in version)
    assert len([r for r in results if r.metadata.get("plugin_check") == "quality"]) == 2
    assert not any("skillevaluator-private-tree-" in str(r) for r in results)


def test_default_static_path_does_not_run_optional_collectors(tmp_path, monkeypatch):
    root = bundle(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("optional collector invoked")

    monkeypatch.setattr("skillevaluator.tier1.plugin_checks.run_dependency_audits", forbidden)
    monkeypatch.setattr("skillevaluator.tier1.plugin_checks.EndpointChecker.check", forbidden)
    monkeypatch.setattr("skillevaluator.tier1.plugin_checks.ClaudePluginValidateParity.validate", forbidden)
    result = run_plugin_validation(root, options=PrivatePluginChecks(checks=frozenset({"schema"})))
    assert result


def test_unsafe_tree_prevents_external_scanners(tmp_path, monkeypatch):
    root = bundle(tmp_path / "bundle")
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    (root / "skills/a/link.txt").symlink_to(outside)

    def forbidden(*args, **kwargs):
        raise AssertionError("scanner invoked on unsafe source")

    monkeypatch.setattr("skillevaluator.validators.secrets.SecretsValidator._validate_single_skill", forbidden)
    results = run_plugin_validation(root, options=PrivatePluginChecks(checks=frozenset({"schema", "secrets"})))
    assert any(not r.passed for r in results)
    assert any(r.is_incomplete for r in results)


def test_opt_in_endpoint_missing_evidence_stays_incomplete(tmp_path, monkeypatch):
    root = bundle(tmp_path)
    monkeypatch.setattr(
        "skillevaluator.tier1.plugin_checks.EndpointChecker.check", lambda _self, _targets: ({"incomplete": True}, [])
    )
    results = run_plugin_validation(
        root, options=PrivatePluginChecks(checks=frozenset({"schema"}), resolve_endpoints=True)
    )
    assert any(r.is_incomplete and r.validator_name == "Private Endpoint Checks" for r in results)
