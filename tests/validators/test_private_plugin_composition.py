# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_declared_skill_in_excluded_directory_receives_all_checks(tmp_path):
    root = bundle(tmp_path)
    manifest = root / ".claude-plugin/plugin.json"
    manifest.write_text(json.dumps({"name": "demo", "skills": ["./evals/client-skill"]}))
    skill = root / "evals/client-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: client-skill\ndescription: Demo\nmetadata:\n  author: Demo <demo@example.com>\n  version: invalid\n---\n[missing](client-absent.md)\n"
    )
    results = run_plugin_validation(root, options=PrivatePluginChecks(checks=frozenset({"version", "hygiene"})))
    assert any(r.metadata.get("plugin_member") == "evals/client-skill" and r.findings for r in results)
    assert sum("client-absent.md" in error for r in results for error in r.errors) == 1


def test_staged_security_view_keeps_bytecode(tmp_path, monkeypatch):
    root = bundle(tmp_path)
    bytecode = root / "__pycache__/module.pyc"
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"bytecode")
    from types import SimpleNamespace

    from skillevaluator.models.result import ValidationResult

    seen = []

    def scan(path):
        seen.extend(p.relative_to(path).as_posix() for p in path.rglob("*.pyc"))
        return ValidationResult()

    monkeypatch.setattr(
        "skillevaluator.tier1.plugin_checks._tree_checks",
        lambda: {"security": (SimpleNamespace(name="Security", description="test"), scan)},
    )
    run_plugin_validation(root, options=PrivatePluginChecks(checks=frozenset({"security"})))
    assert seen == ["__pycache__/module.pyc"]


def test_each_bundled_skill_license_is_checked_beside_root_license(tmp_path):
    root = bundle(tmp_path)
    (root / "LICENSE").write_text("SPDX-License-Identifier: MIT\n")
    manifest = root / "skills/a/SKILL.md"
    manifest.write_text(
        manifest.read_text().replace("description: Demo skill", "description: Demo skill\nlicense: AGPL-3.0-only")
    )
    results = run_plugin_validation(root, options=PrivatePluginChecks(checks=frozenset({"license"})))
    assert any(r.metadata.get("plugin_member") == "skills/a" and not r.passed for r in results)


def test_bundled_skill_pii_uses_its_own_author_without_sharing_identity(tmp_path, monkeypatch):
    from skillevaluator.validators.security import SecurityValidator

    root = bundle(tmp_path)
    for name, author, other in (("a", "alice", "bob"), ("b", "bob", "alice")):
        manifest = root / "skills" / name / "SKILL.md"
        manifest.write_text(manifest.read_text().replace("demo@example.com", f"{author}@example.com"))
        (manifest.parent / "notes.md").write_text(
            f"Personal: /home/{author}/projects/private\nOther: /home/{other}/examples\n"
        )
    (root / "README.md").write_text("Root-owned portable example: /home/alice/examples\n")
    validator = SecurityValidator(submitter_usernames=[])
    monkeypatch.setattr(
        "skillevaluator.tier1.plugin_checks._tree_checks", lambda: {"pii": (validator, validator._scan_for_pii)}
    )
    results = run_plugin_validation(root, options=PrivatePluginChecks(checks=frozenset({"pii"})))
    found = [
        (result.metadata.get("plugin_member"), finding)
        for result in results
        for finding in result.findings
        if finding.check_name == "home_paths"
    ]
    assert {member for member, _ in found} == {"skills/a", "skills/b"}
    assert len(found) == 2
    for member, finding in found:
        assert finding.file_path == f"{member}/notes.md"
        assert finding.metadata["occurrence_count"] == 1
        assert finding.metadata["occurrences"][0]["file"] == f"{member}/notes.md"


def test_hygiene_audits_dependencies_and_test_presence_for_each_skill(tmp_path):
    from skillevaluator.constants import BANNED_PACKAGES

    root = bundle(tmp_path)
    (root / "requirements.txt").write_text("root-unpinned\n")
    (root / "skills/a/requirements.txt").write_text(BANNED_PACKAGES[0] + "==1.0\n")
    (root / "skills/b/requirements-dev.txt").write_text("bundled-unpinned\n")
    (root / "skills/a/test_real.py").write_text("def test_example():\n    assert True\n")
    results = run_plugin_validation(root, options=PrivatePluginChecks(checks=frozenset({"hygiene"})))
    hygiene = next(result for result in results if result.validator_name == "Code Integrity & Hygiene")
    assert "plugin_member" not in hygiene.metadata
    assert sum("Banned package" in error and "skills/a" in error for error in hygiene.errors) == 1
    assert sum("root-unpinned" in warning for warning in hygiene.warnings) == 1
    assert sum("bundled-unpinned" in warning and "skills/b" in warning for warning in hygiene.warnings) == 1
    discovery = [detail for detail in hygiene.success_details if detail.check_name == "test_discovery"]
    assert {detail.metadata.get("plugin_member"): detail.metadata["test_count"] for detail in discovery} == {
        ".": 0,
        "skills/a": 1,
        "skills/b": 0,
    }


def test_hygiene_cross_skill_links_use_the_complete_snapshot_once(tmp_path):
    root = bundle(tmp_path)
    (root / "README.md").write_text("[skill](skills/a/SKILL.md)\n")
    (root / "skills/a/notes.md").write_text(
        "[peer](../b/SKILL.md)\n[root](../../README.md)\n[missing](absent-cross.md)\n"
    )
    results = run_plugin_validation(root, options=PrivatePluginChecks(checks=frozenset({"hygiene"})))
    hygiene = next(result for result in results if result.validator_name == "Code Integrity & Hygiene")
    assert sum("absent-cross.md" in error for error in hygiene.errors) == 1
    assert not any(
        "../b/SKILL.md" in error or "../../README.md" in error or "skills/a/SKILL.md" in error
        for error in hygiene.errors
    )


def test_scanner_partitions_own_each_file_at_the_deepest_skill_once(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from skillevaluator.models.result import ValidationResult

    root = bundle(tmp_path)
    manifest = root / ".claude-plugin/plugin.json"
    manifest.write_text(json.dumps({"name": "demo", "skills": ["./skills/a", "./skills/a/child", "./skills/b"]}))
    child = root / "skills/a/child"
    child.mkdir()
    (child / "SKILL.md").write_text("---\nname: child\ndescription: Child\n---\n# Child\n")
    (root / "README.md").write_text("ROOT_MARKER")
    (root / "skills/a/notes.md").write_text("PARENT_MARKER")
    (child / "notes.md").write_text("CHILD_MARKER")
    seen = []

    def scan(path):
        markers = [file.read_text() for file in path.rglob("notes.md")]
        markers.extend(file.read_text() for file in path.glob("README.md"))
        seen.append(markers)
        return ValidationResult()

    monkeypatch.setattr(
        "skillevaluator.tier1.plugin_checks._tree_checks",
        lambda: {"security": (SimpleNamespace(name="Security", description="partition contract"), scan)},
    )
    results = run_plugin_validation(root, options=PrivatePluginChecks(checks=frozenset({"security"})))
    assert {
        marker: sum(marker in view for view in seen) for marker in ("ROOT_MARKER", "PARENT_MARKER", "CHILD_MARKER")
    } == {"ROOT_MARKER": 1, "PARENT_MARKER": 1, "CHILD_MARKER": 1}
    assert all(not ("PARENT_MARKER" in view and "CHILD_MARKER" in view) for view in seen)
    assert {
        result.metadata.get("plugin_member") for result in results if result.metadata.get("plugin_check") == "security"
    } == {None, "skills/a", "skills/a/child", "skills/b"}


def test_unrelated_numeric_occurrence_metadata_is_preserved(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from skillevaluator.models.result import Finding, Severity, ValidationResult

    root = bundle(tmp_path)

    def scan(_path):
        result = ValidationResult()
        result.add_finding(
            Finding(
                category="TEST",
                severity=Severity.LOW,
                check_name="test",
                message="fixture",
                file_path="SKILL.md",
                metadata={"occurrences": 2},
            )
        )
        return result

    monkeypatch.setattr(
        "skillevaluator.tier1.plugin_checks._tree_checks",
        lambda: {"security": (SimpleNamespace(name="Security", description="metadata compatibility"), scan)},
    )
    results = run_plugin_validation(root, options=PrivatePluginChecks(checks=frozenset({"security"})))
    scans = [result for result in results if result.metadata.get("plugin_check") == "security"]
    assert len(scans) == 3
    assert all(result.findings[0].metadata["occurrences"] == 2 for result in scans)


def test_one_unit_scanner_failure_is_incomplete_and_keeps_later_coverage(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from skillevaluator.models.result import ValidationResult

    root = bundle(tmp_path)
    seen = []

    def scan(path):
        seen.append(path.name)
        if path.name == "a":
            raise OSError("private diagnostic must not be published")
        return ValidationResult()

    monkeypatch.setattr(
        "skillevaluator.tier1.plugin_checks._tree_checks",
        lambda: {"security": (SimpleNamespace(name="Security", description="failure isolation"), scan)},
    )
    results = run_plugin_validation(root, options=PrivatePluginChecks(checks=frozenset({"security"})))
    assert "a" in seen and "b" in seen
    assert any(result.metadata.get("plugin_member") == "skills/a" and result.is_incomplete for result in results)
    assert any(result.metadata.get("plugin_member") == "skills/b" and not result.is_incomplete for result in results)
    assert "private diagnostic" not in str(results)


def test_banned_dependency_remains_confirmed_when_another_unit_is_incomplete(tmp_path, monkeypatch):
    from skillevaluator.constants import BANNED_PACKAGES
    from skillevaluator.validators.hygiene import HygieneValidator

    root = bundle(tmp_path)
    (root / "skills/a/requirements.txt").write_text(BANNED_PACKAGES[0] + "==1.0\n")
    original = HygieneValidator._check_test_presence

    def test_presence(self, path):
        if path.name == "b":
            raise OSError("fixture discovery unavailable")
        return original(self, path)

    monkeypatch.setattr(HygieneValidator, "_check_test_presence", test_presence)
    results = run_plugin_validation(root, options=PrivatePluginChecks(checks=frozenset({"hygiene"})))
    hygiene = next(result for result in results if result.validator_name == "Code Integrity & Hygiene")
    assert hygiene.is_incomplete
    assert any("Banned package" in error for error in hygiene.errors)
    assert hygiene.metadata.get("confirmed_failure") is True


def test_hygiene_scanner_failure_alone_is_not_a_confirmed_content_failure(tmp_path, monkeypatch):
    from skillevaluator.validators.hygiene import HygieneValidator

    root = bundle(tmp_path)

    def unavailable(*_args):
        raise OSError("fixture scanner unavailable")

    monkeypatch.setattr(HygieneValidator, "_check_test_presence", unavailable)
    results = run_plugin_validation(root, options=PrivatePluginChecks(checks=frozenset({"hygiene"})))
    hygiene = next(result for result in results if result.validator_name == "Code Integrity & Hygiene")
    assert hygiene.is_incomplete
    assert hygiene.metadata.get("confirmed_failure") is not True


def test_dead_link_is_confirmed_when_test_discovery_is_incomplete(tmp_path, monkeypatch):
    from skillevaluator.validators.hygiene import HygieneValidator

    root = bundle(tmp_path)
    (root / "README.md").write_text("[missing](absent.md)")

    def unavailable(*_args):
        raise OSError("fixture scanner unavailable")

    monkeypatch.setattr(HygieneValidator, "_check_test_presence", unavailable)
    results = run_plugin_validation(root, options=PrivatePluginChecks(checks=frozenset({"hygiene"})))
    hygiene = next(result for result in results if result.validator_name == "Code Integrity & Hygiene")
    assert hygiene.is_incomplete and any("absent.md" in error for error in hygiene.errors)
    assert hygiene.metadata.get("confirmed_failure") is True


@pytest.mark.parametrize(
    "source_mode,expected_mode", [(0o755, 0o755), (0o444, 0o444), (0o4755, 0o755), (0o2755, 0o755)]
)
def test_scan_views_preserve_permissions_without_privilege_bits(tmp_path, monkeypatch, source_mode, expected_mode):
    import os
    from types import SimpleNamespace

    from skillevaluator.models.result import ValidationResult

    if os.name != "posix":
        pytest.skip("POSIX file permission semantics")
    root = bundle(tmp_path)
    for script in (root / "root-probe.sh", root / "skills/a/probe.sh"):
        script.write_text("#!/bin/sh\nprintf fixture\n")
        script.chmod(source_mode)
    observed = []

    def inspect(path, scope):
        for script in path.rglob("*probe.sh"):
            observed.append((scope, script.name, script.stat().st_mode & 0o7777))
        return ValidationResult()

    tree_validator = SimpleNamespace(name="Security", description="Mode-observing fixture")
    skill_validator = SimpleNamespace(
        name="Version", description="Snapshot-observing fixture", validate=lambda path: inspect(path, "snapshot")
    )
    monkeypatch.setattr(
        "skillevaluator.tier1.plugin_checks._tree_checks",
        lambda: {"security": (tree_validator, lambda path: inspect(path, "partition"))},
    )
    monkeypatch.setattr("skillevaluator.tier1.plugin_checks._skill_checks", lambda: {"version": skill_validator})
    run_plugin_validation(root, options=PrivatePluginChecks(checks=frozenset({"security", "version"})))
    assert sorted(observed) == sorted(
        [
            ("partition", "root-probe.sh", expected_mode),
            ("partition", "probe.sh", expected_mode),
            ("snapshot", "probe.sh", expected_mode),
        ]
    )
