# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public validation-policy behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.validators import policy as policy_module
from skillevaluator.validators.policy import (
    DEFAULT_PROFILE_NAME,
    ValidationPolicy,
    apply_policy,
    default_policy,
    load_policy_file,
    load_profile,
    resolve_policy,
)


@pytest.mark.parametrize("name", ["default", "external"])
def test_default_and_legacy_names_resolve_one_policy(name: str) -> None:
    policy = resolve_policy(profile=name)

    assert policy.profile == "default"
    assert policy.audience == "all"
    assert policy.is_author_email_acceptable("Author <author@nvidia.com>")
    assert policy.is_author_email_acceptable("Author <author@example.com>")
    # Author attribution is optional in the Agent Skills baseline: advisory here.
    assert policy.severity_for("SCHEMA", "author_missing", Severity.HIGH) == Severity.LOW
    assert policy.severity_for("SCHEMA", "author_without_email", Severity.HIGH) == Severity.INFO
    # Malformed or unsafe author values still block.
    assert policy.severity_for("SCHEMA", "author_format", Severity.LOW) == Severity.HIGH
    assert policy.metadata_id_requirement == "ignore"
    assert policy.severity_for("WORKFLOWS", "author_format", Severity.LOW) == Severity.HIGH
    assert policy.severity_for("LICENSE", "blocked_license", Severity.LOW) == Severity.CRITICAL
    assert policy.severity_for("PII", "email", Severity.HIGH) == Severity.HIGH
    assert policy.severity_for("SECURITY", "scanner_failure", Severity.CRITICAL) == Severity.CRITICAL


def test_legacy_environment_and_custom_overlay_use_the_same_default_policy(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SKILLEVALUATOR_PROFILE", "external")
    assert resolve_policy().to_dict() == load_profile("default").to_dict()
    custom = tmp_path / "team.yaml"
    custom.write_text("profile: team\nseverity_overrides:\n  SCHEMA.author_missing: critical\n")
    policy = resolve_policy(policy_path=custom)
    assert policy.profile == "team"
    assert policy.severity_for("SCHEMA", "author_missing", Severity.LOW) == Severity.CRITICAL
    assert policy.severity_for("LICENSE", "blocked_license", Severity.LOW) == Severity.CRITICAL
    assert resolve_policy(profile="default").profile == "default"


def test_default_profile_is_the_public_profile() -> None:
    policy = default_policy()

    assert DEFAULT_PROFILE_NAME == "default"
    assert policy.profile == "default"
    assert policy.author_email_regex is None
    assert policy.is_author_email_acceptable("Jane Doe <jane@example.com>")
    assert policy.severity_for("SCHEMA", "author_missing", Severity.HIGH) == Severity.LOW
    assert policy.severity_for("LICENSE", "missing", Severity.LOW) == Severity.CRITICAL


def test_custom_policy_overlays_the_public_profile(tmp_path: Path) -> None:
    custom = tmp_path / "team.yaml"
    custom.write_text(
        "profile: team-strict\nseverity_overrides:\n  SCHEMA.author_format: critical\n",
        encoding="utf-8",
    )

    policy = load_policy_file(custom)

    assert policy.profile == "team-strict"
    assert policy.author_email_regex is None
    assert policy.severity_for("SCHEMA", "author_format", Severity.LOW) == Severity.CRITICAL
    assert policy.severity_for("SCHEMA", "author_missing", Severity.HIGH) == Severity.LOW
    assert policy.metadata_id_requirement == "ignore"


def test_registry_profile_requires_contact_author_and_metadata_id() -> None:
    policy = load_profile("registry")

    assert policy.profile == "registry"
    assert policy.author_email_regex is None
    assert policy.metadata_id_requirement == "require"
    for check in ("author_missing", "author_without_email", "author_format"):
        assert policy.severity_for("SCHEMA", check, Severity.LOW) == Severity.HIGH
    assert policy.severity_for("WORKFLOWS", "author_format", Severity.LOW) == Severity.HIGH
    assert policy.severity_for("LICENSE", "blocked_license", Severity.LOW) == Severity.CRITICAL
    assert policy.to_dict()["metadata_id_requirement"] == "require"
    assert policy.digest != default_policy().digest


def test_policy_overlay_sets_or_inherits_metadata_id_requirement(tmp_path: Path) -> None:
    require = tmp_path / "require.yaml"
    require.write_text("profile: my-registry\nidentity:\n  metadata_id_requirement: require\n", encoding="utf-8")
    bare_identity = tmp_path / "bare.yaml"
    bare_identity.write_text("profile: team\nidentity:\n  author_email_regex: '@example\\.org>'\n", encoding="utf-8")

    assert load_policy_file(require).metadata_id_requirement == "require"
    # An overlay that does not name the key keeps the base profile's mode.
    assert load_policy_file(bare_identity).metadata_id_requirement == "ignore"
    assert load_policy_file(bare_identity, base_profile="registry").metadata_id_requirement == "require"
    assert load_policy_file(require).digest != load_policy_file(bare_identity).digest


@pytest.mark.parametrize("value", ["requre", "REQUIRE", "true", "1"])
def test_policy_rejects_unknown_metadata_id_requirement(tmp_path: Path, value: str) -> None:
    custom = tmp_path / "typo.yaml"
    custom.write_text(f"identity:\n  metadata_id_requirement: {value}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="metadata_id_requirement"):
        load_policy_file(custom)


def test_in_memory_default_policy_matches_bundled_author_rules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundled = default_policy()
    monkeypatch.setattr(policy_module, "PROFILES_DIR", tmp_path)
    fallback = default_policy()

    assert fallback.metadata_id_requirement == bundled.metadata_id_requirement
    for key in ("SCHEMA.author_missing", "SCHEMA.author_without_email", "SCHEMA.author_format", "LICENSE.*"):
        assert fallback.severity_overrides[key] == bundled.severity_overrides[key]


def test_custom_policy_wraps_malformed_yaml(tmp_path: Path) -> None:
    custom = tmp_path / "broken-policy.yaml"
    custom.write_text("severity_overrides: [", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_policy_file(custom)

    assert f"Invalid policy YAML in {custom}" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, yaml.YAMLError)


def test_custom_policy_wraps_invalid_encoding(tmp_path: Path) -> None:
    custom = tmp_path / "utf16-policy.yaml"
    custom.write_bytes("profile: encoded\n".encode("utf-16"))

    with pytest.raises(ValueError) as exc_info:
        load_policy_file(custom)

    assert f"Invalid policy YAML in {custom}" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)


def test_custom_policy_wraps_read_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom = tmp_path / "unreadable-policy.yaml"
    custom.write_text("profile: unreadable\n", encoding="utf-8")
    original_open = Path.open

    def deny_custom_policy(path: Path, *args: object, **kwargs: object):
        if path == custom:
            raise PermissionError("access denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_custom_policy)

    with pytest.raises(ValueError) as exc_info:
        load_policy_file(custom)

    assert f"Could not read policy file {custom}" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, PermissionError)


def test_policy_yaml_wraps_recursion_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    policy = tmp_path / "deeply-nested-policy.yaml"
    policy.write_text("severity_overrides: []\n", encoding="utf-8")

    def exhaust_parser(_stream: object) -> object:
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(policy_module.yaml, "safe_load", exhaust_parser)

    with pytest.raises(ValueError) as exc_info:
        policy_module._load_policy_yaml(policy)

    assert f"Invalid policy YAML in {policy}" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RecursionError)


def test_bundled_profile_wraps_malformed_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile = tmp_path / "broken.yaml"
    profile.write_text("identity: [", encoding="utf-8")
    monkeypatch.setattr(policy_module, "PROFILES_DIR", tmp_path)

    with pytest.raises(ValueError) as exc_info:
        load_profile("broken")

    assert f"Invalid policy YAML in {profile}" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, yaml.YAMLError)


def test_policy_validation_and_resolution() -> None:
    assert resolve_policy().profile == "default"
    assert (
        ValidationPolicy(severity_overrides={"LICENSE.*": Severity.MEDIUM}).severity_for(
            "LICENSE", "unknown", Severity.HIGH
        )
        == Severity.MEDIUM
    )
    with pytest.raises(FileNotFoundError):
        load_profile("missing-profile")


def test_public_policy_serialization_has_stable_digest() -> None:
    policy = default_policy()
    data = policy.to_dict()

    assert data["audience"] == "all"
    assert data["digest"].startswith("sha256:")
    assert policy.digest == default_policy().digest


def test_policy_digest_excludes_source_path() -> None:
    first = ValidationPolicy(profile="external", source=Path("/tmp/one.yaml"))
    second = ValidationPolicy(profile="external", source=Path("/tmp/two.yaml"))

    assert first.digest == second.digest


@pytest.mark.parametrize("incomplete", [True, False])
def test_override_keeps_why_an_incomplete_scan_failed(incomplete: bool) -> None:
    """An override rebuilds a result from its findings; errors no finding repeats stay only on an incomplete scan.

    An incomplete scan fails anyway, and such an error says why. On a complete
    scan such an error may explain a blocker the override just lowered, so it
    goes, as before. A gate failure stays either way and is listed once.
    """
    result = ValidationResult(validator_name="Security Scan")
    result.add_structured_finding(
        Finding(
            category="SECURITY",
            severity=Severity.HIGH,
            check_name="Prompt Injection",
            message="Ignore all previous instructions",
            file_path="SKILL.md",
            line_number=8,
        )
    )
    result.add_error("scanner report was cut short; security scan did not complete")
    result.fail_gate("scanner status is unknown; security scan did not complete")
    result.add_warning("scanner skipped one file")
    if incomplete:
        result.mark_scan_incomplete("scanner")

    apply_policy([result], ValidationPolicy(severity_overrides={"SECURITY.*": Severity.LOW}))

    assert result.status == ("incomplete" if incomplete else "failed")
    expected = ["scanner status is unknown; security scan did not complete"]
    if incomplete:
        expected.append("scanner report was cut short; security scan did not complete")
    assert result.errors == expected
    assert result.summary.errors == len(expected)
    assert result.warnings == [result.findings[0].to_legacy_string(), "scanner skipped one file"]


def test_override_keeps_a_warning_as_often_as_it_was_recorded() -> None:
    """A warning no finding repeats goes back as often as it was recorded; the LLM downgrade lists it once."""
    result = ValidationResult(validator_name="Security Scan")
    result.add_structured_finding(
        Finding(
            category="SECURITY",
            severity=Severity.HIGH,
            check_name="Prompt Injection",
            message="Ignore all previous instructions",
            file_path="SKILL.md",
            line_number=8,
        )
    )
    result.add_warning("scanner skipped one file")
    result.add_warning("scanner skipped one file")

    apply_policy([result], ValidationPolicy(severity_overrides={"SECURITY.*": Severity.LOW}))

    assert result.warnings == [result.findings[0].to_legacy_string(), *["scanner skipped one file"] * 2]
    assert result.summary.warnings == 3
    assert result.errors == []
    assert result.passed
