# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public validation-policy behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from skillevaluator.models.result import Severity
from skillevaluator.validators import policy as policy_module
from skillevaluator.validators.policy import (
    DEFAULT_PROFILE_NAME,
    ValidationPolicy,
    default_policy,
    load_policy_file,
    load_profile,
    resolve_policy,
)


def test_default_profile_is_the_public_profile() -> None:
    policy = default_policy()

    assert DEFAULT_PROFILE_NAME == "external"
    assert policy.profile == "external"
    assert policy.author_email_regex is None
    assert policy.is_author_email_acceptable("Jane Doe <jane@example.com>")
    assert policy.severity_for("SCHEMA", "author_missing", Severity.LOW) == Severity.HIGH
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
    assert policy.severity_for("SCHEMA", "author_missing", Severity.LOW) == Severity.HIGH


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
    assert resolve_policy().profile == "external"
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

    assert data["audience"] == "external"
    assert data["digest"].startswith("sha256:")
    assert policy.digest == default_policy().digest


def test_policy_digest_excludes_source_path() -> None:
    first = ValidationPolicy(profile="external", source=Path("/tmp/one.yaml"))
    second = ValidationPolicy(profile="external", source=Path("/tmp/two.yaml"))

    assert first.digest == second.digest
