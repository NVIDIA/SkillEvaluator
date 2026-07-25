# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public validation-policy behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from skillevaluator.models.result import Severity
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
