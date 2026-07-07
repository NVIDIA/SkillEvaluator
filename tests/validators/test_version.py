# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for optional semantic resource version validation."""

from pathlib import Path

import pytest

from skillevaluator.validators.version import VersionValidator


def _write_skill(
    tmp_path: Path,
    version_line: str = "",
    *,
    top_level_version: str = "",
) -> Path:
    skill_dir = tmp_path / "versioned-skill"
    skill_dir.mkdir()

    top_level = f"version: {top_level_version}\n" if top_level_version else ""
    metadata = "metadata:\n  author: Test User <testuser@nvidia.com>\n"
    if version_line:
        metadata += f"  version: {version_line}\n"

    (skill_dir / "SKILL.md").write_text(
        f"""---
name: versioned-skill
description: A skill for testing optional semantic version validation.
{top_level}{metadata}---

# Versioned Skill

Use this skill to exercise version validation.
""",
        encoding="utf-8",
    )
    return skill_dir


def _finding_names(result) -> set[str]:
    return {finding.check_name for finding in result.findings}


def test_missing_version_is_valid_commit_hash_only_history(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path)

    result = VersionValidator(previous_version="1.2.0").validate(skill_dir)

    assert result.passed
    assert _finding_names(result) == set()
    assert any(detail.check_name == "version_optional" for detail in result.success_details)


@pytest.mark.parametrize("version", ["1.2", "1.2.3-alpha", "1.2.3+build", "v1.2.3"])
def test_malformed_version_is_rejected_when_present(
    tmp_path: Path,
    version: str,
) -> None:
    skill_dir = _write_skill(tmp_path, version)

    result = VersionValidator().validate(skill_dir)

    assert not result.passed
    assert "version_semver" in _finding_names(result)


def test_equal_version_label_is_rejected(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "1.2.0")

    result = VersionValidator(previous_version="1.2.0").validate(skill_dir)

    assert not result.passed
    assert "version_monotonic" in _finding_names(result)


def test_version_bump_is_allowed(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "1.2.1")

    result = VersionValidator(previous_version="1.2.0").validate(skill_dir)

    assert result.passed
    assert _finding_names(result) == set()


def test_version_downgrade_is_rejected(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "1.1.9")

    result = VersionValidator(previous_version="1.2.0").validate(skill_dir)

    assert not result.passed
    assert "version_monotonic" in _finding_names(result)


def test_malformed_previous_version_is_rejected(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "1.2.0")

    result = VersionValidator(previous_version="1.2").validate(skill_dir)

    assert not result.passed
    assert "previous_version_semver" in _finding_names(result)


def test_top_level_version_is_ignored_when_metadata_version_missing(tmp_path: Path) -> None:
    """Legacy top-level ``version`` (e.g., ``1.0``) must not trigger semver checks.

    Pre-proposal, quality-score recommended a top-level ``version`` without
    enforcing strict semver, so existing skills may carry non-semver values.
    The validator must scope the check to ``metadata.version`` only.
    """
    skill_dir = _write_skill(tmp_path, top_level_version='"1.0"')

    result = VersionValidator(previous_version="1.2.0").validate(skill_dir)

    assert result.passed
    assert _finding_names(result) == set()
    assert any(detail.check_name == "version_optional" for detail in result.success_details)


def test_top_level_version_is_ignored_when_metadata_version_present(tmp_path: Path) -> None:
    """A top-level ``version`` is shadowed by ``metadata.version``.

    Only ``metadata.version`` is authoritative, even when a legacy top-level
    ``version`` exists alongside it.
    """
    skill_dir = _write_skill(tmp_path, "1.2.0", top_level_version='"9.9.9"')

    result = VersionValidator(previous_version="1.1.9").validate(skill_dir)

    assert result.passed
    assert _finding_names(result) == set()
    assert any(
        detail.check_name == "version_semver" and detail.metadata.get("version") == "1.2.0"
        for detail in result.success_details
    )


def test_previous_version_falls_back_to_env_var(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``SKILLEVALUATOR_PREVIOUS_VERSION`` is the documented CI integration path.

    CI can set ``SKILLEVALUATOR_PREVIOUS_VERSION`` from the upstream branch tag
    and run ``skillevaluator validate`` without passing ``--previous-version``.
    ``VersionValidator()`` must therefore fall back to
    the env var when no constructor argument is provided, and downgrade
    detection has to fire just like the explicit-flag path. Regression
    test for ``previous_version or os.getenv(\"SKILLEVALUATOR_PREVIOUS_VERSION\")``
    in ``skillevaluator/validators/version.py``.
    """
    monkeypatch.setenv("SKILLEVALUATOR_PREVIOUS_VERSION", "1.2.0")
    skill_dir = _write_skill(tmp_path, "1.1.9")

    result = VersionValidator().validate(skill_dir)

    assert not result.passed
    assert "version_monotonic" in _finding_names(result)


def test_previous_version_env_var_allows_valid_bump(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clean bump under ``SKILLEVALUATOR_PREVIOUS_VERSION`` must pass.

    Mirrors :func:`test_previous_version_falls_back_to_env_var` for the
    happy path so we cover both sides of the env-var contract: CI sets
    the variable, the contributor bumps the patch, and validation
    succeeds.
    """
    monkeypatch.setenv("SKILLEVALUATOR_PREVIOUS_VERSION", "1.2.0")
    skill_dir = _write_skill(tmp_path, "1.2.1")

    result = VersionValidator().validate(skill_dir)

    assert result.passed
    assert _finding_names(result) == set()


def test_explicit_previous_version_overrides_env_var(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructor arg wins over the env var.

    Local ``skillevaluator validate --previous-version=...`` runs must always
    win over a stale ``SKILLEVALUATOR_PREVIOUS_VERSION`` left in the user's
    shell. Asserting precedence here also locks the documented
    ``--previous-version`` flag semantics.
    """
    monkeypatch.setenv("SKILLEVALUATOR_PREVIOUS_VERSION", "9.9.9")
    skill_dir = _write_skill(tmp_path, "1.2.1")

    result = VersionValidator(previous_version="1.2.0").validate(skill_dir)

    assert result.passed
    assert _finding_names(result) == set()


def test_previous_version_unset_skips_monotonic_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``--previous-version`` and no env var, no downgrade check runs.

    Ensures the env-var fallback doesn't accidentally pick up an empty
    string or a left-over value from another test, and that a fresh CI
    run (no previous tag yet) still passes when only a current version
    is supplied.
    """
    monkeypatch.delenv("SKILLEVALUATOR_PREVIOUS_VERSION", raising=False)
    skill_dir = _write_skill(tmp_path, "1.2.1")

    result = VersionValidator().validate(skill_dir)

    assert result.passed
    assert "version_monotonic" not in _finding_names(result)
    assert "previous_version_semver" not in _finding_names(result)


def test_removing_version_label_is_allowed_even_with_previous_version(tmp_path: Path) -> None:
    """Opting back out of a semver label is intentional and must pass.

    The proposal models ``metadata.version`` as an **opt-in re-tag** on
    top of immutable commit-hash history. A skill that previously
    published ``metadata.version: 1.2.0`` and now removes the label
    entirely is therefore allowed, even when ``--previous-version`` is
    supplied: the contributor is choosing to fall back to commit-hash
    history rather than continue re-tagging. The validator must short-
    circuit on the empty label and never evaluate the previous-version
    bound.

    Locks in the design decision documented on
    :meth:`VersionValidator._validate_single_skill`. If we ever decide
    that removing a published label should be a regression, this test
    must change in lockstep with that policy shift so the behavior
    change is explicit rather than incidental.
    """
    skill_dir = _write_skill(tmp_path)

    result = VersionValidator(previous_version="1.2.0").validate(skill_dir)

    assert result.passed
    assert _finding_names(result) == set(), (
        f"removing metadata.version must not emit any finding; got {_finding_names(result)}"
    )
    optional_details = [d for d in result.success_details if d.check_name == "version_optional"]
    assert optional_details, "expected a version_optional success detail"
    assert "opting back out" in optional_details[0].message.lower()
    assert optional_details[0].metadata.get("previous_version") == "1.2.0", (
        "the previous_version bound should still be reflected on the success detail "
        "for observability, even though it does not gate the result"
    )
