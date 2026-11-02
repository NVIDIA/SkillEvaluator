# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline regressions for PR243 follow-up discussion_r4425167476."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from skillevaluator.constants import CONTENT_DEDUP_MAX_FILE_BYTES
from skillevaluator.deduplication.plugin.profile import PluginProfileError
from skillevaluator.embedding.client import EmbeddingClient
from skillevaluator.embedding.registry import EmbeddingRegistry
from skillevaluator.inference.client import LLMClient
from skillevaluator.models.result import Severity, ValidationResult
from skillevaluator.plugin_skills import discover_openclaw_skill_manifests
from skillevaluator.tier2 import commands
from skillevaluator.utils.secure_fs import SecurePathError

_INVALID_MANIFESTS = [
    pytest.param(json.dumps({"id": "demo", "configSchema": {}, "skills": "./portable"}), id="skills-string"),
    pytest.param(json.dumps({"id": "demo", "configSchema": {}, "skills": None}), id="skills-null"),
    pytest.param(json.dumps({"id": "demo", "configSchema": {}, "skills": ["portable", 7]}), id="skills-nonstring"),
    pytest.param("[]", id="nonmapping"),
    pytest.param('{"id":', id="malformed-json"),
]


@pytest.fixture(autouse=True)
def refuse_paid_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid manifests must be classified before any provider or catalog work."""

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("invalid/unsafe input started embedding, LLM, context, or catalog work")

    monkeypatch.setattr(commands, "IntraSkillValidator", unexpected)
    monkeypatch.setattr(EmbeddingClient, "__init__", unexpected)
    monkeypatch.setattr(EmbeddingRegistry, "__init__", unexpected)
    monkeypatch.setattr(LLMClient, "__init__", unexpected)


def _manifest(root: Path, raw: str) -> Path:
    root.mkdir()
    (root / "openclaw.plugin.json").write_text(raw, encoding="utf-8")
    return root


def _assert_advisory_skip(result: ValidationResult) -> None:
    assert result.passed
    assert result.metadata["advisory_tier2"] is True
    assert result.metadata["skipped"] is True
    assert result.metadata["execution_status"] == "skipped"
    assert result.metadata["optional"] is True
    assert not result.metadata.get("security_failure")
    assert not result.findings
    assert not result.errors
    assert result.metadata["skip_reason"] in result.warnings
    assert "Unsafe plugin filesystem" not in result.metadata["skip_reason"]


@pytest.mark.parametrize("raw", _INVALID_MANIFESTS)
def test_cintra_invalid_native_declarations_have_profile_error_cause(tmp_path: Path, raw: str) -> None:
    plugin = _manifest(tmp_path / "plugin", raw)
    with pytest.raises(PluginProfileError):
        commands._openclaw_skill_dirs(plugin)


@pytest.mark.parametrize("raw", _INVALID_MANIFESTS)
def test_cintra_direct_invalid_manifest_is_advisory_without_paid_work(tmp_path: Path, raw: str) -> None:
    plugin = _manifest(tmp_path / "plugin", raw)
    # A supplied inventory cannot override an invalid native declaration.
    [result] = commands.run_plugin_skill_context_dedup(plugin, skill_dirs=[plugin / "skills/implicit"])
    assert result.validator_name == "Context Deduplication"
    _assert_advisory_skip(result)
    assert "manifest" in result.metadata["skip_reason"].lower()


@pytest.mark.parametrize("raw", _INVALID_MANIFESTS)
@pytest.mark.parametrize("with_catalog", [False, True])
@pytest.mark.parametrize("run_context", [False, True])
def test_cintra_invalid_manifest_keeps_all_four_real_checks(
    tmp_path: Path, raw: str, with_catalog: bool, run_context: bool
) -> None:
    plugin = _manifest(tmp_path / "plugin", raw)
    # A real profile rejection must precede opening this deliberately absent catalog.
    catalog = tmp_path / "never-read.json" if with_catalog else None
    results = commands.run_plugin_dedup_scan(plugin, catalog=catalog, run_context=run_context)
    assert [result.validator_name for result in results] == [
        "Plugin Dependency Deduplication",
        "Context Deduplication",
        "Inter-Skill Deduplication",
        "Inter-Plugin Deduplication",
    ]
    assert results[0].passed
    assert not results[0].metadata.get("security_failure")
    for result in results[1:]:
        _assert_advisory_skip(result)
    if with_catalog and run_context:
        assert all("comparable profile" in result.metadata["skip_reason"] for result in results[2:])


def _assert_blocking(result: ValidationResult) -> None:
    assert not result.passed
    assert result.metadata["security_failure"] is True
    assert result.metadata["execution_status"] == "failed"
    assert result.metadata["optional"] is False
    assert not result.metadata.get("skipped")
    assert [(finding.check_name, finding.severity) for finding in result.findings] == [
        ("unsafe_plugin_filesystem", Severity.HIGH)
    ]


@pytest.mark.parametrize(
    "kind", ["linked-manifest", "linked-root", "unreadable", "byte-budget", "path-budget", "traversal"]
)
def test_cintra_real_filesystem_refusals_remain_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    plugin = _manifest(tmp_path / "plugin", json.dumps({"id": "demo", "configSchema": {}, "skills": ["portable"]}))
    manifest = plugin / "openclaw.plugin.json"
    portable = plugin / "portable"
    portable.mkdir()
    if kind == "linked-manifest":
        outside = tmp_path / "outside.json"
        manifest.rename(outside)
        manifest.symlink_to(outside)
    elif kind == "linked-root":
        portable.rmdir()
        portable.symlink_to(tmp_path / "absent", target_is_directory=True)
    elif kind == "unreadable":
        manifest.chmod(0)
        if os.access(manifest, os.R_OK):
            manifest.chmod(0o600)
            pytest.skip("current account bypasses filesystem read permissions")
    elif kind == "byte-budget":
        manifest.write_text(" " * (CONTENT_DEDUP_MAX_FILE_BYTES + 1), encoding="utf-8")
    elif kind == "path-budget":
        (portable / "one.txt").touch()
        (portable / "two.txt").touch()

        def bounded_discovery(root: Path, data: dict) -> list:
            return discover_openclaw_skill_manifests(root, data, max_paths=1)

        monkeypatch.setattr(commands, "discover_openclaw_skill_manifests", bounded_discovery)
    else:
        manifest.write_text(json.dumps({"id": "demo", "configSchema": {}, "skills": ["../escape"]}), encoding="utf-8")
    try:
        with pytest.raises(SecurePathError):
            commands._openclaw_skill_dirs(plugin)
        [direct] = commands.run_plugin_skill_context_dedup(plugin)
        _assert_blocking(direct)
        results = commands.run_plugin_dedup_scan(plugin, catalog=tmp_path / "never-read.json")
        assert len(results) == 2
        _assert_blocking(results[1])
    finally:
        if kind == "unreadable":
            manifest.chmod(0o600)
