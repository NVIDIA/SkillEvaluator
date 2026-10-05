# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check 3 (dependency resolution) at Tier 1: proof M9 (ref limit) and L8 (advice, [OK] rows, case, duplicates).

Fixtures copy the shapes of check-03 ``edge-10-ref-limit-skips-gate``, ``edge-01-no-origin-hides-missing``,
``edge-03-non-canonical-refs``, ``edge-07-origin-host-and-scheme`` (``plain-http``) and the skeptic's ``case``,
``kind`` and ``dup`` fixtures. Every repository is a local ``git init``; nothing is fetched.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from skillevaluator.models.result import Severity, ValidationResult
from skillevaluator.reporting.plugin_sections import MAX_TABLE_ROWS, dependency_view
from skillevaluator.validators.plugin_schema import PluginSchemaValidator

ORIGIN = "https://github.com/acme/agent-catalog.git"
REPO = "acme/agent-catalog"

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is unavailable")


def _clone(root: Path, origin: str | None = ORIGIN) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    if origin:
        subprocess.run(["git", "-C", str(root), "remote", "add", "origin", origin], check=True)
    return root


def _skill(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {directory.name.lower()}\ndescription: Shared skill.\n---\n# Skill\n", encoding="utf-8"
    )
    return directory


def _plugin(repo: Path, refs: list) -> Path:
    plugin = repo / "plugins" / "demo"
    plugin.mkdir(parents=True, exist_ok=True)
    manifest = {"name": "demo-plugin", "version": "1.0.0", "author": {"email": "dev@example.com"}}
    manifest["skills"] = {"refs": refs}
    (plugin / "agent_plugin.yaml").write_text(json.dumps(manifest), encoding="utf-8")
    return plugin


def _findings(result: ValidationResult, check: str) -> list:
    return [finding for finding in result.findings if finding.check_name == check]


def _ok_rows(result: ValidationResult) -> list[str]:
    return [detail.message for detail in result.success_details if detail.check_name == "plugin_dependencies"]


@requires_git
def test_edge10_more_than_256_refs_keeps_the_gate_and_the_absent_ref(tmp_path: Path) -> None:
    """Proof M9, check-03 edge-10: 257 refs turned the gate off; the absent ghost-skill vanished; exit 0."""
    repo = _clone(tmp_path / "repo")
    _skill(repo / "skills" / "shared-skill")
    refs = [f"github::other-org/repo-{index}::skills::vendor" for index in range(256)]
    refs.append(f"github::{REPO}::skills::ghost-skill")

    result = PluginSchemaValidator().validate(_plugin(repo, refs))

    [missing] = _findings(result, "plugin_dependency_missing")
    assert missing.severity == Severity.HIGH
    assert missing.metadata["ref"] == f"github::{REPO}::skills::ghost-skill"
    assert not result.passed
    [limit] = _findings(result, "plugin_dependency_limit")
    assert limit.severity == Severity.MEDIUM
    assert limit.metadata == {"section": "skills", "declared": 257, "limit": 256}
    counts = result.metadata["plugin"]["dependency_status_counts"]
    assert (counts["external"], counts["missing"]) == (256, 1)
    # The report table is cut at MAX_TABLE_ROWS rows, but never drops the blocking ref.
    view = dependency_view(result.metadata["plugin"])
    assert view is not None and len(view["rows"]) == MAX_TABLE_ROWS
    assert any(row["state"] == "missing" for row in view["rows"])


@requires_git
def test_edge01_gate_that_could_not_run_is_a_finding_not_an_ok_row(tmp_path: Path) -> None:
    """Proof L8, check-03 edge-01: "the gate could not be evaluated" was an [OK] row, and the advice was generic."""
    repo = _clone(tmp_path / "repo", origin=None)
    _skill(repo / "skills" / "shared-skill")
    refs = [f"github::{REPO}::skills::shared-skill", f"github::{REPO}::skills::ghost-skill"]

    result = PluginSchemaValidator().validate(_plugin(repo, refs))

    assert _ok_rows(result) == []
    [unverified] = _findings(result, "plugin_dependency_unverified")
    assert unverified.severity == Severity.MEDIUM
    assert "has no 'origin' remote" in unverified.message
    assert unverified.metadata["ref_count"] == 2
    assert result.passed  # advisory, never blocking


@requires_git
def test_edge03_malformed_refs_in_a_good_clone_get_advice_about_the_ref(tmp_path: Path) -> None:
    """Proof L8, check-03 edge-03: malformed refs told the user to add an origin, which the clone already had."""
    repo = _clone(tmp_path / "repo")
    _skill(repo / "skills" / "shared-skill")
    refs = [
        f"github::{REPO}::skills",
        f"github::{REPO}::skills::shared-skill::extra",
        f"github::{REPO}::skills::../rules/style.md",
        f"github::{REPO}::secrets::token",
    ]

    result = PluginSchemaValidator().validate(_plugin(repo, refs))

    unverified = _findings(result, "plugin_dependency_unverified")
    assert len(unverified) == 4
    assert all("origin" not in f.message and "origin" not in (f.suggestion or "") for f in unverified)
    assert {f.metadata["cause"] for f in unverified} == {"malformed", "unsafe", "content_root"}
    assert _ok_rows(result) == []


@requires_git
@pytest.mark.parametrize(
    ("origin", "state"),
    [
        ("http://git.example.com/acme/agent-catalog.git", "missing"),
        ("git@git.example.com:acme/agent-catalog.git", "missing"),
        ("/srv/git/agent-catalog.git", "unresolved"),
    ],
    ids=["http", "scp", "local-path"],
)
def test_edge07_origin_scheme_reason_and_identity(tmp_path: Path, origin: str, state: str) -> None:
    """Proof L8, check-03 edge-07 plain-http: an http origin was called "not a git top-level with an origin"."""
    repo = _clone(tmp_path / "repo", origin=origin)

    result = PluginSchemaValidator().validate(_plugin(repo, [f"github::{REPO}::skills::ghost-skill"]))

    [row] = result.metadata["plugin"]["dependency_resolution"]["skills"]
    assert row["state"] == state
    if state == "unresolved":
        assert "is a local path" in row["reason"]
        assert "not a git top-level" not in row["reason"]


@requires_git
@pytest.mark.parametrize(
    ("folder", "ref_tail"),
    [("skills/shared-skill", "skills::Shared-Skill"), ("Skills/helper", "Skills::helper")],
    ids=["name-case", "kind-case"],
)
def test_case_mismatch_is_missing_on_every_filesystem(tmp_path: Path, folder: str, ref_tail: str) -> None:
    """Proof L8, skeptic case/kind: macOS said referenced, Linux said missing, for the same clone."""
    repo = _clone(tmp_path / "repo")
    _skill(repo / folder)

    result = PluginSchemaValidator().validate(_plugin(repo, [f"github::{REPO}::{ref_tail}"]))

    [row] = result.metadata["plugin"]["dependency_resolution"]["skills"]
    assert row["state"] == "missing"
    assert "differs only in letter case" in row["reason"]
    assert not result.passed


@requires_git
def test_duplicate_refs_are_counted_and_gated_once(tmp_path: Path) -> None:
    """Proof L8, skeptic dup: a ref and its equivalent selector gave two HIGH findings and missing: 2."""
    repo = _clone(tmp_path / "repo")
    _skill(repo / "skills" / "shared-skill")
    refs = [
        f"github::{REPO}::skills::shared-skill",
        f"github::{REPO}::skills::ghost-skill",
        {"source": "github", "repo": "Acme/Agent-Catalog.git", "path": "skills/ghost-skill"},
    ]

    result = PluginSchemaValidator().validate(_plugin(repo, refs))

    [missing] = _findings(result, "plugin_dependency_missing")
    assert "declared 2 times" in missing.message
    counts = result.metadata["plugin"]["dependency_status_counts"]
    assert (counts["referenced"], counts["missing"]) == (1, 1)
    ghost = next(r for r in result.metadata["plugin"]["dependency_resolution"]["skills"] if "ghost" in r["ref"])
    assert ghost["declared"] == 2


@requires_git
def test_clean_refs_still_get_the_passing_row(tmp_path: Path) -> None:
    repo = _clone(tmp_path / "repo")
    _skill(repo / "skills" / "shared-skill")

    result = PluginSchemaValidator().validate(_plugin(repo, [f"github::{REPO}::skills::shared-skill"]))

    assert _ok_rows(result) == ["Declared dependencies: 1 referenced"]
    assert not _findings(result, "plugin_dependency_unverified")
    assert result.passed
