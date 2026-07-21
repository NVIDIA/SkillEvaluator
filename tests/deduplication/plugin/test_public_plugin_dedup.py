# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from skillevaluator.deduplication.plugin.intra_plugin_validator import IntraPluginValidator
from skillevaluator.deduplication.plugin.ref_utils import find_duplicate_refs, normalize_ref
from skillevaluator.models.result import Severity
from skillevaluator.tier2.commands import run_plugin_dedup_scan


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


def test_public_plugin_scan_never_requires_remote_catalog(tmp_path: Path) -> None:
    (tmp_path / "agent_plugin.yaml").write_text(
        "name: p\nauthor: {email: a@example.com}\nskills:\n  refs: [github::example/repo::skills::a]\n",
        encoding="utf-8",
    )
    results = run_plugin_dedup_scan(tmp_path, run_context=False)
    assert len(results) == 2
    assert all(result.passed for result in results)
    assert all(result.metadata.get("advisory_tier2") for result in results)
    assert results[1].metadata["execution_status"] == "skipped"
