# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ContentExtractor -- unified content extraction for all 3 types."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from skillevaluator.embedding import extractor as extractor_module
from skillevaluator.embedding.extractor import (
    discover_and_extract,
    extract_from_rule,
    extract_from_skill,
    extract_from_workflow,
)

VALID_SKILL_MD = """\
---
name: test-skill
description: A test skill for unit testing extraction logic
metadata:
  author: Tester <tester@nvidia.com>
---

# Test Skill

Body content here.
"""

VALID_RULE_MDC = """\
---
alwaysApply: false
title: python-standards
description: Enforce Python coding standards for the project
---

# Python Standards

Follow PEP-8.
"""

VALID_WORKFLOW_MDC = """\
---
alwaysApply: false
title: fastapi-setup
description: Scaffold a new FastAPI service with best practices
metadata:
  author: Tester <tester@nvidia.com>
---

# FastAPI Setup Workflow

Step-by-step instructions.
"""


def _write_rule(root: Path, filename: str, title: str, description: str) -> Path:
    rule_file = root / filename
    rule_file.parent.mkdir(parents=True, exist_ok=True)
    rule_file.write_text(f"---\nalwaysApply: false\ntitle: {title}\ndescription: {description}\n---\n")
    return rule_file


def _write_workflow(root: Path, name: str, title: str, description: str) -> Path:
    wf_dir = root / name
    wf_dir.mkdir(parents=True)
    (wf_dir / "workflow-rules.mdc").write_text(
        f"---\nalwaysApply: false\ntitle: {title}\ndescription: {description}\n"
        f"metadata:\n  author: Test <test@nvidia.com>\n---\n"
    )
    return wf_dir


def _alias_frontmatter(field: str, levels: int = 18) -> str:
    lines = ["seed: &a0 [safe, safe]"]
    lines.extend(f"a{i}: &a{i} [*a{i - 1}, *a{i - 1}]" for i in range(1, levels + 1))
    lines.extend(
        [
            "name: safe-name" if field != "name" else f"name: *a{levels}",
            "description: Safe description" if field != "description" else f"description: *a{levels}",
        ]
    )
    return "---\n" + "\n".join(lines) + "\n---\n# Body\n"


class TestExtractFromSkill:
    def test_valid_skill(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(VALID_SKILL_MD)

        entry = extract_from_skill(skill_dir)

        assert entry is not None
        assert entry.name == "test-skill"
        assert entry.description == "A test skill for unit testing extraction logic"
        assert entry.content_type == "skill"
        assert entry.path == str(skill_dir)

    def test_embedding_text_format(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "fmt-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(VALID_SKILL_MD)

        entry = extract_from_skill(skill_dir)
        assert entry is not None
        assert entry.embedding_text == "test-skill: A test skill for unit testing extraction logic"

    def test_full_text_includes_body(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "body-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(VALID_SKILL_MD)

        entry = extract_from_skill(skill_dir)
        assert entry is not None
        assert "Body content here." in entry.full_text
        assert "---" in entry.full_text

    def test_missing_skill_md_returns_none(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        assert extract_from_skill(empty_dir) is None

    def test_missing_description_returns_none(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "no-desc"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: no-desc\n---\n")
        assert extract_from_skill(skill_dir) is None

    @pytest.mark.parametrize("field", ["name", "description"])
    def test_rejects_alias_amplified_nonstring_fields(self, tmp_path: Path, field: str) -> None:
        skill_dir = tmp_path / "alias-dag"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(_alias_frontmatter(field))

        with pytest.raises(ValueError, match=rf"{field}.*string|complexity.*limit"):
            extract_from_skill(skill_dir)

    def test_rejects_deep_frontmatter_without_recursion_error(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "deep"
        skill_dir.mkdir()
        nested = "[" * 1_500 + "safe" + "]" * 1_500
        (skill_dir / "SKILL.md").write_text(f"---\nname: safe\ndescription: {nested}\n---\n# Body\n")

        with pytest.raises(ValueError, match=r"complexity|depth|limit"):
            extract_from_skill(skill_dir)

    def test_relative_input_preserves_relative_report_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        skill_dir = tmp_path / "relative-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(VALID_SKILL_MD)
        monkeypatch.chdir(tmp_path)

        entry = extract_from_skill(Path("relative-skill"))

        assert entry is not None
        assert entry.path == "relative-skill"


class TestExtractFromRule:
    def test_valid_rule(self, tmp_path: Path) -> None:
        rule_file = tmp_path / "python-standards.mdc"
        rule_file.write_text(VALID_RULE_MDC)

        entry = extract_from_rule(rule_file)

        assert entry is not None
        assert entry.name == "python-standards"
        assert entry.description == "Enforce Python coding standards for the project"
        assert entry.content_type == "rules"

    def test_non_mdc_file_returns_none(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("not a rule")
        assert extract_from_rule(txt_file) is None

    def test_missing_title_returns_none(self, tmp_path: Path) -> None:
        rule_file = tmp_path / "bad.mdc"
        rule_file.write_text("---\nalwaysApply: false\ndescription: no title\n---\n")
        assert extract_from_rule(rule_file) is None


class TestExtractFromWorkflow:
    def test_valid_workflow(self, tmp_path: Path) -> None:
        wf_dir = tmp_path / "fastapi-setup"
        wf_dir.mkdir()
        (wf_dir / "workflow-rules.mdc").write_text(VALID_WORKFLOW_MDC)

        entry = extract_from_workflow(wf_dir)

        assert entry is not None
        assert entry.name == "fastapi-setup"
        assert entry.description == "Scaffold a new FastAPI service with best practices"
        assert entry.content_type == "workflows"

    def test_missing_manifest_returns_none(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty-wf"
        empty_dir.mkdir()
        assert extract_from_workflow(empty_dir) is None


class TestDiscoverAndExtract:
    def test_discover_skills_in_folder(self, tmp_path: Path, write_skill) -> None:
        write_skill(tmp_path, "skill-a", "First skill for testing")
        write_skill(tmp_path, "skill-b", "Second skill for testing")

        entries = discover_and_extract(tmp_path, "skill")

        assert len(entries) == 2
        names = {e.name for e in entries}
        assert names == {"skill-a", "skill-b"}

    def test_discover_rules_in_folder(self, tmp_path: Path) -> None:
        rules_dir = tmp_path / "team-rules"
        rules_dir.mkdir()
        _write_rule(rules_dir, "lint.mdc", "Lint Rules", "Enforce lint rules")
        _write_rule(rules_dir, "format.mdc", "Format Rules", "Enforce formatting")

        entries = discover_and_extract(rules_dir, "rules")

        assert len(entries) == 2
        names = {e.name for e in entries}
        assert names == {"Lint Rules", "Format Rules"}

    def test_discover_workflows_in_folder(self, tmp_path: Path) -> None:
        _write_workflow(tmp_path, "wf-a", "Workflow A", "First workflow")
        _write_workflow(tmp_path, "wf-b", "Workflow B", "Second workflow")

        entries = discover_and_extract(tmp_path, "workflows")

        assert len(entries) == 2

    def test_empty_directory_returns_empty(self, tmp_path: Path) -> None:
        entries = discover_and_extract(tmp_path, "skill")
        assert entries == []

    def test_unknown_type_returns_empty(self, tmp_path: Path) -> None:
        entries = discover_and_extract(tmp_path, "unknown_type")
        assert entries == []


class TestExtractorSecurityContract:
    @pytest.mark.parametrize("variant", ["SKILL.md", "skill.md"])
    def test_rejects_all_linked_skill_manifest_variants(self, tmp_path: Path, variant: str) -> None:
        catalog = tmp_path / "catalog"
        skill = catalog / "linked-skill"
        skill.mkdir(parents=True)
        outside = tmp_path / "outside.md"
        outside.write_text(VALID_SKILL_MD)
        (skill / variant).symlink_to(outside)

        with pytest.raises(ValueError, match=r"manifest|symlink|reparse|unsafe"):
            discover_and_extract(catalog, "skill")

    def test_rejects_linked_discovery_root(self, tmp_path: Path, write_skill) -> None:
        real_catalog = tmp_path / "real-catalog"
        real_catalog.mkdir()
        write_skill(real_catalog, "skill-a", "A real skill")
        linked_catalog = tmp_path / "linked-catalog"
        linked_catalog.symlink_to(real_catalog, target_is_directory=True)

        with pytest.raises(ValueError, match=r"root|symlink|reparse"):
            discover_and_extract(linked_catalog, "skill")

    def test_rejects_linked_directory_before_manifest_discovery(self, tmp_path: Path) -> None:
        catalog = tmp_path / "catalog"
        catalog.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "SKILL.md").write_text(VALID_SKILL_MD)
        (catalog / "linked-skill").symlink_to(outside, target_is_directory=True)

        with pytest.raises(ValueError, match=r"directory|symlink|reparse|unsafe"):
            discover_and_extract(catalog, "skill")

    def test_rejects_linked_directory_before_excluded_name_pruning(self, tmp_path: Path) -> None:
        catalog = tmp_path / "catalog"
        catalog.mkdir()
        outside = tmp_path / "outside-evals"
        outside.mkdir()
        (outside / "SKILL.md").write_text(VALID_SKILL_MD)
        (catalog / "evals").symlink_to(outside, target_is_directory=True)

        with pytest.raises(ValueError, match=r"directory|symlink|reparse|unsafe"):
            discover_and_extract(catalog, "skill")

    @pytest.mark.parametrize("target_kind", ["contained", "escaping", "broken", "cyclic"])
    def test_rejects_non_compatibility_file_redirects(self, tmp_path: Path, target_kind: str) -> None:
        catalog = tmp_path / "catalog"
        catalog.mkdir()
        linked = catalog / "irrelevant.bin"
        if target_kind == "contained":
            target = catalog / "payload.dat"
            target.write_bytes(b"payload")
        elif target_kind == "escaping":
            target = tmp_path / "outside.dat"
            target.write_bytes(b"outside")
        elif target_kind == "cyclic":
            target = linked
        else:
            target = catalog / "missing.dat"
        linked.symlink_to(target)

        with pytest.raises(ValueError, match=r"symlink|reparse|unsafe"):
            discover_and_extract(catalog, "skill")

    def test_bounds_irrelevant_paths_and_prunes_generated_directories(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(extractor_module, "MAX_DISCOVERED_PATHS", 2, raising=False)
        for name in ("a.bin", "b.bin", "c.bin"):
            (tmp_path / name).write_bytes(b"x")

        with pytest.raises(ValueError, match=r"path.*limit"):
            discover_and_extract(tmp_path, "skill")

        for path in tmp_path.glob("*.bin"):
            path.unlink()
        hidden = tmp_path / "evals" / "results"
        hidden.mkdir(parents=True)
        for index in range(10):
            (hidden / f"generated-{index}.md").write_text("generated")
        visible = tmp_path / "visible"
        visible.mkdir()
        (visible / "SKILL.md").write_text(VALID_SKILL_MD)

        assert [entry.name for entry in discover_and_extract(tmp_path, "skill")] == ["test-skill"]

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are unavailable on this platform")
    def test_rejects_special_selected_rule(self, tmp_path: Path) -> None:
        fifo = tmp_path / "special.mdc"
        os.mkfifo(fifo)

        with pytest.raises(ValueError, match=r"special|non-regular|regular file"):
            discover_and_extract(tmp_path, "rules")
