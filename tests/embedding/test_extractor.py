# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ContentExtractor -- unified content extraction for all 3 types."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import skillevaluator.embedding.extractor as extractor_module
from skillevaluator.embedding.extractor import (
    _is_symlink_or_reparse,
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


def test_discovery_debug_log_uses_root_label_without_host_path(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "nested" / "external-skills"
    root.mkdir(parents=True)
    debug = MagicMock()
    monkeypatch.setattr(extractor_module.logger, "debug", debug)

    assert discover_and_extract(root, "skill") == []

    rendered = "\n".join(call.args[0] % call.args[1:] for call in debug.call_args_list)
    assert str(root) not in rendered
    assert root.name in rendered


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

    def test_manifest_uses_one_bounded_os_read(self, tmp_path: Path, monkeypatch) -> None:
        skill_dir = tmp_path / "single-read"
        skill_dir.mkdir()
        manifest = skill_dir / "SKILL.md"
        manifest.write_bytes(VALID_SKILL_MD.encode("utf-8"))
        real_os_open = os.open
        real_read_text = Path.read_text
        manifest_open_calls: list[Path] = []

        def tracked_os_open(path, flags, *, dir_fd=None):
            if Path(path).name == manifest.name:
                manifest_open_calls.append(Path(path))
            if dir_fd is None:
                return real_os_open(path, flags)
            return real_os_open(path, flags, dir_fd=dir_fd)

        def reject_unbounded_read(path: Path, *_args, **_kwargs):
            if path == manifest:
                raise AssertionError("manifest must not use Path.read_text")
            return real_read_text(path, *_args, **_kwargs)

        monkeypatch.setattr(extractor_module, "os", os, raising=False)
        monkeypatch.setattr(os, "open", tracked_os_open)
        monkeypatch.setattr(Path, "read_text", reject_unbounded_read)

        entry = extract_from_skill(skill_dir)

        assert entry is not None
        assert entry.full_text == VALID_SKILL_MD
        assert len(manifest_open_calls) == 1

    @pytest.mark.skipif(os.name != "posix", reason="descriptor-anchored openat regression is POSIX-specific")
    def test_ancestor_swap_never_reads_outside_manifest_root(self, tmp_path: Path, monkeypatch) -> None:
        skill_dir = tmp_path / "catalog" / "safe-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: safe-skill\ndescription: Safe skill\n---\nSAFE_CONTENT\n")

        outside = tmp_path / "outside-skill"
        outside.mkdir()
        (outside / "SKILL.md").write_text("---\nname: exfiltrated\ndescription: Outside secret\n---\nSECRET_CANARY\n")

        original_skill = skill_dir.with_name("safe-skill-original")
        real_open = os.open
        swapped = False

        def swapping_open(path, flags, *, dir_fd=None):
            nonlocal swapped
            if Path(path).name == "SKILL.md" and not swapped:
                skill_dir.rename(original_skill)
                skill_dir.symlink_to(outside, target_is_directory=True)
                swapped = True
            if dir_fd is None:
                return real_open(path, flags)
            return real_open(path, flags, dir_fd=dir_fd)

        monkeypatch.setattr(extractor_module.os, "open", swapping_open)

        try:
            entry = extract_from_skill(skill_dir)
        except ValueError:
            assert swapped
            return

        assert swapped
        assert entry is not None
        assert entry.name == "safe-skill"
        assert "SECRET_CANARY" not in entry.full_text
        assert "SAFE_CONTENT" in entry.full_text

    @pytest.mark.skipif(os.name != "posix", reason="descriptor-anchored openat regression is POSIX-specific")
    def test_directory_replacement_during_validation_never_reads_outside_manifest(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        skill_dir = tmp_path / "catalog" / "safe-skill"
        skill_dir.mkdir(parents=True)
        manifest = skill_dir / "SKILL.md"
        manifest.write_text("---\nname: safe-skill\ndescription: Safe skill\n---\nSAFE_CONTENT\n")

        outside = tmp_path / "outside-skill"
        outside.mkdir()
        (outside / "SKILL.md").write_text("---\nname: exfiltrated\ndescription: Outside secret\n---\nSECRET_CANARY\n")

        original_skill = skill_dir.with_name("safe-skill-original")
        real_lstat = Path.lstat
        swapped = False

        def swapping_lstat(path: Path):
            nonlocal swapped
            if path == manifest and not swapped:
                skill_dir.rename(original_skill)
                outside.rename(skill_dir)
                swapped = True
            return real_lstat(path)

        monkeypatch.setattr(Path, "lstat", swapping_lstat)

        entry = extract_from_skill(skill_dir)

        assert swapped
        assert entry is not None
        assert entry.name == "safe-skill"
        assert "SECRET_CANARY" not in entry.full_text
        assert "SAFE_CONTENT" in entry.full_text


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

    def test_broken_symlinked_rule_is_rejected(self, tmp_path: Path) -> None:
        linked = tmp_path / "broken.mdc"
        try:
            linked.symlink_to(tmp_path / "missing.mdc")
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

        with pytest.raises(ValueError, match=r"symlink|reparse|escape"):
            extract_from_rule(linked)

    def test_non_regular_rule_is_rejected_before_read(self, tmp_path: Path) -> None:
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFOs are unavailable on this platform")
        fifo = tmp_path / "named-pipe.mdc"
        os.mkfifo(fifo)

        with pytest.raises(ValueError, match="non-regular"):
            extract_from_rule(fifo)


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

    def test_broken_symlinked_workflow_manifest_is_rejected(self, tmp_path: Path) -> None:
        workflow_dir = tmp_path / "broken-workflow"
        workflow_dir.mkdir()
        try:
            (workflow_dir / "workflow-rules.mdc").symlink_to(tmp_path / "missing.mdc")
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

        with pytest.raises(ValueError, match=r"symlink|reparse|escape"):
            extract_from_workflow(workflow_dir)


def test_reparse_point_is_treated_as_unsafe(monkeypatch, tmp_path: Path) -> None:
    class ReparseStat:
        st_mode = stat.S_IFREG
        st_file_attributes = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

    monkeypatch.setattr(Path, "lstat", lambda _self: ReparseStat())

    assert _is_symlink_or_reparse(tmp_path / "junction")


class TestDiscoverAndExtract:
    @pytest.mark.skipif(os.name != "posix", reason="descriptor-anchored openat regression is POSIX-specific")
    def test_collection_root_swap_never_reads_outside_manifest(self, tmp_path: Path, monkeypatch) -> None:
        collection = tmp_path / "collection"
        safe_skill = collection / "safe-skill"
        safe_skill.mkdir(parents=True)
        (safe_skill / "SKILL.md").write_text(
            "---\nname: safe-skill\ndescription: Safe collection skill\n---\nSAFE_CONTENT\n"
        )

        outside_collection = tmp_path / "outside-collection"
        outside_skill = outside_collection / "safe-skill"
        outside_skill.mkdir(parents=True)
        (outside_skill / "SKILL.md").write_text(
            "---\nname: exfiltrated\ndescription: Outside collection secret\n---\nSECRET_CANARY\n"
        )

        original_collection = tmp_path / "collection-original"
        real_open = os.open
        swapped = False

        def swapping_open(path, flags, *, dir_fd=None):
            nonlocal swapped
            if Path(path).name == "safe-skill" and not swapped:
                collection.rename(original_collection)
                collection.symlink_to(outside_collection, target_is_directory=True)
                swapped = True
            if dir_fd is None:
                return real_open(path, flags)
            return real_open(path, flags, dir_fd=dir_fd)

        monkeypatch.setattr(extractor_module.os, "open", swapping_open)

        try:
            entries = discover_and_extract(collection, "skill")
        except ValueError:
            assert swapped
            return

        assert swapped
        assert [entry.name for entry in entries] == ["safe-skill"]
        assert all("SECRET_CANARY" not in entry.full_text for entry in entries)

    def test_discovery_bounds_irrelevant_paths(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(extractor_module, "MAX_DISCOVERED_PATHS", 2, raising=False)
        for name in ("a.bin", "b.bin", "c.bin"):
            (tmp_path / name).write_bytes(b"x")

        with pytest.raises(ValueError, match=r"path.*limit"):
            discover_and_extract(tmp_path, "skill")

    def test_discovery_prunes_standard_excluded_directories(self, tmp_path: Path, write_skill, monkeypatch) -> None:
        hidden_skill = tmp_path / ".git" / "nested-skill"
        hidden_skill.mkdir(parents=True)
        (hidden_skill / "SKILL.md").write_text("---\nname: hidden\ndescription: Must not be discovered\n---\n")
        write_skill(tmp_path, "visible-skill", "Visible skill")
        monkeypatch.setattr(extractor_module, "MAX_DISCOVERED_PATHS", 3, raising=False)

        entries = discover_and_extract(tmp_path, "skill")

        assert [entry.name for entry in entries] == ["visible-skill"]

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

    def test_rejects_symlinked_skill_manifest(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside.md"
        outside.write_text(VALID_SKILL_MD)
        skill_dir = tmp_path / "catalog" / "linked-skill"
        skill_dir.mkdir(parents=True)
        try:
            (skill_dir / "SKILL.md").symlink_to(outside)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

        with pytest.raises(ValueError, match=r"symlink|reparse|escape"):
            discover_and_extract(tmp_path / "catalog", "skill")

    def test_rejects_skill_directory_symlink_escape(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside" / "skill"
        outside.mkdir(parents=True)
        (outside / "SKILL.md").write_text(VALID_SKILL_MD)
        catalog = tmp_path / "catalog"
        catalog.mkdir()
        try:
            (catalog / "linked-skill").symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

        with pytest.raises(ValueError, match=r"symlink|reparse|escape"):
            extract_from_skill(catalog / "linked-skill")

    def test_rejects_broken_symlinked_manifest(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "broken-skill"
        skill_dir.mkdir()
        try:
            (skill_dir / "SKILL.md").symlink_to(tmp_path / "missing.md")
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

        with pytest.raises(ValueError, match=r"symlink|reparse|escape"):
            extract_from_skill(skill_dir)
