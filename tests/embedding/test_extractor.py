# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ContentExtractor -- unified content extraction for all 3 types."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import skillevaluator.embedding.extractor as extractor_module
import skillevaluator.embedding.registry as registry_module
from skillevaluator.embedding.extractor import (
    _is_symlink_or_reparse,
    discover_and_extract,
    discover_and_extract_detailed,
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


@pytest.mark.parametrize("field", ["name", "description"])
def test_non_string_frontmatter_debug_log_omits_host_path(tmp_path: Path, monkeypatch, field: str) -> None:
    skill_dir = tmp_path / "private-ancestor" / "typed-skill"
    skill_dir.mkdir(parents=True)
    values = {"name": "typed-skill", "description": "A test skill"}
    values[field] = "[not, a, string]"
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {values['name']}\ndescription: {values['description']}\n---\n", encoding="utf-8"
    )
    debug = MagicMock()
    monkeypatch.setattr(extractor_module.logger, "debug", debug)

    assert extract_from_skill(skill_dir) is None

    rendered = "\n".join(call.args[0] % call.args[1:] for call in debug.call_args_list)
    assert "is not a string" in rendered
    assert str(tmp_path) not in rendered
    assert "private-ancestor" not in rendered


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

    @pytest.mark.parametrize("content_type", ["skill", "rules", "workflows"])
    def test_discovery_path_budget_counts_directories_and_unselected_files(
        self, tmp_path: Path, monkeypatch, content_type: str
    ) -> None:
        monkeypatch.setattr(extractor_module, "MAX_DISCOVERED_PATHS", 4)
        manifest_name, manifest_text = {
            "skill": ("SKILL.md", VALID_SKILL_MD),
            "rules": ("rule.mdc", VALID_RULE_MDC),
            "workflows": ("workflow-rules.mdc", VALID_WORKFLOW_MDC),
        }[content_type]
        item_dir = tmp_path / "nested" / "item"
        item_dir.mkdir(parents=True)
        (item_dir / manifest_name).write_text(manifest_text)
        (item_dir / "a.bin").write_bytes(b"x")

        assert len(discover_and_extract(tmp_path, content_type, max_entries=5_000)) == 1

        (item_dir / "b.bin").write_bytes(b"x")
        with pytest.raises(ValueError, match=r"Collection path limit exceeded \(4\)"):
            discover_and_extract(tmp_path, content_type, max_entries=5_000)

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

    def test_openclaw_compatibility_alias_does_not_abort_skill_discovery(self, tmp_path: Path, write_skill) -> None:
        skill_dir = write_skill(tmp_path, "autoreview", "Review changes with multiple agents")
        (skill_dir / "AGENTS.md").write_text("# Shared agent instructions\n")
        try:
            (skill_dir / "CLAUDE.md").symlink_to("AGENTS.md")
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

        entries = discover_and_extract(tmp_path, "skill")

        assert [entry.name for entry in entries] == ["autoreview"]

    def test_rejects_openclaw_alias_when_regular_target_is_hard_linked(
        self,
        tmp_path: Path,
        write_skill,
    ) -> None:
        collection = tmp_path / "skills"
        skill_dir = write_skill(collection, "autoreview", "Review changes with multiple agents")
        outside_target = tmp_path / "outside-agents.md"
        outside_target.write_text("# Outside agent instructions\n")
        try:
            os.link(outside_target, skill_dir / "AGENTS.md")
            (skill_dir / "CLAUDE.md").symlink_to("AGENTS.md")
        except OSError as exc:
            pytest.skip(f"hard links or symlinks unavailable: {exc}")

        with pytest.raises(ValueError, match=r"symlink|reparse|unsafe"):
            discover_and_extract(collection, "skill")

    def test_rejects_hard_linked_skill_manifest(self, tmp_path: Path) -> None:
        collection = tmp_path / "skills"
        skill_dir = collection / "hard-linked-skill"
        skill_dir.mkdir(parents=True)
        outside_manifest = tmp_path / "outside-SKILL.md"
        outside_manifest.write_text(VALID_SKILL_MD)
        try:
            os.link(outside_manifest, skill_dir / "SKILL.md")
        except OSError as exc:
            pytest.skip(f"hard links unavailable: {exc}")

        with pytest.raises(ValueError, match=r"hard-linked|unsafe"):
            discover_and_extract(collection, "skill")

    def test_rejects_openclaw_alias_when_target_entry_case_does_not_match(
        self,
        tmp_path: Path,
        write_skill,
    ) -> None:
        collection = tmp_path / "skills"
        skill_dir = write_skill(collection, "autoreview", "Review changes with multiple agents")
        (skill_dir / "agents.md").write_text("# Lowercase agent instructions\n")
        try:
            (skill_dir / "CLAUDE.md").symlink_to("AGENTS.md")
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

        with pytest.raises(ValueError, match=r"symlink|reparse|unsafe"):
            discover_and_extract(collection, "skill")

    def test_rejects_alias_target_created_after_discovery_snapshot(self, tmp_path: Path, monkeypatch) -> None:
        collection = tmp_path / "skills"
        skill_dir = collection / "autoreview"
        skill_dir.mkdir(parents=True)
        try:
            (skill_dir / "CLAUDE.md").symlink_to("AGENTS.md")
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

        real_walk = os.walk

        def walk_with_late_target(*args, **kwargs):
            for dirpath, dirnames, filenames in real_walk(*args, **kwargs):
                if Path(dirpath) == skill_dir:
                    assert "AGENTS.md" not in filenames
                    (skill_dir / "AGENTS.md").write_text("# Late agent instructions\n")
                yield dirpath, dirnames, filenames

        monkeypatch.setattr(extractor_module.os, "walk", walk_with_late_target)

        with pytest.raises(ValueError, match=r"symlink|reparse|unsafe"):
            discover_and_extract(collection, "skill")

    @pytest.mark.parametrize("target", ["./AGENTS.md", "../AGENTS.md", "missing.md"])
    def test_rejects_non_exact_openclaw_alias_during_skill_discovery(
        self,
        tmp_path: Path,
        write_skill,
        target: str,
    ) -> None:
        skill_dir = write_skill(tmp_path, "autoreview", "Review changes with multiple agents")
        (skill_dir / "AGENTS.md").write_text("# Shared agent instructions\n")
        try:
            (skill_dir / "CLAUDE.md").symlink_to(target)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

        with pytest.raises(ValueError, match=r"symlink|reparse|unsafe"):
            discover_and_extract(tmp_path, "skill")

    def test_contained_directory_alias_is_skipped_and_target_discovered_once(self, tmp_path: Path, write_skill) -> None:
        collection = tmp_path / "skills"
        write_skill(collection / "shared-skills", "helper", "Shared helper for every coding agent")
        (collection / ".claude").mkdir()
        try:
            # A Path target uses native separators, so the link also resolves on Windows.
            (collection / ".claude" / "skills").symlink_to(Path("..") / "shared-skills", target_is_directory=True)
            (collection / ".agents").symlink_to(".claude", target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

        entries = discover_and_extract(collection, "skill")

        assert [Path(entry.path).relative_to(collection).as_posix() for entry in entries] == ["shared-skills/helper"]

    def test_directory_alias_to_pruned_regular_directory_is_skipped(self, tmp_path: Path, write_skill) -> None:
        collection = tmp_path / "skills"
        write_skill(collection, "visible-skill", "Visible skill")
        write_skill(collection / "evals", "generated-skill", "Generated snapshot")
        try:
            (collection / "latest-evals").symlink_to("evals", target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

        entries = discover_and_extract(collection, "skill")

        assert [entry.name for entry in entries] == ["visible-skill"]

    def test_rejects_directory_alias_chain(self, tmp_path: Path, write_skill) -> None:
        collection = tmp_path / "skills"
        write_skill(collection / "canonical", "helper", "Shared helper for every coding agent")
        try:
            (collection / ".claude").symlink_to("canonical", target_is_directory=True)
            (collection / ".agents").symlink_to(".claude", target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

        with pytest.raises(ValueError, match=r"symlink or reparse point.*\.agents -> \.claude"):
            discover_and_extract(collection, "skill")

    @pytest.mark.parametrize("alias_name", ["SKILL.md", "skill.md", "team.mdc", "workflow-rules.mdc"])
    def test_rejects_directory_alias_with_manifest_name(self, tmp_path: Path, write_skill, alias_name: str) -> None:
        collection = tmp_path / "skills"
        write_skill(collection / "canonical", "helper", "Shared helper for every coding agent")
        try:
            (collection / alias_name).symlink_to("canonical", target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

        with pytest.raises(ValueError, match=r"symlink or reparse point"):
            discover_and_extract(collection, "skill")

    @pytest.mark.parametrize("target", ["missing", ".", "..", "../outside", "absolute"])
    def test_rejects_directory_alias_without_contained_regular_target(
        self, tmp_path: Path, write_skill, target: str
    ) -> None:
        collection = tmp_path / "skills"
        write_skill(collection, "visible-skill", "Visible skill")
        (tmp_path / "outside").mkdir()
        link_text = str(collection / "visible-skill") if target == "absolute" else target
        try:
            (collection / ".agents").symlink_to(link_text, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

        with pytest.raises(ValueError, match=r"symlink or reparse point"):
            discover_and_extract(collection, "skill")

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


def _alias_dag(field: str, *, levels: int, fanout: int) -> str:
    """Return frontmatter whose *field* is a list built from chained YAML aliases."""
    lines = ["seed: &a0 safe"]
    for level in range(1, levels + 1):
        lines.append(f"a{level}: &a{level} [{', '.join([f'*a{level - 1}'] * fanout)}]")
    lines.append("name: safe-name" if field != "name" else f"name: *a{levels}")
    lines.append("description: Safe description" if field != "description" else f"description: *a{levels}")
    return "---\n" + "\n".join(lines) + "\n---\n# Body\n"


def _merge_chain(*, levels: int, fanout: int) -> str:
    """Return frontmatter whose merge keys repeat earlier mappings at every level."""
    lines = ["m0: &m0 {" + ", ".join(f"k{index}: {index}" for index in range(fanout)) + "}"]
    for level in range(1, levels + 1):
        lines.append(f"m{level}: &m{level} {{<<: [{', '.join([f'*m{level - 1}'] * fanout)}]}}")
    lines += ["name: merge-chain", "description: Merge chain"]
    return "---\n" + "\n".join(lines) + "\n---\n# Body\n"


# Markdown whose text between two ``---`` lines is not a YAML mapping, so it is
# not frontmatter. Each is over a structure limit if parsed as YAML: a list of
# 1,100 items, a plain scalar of about 70,400 characters, a sequence of mappings
# that chains merge keys, and a 1,100-item ``!!set``, which loads as a set.
NON_MAPPING_BLOCKS = [
    pytest.param("\n".join(f"- item {index}" for index in range(1_100)), id="thematic-break-list"),
    pytest.param("A line of prose that holds no key or colon.\n" * 1_600, id="thematic-break-prose"),
    pytest.param(
        "\n".join(
            ["- &m0 {" + ", ".join(f"k{index}: {index}" for index in range(10)) + "}"]
            + [f"- &m{level} {{<<: [{', '.join([f'*m{level - 1}'] * 10)}]}}" for level in range(1, 5)]
        ),
        id="sequence-merge-chain",
    ),
    pytest.param("!!set\n" + "\n".join(f"? item{index}" for index in range(1_100)), id="set-root"),
]


def _write_skill_manifest(root: Path, name: str, manifest: str) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_bytes(manifest.encode("utf-8"))
    return skill_dir


@pytest.fixture
def default_int_digit_limit():
    """Pin Python's default 4,300-digit limit for converting text to an int."""
    previous = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(4_300)
    yield
    sys.set_int_max_str_digits(previous)


class TestBoundedFrontmatter:
    """Frontmatter is parsed with structured-data limits and never coerced to text."""

    @pytest.mark.parametrize("field", ["name", "description"])
    def test_alias_amplified_field_fails_closed(self, tmp_path: Path, field: str) -> None:
        # 2**18 list items from under 450 bytes; str() of the list would be megabytes.
        skill_dir = _write_skill_manifest(tmp_path, "alias-dag", _alias_dag(field, levels=18, fanout=2))

        with pytest.raises(ValueError, match=r"structured-data complexity limit exceeded: alias-dag/SKILL\.md$"):
            extract_from_skill(skill_dir)

    def test_merge_key_chain_fails_closed(self, tmp_path: Path) -> None:
        skill_dir = _write_skill_manifest(tmp_path, "merge-chain", _merge_chain(levels=4, fanout=10))

        with pytest.raises(ValueError, match=r"structured-data complexity limit exceeded: merge-chain/SKILL\.md$"):
            extract_from_skill(skill_dir)

    @pytest.mark.usefixtures("default_int_digit_limit")
    @pytest.mark.parametrize(
        "value",
        [
            # 3,600 hex digits are 14,400 bits, over the 14,000-bit integer limit.
            pytest.param("0x" + "f" * 3_600, id="hex"),
            # 4,250 decimal digits are about 14,118 bits, within Python's 4,300-digit limit.
            pytest.param("9" * 4_250, id="decimal"),
        ],
    )
    def test_oversized_integer_fails_closed(self, tmp_path: Path, value: str) -> None:
        skill_dir = _write_skill_manifest(
            tmp_path, "big-int", f"---\nname: big-int\ndescription: Big\nversion: {value}\n---\n"
        )

        with pytest.raises(ValueError, match=r"structured-data complexity limit exceeded: big-int/SKILL\.md$"):
            extract_from_skill(skill_dir)

    def test_deep_nesting_fails_closed_without_recursion_error(self, tmp_path: Path) -> None:
        nested = "[" * 1_500 + "safe" + "]" * 1_500
        skill_dir = _write_skill_manifest(
            tmp_path, "deep", f"---\nname: deep\ndescription: Deep\nextra: {nested}\n---\n# Body\n"
        )

        with pytest.raises(ValueError, match=r"structured-data complexity limit exceeded: deep/SKILL\.md$"):
            extract_from_skill(skill_dir)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("description", "*items"),
            ("description", "{key: value}"),
            ("description", "3.5"),
            ("name", "2048"),
            ("name", "true"),
        ],
    )
    def test_non_string_field_skips_the_manifest_without_coercion(
        self, tmp_path: Path, monkeypatch, field: str, value: str
    ) -> None:
        # Three alias levels stay within the graph limits, yet str() of *items
        # would turn this 278-byte frontmatter into an 8,222-character description.
        lines = ["seed: &a0 safe"]
        lines += [f"a{level}: &a{level} [{', '.join([f'*a{level - 1}'] * 10)}]" for level in (1, 2, 3)]
        lines += ["anchor: &items [*a3]", "name: safe-name", "description: Safe description", f"{field}: {value}"]
        skill_dir = _write_skill_manifest(tmp_path, "typed", "---\n" + "\n".join(lines) + "\n---\n# Body\n")
        monkeypatch.setattr(extractor_module, "ContentEntry", MagicMock(side_effect=AssertionError("entry built")))

        assert extract_from_skill(skill_dir) is None

    @pytest.mark.parametrize("field", ["name", "description"])
    def test_field_length_matches_the_catalog_text_limit(self, tmp_path: Path, field: str) -> None:
        limit = extractor_module.MAX_FRONTMATTER_FIELD_CHARS
        assert limit == registry_module.MAX_CATALOG_TEXT_LENGTH == 16_384
        values = {"name": "safe-name", "description": "Safe description"}

        values[field] = "x" * limit
        accepted = _write_skill_manifest(
            tmp_path, "at-limit", f"---\nname: {values['name']}\ndescription: {values['description']}\n---\n"
        )
        entry = extract_from_skill(accepted)
        assert entry is not None
        assert len(getattr(entry, field)) == limit

        values[field] = "x" * (limit + 1)
        rejected = _write_skill_manifest(
            tmp_path, "over-limit", f"---\nname: {values['name']}\ndescription: {values['description']}\n---\n"
        )
        with pytest.raises(
            ValueError, match=rf"frontmatter {field} exceeds the 16384-character limit: over-limit/SKILL\.md$"
        ):
            extract_from_skill(rejected)

    def test_single_workflow_error_names_its_directory(self, tmp_path: Path) -> None:
        workflow_dir = tmp_path / "long-workflow"
        workflow_dir.mkdir()
        (workflow_dir / "workflow-rules.mdc").write_bytes(
            ("---\ntitle: Workflow\ndescription: " + "x" * 16_385 + "\n---\n").encode("utf-8")
        )

        with pytest.raises(
            ValueError, match=r"description exceeds the 16384-character limit: long-workflow/workflow-rules\.mdc$"
        ):
            extract_from_workflow(workflow_dir)

    def test_single_rule_error_names_its_directory(self, tmp_path: Path) -> None:
        rule_dir = tmp_path / "rules-dir"
        rule_dir.mkdir()
        (rule_dir / "long.mdc").write_bytes(
            ("---\ntitle: " + "x" * 16_385 + "\ndescription: Rule\n---\n").encode("utf-8")
        )

        with pytest.raises(ValueError, match=r"title exceeds the 16384-character limit: rules-dir/long\.mdc$"):
            extract_from_rule(rule_dir / "long.mdc")

    @pytest.mark.parametrize("relative", [".", "nested/.."])
    def test_relative_directory_is_named_by_its_own_name(self, tmp_path: Path, monkeypatch, relative: str) -> None:
        skill_dir = _write_skill_manifest(tmp_path, "here", _merge_chain(levels=4, fanout=10))
        (skill_dir / "nested").mkdir()
        monkeypatch.chdir(skill_dir)

        with pytest.raises(ValueError, match=r"structured-data complexity limit exceeded: here/SKILL\.md$"):
            extract_from_skill(Path(*relative.split("/")))

    def test_repeated_key_keeps_the_last_value_like_agent_clients(self, tmp_path: Path) -> None:
        skill_dir = _write_skill_manifest(
            tmp_path, "repeated", "---\nname: first\nname: second\ndescription: Repeated name\n---\n"
        )

        entry = extract_from_skill(skill_dir)

        assert entry is not None
        assert entry.name == "second"

    @pytest.mark.usefixtures("default_int_digit_limit")
    @pytest.mark.parametrize(
        "line",
        [
            pytest.param('version: !!int ""', id="tagged-int"),
            pytest.param("version: !!bool nope", id="tagged-bool"),
            pytest.param("version: !!float x", id="tagged-float"),
            pytest.param("last_updated: 2024-02-30", id="impossible-date"),
            # int() rejects more than 4,300 decimal digits before the bit limit applies.
            pytest.param("version: " + "9" * 5_000, id="decimal-over-digit-limit"),
        ],
    )
    def test_unconstructible_value_is_invalid_frontmatter(self, tmp_path: Path, line: str) -> None:
        skill_dir = _write_skill_manifest(tmp_path, "tagged", f"---\nname: tagged\ndescription: Tagged\n{line}\n---\n")

        assert extract_from_skill(skill_dir) is None

    @pytest.mark.parametrize("value", ['""', "[]", "{}", "0", "false"])
    def test_empty_field_still_skips_the_manifest(self, tmp_path: Path, value: str) -> None:
        skill_dir = _write_skill_manifest(tmp_path, "empty", f"---\nname: empty\ndescription: {value}\n---\n")

        assert extract_from_skill(skill_dir) is None

    @pytest.mark.parametrize("block", NON_MAPPING_BLOCKS)
    def test_block_that_is_not_a_mapping_skips_the_manifest_unparsed(self, tmp_path: Path, block: str) -> None:
        skill_dir = _write_skill_manifest(tmp_path, "not-a-mapping", f"---\n{block}\n---\n# Body\n")

        assert extract_from_skill(skill_dir) is None

    @pytest.mark.parametrize(
        ("content_type", "relative", "manifest", "message"),
        [
            pytest.param(
                "skill",
                "nested/bomb/SKILL.md",
                _alias_dag("description", levels=18, fanout=2),
                "complexity limit exceeded",
                id="skill",
            ),
            pytest.param(
                "rules",
                "nested/long.mdc",
                "---\ntitle: " + "x" * 16_385 + "\ndescription: Rule\n---\n",
                "title exceeds the 16384-character limit",
                id="rules",
            ),
            pytest.param(
                "workflows",
                "nested/long/workflow-rules.mdc",
                "---\ntitle: Workflow\ndescription: " + "x" * 16_385 + "\n---\n",
                "description exceeds the 16384-character limit",
                id="workflows",
            ),
        ],
    )
    def test_collection_error_names_the_manifest(
        self, tmp_path: Path, write_skill, content_type: str, relative: str, manifest: str, message: str
    ) -> None:
        write_skill(tmp_path, "valid-skill", "A valid skill")
        _write_rule(tmp_path, "valid.mdc", "valid-rule", "A valid rule")
        _write_workflow(tmp_path, "valid-workflow", "valid-workflow", "A valid workflow")
        target = tmp_path.joinpath(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(manifest.encode("utf-8"))

        with pytest.raises(ValueError, match=rf"{message}: {relative}$"):
            discover_and_extract(tmp_path, content_type)

    @pytest.mark.parametrize(
        ("content_type", "relative", "manifest", "expected", "reason"),
        [
            pytest.param(
                "skill",
                "nested/odd/SKILL.md",
                "---\nname: 2048\ndescription: Odd\n---\n",
                "valid-skill",
                "non_string_name_or_description",
                id="skill",
            ),
            pytest.param(
                "skill",
                "nested/odd/SKILL.md",
                "---\n- item\n- item\n---\n# Body\n",
                "valid-skill",
                "invalid_frontmatter_mapping",
                id="skill-list",
            ),
            pytest.param(
                "rules",
                "nested/odd.mdc",
                "---\ntitle: [a, b]\ndescription: Rule\n---\n",
                "valid-rule",
                "non_string_name_or_description",
                id="rules",
            ),
            pytest.param(
                "workflows",
                "nested/odd/workflow-rules.mdc",
                "---\ntitle: Workflow\ndescription: {a: b}\n---\n",
                "valid-workflow",
                "non_string_name_or_description",
                id="workflows",
            ),
        ],
    )
    def test_collection_excludes_a_manifest_whose_field_is_not_a_string(
        self, tmp_path: Path, write_skill, content_type: str, relative: str, manifest: str, expected: str, reason: str
    ) -> None:
        write_skill(tmp_path, "valid-skill", "A valid skill")
        _write_rule(tmp_path, "valid.mdc", "valid-rule", "A valid rule")
        _write_workflow(tmp_path, "valid-workflow", "valid-workflow", "A valid workflow")
        target = tmp_path.joinpath(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(manifest.encode("utf-8"))

        result = discover_and_extract_detailed(tmp_path, content_type)

        assert [entry.name for entry in result.entries] == [expected]
        assert result.selected_count == 2
        assert result.exclusions == [{"path": relative, "reason": reason}]

    @pytest.mark.parametrize(
        ("frontmatter", "reason"),
        [
            pytest.param('name: odd\ndescription: Odd\nversion: !!int ""', "invalid_yaml", id="tagged-int"),
            pytest.param("name: odd\ndescription: Odd\nlast_updated: 2024-02-30", "invalid_yaml", id="impossible-date"),
            # The YAML is malformed before its root node, so it is invalid, not a non-mapping root.
            pytest.param("\tname: odd\ndescription: Odd", "invalid_yaml", id="tab-before-root"),
            pytest.param("- item\n- item", "invalid_frontmatter_mapping", id="list-root"),
            # A list root is never built, so YAML errors after it do not change the reason.
            pytest.param("- item\n- key: a: b", "invalid_frontmatter_mapping", id="list-root-then-invalid"),
            pytest.param("name: ''\ndescription: Odd", "missing_name_or_description", id="empty-name"),
        ],
    )
    def test_collection_names_why_frontmatter_is_excluded(
        self, tmp_path: Path, write_skill, frontmatter: str, reason: str
    ) -> None:
        write_skill(tmp_path, "valid-skill", "A valid skill")
        _write_skill_manifest(tmp_path, "odd", f"---\n{frontmatter}\n---\n# Body\n")

        result = discover_and_extract_detailed(tmp_path, "skill")

        assert [entry.name for entry in result.entries] == ["valid-skill"]
        assert result.exclusions == [{"path": "odd/SKILL.md", "reason": reason}]
