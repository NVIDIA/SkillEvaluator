# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for skillevaluator.deduplication.utils.skill_collector."""

from __future__ import annotations

import logging
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from skillevaluator.constants import CONTENT_DEDUP_EXCLUDED_FILES
from skillevaluator.deduplication.utils import skill_collector
from skillevaluator.deduplication.utils.skill_collector import (
    CollectedFile,
    SkillCollectionError,
    collect_files,
)


class TestCollectedFile:
    def test_dataclass_fields(self, tmp_path: Path) -> None:
        cf = CollectedFile(
            path=tmp_path / "test.md",
            rel_path="test.md",
            extension=".md",
            content="hello",
            line_count=1,
        )
        assert cf.rel_path == "test.md"
        assert cf.extension == ".md"
        assert cf.content == "hello"
        assert cf.line_count == 1


class TestCollectFiles:
    def test_collects_markdown(self, skill_root: Path) -> None:
        (skill_root / "SKILL.md").write_text("---\nname: test\n---\n# Body\nContent here.")
        result = collect_files(skill_root)
        assert len(result) == 1
        assert result[0].extension == ".md"
        assert result[0].rel_path == "SKILL.md"

    def test_collects_python(self, skill_root: Path) -> None:
        (skill_root / "helper.py").write_text("def foo():\n    pass\n")
        result = collect_files(skill_root)
        assert len(result) == 1
        assert result[0].extension == ".py"

    def test_collects_shell(self, skill_root: Path) -> None:
        (skill_root / "setup.sh").write_text("#!/bin/bash\necho hello\n")
        result = collect_files(skill_root)
        assert len(result) == 1
        assert result[0].extension == ".sh"

    def test_skips_non_scannable_extensions(self, skill_root: Path) -> None:
        (skill_root / "image.png").write_bytes(b"\x89PNG")
        (skill_root / "data.json").write_text('{"key": "value"}')
        (skill_root / "config.yaml").write_text("key: value")
        (skill_root / "notes.txt").write_text("some notes")
        result = collect_files(skill_root)
        assert len(result) == 0

    def test_strips_frontmatter_from_markdown(self, skill_root: Path) -> None:
        (skill_root / "SKILL.md").write_text("---\nname: test\ndescription: a skill\n---\n# Body\nActual content.")
        result = collect_files(skill_root)
        assert len(result) == 1
        assert "---" not in result[0].content
        assert "name: test" not in result[0].content
        assert "Actual content" in result[0].content

    def test_line_count_includes_frontmatter(self, skill_root: Path) -> None:
        text = "---\nname: test\n---\n# Body\nContent."
        (skill_root / "SKILL.md").write_text(text)
        result = collect_files(skill_root)
        assert result[0].line_count == len(text.splitlines())
        assert result[0].line_offset == 3

    def test_does_not_strip_frontmatter_from_python(self, skill_root: Path) -> None:
        py_content = '---\nname: not-frontmatter\n---\nprint("hello")'
        (skill_root / "script.py").write_text(py_content)
        result = collect_files(skill_root)
        assert "---" in result[0].content

    def test_returns_sorted_list(self, skill_root: Path) -> None:
        (skill_root / "z_last.md").write_text("# Z")
        (skill_root / "a_first.md").write_text("# A")
        (skill_root / "m_middle.py").write_text("pass")
        result = collect_files(skill_root)
        rel_paths = [f.rel_path for f in result]
        assert rel_paths == sorted(rel_paths)

    def test_empty_directory(self, skill_root: Path) -> None:
        result = collect_files(skill_root)
        assert result == []

    def test_relative_path_is_relative_to_root(self, skill_root: Path) -> None:
        refs_dir = skill_root / "references"
        refs_dir.mkdir()
        (refs_dir / "guide.md").write_text("# Guide")
        result = collect_files(skill_root)
        assert result[0].rel_path == "references/guide.md"

    def test_markdown_without_frontmatter_uses_full_content(self, skill_root: Path) -> None:
        (skill_root / "notes.md").write_text("# No Frontmatter\nJust plain markdown.")
        result = collect_files(skill_root)
        assert "# No Frontmatter" in result[0].content

    def test_collects_mdc_files(self, skill_root: Path) -> None:
        (skill_root / "rule.mdc").write_text("---\ntitle: A Rule\n---\nRule body.")
        result = collect_files(skill_root)
        assert len(result) == 1
        assert result[0].extension == ".mdc"


class TestCollectFilesSafety:
    @pytest.mark.skipif(os.name != "posix", reason="descriptor-anchored availability is a POSIX-specific contract")
    def test_fails_closed_without_descriptor_anchored_reads(self, skill_root: Path, monkeypatch) -> None:
        (skill_root / "SKILL.md").write_text("# Skill\nSAFE_CONTENT")
        monkeypatch.setattr(
            skill_collector,
            "_supports_descriptor_anchored_reads",
            lambda: False,
            raising=False,
        )

        with pytest.raises(SkillCollectionError) as exc_info:
            collect_files(skill_root)

        assert exc_info.value.check_name == "secure_open_unavailable"
        assert "secure" in str(exc_info.value).lower()

    @pytest.mark.skipif(os.name != "posix", reason="descriptor lifecycle regression is POSIX-specific")
    def test_secure_root_closes_descriptor_when_root_verification_fails(self, skill_root: Path, monkeypatch) -> None:
        real_open = os.open
        real_fstat = os.fstat
        real_close = os.close
        root_fd: int | None = None
        closed_fds: list[int] = []

        def tracked_open(path, flags, *, dir_fd=None):
            nonlocal root_fd
            if dir_fd is None:
                fd = real_open(path, flags)
            else:
                fd = real_open(path, flags, dir_fd=dir_fd)
            if dir_fd is None and Path(path) == skill_root:
                root_fd = fd
            return fd

        def failing_fstat(fd: int):
            if fd == root_fd:
                raise OSError("simulated root fstat failure")
            return real_fstat(fd)

        def tracked_close(fd: int) -> None:
            closed_fds.append(fd)
            real_close(fd)

        monkeypatch.setattr(skill_collector.os, "open", tracked_open)
        monkeypatch.setattr(skill_collector.os, "fstat", failing_fstat)
        monkeypatch.setattr(skill_collector.os, "close", tracked_close)

        with (
            pytest.raises(skill_collector._SecureReadError, match="verify"),
            skill_collector._SecureRoot(skill_root),
        ):
            pass

        assert root_fd is not None
        assert root_fd in closed_fds

    @pytest.mark.skipif(os.name != "posix", reason="descriptor lifecycle regression is POSIX-specific")
    def test_secure_root_closes_ancestor_descriptor_when_verification_fails(
        self, skill_root: Path, monkeypatch
    ) -> None:
        references = skill_root / "references"
        references.mkdir()
        (references / "guide.md").write_text("SAFE_CONTENT")

        real_open = os.open
        real_fstat = os.fstat
        real_close = os.close
        ancestor_fd: int | None = None
        closed_fds: list[int] = []

        def tracked_open(path, flags, *, dir_fd=None):
            nonlocal ancestor_fd
            if dir_fd is None:
                fd = real_open(path, flags)
            else:
                fd = real_open(path, flags, dir_fd=dir_fd)
            if dir_fd is not None and Path(path).name == "references":
                ancestor_fd = fd
            return fd

        def failing_fstat(fd: int):
            if fd == ancestor_fd:
                raise OSError("simulated ancestor fstat failure")
            return real_fstat(fd)

        def tracked_close(fd: int) -> None:
            closed_fds.append(fd)
            real_close(fd)

        monkeypatch.setattr(skill_collector.os, "open", tracked_open)
        monkeypatch.setattr(skill_collector.os, "fstat", failing_fstat)
        monkeypatch.setattr(skill_collector.os, "close", tracked_close)

        with (
            pytest.raises(skill_collector._SecureReadError, match="verify"),
            skill_collector._SecureRoot(skill_root) as secure_root,
        ):
            secure_root.read_bounded(Path("references/guide.md"), 1024)

        assert ancestor_fd is not None
        assert ancestor_fd in closed_fds

    @pytest.mark.skipif(os.name != "posix", reason="descriptor-anchored openat regression is POSIX-specific")
    def test_ancestor_swap_never_reads_outside_skill_root(self, skill_root: Path, tmp_path: Path, monkeypatch) -> None:
        references = skill_root / "references"
        references.mkdir()
        target = references / "guide.md"
        target.write_text("# Safe guide\nSAFE_CONTENT")

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "guide.md").write_text("# Outside\nSECRET_CANARY")

        original_references = skill_root / "references-original"
        real_open = os.open
        swapped = False

        def swapping_open(path, flags, *, dir_fd=None):
            nonlocal swapped
            if Path(path).name == "guide.md" and not swapped:
                references.rename(original_references)
                references.symlink_to(outside, target_is_directory=True)
                swapped = True
            if dir_fd is None:
                return real_open(path, flags)
            return real_open(path, flags, dir_fd=dir_fd)

        monkeypatch.setattr(skill_collector.os, "open", swapping_open)

        try:
            result = collect_files(skill_root)
        except SkillCollectionError:
            assert swapped
            return

        assert swapped
        collected_text = "\n".join(item.content for item in result)
        assert "SECRET_CANARY" not in collected_text
        assert "SAFE_CONTENT" in collected_text

    def test_rejects_unbounded_directory_traversal(self, skill_root: Path, monkeypatch) -> None:
        monkeypatch.setattr(skill_collector, "CONTENT_DEDUP_MAX_DISCOVERED_PATHS", 2)
        for name in ("a.bin", "b.bin", "c.bin"):
            (skill_root / name).write_bytes(b"x")

        with pytest.raises(SkillCollectionError) as exc_info:
            collect_files(skill_root)

        assert exc_info.value.check_name == "path_count_limit"
        assert exc_info.value.metadata == {"actual": 3, "limit": 2}

    def test_directory_traversal_error_is_actionable(self, skill_root: Path, monkeypatch) -> None:
        def deny_traversal(_root: Path, *, onerror=None, **_kwargs):
            assert onerror is not None
            onerror(PermissionError("permission denied"))
            return iter(())

        monkeypatch.setattr(os, "walk", deny_traversal)

        with pytest.raises(SkillCollectionError) as exc_info:
            collect_files(skill_root)

        assert exc_info.value.check_name == "path_access_error"
        assert exc_info.value.rel_path == "."
        assert "traverse" in str(exc_info.value).lower()
        assert "readable" in exc_info.value.suggestion.lower()

    def test_rejects_symlinked_scannable_file(self, skill_root: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside.md"
        outside.write_text("private host content")
        link = skill_root / "references" / "outside.md"
        link.parent.mkdir()
        link.symlink_to(outside)

        with pytest.raises(SkillCollectionError) as exc_info:
            collect_files(skill_root)

        assert exc_info.value.check_name == "unsafe_path"
        assert exc_info.value.rel_path == "references/outside.md"
        assert "symbolic link or reparse point" in str(exc_info.value)
        assert "replace" in exc_info.value.suggestion.lower()

    def test_rejects_reparse_directory_before_descending(self, skill_root: Path) -> None:
        target = skill_root / "reparse-directory"
        target.mkdir()
        (target / "private.md").write_text("private host content")
        original_lstat = Path.lstat

        def guarded_walk(root: Path, **_kwargs):
            yield root, [target.name], []
            raise AssertionError("walk descended into a reparse directory")

        def fake_lstat(path: Path):
            if path == target:
                return SimpleNamespace(
                    st_mode=stat.S_IFDIR,
                    st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT,
                )
            return original_lstat(path)

        with (
            patch.object(Path, "lstat", fake_lstat),
            patch.object(os, "walk", guarded_walk),
            pytest.raises(SkillCollectionError) as exc_info,
        ):
            collect_files(skill_root)

        assert exc_info.value.check_name == "unsafe_path"
        assert exc_info.value.rel_path == "reparse-directory"
        assert "symbolic link or reparse point" in str(exc_info.value)

    @pytest.mark.skipif(os.name != "nt", reason="directory junctions are Windows-specific")
    def test_rejects_windows_junction_before_walking_target(
        self, skill_root: Path, tmp_path: Path, monkeypatch
    ) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "private.md").write_text("private host content")
        junction = skill_root / "junction"
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
            check=True,
            capture_output=True,
            text=True,
        )
        real_walk = os.walk
        visited: list[Path] = []

        def recording_walk(*args, **kwargs):
            for entry in real_walk(*args, **kwargs):
                visited.append(Path(entry[0]))
                yield entry

        monkeypatch.setattr(os, "walk", recording_walk)

        with pytest.raises(SkillCollectionError) as exc_info:
            collect_files(skill_root)

        assert exc_info.value.check_name == "unsafe_path"
        assert exc_info.value.rel_path == "junction"
        assert "symbolic link or reparse point" in str(exc_info.value)
        assert junction not in visited

    def test_rejects_windows_reparse_point_even_when_not_a_symlink(self, skill_root: Path) -> None:
        target = skill_root / "reparse.md"
        target.write_text("content")
        original_lstat = Path.lstat

        def fake_lstat(path: Path):
            if path == target:
                return SimpleNamespace(
                    st_mode=stat.S_IFREG,
                    st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT,
                )
            return original_lstat(path)

        with (
            patch.object(Path, "lstat", fake_lstat),
            pytest.raises(SkillCollectionError) as exc_info,
        ):
            collect_files(skill_root)

        assert exc_info.value.check_name == "unsafe_path"
        assert exc_info.value.rel_path == "reparse.md"
        assert "symbolic link or reparse point" in str(exc_info.value)

    def test_rejects_resolved_path_outside_skill_root_even_if_link_check_is_bypassed(
        self, skill_root: Path, tmp_path: Path
    ) -> None:
        outside = tmp_path / "outside.md"
        outside.write_text("private host content")
        link = skill_root / "outside.md"
        link.symlink_to(outside)

        with (
            patch.object(skill_collector, "_is_link_or_reparse", return_value=False),
            pytest.raises(SkillCollectionError) as exc_info,
        ):
            collect_files(skill_root)

        assert exc_info.value.check_name == "unsafe_path"
        assert exc_info.value.rel_path == "outside.md"
        assert "resolves outside" in str(exc_info.value)

    def test_rejects_more_than_maximum_scannable_files(self, skill_root: Path, monkeypatch) -> None:
        monkeypatch.setattr(skill_collector, "CONTENT_DEDUP_MAX_FILES", 1)
        (skill_root / "a.md").write_text("a")
        (skill_root / "b.md").write_text("b")

        with pytest.raises(SkillCollectionError) as exc_info:
            collect_files(skill_root)

        assert exc_info.value.check_name == "file_count_limit"
        assert exc_info.value.metadata == {"actual": 2, "limit": 1}

    def test_rejects_file_larger_than_per_file_byte_limit(self, skill_root: Path, monkeypatch) -> None:
        monkeypatch.setattr(skill_collector, "CONTENT_DEDUP_MAX_FILE_BYTES", 4)
        target = skill_root / "oversized.md"
        target.write_bytes(b"12345")

        with pytest.raises(SkillCollectionError) as exc_info:
            collect_files(skill_root)

        assert exc_info.value.check_name == "file_size_limit"
        assert exc_info.value.rel_path == "oversized.md"
        assert exc_info.value.metadata == {"actual_bytes": 5, "limit_bytes": 4}

    def test_rejects_combined_content_above_total_byte_limit(self, skill_root: Path, monkeypatch) -> None:
        monkeypatch.setattr(skill_collector, "CONTENT_DEDUP_MAX_FILE_BYTES", 10)
        monkeypatch.setattr(skill_collector, "CONTENT_DEDUP_MAX_TOTAL_BYTES", 7)
        (skill_root / "a.md").write_bytes(b"1234")
        (skill_root / "b.md").write_bytes(b"5678")

        with pytest.raises(SkillCollectionError) as exc_info:
            collect_files(skill_root)

        assert exc_info.value.check_name == "total_size_limit"
        assert exc_info.value.metadata == {"actual_bytes": 8, "limit_bytes": 7}


class TestCollectFilesExclusions:
    def test_excluded_path_debug_log_is_relative(self, skill_root: Path, caplog) -> None:
        excluded = skill_root / "references" / "evals"
        excluded.mkdir(parents=True)
        (excluded / "fixture.md").write_text("# fixture\nbody")

        with caplog.at_level(logging.DEBUG, logger=skill_collector.__name__):
            collect_files(skill_root)

        assert str(excluded) not in caplog.text
        assert "references/evals" in caplog.text

    """Tier 2 dedup must ignore evaluation harness output and version snapshots.

    Both the live skill and its meta-folders (``references/``, ``scripts/``,
    ``assets/``) feed the dedup pipeline, but ``evals/`` and ``.versions/``
    contain near-copies of the live skill (Harbor task environments and
    historical snapshots) that would otherwise dominate every dedup report
    with self-matches.
    """

    def test_keeps_meta_folders(self, skill_root: Path) -> None:
        """Standard meta-folders (``references``, ``scripts``, ``assets``) stay in scope.

        The exclusion is targeted at evaluation/version artifacts only —
        every other meta-folder still feeds the dedup pipeline so we can
        catch real cross-file duplication within the live skill.
        """
        (skill_root / "SKILL.md").write_text("# Skill")
        for sub in ("references", "scripts", "assets"):
            d = skill_root / sub
            d.mkdir()
            (d / "doc.md").write_text(f"# {sub}\nbody")

        result = collect_files(skill_root)
        rel_paths = sorted(f.rel_path for f in result)
        assert rel_paths == [
            "SKILL.md",
            "assets/doc.md",
            "references/doc.md",
            "scripts/doc.md",
        ]

    def test_excludes_evals_at_skill_root(self, skill_root: Path) -> None:
        """Top-level ``evals/`` must not contribute files to the dedup pass."""
        (skill_root / "SKILL.md").write_text("# Live\nReal content.")
        evals_dir = skill_root / "evals"
        evals_dir.mkdir()
        (evals_dir / "evals.json").write_text('{"cases": []}')
        (evals_dir / "fixture.md").write_text("# Eval fixture\nShould be ignored.")

        result = collect_files(skill_root)
        rel_paths = [f.rel_path for f in result]
        assert rel_paths == ["SKILL.md"]

    def test_excludes_harbor_results_under_evals(self, skill_root: Path) -> None:
        """Reproduces the user-reported case: harbor run snapshots under ``evals/results/``.

        The Tier 3 harbor runner copies the entire skill into each task's
        environment, producing files that match the live skill byte-for-byte.
        Including them would flag the live SKILL.md as a duplicate of every
        per-task copy.
        """
        (skill_root / "SKILL.md").write_text("# Build with kdb (~8 hrs)\nRun the full build.")
        refs = skill_root / "references"
        refs.mkdir()
        (refs / "build-reference.md").write_text("# Build reference\nDetailed build docs.")

        snapshot = (
            skill_root
            / "evals"
            / "results"
            / "20260505_001822"
            / "_harbor-tasks"
            / "nvgpu-skill-001"
            / "environment"
            / "skills"
            / "nvgpu-skill"
        )
        snapshot.mkdir(parents=True)
        (snapshot / "SKILL.md").write_text("# Build with kdb (~8 hrs)\nRun the full build.")
        snapshot_refs = snapshot / "references"
        snapshot_refs.mkdir()
        (snapshot_refs / "build-reference.md").write_text("# Build reference\nDetailed build docs.")

        result = collect_files(skill_root)
        rel_paths = sorted(f.rel_path for f in result)
        assert rel_paths == ["SKILL.md", "references/build-reference.md"]

    def test_excludes_versions_snapshot(self, skill_root: Path) -> None:
        """``.versions/<version>/`` snapshots are mirrors of the live skill."""
        (skill_root / "SKILL.md").write_text("## Purpose\nLive purpose.")
        scripts = skill_root / "scripts"
        scripts.mkdir()
        (scripts / "search.py").write_text('"""Search bugs."""\n\ndef search_bugs():\n    pass\n')

        version_dir = skill_root / ".versions" / "1.0.0"
        version_dir.mkdir(parents=True)
        (version_dir / "SKILL.md").write_text("## Purpose\nLive purpose.")
        version_scripts = version_dir / "scripts"
        version_scripts.mkdir()
        (version_scripts / "search.py").write_text('"""Search bugs."""\n\ndef search_bugs():\n    pass\n')
        version_refs = version_dir / "references"
        version_refs.mkdir()
        (version_refs / "search-parameters.md").write_text("# Params\nDetails.")

        result = collect_files(skill_root)
        rel_paths = sorted(f.rel_path for f in result)
        assert rel_paths == ["SKILL.md", "scripts/search.py"]

    def test_excludes_generated_skill_card(self, skill_root: Path) -> None:
        """``skill-card.md`` is generated from the manifest and signed downstream."""
        (skill_root / "SKILL.md").write_text(
            "---\n"
            "name: cuopt-developer\n"
            "description: Helps developers build and debug cuOpt integrations.\n"
            "---\n"
            "## Workflow\n"
            "Use this skill when building and debugging cuOpt integrations.\n"
        )
        (skill_root / "skill-card.md").write_text(
            "## Description:\n"
            "Helps developers build and debug cuOpt integrations.\n\n"
            "## Use Case:\n"
            "Use this skill when building and debugging cuOpt integrations.\n"
        )
        refs = skill_root / "references"
        refs.mkdir()
        (refs / "guide.md").write_text("# Guide\nReal author-owned context.")

        result = collect_files(skill_root)
        rel_paths = sorted(f.rel_path for f in result)
        assert rel_paths == ["SKILL.md", "references/guide.md"]

    def test_excludes_generated_benchmark_report(self, skill_root: Path) -> None:
        """``BENCHMARK.md`` is generated and refreshed from validation output."""
        (skill_root / "SKILL.md").write_text(
            "---\n"
            "name: cuopt-developer\n"
            "description: Helps developers build and debug cuOpt integrations.\n"
            "---\n"
            "## Workflow\n"
            "Use this skill when building and debugging cuOpt integrations.\n"
        )
        (skill_root / "BENCHMARK.md").write_text(
            "# Evaluation Report\n\n"
            "This benchmark summarizes validation and Tier 3 live agent results.\n\n"
            "Use this skill when building and debugging cuOpt integrations.\n"
        )
        refs = skill_root / "references"
        refs.mkdir()
        (refs / "guide.md").write_text("# Guide\nReal author-owned context.")

        result = collect_files(skill_root)
        rel_paths = sorted(f.rel_path for f in result)
        assert rel_paths == ["SKILL.md", "references/guide.md"]

    def test_keeps_plural_benchmarks_report(self, skill_root: Path) -> None:
        """``benchmarks.md`` is not a generated artifact."""
        assert "benchmarks.md" not in CONTENT_DEDUP_EXCLUDED_FILES
        (skill_root / "SKILL.md").write_text("# Skill")
        (skill_root / "benchmarks.md").write_text("# Author benchmark notes\nReal context.")

        result = collect_files(skill_root)
        rel_paths = sorted(f.rel_path for f in result)
        assert rel_paths == ["SKILL.md", "benchmarks.md"]

    def test_dedup_exclusion_list_includes_generated_signature(self) -> None:
        """``skill.oms.sig`` is generated signing output and should stay out of dedup."""
        assert "skill.oms.sig" in CONTENT_DEDUP_EXCLUDED_FILES

    def test_excludes_nested_evals_inside_meta_folder(self, skill_root: Path) -> None:
        """An ``evals/`` directory nested inside another folder is also excluded.

        Matching against any path component (rather than just the top level)
        keeps the filter robust against unusual layouts where a meta-folder
        carries its own evaluation fixtures.
        """
        (skill_root / "SKILL.md").write_text("# Skill")
        nested_evals = skill_root / "references" / "evals"
        nested_evals.mkdir(parents=True)
        (nested_evals / "fixture.md").write_text("# fixture\nbody")

        keep = skill_root / "references" / "guide.md"
        keep.write_text("# guide\nbody")

        result = collect_files(skill_root)
        rel_paths = sorted(f.rel_path for f in result)
        assert rel_paths == ["SKILL.md", "references/guide.md"]

    def test_custom_exclusion_set_overrides_default(self, skill_root: Path) -> None:
        """Callers can opt out of the default filter (e.g. for diagnostic dumps)."""
        (skill_root / "SKILL.md").write_text("# Skill")
        evals_dir = skill_root / "evals"
        evals_dir.mkdir()
        (evals_dir / "fixture.md").write_text("# fixture\nbody")

        result = collect_files(skill_root, excluded_dirs=())
        rel_paths = sorted(f.rel_path for f in result)
        assert rel_paths == ["SKILL.md", "evals/fixture.md"]

    def test_custom_exclusion_set_extends_filter(self, skill_root: Path) -> None:
        """Callers can swap in a different set when scanning non-skill trees."""
        (skill_root / "SKILL.md").write_text("# Skill")
        cache = skill_root / "build_cache"
        cache.mkdir()
        (cache / "stale.md").write_text("# stale\nbody")

        result = collect_files(skill_root, excluded_dirs={"build_cache"})
        rel_paths = sorted(f.rel_path for f in result)
        assert rel_paths == ["SKILL.md"]


class TestPathBudgetExcludesArtifacts:
    def test_path_limit_ignores_generated_artifact_trees(self, tmp_path: Path, monkeypatch) -> None:
        # Live regression: a well-used skill accumulates thousands of trial
        # artifacts under evals/results/. They are excluded content and must
        # not consume the path-count budget (managing-calendar failed Tier 2
        # with "more than 4096 paths" on generated files it never scans).
        monkeypatch.setattr(skill_collector, "CONTENT_DEDUP_MAX_DISCOVERED_PATHS", 8)
        skill = tmp_path / "busy-skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text("# Busy skill\n\nAuthored content.", encoding="utf-8")
        trials = skill / "evals" / "results" / "20260101_000000" / "trials"
        trials.mkdir(parents=True)
        for index in range(10):
            (trials / f"artifact-{index}.json").write_text("{}", encoding="utf-8")

        result = collect_files(skill)

        assert [f.rel_path for f in result] == ["SKILL.md"]

    def test_path_limit_still_applies_to_authored_content(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(skill_collector, "CONTENT_DEDUP_MAX_DISCOVERED_PATHS", 8)
        skill = tmp_path / "huge-skill"
        (skill / "docs").mkdir(parents=True)
        for index in range(10):
            (skill / "docs" / f"note-{index}.md").write_text("hi", encoding="utf-8")

        with pytest.raises(SkillCollectionError) as exc_info:
            collect_files(skill)

        assert exc_info.value.check_name == "path_count_limit"
        assert exc_info.value.metadata == {"actual": 9, "limit": 8}
