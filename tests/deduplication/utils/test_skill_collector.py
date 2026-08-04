# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for skillevaluator.deduplication.utils.skill_collector."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from skillevaluator.constants import CONTENT_DEDUP_EXCLUDED_FILES
from skillevaluator.deduplication.utils import skill_collector
from skillevaluator.deduplication.utils.skill_collector import CollectedFile, collect_files
from skillevaluator.utils import secure_fs


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


class TestCollectFilesSecurityContract:
    def test_rejects_symlinked_root(self, tmp_path: Path) -> None:
        real_root = tmp_path / "real-skill"
        real_root.mkdir()
        (real_root / "SKILL.md").write_text("# Real skill")
        linked_root = tmp_path / "linked-skill"
        linked_root.symlink_to(real_root, target_is_directory=True)

        with pytest.raises(ValueError, match=r"root|symlink|reparse"):
            collect_files(linked_root)

    def test_rejects_linked_directory_before_descent(self, skill_root: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.md").write_text("SECRET_CANARY")
        (skill_root / "references").symlink_to(outside, target_is_directory=True)

        with pytest.raises(ValueError, match=r"directory|symlink|reparse|unsafe"):
            collect_files(skill_root)

    def test_rejects_linked_directory_before_excluded_name_pruning(self, skill_root: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside-evals"
        outside.mkdir()
        (outside / "secret.md").write_text("SECRET_CANARY")
        (skill_root / "evals").symlink_to(outside, target_is_directory=True)

        with pytest.raises(ValueError, match=r"directory|symlink|reparse|unsafe"):
            collect_files(skill_root)

    def test_rejects_selected_file_symlink_even_when_contained(self, skill_root: Path) -> None:
        target = skill_root / "AGENTS.md"
        target.write_text("# independently selected target")
        (skill_root / "guide.md").symlink_to(target.name)

        with pytest.raises(ValueError, match=r"guide\.md|symlink|reparse|unsafe"):
            collect_files(skill_root)

    def test_ignores_irrelevant_links_without_resolving_them(
        self, skill_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contained = skill_root / "contained.bin"
        contained_target = skill_root / "payload.dat"
        contained_target.write_bytes(b"contained")
        contained.symlink_to(contained_target.name)
        escaping = skill_root / "escaping.bin"
        outside = tmp_path / "outside.dat"
        outside.write_bytes(b"outside")
        escaping.symlink_to(outside)
        broken = skill_root / "broken.bin"
        broken.symlink_to("missing.dat")

        real_resolve = Path.resolve

        def reject_link_resolution(path: Path, *args, **kwargs):
            if path in {contained, escaping, broken}:
                raise AssertionError(f"irrelevant link was resolved: {path.name}")
            return real_resolve(path, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", reject_link_resolution)

        assert collect_files(skill_root) == []

    def test_ignores_suffixless_irrelevant_file_alias(self, skill_root: Path) -> None:
        target = skill_root / "LICENSE.txt"
        target.write_text("license text")
        (skill_root / "LICENSE").symlink_to(target.name)

        assert collect_files(skill_root) == []

    def test_deduplicates_contained_claude_agents_compatibility_alias(self, skill_root: Path) -> None:
        agents = skill_root / "AGENTS.md"
        agents.write_text("# Shared agent context")
        (skill_root / "CLAUDE.md").symlink_to(agents.name)

        result = collect_files(skill_root)

        assert [item.rel_path for item in result] == ["AGENTS.md"]

    @pytest.mark.parametrize("target", ["./AGENTS.md", "AGENTS.md/"])
    def test_rejects_non_exact_or_broken_compatibility_alias(self, skill_root: Path, target: str) -> None:
        (skill_root / "AGENTS.md").write_text("# Shared agent context")
        (skill_root / "CLAUDE.md").symlink_to(target)

        with pytest.raises(ValueError, match=r"CLAUDE|symlink|unsafe|target"):
            collect_files(skill_root)

    def test_rejects_hardlinked_compatibility_alias_inode(self, skill_root: Path) -> None:
        agents = skill_root / "AGENTS.md"
        agents.write_text("# Shared agent context")
        claude = skill_root / "CLAUDE.md"
        claude.symlink_to(agents.name)
        second_name = skill_root / "irrelevant-alias.bin"
        try:
            os.link(claude, second_name, follow_symlinks=False)
        except (NotImplementedError, OSError) as exc:
            pytest.skip(f"hardlinking a symlink inode is unavailable: {exc}")
        if claude.lstat().st_nlink == 1:
            pytest.skip("platform followed the symlink while creating the hardlink")

        with pytest.raises(ValueError, match=r"hard.?link|link count"):
            collect_files(skill_root)

    def test_relative_root_preserves_relative_collected_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        skill = tmp_path / "relative-skill"
        skill.mkdir()
        (skill / "guide.md").write_text("guide")
        monkeypatch.chdir(tmp_path)

        result = collect_files(Path("relative-skill"))

        assert result[0].path == Path("relative-skill/guide.md")

    def test_rejects_hard_linked_selected_file(self, skill_root: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside.md"
        outside.write_text("SECRET_CANARY")
        os.link(outside, skill_root / "linked.md")

        with pytest.raises(ValueError, match=r"hard.?link|link count|regular"):
            collect_files(skill_root)

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are unavailable on this platform")
    def test_rejects_special_selected_file(self, skill_root: Path) -> None:
        fifo = skill_root / "stream.md"
        os.mkfifo(fifo)

        with pytest.raises(ValueError, match=r"special|non-regular|regular file"):
            collect_files(skill_root)

    def test_windows_reparse_file_is_rejected(self, skill_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        selected = skill_root / "reparse.md"
        selected.write_text("selected content")
        selected_metadata = selected.lstat()
        original_check = secure_fs.stat_is_link_or_reparse

        def fake_reparse_check(metadata):
            return original_check(metadata) or os.path.samestat(metadata, selected_metadata)

        monkeypatch.setattr(secure_fs, "stat_is_link_or_reparse", fake_reparse_check)
        with pytest.raises(ValueError, match=r"reparse|symlink|unsafe"):
            collect_files(skill_root)

    def test_windows_reparse_directory_rejected_before_descent(
        self, skill_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = skill_root / "junction"
        directory.mkdir()
        (directory / "secret.md").write_text("SECRET_CANARY")
        directory_metadata = directory.lstat()
        original_check = secure_fs.stat_is_link_or_reparse
        real_open = os.open
        opened: list[str] = []

        def fake_reparse_check(metadata):
            return original_check(metadata) or os.path.samestat(metadata, directory_metadata)

        def recording_open(path, flags, mode=0o777, *, dir_fd=None):
            opened.append(str(path))
            if dir_fd is None:
                return real_open(path, flags, mode)
            return real_open(path, flags, mode, dir_fd=dir_fd)

        monkeypatch.setattr(secure_fs, "stat_is_link_or_reparse", fake_reparse_check)
        monkeypatch.setattr(secure_fs.os, "open", recording_open)
        with pytest.raises(ValueError, match=r"junction|reparse|symlink|unsafe"):
            collect_files(skill_root)
        assert "junction" not in opened

    def test_bounds_irrelevant_authored_paths(self, skill_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(skill_collector, "CONTENT_DEDUP_MAX_DISCOVERED_PATHS", 2, raising=False)
        for name in ("a.bin", "b.bin", "c.bin"):
            (skill_root / name).write_bytes(b"x")

        with pytest.raises(ValueError, match=r"path.*limit|more than 2 paths"):
            collect_files(skill_root)

    def test_prunes_excluded_tree_before_path_budget(self, skill_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(skill_collector, "CONTENT_DEDUP_MAX_DISCOVERED_PATHS", 2, raising=False)
        (skill_root / "SKILL.md").write_text("# Skill")
        generated = skill_root / "evals" / "results"
        generated.mkdir(parents=True)
        for index in range(10):
            (generated / f"artifact-{index}.md").write_text("generated")

        assert [item.rel_path for item in collect_files(skill_root)] == ["SKILL.md"]

    def test_enforces_per_file_and_total_read_budgets(self, skill_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(skill_collector, "CONTENT_DEDUP_MAX_FILE_BYTES", 4, raising=False)
        monkeypatch.setattr(skill_collector, "CONTENT_DEDUP_MAX_TOTAL_BYTES", 7, raising=False)
        (skill_root / "oversized.md").write_bytes(b"12345")

        with pytest.raises(ValueError, match=r"file.*limit|byte.*limit|exceeds"):
            collect_files(skill_root)

        (skill_root / "oversized.md").unlink()
        (skill_root / "a.md").write_bytes(b"1234")
        (skill_root / "b.md").write_bytes(b"5678")

        with pytest.raises(ValueError, match=r"total.*limit|byte.*limit|exceeds"):
            collect_files(skill_root)


class TestCollectFilesExclusions:
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
