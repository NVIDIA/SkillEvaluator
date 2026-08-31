# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for hygiene validator."""

from pathlib import Path

import pytest

from skillevaluator.validators.hygiene import HygieneValidator


class TestHygieneValidator:
    """Test cases for HygieneValidator."""

    @staticmethod
    def _test_discovery_detail(result):
        return next(detail for detail in result.success_details if detail.check_name == "test_discovery")

    @pytest.mark.parametrize("payload_path", ["pytest.py", "pytest/__main__.py"])
    def test_hygiene_never_executes_target_pytest_module(self, tmp_path: Path, payload_path: str):
        """Default Tier 1 must not execute a checkout-provided pytest module."""
        skill = tmp_path / "skill"
        tests_dir = skill / "tests"
        tests_dir.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: skill\ndescription: test\n---\n", encoding="utf-8")
        (tests_dir / "test_present.py").write_text("def test_present():\n    assert True\n", encoding="utf-8")
        marker = tmp_path / "target-code-executed"
        payload = skill / payload_path
        payload.parent.mkdir(parents=True, exist_ok=True)
        if payload_path.endswith("__main__.py"):
            (payload.parent / "__init__.py").write_text("", encoding="utf-8")
        payload.write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
            encoding="utf-8",
        )

        result = HygieneValidator().validate(skill)

        assert result.passed
        assert not marker.exists()
        detail = self._test_discovery_detail(result)
        assert detail.metadata["execution_performed"] is False
        assert detail.metadata["coverage_measured"] is False

    def test_static_discovery_counts_filename_candidates_without_parsing_or_execution(self, tmp_path: Path):
        skill = tmp_path / "skill"
        tests_dir = skill / "tests"
        tests_dir.mkdir(parents=True)
        marker = tmp_path / "test-module-executed"
        (tests_dir / "test_prefix.py").write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
            encoding="utf-8",
        )
        # Invalid UTF-8 proves discovery does not parse or import candidates.
        (tests_dir / "suffix_test.py").write_bytes(b"\xff\xfe\x00")

        result = HygieneValidator()._check_test_presence(skill)
        detail = self._test_discovery_detail(result)

        assert not marker.exists()
        assert detail.metadata == {
            "test_count": 2,
            "execution_performed": False,
            "coverage_measured": False,
            "patterns": ["test_*.py", "*_test.py"],
        }
        assert "were not executed" in detail.message
        assert "coverage was not measured" in detail.message

    def test_no_standard_python_tests_reports_unmeasured_coverage_truthfully(self, tmp_path: Path):
        skill = tmp_path / "skill"
        (skill / "tests").mkdir(parents=True)

        result = HygieneValidator()._check_test_presence(skill)
        detail = self._test_discovery_detail(result)

        assert detail.metadata["test_count"] == 0
        assert detail.metadata["execution_performed"] is False
        assert detail.metadata["coverage_measured"] is False
        assert "No standard Python test-file candidates found" in detail.message
        assert "coverage was not measured" in detail.message
        assert any("consider adding tests" in warning.lower() for warning in result.warnings)

    def test_static_discovery_excludes_artifacts_directories_and_symlinks(self, tmp_path: Path):
        skill = tmp_path / "skill"
        tests_dir = skill / "tests"
        tests_dir.mkdir(parents=True)
        (tests_dir / "test_regular.py").write_text("def test_regular():\n    assert True\n", encoding="utf-8")
        (skill / "evals" / "results").mkdir(parents=True)
        (skill / "evals" / "results" / "test_snapshot.py").write_text("raise RuntimeError\n", encoding="utf-8")
        (skill / "test_directory.py").mkdir()

        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside_test = outside_dir / "outside_test.py"
        outside_test.write_text("raise RuntimeError\n", encoding="utf-8")
        try:
            (tests_dir / "test_file_link.py").symlink_to(outside_test)
            (skill / "linked_tests").symlink_to(outside_dir, target_is_directory=True)
            (tests_dir / "test_broken_link.py").symlink_to(outside_dir / "missing.py")
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

        result = HygieneValidator()._check_test_presence(skill)
        detail = self._test_discovery_detail(result)

        assert detail.metadata["test_count"] == 1
        assert result.summary.files_scanned == 1

    def test_hygiene_never_loads_conftest_plugins_or_test_modules(self, tmp_path: Path, monkeypatch) -> None:
        skill = tmp_path / "skill"
        tests_dir = skill / "tests"
        tests_dir.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: skill\ndescription: test\n---\n", encoding="utf-8")
        monkeypatch.setenv("SEC001_TEST_SECRET", "must-not-be-read")
        conftest_marker = tmp_path / "conftest-executed"
        plugin_marker = tmp_path / "plugin-executed"
        test_marker = tmp_path / "test-module-executed"
        (skill / "conftest.py").write_text(
            "import os\n"
            "import socket\n"
            "from pathlib import Path\n"
            "pytest_plugins = ('target_plugin',)\n"
            "try:\n"
            "    socket.create_connection(('127.0.0.1', 9), timeout=0.01)\n"
            "except OSError:\n"
            "    pass\n"
            f"Path({str(conftest_marker)!r}).write_text(os.environ['SEC001_TEST_SECRET'], encoding='utf-8')\n",
            encoding="utf-8",
        )
        (skill / "target_plugin.py").write_text(
            f"from pathlib import Path\nPath({str(plugin_marker)!r}).write_text('executed', encoding='utf-8')\n",
            encoding="utf-8",
        )
        (tests_dir / "test_payload.py").write_text(
            f"from pathlib import Path\nPath({str(test_marker)!r}).write_text('executed', encoding='utf-8')\n",
            encoding="utf-8",
        )

        result = HygieneValidator().validate(skill)

        assert result.passed
        assert not conftest_marker.exists()
        assert not plugin_marker.exists()
        assert not test_marker.exists()
        assert self._test_discovery_detail(result).metadata["test_count"] == 1

    def test_valid_skill_passes(self, sample_skill_dir: Path):
        """Test validation passes for clean skill."""
        validator = HygieneValidator()
        result = validator.validate(sample_skill_dir)

        # Should pass (may have warnings about no tests)
        assert len(result.errors) == 0, f"Unexpected errors: {result.errors}"

    def test_detects_dead_links(self, tmp_path: Path):
        """Test detection of dead/broken links in markdown."""
        skill_dir = tmp_path / "dead-links-skill"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: dead-links-skill
description: A skill with dead links for testing
version: 1.0.0
---

# Dead Links Test

See the [documentation](./docs/README.md) for more info.
Check the [config file](./config/settings.yaml) for settings.
""")

        validator = HygieneValidator()
        result = validator.validate(skill_dir)

        # Should detect dead links
        assert any(
            "dead link" in e.lower() or "docs/README.md" in e or "config/settings.yaml" in e for e in result.errors
        )

    @pytest.mark.parametrize(
        ("content", "target"),
        [
            ("See [the guide][guide].\n\n[guide]: missing-reference.md\n", "missing-reference.md"),
            ("See [guide][].\n\n[guide]: missing-collapsed.md\n", "missing-collapsed.md"),
            ("See [guide].\n\n[guide]: missing-shortcut.md\n", "missing-shortcut.md"),
            ('See <a href="missing-html.md">the guide</a>.\n', "missing-html.md"),
            ("![diagram](missing-image.png)\n", "missing-image.png"),
        ],
    )
    def test_commonmark_dead_links_are_detected(self, tmp_path: Path, content: str, target: str):
        """Navigation and image targets discovered by CommonMark are checked."""
        skill_dir = tmp_path / "commonmark-dead-links"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

        result = HygieneValidator()._check_dead_links(skill_dir)

        assert result.errors == [f"Dead link in SKILL.md: {target}"]

    def test_commonmark_non_navigation_contexts_are_ignored(self, tmp_path: Path):
        """Frontmatter, code, comments, and non-navigation HTML stay inert."""
        skill_dir = tmp_path / "inert-links"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            '---\ndescription: "[metadata](frontmatter.md)"\n---\n\n'
            '`[inline](inline.md)`\n\n'
            "```markdown\n[fenced](fenced.md)\n```\n\n"
            "<!-- <a href='comment.md'>comment</a> -->\n"
            "<template><a href='template.md'>template</a></template>\n",
            encoding="utf-8",
        )

        result = HygieneValidator()._check_dead_links(skill_dir)

        assert result.errors == []

    def test_commonmark_local_targets_are_normalized_before_lookup(self, tmp_path: Path):
        """Encoded spaces, queries, fragments, and directory links resolve locally."""
        skill_dir = tmp_path / "normalized-links"
        skill_dir.mkdir()
        (skill_dir / "my file.md").write_text("# Existing\n", encoding="utf-8")
        (skill_dir / "guide.md").write_text("# Existing\n", encoding="utf-8")
        (skill_dir / "docs").mkdir()
        (skill_dir / "SKILL.md").write_text(
            "[space](<my file.md>)\n[query](guide.md?view=1)\n"
            "[fragment](guide.md#overview)\n[directory](docs/)\n",
            encoding="utf-8",
        )

        result = HygieneValidator()._check_dead_links(skill_dir)

        assert result.errors == []

    def test_missing_directory_target_is_reported(self, tmp_path: Path):
        """Trailing slashes remain checkable for dead directory links."""
        skill_dir = tmp_path / "missing-directory"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("[docs](missing-dir/)\n", encoding="utf-8")

        result = HygieneValidator()._check_dead_links(skill_dir)

        assert result.errors == ["Dead link in SKILL.md: missing-dir/"]

    def test_non_local_and_root_absolute_targets_are_ignored(self, tmp_path: Path):
        """External, anchor-only, and root-absolute destinations are not host paths."""
        skill_dir = tmp_path / "non-local-links"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "[root](/config/settings.yaml)\n[protocol](//example.com/missing.md)\n"
            "[web](https://example.com/missing.md)\n[email](mailto:user@example.com)\n"
            "[anchor](#overview)\n",
            encoding="utf-8",
        )

        result = HygieneValidator()._check_dead_links(skill_dir)

        assert result.errors == []

    def test_repeated_dead_targets_are_reported_once(self, tmp_path: Path):
        """Repeated links do not generate duplicate findings without line metadata."""
        skill_dir = tmp_path / "duplicate-links"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "[first](missing.md), [second](./missing.md#part), and [third][same].\n\n"
            "[same]: missing.md?view=1\n",
            encoding="utf-8",
        )

        result = HygieneValidator()._check_dead_links(skill_dir)

        assert result.errors == ["Dead link in SKILL.md: missing.md"]

    def test_valid_relative_links_pass(self, tmp_path: Path):
        """Test that valid relative links pass."""
        skill_dir = tmp_path / "valid-links-skill"
        skill_dir.mkdir()

        # Create the linked file
        docs_dir = skill_dir / "docs"
        docs_dir.mkdir()
        (docs_dir / "README.md").write_text("# Documentation")

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: valid-links-skill
description: A skill with valid links
version: 1.0.0
---

# Valid Links Test

See the [documentation](./docs/README.md) for more info.
""")

        validator = HygieneValidator()
        result = validator.validate(skill_dir)

        # Should not report dead links for existing files
        dead_link_errors = [e for e in result.errors if "dead link" in e.lower()]
        assert len(dead_link_errors) == 0

    def test_external_links_ignored(self, tmp_path: Path):
        """Test that external links are not checked."""
        skill_dir = tmp_path / "external-links-skill"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: external-links-skill
description: A skill with external links
version: 1.0.0
---

# External Links Test

Visit [NVIDIA](https://nvidia.com) for more info.
Check [GitHub](https://github.com/nvidia) for code.
Send [email](mailto:support@nvidia.com) for help.
""")

        validator = HygieneValidator()
        result = validator.validate(skill_dir)

        # Should not report errors for external links
        assert len(result.errors) == 0

    def test_detects_unpinned_dependencies(self, tmp_path: Path):
        """Test detection of unpinned dependencies."""
        skill_dir = tmp_path / "unpinned-deps-skill"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: unpinned-deps-skill
description: A skill with unpinned dependencies
version: 1.0.0
---

# Unpinned Dependencies Test
""")

        requirements = skill_dir / "requirements.txt"
        requirements.write_text("""
# Pinned
requests==2.28.0
pydantic>=2.0.0

# Unpinned (should warn)
numpy
pandas
""")

        validator = HygieneValidator()
        result = validator.validate(skill_dir)

        # Should warn about unpinned packages
        all_messages = result.errors + result.warnings
        assert any("unpinned" in m.lower() for m in all_messages)

    def test_detects_banned_packages(self, tmp_path: Path):
        """Test detection of banned/deprecated packages."""
        skill_dir = tmp_path / "banned-deps-skill"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: banned-deps-skill
description: A skill with banned dependencies
version: 1.0.0
---

# Banned Dependencies Test
""")

        requirements = skill_dir / "requirements.txt"
        requirements.write_text("""
pycrypto==2.6.1
subprocess32==3.5.4
""")

        validator = HygieneValidator()
        result = validator.validate(skill_dir)

        # Should error on banned packages
        assert any(
            "banned" in e.lower() or "pycrypto" in e.lower() or "subprocess32" in e.lower() for e in result.errors
        )

    def test_no_requirements_is_ok(self, sample_skill_dir: Path):
        """Test that missing requirements.txt is not an error."""
        validator = HygieneValidator()
        result = validator.validate(sample_skill_dir)

        # Should not error if no requirements.txt
        req_errors = [e for e in result.errors if "requirements" in e.lower()]
        assert len(req_errors) == 0
