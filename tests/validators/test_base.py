# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the refactored ValidatorBase class."""

from pathlib import Path

from skillevaluator.validators.base import ValidationResult, ValidatorBase, iter_scannable_files


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_default_state(self):
        """Test default ValidationResult is passing."""
        result = ValidationResult()
        assert result.passed is True
        assert result.errors == []
        assert result.warnings == []
        assert result.messages == []

    def test_add_error_fails_result(self):
        """Test that adding an error marks result as failed."""
        result = ValidationResult()
        result.add_error("Test error")

        assert result.passed is False
        assert "Test error" in result.errors

    def test_add_warning_does_not_fail(self):
        """Test that warnings don't fail the result."""
        result = ValidationResult()
        result.add_warning("Test warning")

        assert result.passed is True
        assert "Test warning" in result.warnings

    def test_add_message(self):
        """Test adding informational messages."""
        result = ValidationResult()
        result.add_message("Info message")

        assert result.passed is True
        assert "Info message" in result.messages

    def test_merge_results(self):
        """Test merging two ValidationResults."""
        result1 = ValidationResult()
        result1.add_message("Message 1")
        result1.add_warning("Warning 1")

        result2 = ValidationResult()
        result2.add_error("Error from result2")
        result2.add_message("Message 2")

        result1.merge(result2)

        assert result1.passed is False
        assert "Message 1" in result1.messages
        assert "Message 2" in result1.messages
        assert "Warning 1" in result1.warnings
        assert "Error from result2" in result1.errors

    def test_merge_with_prefix(self):
        """Test merge_with_prefix prefixes errors/warnings."""
        result1 = ValidationResult()

        result2 = ValidationResult()
        result2.add_error("Some error")
        result2.add_warning("Some warning")

        result1.merge_with_prefix(result2, "skill-name")

        assert "[skill-name] Some error" in result1.errors
        assert "[skill-name] Some warning" in result1.warnings
        assert result1.passed is False


class ConcreteValidator(ValidatorBase):
    """Concrete implementation of ValidatorBase for testing."""

    @property
    def name(self) -> str:
        return "Test Validator"

    @property
    def description(self) -> str:
        return "A test validator"

    def validate(self, skill_path):
        return self._validate_folder_or_skill(
            skill_path,
            self._validate_single,
            action_description="Testing",
        )

    def _validate_single(self, skill_path):
        result = ValidationResult()
        if self._find_skill_manifest(skill_path):
            result.add_message(f"Found skill at {skill_path.name}")
        else:
            result.add_error(f"No SKILL.md in {skill_path.name}")
        return result


class TestValidatorBase:
    """Tests for ValidatorBase functionality."""

    def test_find_skill_manifest_uppercase(self, tmp_path):
        """Test finding SKILL.md (uppercase)."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test\n---")

        validator = ConcreteValidator()
        manifest = validator._find_skill_manifest(skill_dir)

        assert manifest is not None
        assert manifest.name == "SKILL.md"

    def test_find_skill_manifest_lowercase(self, tmp_path):
        """Test finding skill.md (lowercase)."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text("---\nname: test\n---")

        validator = ConcreteValidator()
        manifest = validator._find_skill_manifest(skill_dir)

        assert manifest is not None
        # macOS APFS is case-insensitive: _find_skill_manifest iterates
        # SKILL_MANIFEST_VARIANTS in order, so "SKILL.md" matches first
        assert manifest.name.lower() == "skill.md"

    def test_find_skill_manifest_not_found(self, tmp_path):
        """Test when no SKILL.md exists."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()

        validator = ConcreteValidator()
        manifest = validator._find_skill_manifest(skill_dir)

        assert manifest is None

    def test_is_skill_directory_true(self, tmp_path):
        """Test _is_skill_directory returns True when SKILL.md exists."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("test")

        validator = ConcreteValidator()
        assert validator._is_skill_directory(skill_dir) is True

    def test_is_skill_directory_false(self, tmp_path):
        """Test _is_skill_directory returns False when no SKILL.md."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()

        validator = ConcreteValidator()
        assert validator._is_skill_directory(skill_dir) is False

    def test_find_all_skills(self, tmp_path):
        """Test finding multiple skills in a directory tree."""
        root = tmp_path / "project"
        root.mkdir()

        # Create multiple skills
        for name in ["skill-a", "skill-b", "skill-c"]:
            skill_dir = root / name
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\n---")

        validator = ConcreteValidator()
        skills = validator._find_all_skills(root)

        assert len(skills) == 3
        names = [s.name for s in skills]
        assert "skill-a" in names
        assert "skill-b" in names
        assert "skill-c" in names

    def test_find_all_skills_nested(self, tmp_path):
        """Test finding skills in nested directory structure."""
        root = tmp_path / "project"
        root.mkdir()

        # Create nested skills
        (root / "skills").mkdir()
        (root / "skills" / "skill-one").mkdir(parents=True)
        (root / "skills" / "skill-one" / "SKILL.md").write_text("---\nname: skill-one\n---")

        (root / "team-skills" / "team-a" / "skill-two").mkdir(parents=True)
        (root / "team-skills" / "team-a" / "skill-two" / "SKILL.md").write_text("---\nname: skill-two\n---")

        validator = ConcreteValidator()
        skills = validator._find_all_skills(root)

        assert len(skills) == 2

    def test_find_all_skills_skips_evals_artifacts(self, tmp_path):
        """Eval artifacts under ``evals/`` must not be re-validated as skills.

        Tier 3 evaluation runs drop their per-trial output (including, in
        some agents, full skill snapshots) under ``evals/results/<run-id>/``.
        Without this exclusion, ``_find_all_skills`` would re-discover the
        snapshot copy of every skill and re-run the entire Tier 1 pipeline
        on the LLM transcripts, surfacing PII / credit-card / phone-number
        false positives that the user already filed feedback on.
        """
        root = tmp_path / "project"
        root.mkdir()
        live = root / "skill-a"
        live.mkdir()
        (live / "SKILL.md").write_text("---\nname: skill-a\n---")

        # Synthetic evals/results/.../trials/.../SKILL.md snapshot.
        snap = live / "evals" / "results" / "20260520_113343" / "trial-001" / "skill-a"
        snap.mkdir(parents=True)
        (snap / "SKILL.md").write_text("---\nname: skill-a\nversion: snapshot\n---")

        validator = ConcreteValidator()
        skills = validator._find_all_skills(root)

        assert [s.name for s in skills] == ["skill-a"]
        assert all("evals" not in s.parts for s in skills)
        assert all("results" not in s.parts for s in skills)

    def test_find_all_skills_skips_versions_snapshots(self, tmp_path):
        """Historical snapshots under ``.versions/`` must not be re-validated.

        The resource versioning proposal (SkillEvaluator) lets contributors keep
        immutable per-commit snapshots of a skill alongside the live
        ``SKILL.md`` (e.g. ``my-skill/.versions/v1.0.0/SKILL.md``).
        ``_find_all_skills`` must treat anything under a ``.versions``
        directory as historical and exclude it, so pre-submit checks only
        run against the live skill the contributor is editing — otherwise
        every snapshot would re-run validation, surface duplicate findings,
        and (worst case) fail CI for snapshots authored on older
        rules. Regression test for the ``.versions`` filter in
        ``skillevaluator/validators/base.py``.
        """
        root = tmp_path / "project"
        root.mkdir()

        live = root / "skill-a"
        live.mkdir()
        (live / "SKILL.md").write_text("---\nname: skill-a\n---")

        # Direct snapshot: my-skill/.versions/<commit>/SKILL.md
        flat_snapshot = live / ".versions" / "abc123"
        flat_snapshot.mkdir(parents=True)
        (flat_snapshot / "SKILL.md").write_text("---\nname: skill-a\nversion: 0.9.0\n---")

        # Deeper snapshot in case the layout nests multiple historical commits.
        nested_snapshot = root / "team-skills" / "team-a" / "skill-b"
        nested_snapshot.mkdir(parents=True)
        (nested_snapshot / "SKILL.md").write_text("---\nname: skill-b\n---")

        nested_history = nested_snapshot / ".versions" / "v1.0.0" / "deep"
        nested_history.mkdir(parents=True)
        (nested_history / "SKILL.md").write_text("---\nname: skill-b\nversion: 1.0.0\n---")

        validator = ConcreteValidator()
        skills = validator._find_all_skills(root)

        names = sorted(s.name for s in skills)
        assert names == ["skill-a", "skill-b"], f"_find_all_skills must skip .versions snapshots, got {names}"
        assert all(".versions" not in skill.parts for skill in skills)

    def test_validate_folder_or_skill_single(self, tmp_path):
        """Test template method with single skill."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test\n---")

        validator = ConcreteValidator()
        result = validator.validate(skill_dir)

        assert result.passed
        assert "Found skill at test-skill" in result.messages

    def test_validate_folder_or_skill_multiple(self, tmp_path):
        """Test template method with folder containing multiple skills."""
        root = tmp_path / "skills"
        root.mkdir()

        for name in ["skill-a", "skill-b"]:
            skill_dir = root / name
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\n---")

        validator = ConcreteValidator()
        result = validator.validate(root)

        assert result.passed
        assert "Testing 2 skill(s)" in result.messages[0]
        assert any("skill-a" in m for m in result.messages)
        assert any("skill-b" in m for m in result.messages)

    def test_validate_folder_or_skill_no_fallback(self, tmp_path):
        """Test template method without fallback for empty folders."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        def validate_single(path):
            result = ValidationResult()
            result.add_message("Validated single")
            return result

        validator = ConcreteValidator()
        result = validator._validate_folder_or_skill(
            empty_dir,
            validate_single,
            action_description="Testing",
            no_skills_fallback=False,  # Don't fall back to single validation
        )

        assert not result.passed
        assert "No skills found" in result.errors[0]


class TestIterScannableFiles:
    """Tests for the shared ``iter_scannable_files`` Tier 1 file walker.

    The Tier 1 PII / unicode / hygiene / code-risk / license validators all
    delegate file collection to this helper so they consistently skip
    evaluation artifacts (``evals/``, ``results/``, ``versions/`` and
    dot-prefixed variants). These tests pin the contract.
    """

    def test_returns_files_with_matching_extensions(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("x")
        (tmp_path / "b.py").write_text("x")
        (tmp_path / "c.bin").write_text("x")

        out = iter_scannable_files(tmp_path, {".md", ".py"})
        names = sorted(p.name for p in out)
        assert names == ["a.md", "b.py"]

    def test_skips_evals_results_versions_dirs(self, tmp_path: Path):
        (tmp_path / "live.md").write_text("x")
        for excluded in ("evals", "results", "versions", ".evals", ".results", ".versions"):
            sub = tmp_path / excluded / "deep"
            sub.mkdir(parents=True)
            (sub / "snap.md").write_text("x")

        out = iter_scannable_files(tmp_path, {".md"})
        rel_paths = sorted(p.relative_to(tmp_path).as_posix() for p in out)
        assert rel_paths == ["live.md"]

    def test_skips_generated_skill_artifacts_by_default(self, tmp_path: Path):
        (tmp_path / "SKILL.md").write_text("x")
        (tmp_path / "notes.md").write_text("x")
        (tmp_path / "skill-card.md").write_text("generated card")
        (tmp_path / "BENCHMARK.md").write_text("generated benchmark")
        (tmp_path / "benchmarks.md").write_text("author-owned benchmark notes")
        (tmp_path / "skill.oms.sig").write_text("generated signature")

        out = iter_scannable_files(tmp_path, {".md", ".sig"})
        rel_paths = sorted(p.relative_to(tmp_path).as_posix() for p in out)
        assert rel_paths == ["SKILL.md", "benchmarks.md", "notes.md"]

    def test_single_generated_artifact_file_returns_empty(self, tmp_path: Path):
        f = tmp_path / "skill-card.md"
        f.write_text("generated")
        assert iter_scannable_files(f, {".md"}) == []

    def test_skips_excluded_dirs_at_any_depth(self, tmp_path: Path):
        """Re-occurrence of an excluded name at depth must also be skipped."""
        nested = tmp_path / "references" / "evals" / "deep"
        nested.mkdir(parents=True)
        (nested / "stale.md").write_text("x")
        (tmp_path / "live.md").write_text("x")

        out = iter_scannable_files(tmp_path, {".md"})
        rel_paths = sorted(p.relative_to(tmp_path).as_posix() for p in out)
        assert rel_paths == ["live.md"]

    def test_custom_exclusion_overrides_default(self, tmp_path: Path):
        """Callers can opt out of the default filter for diagnostic dumps."""
        evals = tmp_path / "evals"
        evals.mkdir()
        (evals / "snap.md").write_text("x")

        out = iter_scannable_files(tmp_path, {".md"}, excluded_dirs=())
        rel_paths = sorted(p.relative_to(tmp_path).as_posix() for p in out)
        assert rel_paths == ["evals/snap.md"]

    def test_custom_exclusion_extends_filter(self, tmp_path: Path):
        """Callers can pass a different excluded-set when scanning non-skill trees."""
        cache = tmp_path / "build_cache"
        cache.mkdir()
        (cache / "stale.md").write_text("x")
        (tmp_path / "live.md").write_text("x")

        out = iter_scannable_files(tmp_path, {".md"}, excluded_dirs={"build_cache"})
        rel_paths = sorted(str(p.relative_to(tmp_path)) for p in out)
        assert rel_paths == ["live.md"]

    def test_single_file_returns_self_when_extension_matches(self, tmp_path: Path):
        f = tmp_path / "x.md"
        f.write_text("x")
        assert iter_scannable_files(f, {".md"}) == [f]

    def test_single_file_returns_empty_when_extension_mismatch(self, tmp_path: Path):
        f = tmp_path / "x.bin"
        f.write_text("x")
        assert iter_scannable_files(f, {".md"}) == []
