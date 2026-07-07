# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for folder-level validation functionality."""

from skillevaluator.validators.hygiene import HygieneValidator
from skillevaluator.validators.schema import SchemaValidator
from skillevaluator.validators.security import SecurityValidator


class TestFolderLevelValidation:
    """Tests for validating folders containing multiple skills."""

    def test_schema_validator_multiple_skills(self, tmp_path):
        """Test SchemaValidator with folder containing multiple skills."""
        # Create a folder with multiple valid skills
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        # Skill 1
        skill1_dir = skills_dir / "skill-one"
        skill1_dir.mkdir()
        (skill1_dir / "SKILL.md").write_text("""---
name: skill-one
description: First test skill for folder validation
metadata:
  author: Test User <test@nvidia.com>
---

# Skill One

## Instructions

1. First step

## Examples

Example one.
""")

        # Skill 2
        skill2_dir = skills_dir / "skill-two"
        skill2_dir.mkdir()
        (skill2_dir / "SKILL.md").write_text("""---
name: skill-two
description: Second test skill for folder validation
metadata:
  author: Test User <test@nvidia.com>
---

# Skill Two

## Instructions

1. Second step

## Examples

Example two.
""")

        validator = SchemaValidator()
        result = validator.validate(skills_dir)

        assert result.passed
        assert "Found 2 skill(s) in folder" in result.messages[0]
        assert any("skill-one" in m for m in result.messages)
        assert any("skill-two" in m for m in result.messages)

    def test_schema_validator_mixed_valid_invalid(self, tmp_path):
        """Test SchemaValidator with folder containing both valid and invalid skills."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        # Valid skill
        skill1_dir = skills_dir / "valid-skill"
        skill1_dir.mkdir()
        (skill1_dir / "SKILL.md").write_text("""---
name: valid-skill
description: Valid skill
metadata:
  author: Test User <test@nvidia.com>
---

# Valid Skill

## Instructions

1. Do the thing

## Examples

Example.
""")

        # Invalid skill (name mismatch)
        skill2_dir = skills_dir / "invalid-skill"
        skill2_dir.mkdir()
        (skill2_dir / "SKILL.md").write_text("""---
name: wrong-name
description: Invalid skill with name mismatch
---

# Invalid Skill
""")

        validator = SchemaValidator()
        result = validator.validate(skills_dir)

        assert not result.passed
        assert any("valid-skill" in msg and "passed" in msg for msg in result.messages)
        assert any("[invalid-skill]" in error for error in result.errors)

    def test_security_validator_multiple_skills(self, tmp_path):
        """Test SecurityValidator with folder containing multiple skills."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        # Skill 1 - clean
        skill1_dir = skills_dir / "clean-skill"
        skill1_dir.mkdir()
        (skill1_dir / "SKILL.md").write_text("""---
name: clean-skill
description: Clean skill
---

# Clean
No PII here.
""")

        # Skill 2 - with PII
        skill2_dir = skills_dir / "pii-skill"
        skill2_dir.mkdir()
        (skill2_dir / "SKILL.md").write_text("""---
name: pii-skill
description: Skill with PII
---

# PII Skill
Contact: john.doe@personal.com
""")

        validator = SecurityValidator()
        result = validator.validate(skills_dir)

        assert "Scanning 2 skill(s)" in result.messages[0]
        # Check that we found PII in the second skill
        pii_errors = [e for e in result.errors + result.warnings if "pii-skill" in e.lower()]
        assert len(pii_errors) > 0

    def test_hygiene_validator_multiple_skills(self, tmp_path):
        """Test HygieneValidator with folder containing multiple skills."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        # Skill 1 - with valid link
        skill1_dir = skills_dir / "skill-with-link"
        skill1_dir.mkdir()
        ref_dir = skill1_dir / "references"
        ref_dir.mkdir()
        (ref_dir / "guide.md").write_text("# Guide")
        (skill1_dir / "SKILL.md").write_text("""---
name: skill-with-link
description: Skill with valid link
---

See [guide](references/guide.md).
""")

        # Skill 2 - with dead link
        skill2_dir = skills_dir / "skill-with-dead-link"
        skill2_dir.mkdir()
        (skill2_dir / "SKILL.md").write_text("""---
name: skill-with-dead-link
description: Skill with dead link
---

See [missing](missing-file.md).
""")

        validator = HygieneValidator()
        result = validator.validate(skills_dir)

        assert "Checking code integrity for 2 skill(s)" in result.messages[0]
        passing_skill = next(detail for detail in result.success_details if detail.check_name == "skill-with-link")
        test_discovery = next(check for check in passing_skill.metadata["checks"] if check["name"] == "test_discovery")
        assert test_discovery["metadata"]["execution_performed"] is False
        assert test_discovery["metadata"]["coverage_measured"] is False
        assert test_discovery["metadata"]["test_count"] == 0
        assert "coverage was not measured" in test_discovery["description"]
        # Check that dead link was detected
        dead_link_errors = [e for e in result.errors if "skill-with-dead-link" in e.lower()]
        assert len(dead_link_errors) > 0

    def test_empty_folder(self, tmp_path):
        """Test validation of empty folder."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        validator = SchemaValidator()
        result = validator.validate(empty_dir)

        assert not result.passed
        assert "No skills found" in result.errors[0]

    def test_folder_structure_compliance_with_skills_dir(self, tmp_path):
        """Test folder structure compliance for folder with skills/ directory."""
        root = tmp_path / "project"
        root.mkdir()

        # Create skills/ directory
        skills_dir = root / "skills"
        skills_dir.mkdir()

        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: test-skill
description: Test skill
---

# Test Skill

## Instructions

1. Test it

## Examples

Example.
""")

        validator = SchemaValidator()
        result = validator.validate(root)

        assert any("Folder structure compliant with Skill Evaluator guidelines" in m for m in result.messages)

    def test_folder_structure_compliance_with_team_skills(self, tmp_path):
        """Test folder structure compliance for folder with team-skills/ directory."""
        root = tmp_path / "project"
        root.mkdir()

        # Create team-skills/ directory
        team_skills_dir = root / "team-skills"
        team_skills_dir.mkdir()

        team_dir = team_skills_dir / "my-team"
        team_dir.mkdir()

        skill_dir = team_dir / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: test-skill
description: Test skill
---

# Test Skill

## Instructions

1. Test it

## Examples

Example.
""")

        validator = SchemaValidator()
        result = validator.validate(root)

        assert any("Folder structure compliant with Skill Evaluator guidelines" in m for m in result.messages)
        assert any("team-skills" in m for m in result.messages)


class TestOptionalFilesValidation:
    """Tests for optional files validation."""

    def test_optional_files_detected(self, tmp_path):
        """Test detection of optional supporting files."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()

        # Create SKILL.md
        (skill_dir / "SKILL.md").write_text("""---
name: test-skill
description: Test skill with optional files
metadata:
  author: Test User <test@nvidia.com>
---

# Test Skill

## Instructions

1. Use the skill

## Examples

Example.
""")

        # Create optional files
        (skill_dir / "README.md").write_text("# README")

        refs_dir = skill_dir / "references"
        refs_dir.mkdir()
        (refs_dir / "guide.md").write_text("# Guide")

        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "helper.py").write_text("# Helper script")

        assets_dir = skill_dir / "assets"
        assets_dir.mkdir()
        (assets_dir / "diagram.png").write_text("fake image")

        # `evals/` is the canonical location for Tier 3 evaluation datasets
        # and Harbor outputs; it must be recognised as expected, not flagged.
        evals_dir = skill_dir / "evals"
        evals_dir.mkdir()
        (evals_dir / "evals.json").write_text("[]")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        assert result.passed
        optional_msg = [m for m in result.messages if "optional" in m.lower() and "files" in m.lower()]
        assert len(optional_msg) == 1
        assert "README.md" in optional_msg[0]
        assert "references" in optional_msg[0]
        assert "scripts" in optional_msg[0]
        assert "assets" in optional_msg[0]
        assert "evals" in optional_msg[0]
        # And no "Unexpected 'evals'..." warning should be raised for it.
        unexpected = [w for w in result.warnings if "Unexpected" in w and "evals" in w]
        assert unexpected == []

    def test_evals_directory_not_flagged_as_unexpected(self, tmp_path):
        """A skill that only ships an evals/ subdir for Tier 3 must not warn."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: test-skill
description: Test skill that only adds an evals/ directory for Tier 3 datasets
metadata:
  author: Test User <test@nvidia.com>
---

# Test Skill

## Instructions

1. Use it

## Examples

Example.
""")
        evals_dir = skill_dir / "evals"
        evals_dir.mkdir()
        (evals_dir / "evals.json").write_text(
            '[{"id":"t-001","question":"Q?","expected_skill":"test-skill",'
            '"ground_truth":"a","expected_behavior":["Read SKILL.md"]}]'
        )
        (evals_dir / "EVAL.md").write_text("# Eval guidance")
        # Harbor results live under evals/results/<ts>/...; emulate one.
        (evals_dir / "results").mkdir()

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        assert result.passed
        unexpected = [w for w in result.warnings if "Unexpected" in w and "evals" in w]
        assert unexpected == [], f"evals/ should be a recognised optional dir; got warnings: {unexpected}"

    def test_generated_artifacts_not_flagged_as_unexpected(self, tmp_path):
        """Generated validation/signing artifacts are expected in published skill roots."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: test-skill
description: Test skill with generated publishing artifacts
metadata:
  author: Test User <test@nvidia.com>
---

# Test Skill

## Instructions

1. Use it

## Examples

Example.
""")
        (skill_dir / "skill-card.md").write_text("# Generated card")
        (skill_dir / "BENCHMARK.md").write_text("# Generated benchmark")
        (skill_dir / "skill.oms.sig").write_text("generated signature")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        assert result.passed
        unexpected = [
            w
            for w in result.warnings
            if "Unexpected" in w and any(name in w for name in ("skill-card.md", "BENCHMARK.md", "skill.oms.sig"))
        ]
        assert unexpected == []

    def test_plural_benchmarks_file_is_unexpected(self, tmp_path):
        """``benchmarks.md`` is author-owned content, not generated output."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: test-skill
description: Test skill with author benchmark notes
metadata:
  author: Test User <test@nvidia.com>
---

# Test Skill

## Instructions

1. Use it

## Examples

Example.
""")
        (skill_dir / "benchmarks.md").write_text("# Author benchmark notes")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        assert result.passed
        warnings = [w for w in result.warnings if "Unexpected" in w and "benchmarks.md" in w]
        assert warnings

    def test_no_optional_files(self, tmp_path):
        """Test skill with no optional files."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: test-skill
description: Test skill without optional files
metadata:
  author: Test User <test@nvidia.com>
---

# Test Skill

## Instructions

1. Use it

## Examples

Example.
""")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        assert result.passed
        # No "Optional files found" message should appear when no optional files exist
        assert not any("Optional files found" in m for m in result.messages)

    def test_unexpected_files_warning(self, tmp_path):
        """Test warning for unexpected files in skill root."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: test-skill
description: Test skill
metadata:
  author: Test User <test@nvidia.com>
---

# Test Skill

## Instructions

1. Use it

## Examples

Example.
""")

        # Create unexpected file
        (skill_dir / "random_file.txt").write_text("unexpected content")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        assert result.passed
        # Should have warning about unexpected file
        warnings = [w for w in result.warnings if "Unexpected" in w and "random_file.txt" in w]
        assert len(warnings) == 1


class TestSingleSkillValidation:
    """Tests to ensure single skill validation still works."""

    def test_single_skill_directory(self, tmp_path):
        """Test validation of a single skill directory."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: my-skill
description: Single skill test
metadata:
  author: Test User <test@nvidia.com>
---

# My Skill

## Instructions

1. Do it

## Examples

Example.
""")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        assert result.passed
        # Should NOT have "Found N skills" message for single skill
        assert not any("Found" in m and "skill(s) in folder" in m for m in result.messages)

    def test_single_skill_with_file_path(self, tmp_path):
        """Test validation with SKILL.md file path."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: my-skill
description: Single skill test
metadata:
  author: Test User <test@nvidia.com>
---

# My Skill

## Instructions

1. Do it

## Examples

Example.
""")

        # Note: This should be handled by resolve_skill_path in main.py
        # But validators should work with skill_dir directly
        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        assert result.passed


class TestBaseFindAllSkills:
    """Tests for _find_all_skills base method."""

    def test_find_all_skills_nested(self, tmp_path):
        """Test finding all skills in nested structure."""
        root = tmp_path / "project"
        root.mkdir()

        # Create nested skill structure with proper SKILL.md files
        skills = {
            "skills/skill-one": "skill-one",
            "skills/skill-two": "skill-two",
            "team-skills/team-a/skill-three": "skill-three",
            "team-skills/team-b/project-x/skill-four": "skill-four",
        }

        for skill_path, skill_name in skills.items():
            full_path = root / skill_path
            full_path.mkdir(parents=True)
            (full_path / "SKILL.md").write_text(f"""---
name: {skill_name}
description: Test skill
---

# {skill_name}

## Instructions

1. Use {skill_name}

## Examples

Example for {skill_name}.
""")

        # Use a concrete validator (SchemaValidator) to access the base method
        validator = SchemaValidator()
        found_skills = validator._find_all_skills(root)

        assert len(found_skills) == 4
        skill_names = [s.name for s in found_skills]
        assert "skill-one" in skill_names
        assert "skill-two" in skill_names
        assert "skill-three" in skill_names
        assert "skill-four" in skill_names
