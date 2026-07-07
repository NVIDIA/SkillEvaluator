# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Rules schema validation.

Based on Skill Evaluator HOW_TO_CONTRIBUTE_WORKFLOW_RULES.md specification.
"""

from pathlib import Path

import pytest

from skillevaluator.validators.rules_schema import RulesSchemaValidator, parse_rules_manifest


class TestRulesSchemaValidator:
    """Test suite for RulesSchemaValidator."""

    @pytest.fixture
    def validator(self) -> RulesSchemaValidator:
        """Create a RulesSchemaValidator instance."""
        return RulesSchemaValidator()

    @pytest.fixture
    def valid_rule_file(self, tmp_path: Path) -> Path:
        """Create a valid .mdc rule file per Skill Evaluator spec."""
        team_rules_dir = tmp_path / "team-rules" / "test-team"
        team_rules_dir.mkdir(parents=True)

        rule_file = team_rules_dir / "python-standards.mdc"
        rule_file.write_text("""---
alwaysApply: false
title: "Python Coding Standards"
description: "Best practices for Python development in test-team projects"
globs:
  - '*.py'
metadata:
  tags:
    - python
    - best-practices
    - test-team
  language: python
  team: test-team
  domain: backend
---

# Python Coding Standards

## Overview

This rule defines Python coding standards for the test-team.

## Guidelines

### Do This

- Use type hints
- Write docstrings

### Don't Do This

- Ignore linter warnings
- Skip tests
""")
        return rule_file

    @pytest.fixture
    def invalid_rule_missing_required(self, tmp_path: Path) -> Path:
        """Create a rule file missing required fields."""
        rule_file = tmp_path / "invalid-rule.mdc"
        rule_file.write_text("""---
title: "Missing Required Fields"
description: "This rule is missing alwaysApply"
---

# Invalid Rule
""")
        return rule_file

    @pytest.fixture
    def rule_with_invalid_yaml(self, tmp_path: Path) -> Path:
        """Create a rule file with invalid YAML."""
        rule_file = tmp_path / "bad-yaml.mdc"
        rule_file.write_text("""---
alwaysApply: false
title: "Bad YAML
description: unclosed quote
---

# Bad YAML
""")
        return rule_file

    def test_validate_valid_rule(self, validator: RulesSchemaValidator, valid_rule_file: Path):
        """Test validation passes for valid rule file."""
        result = validator.validate(valid_rule_file)
        assert result.passed, f"Expected validation to pass. Errors: {result.errors}"
        assert any("python-standards" in msg.lower() for msg in result.messages)

    def test_validate_missing_required_fields(
        self, validator: RulesSchemaValidator, invalid_rule_missing_required: Path
    ):
        """Test validation fails when required fields are missing."""
        result = validator.validate(invalid_rule_missing_required)
        assert not result.passed
        assert any("alwaysApply" in err for err in result.errors)

    def test_validate_invalid_yaml(self, validator: RulesSchemaValidator, rule_with_invalid_yaml: Path):
        """Test validation fails for invalid YAML."""
        result = validator.validate(rule_with_invalid_yaml)
        assert not result.passed
        assert any("yaml" in err.lower() for err in result.errors)

    def test_validate_folder_structure(self, validator: RulesSchemaValidator, valid_rule_file: Path):
        """Test folder structure validation for team-rules."""
        result = validator.validate(valid_rule_file)
        assert result.passed
        assert any("team-rules" in msg.lower() for msg in result.messages)

    def test_validate_rule_directory(self, validator: RulesSchemaValidator, tmp_path: Path):
        """Test validation of a directory containing multiple rules."""
        team_rules_dir = tmp_path / "team-rules" / "my-team"
        team_rules_dir.mkdir(parents=True)

        # Create multiple rule files
        for name in ["rule-one", "rule-two"]:
            rule_file = team_rules_dir / f"{name}.mdc"
            rule_file.write_text(f"""---
alwaysApply: false
title: "{name.replace("-", " ").title()}"
description: "Description for {name}"
---

# {name.replace("-", " ").title()}
""")

        result = validator.validate(team_rules_dir)
        assert result.passed
        assert any("2 rule file(s)" in msg for msg in result.messages)

    def test_validate_nonexistent_path(self, validator: RulesSchemaValidator, tmp_path: Path):
        """Test validation of nonexistent path."""
        result = validator.validate(tmp_path / "nonexistent.mdc")
        assert not result.passed
        # Error can be "not found" or "No .mdc rule files found"
        assert any("not found" in err.lower() or "no .mdc" in err.lower() for err in result.errors)

    def test_validate_wrong_extension(self, validator: RulesSchemaValidator, tmp_path: Path):
        """Test validation fails for non-.mdc file."""
        wrong_file = tmp_path / "rule.md"
        wrong_file.write_text("# Not a rule file")
        result = validator.validate(wrong_file)
        assert not result.passed
        assert any(".mdc" in err for err in result.errors)

    def test_validate_naming_convention(self, validator: RulesSchemaValidator, tmp_path: Path):
        """Test naming convention validation."""
        # Create rule with non-kebab-case name
        rule_file = tmp_path / "BadName_Rule.mdc"
        rule_file.write_text("""---
alwaysApply: false
title: "Bad Name Rule"
description: "Rule with non-kebab-case filename"
---

# Bad Name Rule
""")
        result = validator.validate(rule_file)
        # Should warn about naming convention but still parse
        assert any("kebab-case" in warn.lower() for warn in result.warnings)

    def test_validate_metadata_fields(self, validator: RulesSchemaValidator, tmp_path: Path):
        """Test that missing metadata generates warnings."""
        rule_file = tmp_path / "no-metadata.mdc"
        rule_file.write_text("""---
alwaysApply: false
title: "No Metadata Rule"
description: "Rule without metadata section"
---

# No Metadata Rule
""")
        result = validator.validate(rule_file)
        assert result.passed  # Should still pass
        assert any("metadata" in warn.lower() for warn in result.warnings)

    def test_parse_rules_manifest(self, valid_rule_file: Path):
        """Test parse_rules_manifest utility function."""
        manifest = parse_rules_manifest(valid_rule_file)
        assert manifest is not None
        assert manifest.frontmatter.title == "Python Coding Standards"
        assert manifest.frontmatter.alwaysApply is False
        assert manifest.has_content

    def test_parse_rules_manifest_invalid(self, tmp_path: Path):
        """Test parse_rules_manifest returns None for invalid file."""
        invalid_file = tmp_path / "invalid.mdc"
        invalid_file.write_text("# No frontmatter")
        manifest = parse_rules_manifest(invalid_file)
        assert manifest is None


class TestRulesRequiredFields:
    """Test that Rules require specific fields (opposite of Skills)."""

    @pytest.fixture
    def validator(self) -> RulesSchemaValidator:
        return RulesSchemaValidator()

    def test_always_apply_is_required(self, validator: RulesSchemaValidator, tmp_path: Path):
        """Test that alwaysApply field is required for Rules."""
        rule_file = tmp_path / "test.mdc"
        rule_file.write_text("""---
title: "Test Rule"
description: "Missing alwaysApply"
---

# Test Rule
""")
        result = validator.validate(rule_file)
        assert not result.passed
        assert any("alwaysApply" in err for err in result.errors)

    def test_title_is_required(self, validator: RulesSchemaValidator, tmp_path: Path):
        """Test that title field is required for Rules."""
        rule_file = tmp_path / "test.mdc"
        rule_file.write_text("""---
alwaysApply: false
description: "Missing title"
---

# Test Rule
""")
        result = validator.validate(rule_file)
        assert not result.passed
        assert any("title" in err for err in result.errors)

    def test_description_is_required(self, validator: RulesSchemaValidator, tmp_path: Path):
        """Test that description field is required for Rules."""
        rule_file = tmp_path / "test.mdc"
        rule_file.write_text("""---
alwaysApply: false
title: "Missing Description"
---

# Test Rule
""")
        result = validator.validate(rule_file)
        assert not result.passed
        assert any("description" in err for err in result.errors)

    def test_globs_is_optional(self, validator: RulesSchemaValidator, tmp_path: Path):
        """Test that globs field is optional for Rules."""
        rule_file = tmp_path / "test.mdc"
        rule_file.write_text("""---
alwaysApply: false
title: "No Globs"
description: "Rule without globs should still be valid"
---

# Test Rule
""")
        result = validator.validate(rule_file)
        assert result.passed

    def test_globs_accepted(self, validator: RulesSchemaValidator, tmp_path: Path):
        """Test that globs field is accepted for Rules (unlike Skills)."""
        rule_file = tmp_path / "test.mdc"
        rule_file.write_text("""---
alwaysApply: false
title: "With Globs"
description: "Rule with globs pattern"
globs:
  - '*.py'
  - 'src/**/*.ts'
---

# Test Rule
""")
        result = validator.validate(rule_file)
        assert result.passed


class TestRulesFolderStructure:
    """Tests for rules folder structure validation."""

    def test_validate_team_rules_folder(self, tmp_path: Path):
        """Test validation of team-rules folder structure."""
        validator = RulesSchemaValidator()

        team_rules_dir = tmp_path / "team-rules" / "my-team"
        team_rules_dir.mkdir(parents=True)

        rule_file = team_rules_dir / "python-standards.mdc"
        rule_file.write_text("""---
alwaysApply: false
title: "Python Standards"
description: "Python coding standards for my team"
metadata:
  team: my-team
  language: python
---

# Python Standards
""")

        result = validator.validate(tmp_path)
        assert result.passed
        assert any("team-rules" in msg.lower() for msg in result.messages)

    def test_validate_team_rules_project_subfolder(self, tmp_path: Path):
        """Test validation of team-rules with project subfolder."""
        validator = RulesSchemaValidator()

        project_dir = tmp_path / "team-rules" / "my-team" / "my-project"
        project_dir.mkdir(parents=True)

        rule_file = project_dir / "project-rules.mdc"
        rule_file.write_text("""---
alwaysApply: false
title: "Project Rules"
description: "Rules for my project"
metadata:
  team: my-team
  project: my-project
---

# Project Rules
""")

        result = validator.validate(tmp_path)
        assert result.passed

    def test_validate_multiple_rules_in_directory(self, tmp_path: Path):
        """Test validation of multiple rules in one directory."""
        validator = RulesSchemaValidator()

        rules_dir = tmp_path / "team-rules" / "my-team"
        rules_dir.mkdir(parents=True)

        for i in range(3):
            rule_file = rules_dir / f"rule-{i}.mdc"
            rule_file.write_text(f"""---
alwaysApply: false
title: "Rule {i}"
description: "Description for rule {i}"
---

# Rule {i}
""")

        result = validator.validate(tmp_path)
        assert result.passed
        # Should report found multiple rules
        assert any("3" in msg or "rule" in msg.lower() for msg in result.messages)


class TestRulesLineCount:
    """Tests for rules line count validation."""

    def test_validate_rule_within_line_limit(self, tmp_path: Path):
        """Test rule within line count limit passes."""
        validator = RulesSchemaValidator()
        rule_file = tmp_path / "short-rule.mdc"

        content = """---
alwaysApply: false
title: "Short Rule"
description: "A short rule"
---

# Short Rule

Some content.
"""
        rule_file.write_text(content)
        result = validator.validate(rule_file)
        assert result.passed

    def test_validate_rule_exceeds_line_limit_warning(self, tmp_path: Path):
        """Test rule exceeding line limit gets warning."""
        validator = RulesSchemaValidator()
        rule_file = tmp_path / "long-rule.mdc"

        # Create a rule with more than 500 lines
        lines = [
            "---",
            "alwaysApply: false",
            'title: "Long Rule"',
            'description: "A very long rule"',
            "---",
            "",
            "# Long Rule",
            "",
        ]
        lines.extend([f"Line {i}" for i in range(510)])

        rule_file.write_text("\n".join(lines))
        result = validator.validate(rule_file)
        # Should warn about line count
        assert any("line" in warn.lower() for warn in result.warnings)


class TestRulesNamingConventions:
    """Tests for rules naming convention validation."""

    def test_valid_kebab_case_filename(self, tmp_path: Path):
        """Test valid kebab-case filename passes."""
        validator = RulesSchemaValidator()
        rule_file = tmp_path / "my-awesome-rule.mdc"
        rule_file.write_text("""---
alwaysApply: false
title: "My Awesome Rule"
description: "Description"
---

# Content
""")
        result = validator.validate(rule_file)
        assert result.passed

    def test_invalid_filename_with_spaces(self, tmp_path: Path):
        """Test filename with spaces fails/warns."""
        validator = RulesSchemaValidator()
        rule_file = tmp_path / "my rule.mdc"
        rule_file.write_text("""---
alwaysApply: false
title: "My Rule"
description: "Description"
---

# Content
""")
        result = validator.validate(rule_file)
        # Should have warning about kebab-case naming
        has_naming_issue = any("kebab-case" in warn.lower() or "my rule" in warn.lower() for warn in result.warnings)
        assert has_naming_issue

    def test_filename_with_consecutive_hyphens(self, tmp_path: Path):
        """Test filename with consecutive hyphens warns."""
        validator = RulesSchemaValidator()
        rule_file = tmp_path / "my--bad--rule.mdc"
        rule_file.write_text("""---
alwaysApply: false
title: "Bad Rule"
description: "Description"
---

# Content
""")
        result = validator.validate(rule_file)
        # Should warn about consecutive hyphens
        assert any("consecutive" in warn.lower() or "hyphen" in warn.lower() for warn in result.warnings)
