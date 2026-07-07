# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for schema validator.

Based on Skill Evaluator HOW_TO_CONTRIBUTE_SKILLS.md specification.
"""

from pathlib import Path

from skillevaluator.validators.schema import SchemaValidator


class TestSchemaValidator:
    """Test cases for SchemaValidator based on Skill Evaluator spec."""

    def test_valid_skill(self, sample_skill_dir: Path):
        """Test validation passes for valid skill with proper Skill Evaluator structure."""
        validator = SchemaValidator()
        result = validator.validate(sample_skill_dir)

        assert result.passed, f"Expected valid skill to pass. Errors: {result.errors}"
        assert len(result.errors) == 0

    def test_valid_team_skill(self, team_skill_dir: Path):
        """Test validation passes for valid team-specific skill."""
        validator = SchemaValidator()
        result = validator.validate(team_skill_dir)

        assert result.passed, f"Expected team skill to pass. Errors: {result.errors}"

    def test_invalid_skill_name(self, invalid_skill_dir: Path):
        """Test validation fails for invalid skill name (not kebab-case)."""
        validator = SchemaValidator()
        result = validator.validate(invalid_skill_dir)

        assert not result.passed
        # Should have error about name format
        assert any("kebab-case" in err.lower() or "name" in err.lower() for err in result.errors)

    def test_missing_skill_md(self, tmp_path: Path):
        """Test validation fails when SKILL.md is missing."""
        empty_dir = tmp_path / "empty-skill"
        empty_dir.mkdir()

        validator = SchemaValidator()
        result = validator.validate(empty_dir)

        assert not result.passed
        assert any("SKILL.md" in err or "not found" in err or "No skills found" in err for err in result.errors)

    def test_missing_required_fields(self, tmp_path: Path):
        """Test validation fails for missing required fields (description)."""
        skill_dir = tmp_path / "incomplete-skill"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: incomplete-skill
---

# Incomplete
""")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        assert not result.passed
        # Should fail for missing description
        assert any("description" in err.lower() for err in result.errors)

    def test_forbidden_fields_rejected(self, skill_with_forbidden_fields: Path):
        """Test validation fails when forbidden fields (alwaysApply, globs) are present."""
        validator = SchemaValidator()
        result = validator.validate(skill_with_forbidden_fields)

        assert not result.passed
        assert any("forbidden" in err.lower() or "alwaysapply" in err.lower() for err in result.errors)

    def test_name_must_match_directory(self, skill_with_name_mismatch: Path):
        """Test validation fails when frontmatter name doesn't match directory name."""
        validator = SchemaValidator()
        result = validator.validate(skill_with_name_mismatch)

        assert not result.passed
        assert any("match" in err.lower() or "directory" in err.lower() for err in result.errors)

    def test_skill_md_case_insensitive(self, tmp_path: Path):
        """Test that skill.md (lowercase) is found but produces a naming warning."""
        skill_dir = tmp_path / "lowercase-skill"
        skill_dir.mkdir()

        skill_md = skill_dir / "skill.md"  # lowercase
        skill_md.write_text("""---
name: lowercase-skill
description: A skill with lowercase skill.md for testing case insensitivity
---

# Lowercase Skill

## Instructions

1. Use this skill

## Examples

Example usage here.
""")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        naming_findings = [f for f in result.findings if f.check_name == "manifest_naming"]
        assert len(naming_findings) == 1, "Expected a HIGH finding for lowercase skill.md naming"
        assert "SKILL.md" in naming_findings[0].message
        assert "agentskills.io" in naming_findings[0].message

    def test_skill_md_uppercase_no_naming_warning(self, tmp_path: Path):
        """Test that SKILL.md (uppercase) does NOT produce a naming warning."""
        skill_dir = tmp_path / "uppercase-skill"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: uppercase-skill
description: A skill with proper SKILL.md naming
---

# Uppercase Skill

## Instructions

1. Use this skill

## Examples

Example usage here.
""")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        naming_findings = [f for f in result.findings if f.check_name == "manifest_naming"]
        assert len(naming_findings) == 0, "SKILL.md (uppercase) should not produce a manifest_naming finding"

    def test_name_length_constraints(self, tmp_path: Path):
        """Test validation enforces name length constraints (1-64 chars)."""
        skill_dir = tmp_path / ("a" * 65 + "-skill")  # 65+ chars
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(f"""---
name: {"a" * 65}-skill
description: A skill with name exceeding 64 character limit
---

# Long Name Skill
""")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        assert not result.passed
        assert any("64" in err or "character" in err.lower() or "length" in err.lower() for err in result.errors)

    def test_consecutive_hyphens_rejected(self, tmp_path: Path):
        """Test validation fails for names with consecutive hyphens."""
        skill_dir = tmp_path / "bad--name"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: bad--name
description: A skill with consecutive hyphens in name
---

# Bad Name Skill
""")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        assert not result.passed
        assert any("consecutive" in err.lower() or "hyphen" in err.lower() for err in result.errors)

    def test_description_length_constraints(self, tmp_path: Path):
        """Test validation enforces description length constraints (1-1024 chars)."""
        skill_dir = tmp_path / "long-description"
        skill_dir.mkdir()

        # Create description over 1024 chars
        long_desc = "A" * 1030

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(f"""---
name: long-description
description: {long_desc}
---

# Long Description Skill
""")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        assert not result.passed
        assert any("1024" in err or "description" in err.lower() for err in result.errors)

    def test_metadata_author_validation(self, tmp_path: Path):
        """Test author format fails for malformed (no email) author under default profile."""
        from skillevaluator.models.result import Severity

        skill_dir = tmp_path / "no-email-author"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: no-email-author
description: A skill with malformed author metadata
metadata:
  author: John Doe
---

# Author Test Skill

## Instructions

1. Test the author field

## Examples

Example usage.
""")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        assert not result.passed
        author_findings = [f for f in result.findings if f.check_name == "author_format"]
        assert len(author_findings) == 1
        # The default public profile keeps malformed author metadata at HIGH severity.
        assert author_findings[0].severity == Severity.HIGH
        # Message changed to be precise about *which* aspect failed (shape vs. domain).
        # "John Doe" fails the shape check (no Name <email> form).
        assert "Name <email" in author_findings[0].message
        assert author_findings[0].metadata.get("shape_ok") is False

    def test_metadata_author_required(self, tmp_path: Path):
        """Test missing metadata.author is a HIGH severity failure under default profile."""
        from skillevaluator.models.result import Severity

        skill_dir = tmp_path / "missing-author"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: missing-author
description: A skill with no metadata author for testing
---

# Missing Author Skill

## Instructions

1. Test that the author field is required

## Examples

Example usage.
""")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        assert not result.passed
        author_findings = [f for f in result.findings if f.check_name == "author_missing"]
        assert len(author_findings) == 1
        assert author_findings[0].severity == Severity.HIGH
        assert "Author not specified" in author_findings[0].message

    def test_valid_metadata_structure(self, tmp_path: Path):
        """Test validation passes with complete valid metadata structure."""
        skill_dir = tmp_path / "complete-metadata"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: complete-metadata
description: A skill with complete metadata structure per Skill Evaluator spec
license: MIT
compatibility: Requires Python 3.11+
metadata:
  author: Test User <testuser@example.com>
  tags:
    - testing
    - metadata
  languages:
    - python
  frameworks:
    - pytest
  domain: testing
---

# Complete Metadata Skill

## Instructions

1. Check metadata fields
2. Verify author format

## Examples

```yaml
metadata:
  author: Test User <testuser@example.com>
```
""")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        assert result.passed, f"Expected complete metadata skill to pass. Errors: {result.errors}"

    # =========================================================================
    # agentskills.io spec + validate_skill_md.py CI checks
    # =========================================================================

    def test_reserved_word_in_name_rejected(self, tmp_path: Path):
        """Test validation fails when name contains reserved words ('anthropic', 'claude')."""
        for word in ("anthropic", "claude"):
            skill_dir = tmp_path / f"my-{word}-skill"
            skill_dir.mkdir(exist_ok=True)

            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text(f"""---
name: my-{word}-skill
description: A skill with a reserved word in its name
---

# Reserved Word Skill

## Instructions

1. Should not pass

## Examples

Example.
""")

            validator = SchemaValidator()
            result = validator.validate(skill_dir)

            assert not result.passed, f"Name containing '{word}' should be rejected"
            assert any("reserved" in err.lower() for err in result.errors)

    def test_xml_tags_in_name_rejected(self, tmp_path: Path):
        """Test validation fails when name contains XML tags."""
        skill_dir = tmp_path / "xml-name"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: "<script>alert</script>"
description: A skill with XML tags in name
---

# XML Name Skill

## Instructions

1. Should fail

## Examples

Example.
""")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        assert not result.passed

    def test_xml_tags_in_description_rejected(self, tmp_path: Path):
        """Test validation fails when description contains XML tags."""
        skill_dir = tmp_path / "xml-desc"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: xml-desc
description: "A skill <script>alert('xss')</script> with injected tags"
---

# XML Description Skill

## Instructions

1. Should fail

## Examples

Example.
""")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        assert not result.passed
        assert any("xml" in err.lower() or "description" in err.lower() for err in result.errors)

    def test_unclosed_xml_tag_in_description_rejected(self, tmp_path: Path):
        """Test validation fails when description contains an unclosed tag-like value."""
        skill_dir = tmp_path / "unclosed-xml-desc"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: unclosed-xml-desc
description: "A skill with an unclosed <script foo tag in the description"
---

# Unclosed XML Description Skill

## Instructions

1. Should fail

## Examples

Example.
""")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        assert not result.passed
        assert any("xml" in err.lower() or "description" in err.lower() for err in result.errors)

    def test_comparison_operators_in_description_allowed(self, tmp_path: Path):
        """Test threshold comparators in descriptions are not treated as XML tags."""
        skill_dir = tmp_path / "threshold-monitor"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: threshold-monitor
description: "Use this skill when free disk space < 10GB and CPU usage > 90% to trigger cleanup"
metadata:
  author: Test User <test@example.com>
---

# Threshold Monitor

## Instructions

1. Check threshold values.

## Examples

Use this skill when cleanup thresholds are exceeded.
""")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        assert result.passed, f"Expected comparator description to pass: {result.errors}"

    def test_missing_top_level_heading_rejected(self, tmp_path: Path):
        """Test validation fails when body has no top-level heading (# Title)."""
        skill_dir = tmp_path / "no-heading"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: no-heading
description: A skill missing top-level heading in body
---

This body has no heading at all.

## Instructions

1. Do things

## Examples

Example.
""")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        assert not result.passed
        assert any("heading" in err.lower() for err in result.errors)

    def test_missing_examples_section_is_medium(self, tmp_path: Path):
        """Missing ## Examples produces a MEDIUM finding (recommended, not required)."""
        from skillevaluator.models.result import Severity

        skill_dir = tmp_path / "no-examples"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: no-examples
description: A skill missing the Examples section
---

# No Examples

## Instructions

1. Do something
""")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        rec_findings = [f for f in result.findings if f.check_name == "body_recommended_section"]
        assert len(rec_findings) == 1
        assert rec_findings[0].severity == Severity.MEDIUM
        assert "Examples" in rec_findings[0].message

    def test_usage_section_accepted_as_instructions_alternative(self, tmp_path: Path):
        """## Usage satisfies the recommended ## Instructions section nudge."""
        skill_dir = tmp_path / "usage-skill"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: usage-skill
description: A skill that uses Usage instead of Instructions
---

# Usage Skill

## Usage

1. Run the tool
2. Check results

## Examples

```bash
run-tool --help
```
""")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        instructions_findings = [
            f for f in result.findings if f.check_name == "body_recommended_section" and "Instructions" in f.message
        ]
        assert len(instructions_findings) == 0, (
            f"## Usage should satisfy the Instructions nudge. Findings: {instructions_findings}"
        )

    def test_allowed_tools_field_accepted(self, tmp_path: Path):
        """Test that the optional 'allowed-tools' field is accepted per agentskills.io spec."""
        skill_dir = tmp_path / "with-tools"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: with-tools
description: A skill that declares allowed tools per agentskills.io spec
allowed-tools: Bash(git:*) Bash(jq:*) Read
metadata:
  author: Tool User <tooluser@example.com>
---

# Skill With Allowed Tools

## Instructions

1. Use git commands
2. Use jq for JSON processing

## Examples

```bash
git status
jq '.name' package.json
```
""")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        assert result.passed, f"Skill with allowed-tools should pass. Errors: {result.errors}"

    def test_all_required_sections_present_passes(self, tmp_path: Path):
        """Test validation passes when all body requirements are met."""
        skill_dir = tmp_path / "complete-body"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: complete-body
description: A skill with all required body sections
metadata:
  author: Body User <bodyuser@example.com>
---

# Complete Body Skill

## Instructions

1. Follow step one
2. Follow step two

## Examples

```python
print("hello")
```
""")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        assert result.passed, f"Skill with complete body should pass. Errors: {result.errors}"

    def test_tools_and_config_dirs_accepted(self, tmp_path: Path):
        """``tools/`` and ``config/`` skill-root dirs must not be flagged.

        The skill_evaluator onboarding renames ``scripts/`` -> ``tools/`` (agentskills.io
        convention) and skills may carry a ``config/`` dir for data-driven runtime
        config. Neither should produce an ``unexpected_file`` finding.
        """
        skill_dir = tmp_path / "spec-layout-skill"
        skill_dir.mkdir()
        (skill_dir / "tools").mkdir()
        (skill_dir / "tools" / "run.py").write_text("print('hi')\n")
        (skill_dir / "config").mkdir()
        (skill_dir / "config" / "settings.json").write_text('{"k": "v"}\n')

        (skill_dir / "SKILL.md").write_text("""---
name: spec-layout-skill
description: A skill using the agentskills.io tools/ layout plus a config/ dir.
metadata:
  author: Spec User <specuser@example.com>
---

# Spec Layout Skill

## Instructions

1. Use the tools/ directory for executables.

## Examples

Example usage.
""")

        result = SchemaValidator().validate(skill_dir)

        unexpected = [f.message for f in result.findings if f.check_name == "unexpected_file"]
        assert not unexpected, f"tools/ and config/ should be allowed, got: {unexpected}"

    def test_allowed_dirs_extended_via_env(self, tmp_path: Path, monkeypatch):
        """``SKILLEVALUATOR_SCHEMA_ALLOWED_DIRS`` extends the allowed skill-root dirs.

        A consumer with a non-canonical directory (e.g. ``data/``) should be able
        to clear the otherwise unavoidable LOW ``unexpected_file`` finding per-repo
        without forking the validator.
        """
        skill_dir = tmp_path / "extra-dirs-skill"
        skill_dir.mkdir()
        (skill_dir / "data").mkdir()
        (skill_dir / "data" / "table.json").write_text('{"k": "v"}\n')
        (skill_dir / "SKILL.md").write_text("""---
name: extra-dirs-skill
description: A skill that ships a data/ directory extended via the schema env knob.
metadata:
  author: Spec User <specuser@example.com>
---

# Extra Dirs Skill

## Instructions

1. Load runtime tables from data/.

## Examples

Example usage.
""")

        def _unexpected(result):
            return [f.message for f in result.findings if f.check_name == "unexpected_file"]

        monkeypatch.delenv("SKILLEVALUATOR_SCHEMA_ALLOWED_DIRS", raising=False)
        flagged = _unexpected(SchemaValidator().validate(skill_dir))
        assert any("data" in m for m in flagged), "data/ should be flagged without the env knob"

        monkeypatch.setenv("SKILLEVALUATOR_SCHEMA_ALLOWED_DIRS", "data, fixtures")
        cleared = _unexpected(SchemaValidator().validate(skill_dir))
        assert not cleared, f"data/ should be allowed via env knob, got: {cleared}"

    def test_missing_instructions_section_is_medium(self, tmp_path: Path):
        """Missing ## Instructions produces a MEDIUM finding (recommended, not required).

        Per agentskills.io spec the body has no format restrictions, so skills
        produced by tools like Codex's skill-creator (which don't emit a
        ## Instructions heading) should not be hard-failed.
        """
        from skillevaluator.models.result import Severity

        skill_dir = tmp_path / "no-instructions"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: no-instructions
description: A skill missing the Instructions section
metadata:
  author: Body User <bodyuser@example.com>
---

# No Instructions

## Examples

Example usage.
""")

        validator = SchemaValidator()
        result = validator.validate(skill_dir)

        rec_findings = [
            f for f in result.findings if f.check_name == "body_recommended_section" and "Instructions" in f.message
        ]
        assert len(rec_findings) == 1
        assert rec_findings[0].severity == Severity.MEDIUM

        body_required_findings = [f for f in result.findings if f.check_name == "body_required_section"]
        assert body_required_findings == [], (
            f"## Instructions must not produce a body_required_section finding anymore. Got: {body_required_findings}"
        )


class TestSchemaValidatorProfiles:
    """Profile-aware author validation for the external public policy."""

    @staticmethod
    def _write_skill(
        tmp_path: Path,
        *,
        name: str,
        author_line: str | None,
    ) -> Path:
        """Write a minimal SKILL.md with the given author (or no metadata block)."""
        skill_dir = tmp_path / name
        skill_dir.mkdir()
        metadata_block = ""
        if author_line is not None:
            metadata_block = f"metadata:\n  author: {author_line}\n"
        (skill_dir / "SKILL.md").write_text(
            f"""---
name: {name}
description: A profile-aware skill used by the test suite for testing severity overrides
{metadata_block}---

# {name}

## Instructions

1. Run.

## Examples

Example.
"""
        )
        return skill_dir

    def test_external_profile_accepts_well_formed_author(self, tmp_path: Path):
        """A well-formed author is accepted under the external profile."""
        from skillevaluator.validators.policy import load_profile

        skill_dir = self._write_skill(
            tmp_path,
            name="external-author-ok",
            author_line="Jane Doe <jane@example.com>",
        )

        result = SchemaValidator(policy=load_profile("external")).validate(skill_dir)

        assert result.passed, f"External profile should accept a well-formed author. Errors: {result.errors}"
        assert not [f for f in result.findings if f.check_name == "author_format"]

    def test_external_profile_requires_author(self, tmp_path: Path):
        """Missing author remains a blocking schema error under the public profile."""
        from skillevaluator.models.result import Severity
        from skillevaluator.validators.policy import load_profile

        skill_dir = self._write_skill(
            tmp_path,
            name="external-no-author",
            author_line=None,
        )

        result = SchemaValidator(policy=load_profile("external")).validate(skill_dir)

        author_findings = [f for f in result.findings if f.check_name == "author_missing"]
        assert len(author_findings) == 1
        assert author_findings[0].severity == Severity.HIGH
        assert not result.passed

    def test_external_profile_rejects_malformed_author_shape(self, tmp_path: Path):
        """Bare 'John Doe' fails the shape check under the external profile."""
        from skillevaluator.validators.policy import load_profile

        skill_dir = self._write_skill(
            tmp_path,
            name="external-malformed-author",
            author_line="John Doe",
        )
        result = SchemaValidator(policy=load_profile("external")).validate(skill_dir)

        author_findings = [f for f in result.findings if f.check_name == "author_format"]
        assert len(author_findings) == 1, (
            f"Expected exactly one author_format finding for malformed shape, got {len(author_findings)}"
        )
        assert author_findings[0].metadata.get("shape_ok") is False

    def test_default_validator_uses_public_profile(self, tmp_path: Path):
        """Constructing SchemaValidator() without a policy accepts public domains."""

        skill_dir = self._write_skill(
            tmp_path,
            name="default-uses-public-profile",
            author_line="Jane Doe <jane@example.com>",
        )

        result = SchemaValidator().validate(skill_dir)

        assert result.passed
