# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for schema validator.

Based on SkillEvaluator HOW_TO_CONTRIBUTE_SKILLS.md specification.
"""

from pathlib import Path

import pytest

from skillevaluator.models.result import Severity
from skillevaluator.models.skill import SkillMetadata
from skillevaluator.validators.schema import SchemaValidator


class TestSchemaValidator:
    """Test cases for SchemaValidator based on SkillEvaluator spec."""

    @pytest.mark.parametrize("key", ["123", "true", "false", "null", "2026-01-01"])
    @pytest.mark.parametrize("nested", [False, True])
    def test_non_string_key_returns_validation_error(self, sample_skill_dir, key, nested):
        manifest = sample_skill_dir / "SKILL.md"
        content = manifest.read_text()
        marker = "metadata:\n" if nested else "---\n"
        indentation = "  " if nested else ""
        manifest.write_text(content.replace(marker, f"{marker}{indentation}{key}: extra value\n", 1))

        result = SchemaValidator().validate(sample_skill_dir)

        assert not result.passed
        field = f"metadata.{key}" if nested else key
        findings = [finding for finding in result.findings if finding.check_name == "frontmatter_field"]
        assert any(finding.message == f"Field '{field}': Keys should be strings" for finding in findings)
        assert any(finding.metadata == {"field": field, "error_type": "invalid_key"} for finding in findings)

    @pytest.mark.parametrize("key", ["123", "true", "false", "null", "2026-01-01"])
    def test_quoted_extra_keys_remain_strings(self, sample_skill_dir, key):
        manifest = sample_skill_dir / "SKILL.md"
        content = manifest.read_text().replace("---\n", f'---\n"{key}": top-level\n', 1)
        manifest.write_text(content.replace("metadata:\n", f'metadata:\n  "{key}": nested\n', 1))

        result = SchemaValidator().validate(sample_skill_dir)

        assert result.passed, result.errors
        assert result.metadata["frontmatter"].model_extra[key] == "top-level"
        assert result.metadata["frontmatter"].metadata.model_extra[key] == "nested"

    def test_valid_skill(self, sample_skill_dir: Path):
        """Test validation passes for valid skill with proper SkillEvaluator structure."""
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
        """A name without an email is advisory attribution under the default profile."""
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

        assert result.passed, result.errors
        assert not [f for f in result.findings if f.check_name == "author_format"]
        author_findings = [f for f in result.findings if f.check_name == "author_without_email"]
        assert len(author_findings) == 1
        assert author_findings[0].severity == Severity.INFO
        assert author_findings[0].message == "Author has no email address: John Doe"
        assert author_findings[0].metadata.get("shape_ok") is False

    def test_metadata_author_missing_is_advisory(self, tmp_path: Path):
        """A missing metadata.author is a LOW advisory under the default profile."""
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

        assert result.passed, result.errors
        author_findings = [f for f in result.findings if f.check_name == "author_missing"]
        assert len(author_findings) == 1
        assert author_findings[0].severity == Severity.LOW
        assert "Author not specified" in author_findings[0].message

    def test_valid_metadata_structure(self, tmp_path: Path):
        """Test validation passes with complete valid metadata structure."""
        skill_dir = tmp_path / "complete-metadata"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: complete-metadata
description: A skill with complete metadata structure per SkillEvaluator spec
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
  domain:
    - testing
    - quality
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
        assert result.metadata["frontmatter"].metadata.domain == ["testing", "quality"]

    @pytest.mark.parametrize(
        ("metadata", "expected"),
        [
            pytest.param({"domain": "backend"}, ["backend"], id="scalar"),
            pytest.param({"domain": ["backend", "ai-ml"]}, ["backend", "ai-ml"], id="list"),
            pytest.param(
                {"domain": [" backend ", "ai-ml", "backend"]},
                ["backend", "ai-ml"],
                id="strip-and-stable-dedupe",
            ),
            pytest.param({}, [], id="missing"),
            pytest.param({"domain": None}, [], id="null"),
            pytest.param({"domain": []}, [], id="empty"),
            pytest.param({"domain": ""}, [], id="blank-scalar"),
            pytest.param({"domain": "   "}, [], id="whitespace-scalar"),
        ],
    )
    def test_metadata_domain_normalization(self, metadata: dict[str, object], expected: list[str]):
        assert SkillMetadata(**metadata).domain == expected

    @pytest.mark.parametrize(
        ("domain", "message"),
        [
            pytest.param(["backend", 1], "domain values must be strings", id="non-string"),
            pytest.param(["backend", "   "], "domain values must not be blank", id="blank-list-entry"),
            pytest.param(["backend"] * 17, "domain must contain at most 16 values", id="too-many-raw-items"),
            pytest.param("x" * 129, "domain values must be at most 128 characters", id="too-long"),
            pytest.param({"name": "backend"}, "domain must be a string or list of strings", id="mapping"),
        ],
    )
    def test_metadata_domain_rejects_invalid_values(self, domain: object, message: str):
        with pytest.raises(ValueError, match=message):
            SkillMetadata(domain=domain)

    @pytest.mark.parametrize(
        ("domain_yaml", "message"),
        [
            pytest.param(
                "  domain:\n    - backend\n    - 42\n",
                "domain values must be strings",
                id="non-string-list-entry",
            ),
            pytest.param(
                f"  domain: {'x' * 129}\n",
                "domain values must be at most 128 characters",
                id="over-long-single-string",
            ),
        ],
    )
    def test_invalid_metadata_domain_reports_field_finding(self, tmp_path: Path, domain_yaml: str, message: str):
        """An invalid domain, including an over-long single string, is a HIGH metadata.domain finding."""
        from skillevaluator.models.result import Severity

        skill_dir = tmp_path / "bad-domain"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: bad-domain\n"
            "description: A skill with an invalid domain\n"
            "metadata:\n"
            "  author: Test User <testuser@example.com>\n"
            f"{domain_yaml}"
            "---\n\n# Bad Domain\n",
            encoding="utf-8",
        )

        result = SchemaValidator().validate(skill_dir)

        assert not result.passed
        findings = [f for f in result.findings if f.check_name == "frontmatter_field"]
        assert len(findings) == 1
        assert findings[0].severity == Severity.HIGH
        assert findings[0].metadata["field"] == "metadata.domain"
        assert message in findings[0].message

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

    def test_reserved_word_substring_in_name_is_allowed(self, tmp_path: Path):
        """Reserved providers only match complete hyphen-delimited name segments."""
        skill_dir = tmp_path / "philanthropic-grant-matcher"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: philanthropic-grant-matcher\n"
            "description: Use when matching philanthropic grants to eligible programs.\n"
            "metadata:\n"
            "  author: Test User <test@nvidia.com>\n"
            "---\n\n"
            "# Philanthropic Grant Matcher\n\n"
            "## Instructions\n\n1. Match each grant to an eligible program.\n\n"
            "## Examples\n\nMatch the example grant.\n"
        )

        result = SchemaValidator().validate(skill_dir)

        assert result.passed, result.errors

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

    def test_closing_xml_tag_in_description_rejected(self, tmp_path: Path):
        """Opening and closing XML tags use the same schema and quality definition."""
        skill_dir = tmp_path / "closing-xml-desc"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: closing-xml-desc\n"
            'description: "Use when validating a closing </tool> tag."\n'
            "metadata:\n"
            "  author: Test User <test@nvidia.com>\n"
            "---\n\n"
            "# Closing XML Description\n\n"
            "## Instructions\n\n1. Run.\n\n"
            "## Examples\n\nExample.\n"
        )

        result = SchemaValidator().validate(skill_dir)

        assert not result.passed
        assert any("xml" in err.lower() for err in result.errors)

    def test_self_closing_xml_tag_in_description_rejected(self, tmp_path: Path):
        """Self-closing XML tags remain forbidden in frontmatter descriptions."""
        skill_dir = tmp_path / "self-closing-xml-desc"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: self-closing-xml-desc\n"
            'description: "Use when validating an injected <tool/> tag."\n'
            "metadata:\n"
            "  author: Test User <test@nvidia.com>\n"
            "---\n\n"
            "# Self-closing XML Description\n\n"
            "## Instructions\n\n1. Run.\n\n"
            "## Examples\n\nExample.\n"
        )

        result = SchemaValidator().validate(skill_dir)

        assert not result.passed
        assert any("xml" in err.lower() for err in result.errors)

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

    def test_utf8_bom_skill_md_passes_schema(self, tmp_path: Path):
        """A valid SKILL.md that only adds a UTF-8 BOM must still pass schema (#91)."""
        skill_dir = tmp_path / "bom-skill"
        skill_dir.mkdir()
        body = """---
name: bom-skill
description: Valid skill whose SKILL.md starts with a UTF-8 BOM
metadata:
  author: Bom User <bomuser@example.com>
---

# BOM Skill

## Instructions

1. Open the file in an editor that writes a BOM.

## Examples

Example usage.
"""
        (skill_dir / "SKILL.md").write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))

        result = SchemaValidator().validate(skill_dir)

        assert result.passed, f"BOM-only difference should still pass schema. Errors: {result.errors}"
        assert all(f.check_name != "frontmatter_format" for f in result.findings)

    def test_canonical_support_dirs_accepted(self, tmp_path: Path):
        """Canonical public skill support directories must not be flagged.

        Skills may carry agent metadata, local verification, executable helpers,
        and runtime configuration without producing an ``unexpected_file`` finding.
        """
        skill_dir = tmp_path / "spec-layout-skill"
        skill_dir.mkdir()
        (skill_dir / "tools").mkdir()
        (skill_dir / "tools" / "run.py").write_text("print('hi')\n")
        (skill_dir / "config").mkdir()
        (skill_dir / "config" / "settings.json").write_text('{"k": "v"}\n')
        (skill_dir / "agents").mkdir()
        (skill_dir / "agents" / "openai.yaml").write_text("name: spec-layout-skill\n")
        (skill_dir / "tests").mkdir()
        (skill_dir / "tests" / "test_skill.py").write_text("def test_skill():\n    assert True\n")

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
        assert not unexpected, f"Canonical support directories should be allowed, got: {unexpected}"

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

    def test_external_profile_reports_missing_author_as_advisory(self, tmp_path: Path):
        """Missing author is reported but does not block under the public profile."""
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
        assert author_findings[0].severity == Severity.LOW
        assert result.passed, result.errors

    def test_external_profile_rejects_malformed_author_shape(self, tmp_path: Path):
        """A bare email fails the shape check under the external profile."""
        from skillevaluator.models.result import Severity
        from skillevaluator.validators.policy import load_profile

        skill_dir = self._write_skill(
            tmp_path,
            name="external-malformed-author",
            author_line="john@example.com",
        )
        result = SchemaValidator(policy=load_profile("external")).validate(skill_dir)

        author_findings = [f for f in result.findings if f.check_name == "author_format"]
        assert len(author_findings) == 1, (
            f"Expected exactly one author_format finding for malformed shape, got {len(author_findings)}"
        )
        assert author_findings[0].severity == Severity.HIGH
        assert author_findings[0].metadata.get("shape_ok") is False
        assert not result.passed

    @pytest.mark.parametrize(
        "author",
        [
            "NVIDIA cuOpt Team",
            "Example Corporation and Affiliates",
            "Jane Doe (@jdoe)",
            "Équipe Données",
        ],
    )
    def test_default_profile_accepts_name_only_attribution(self, tmp_path: Path, author: str):
        """Person, team, and organization names without an email are advisory, not blocking."""
        from skillevaluator.models.result import Severity
        from skillevaluator.validators.policy import load_profile

        skill_dir = self._write_skill(tmp_path, name="team-author", author_line=f'"{author}"')

        result = SchemaValidator(policy=load_profile("default")).validate(skill_dir)

        assert result.passed, result.errors
        findings = [f for f in result.findings if f.check_name.startswith("author")]
        assert [(f.check_name, f.severity) for f in findings] == [("author_without_email", Severity.INFO)]

    @pytest.mark.parametrize(
        ("author", "reason"),
        [
            ("jane@example.com", "contains an email address"),
            ("Jane Doe jane@example.com", "contains an email address"),
            # An intranet host has no dot, but the value is still only an email.
            ("root@buildhost", "contains an email address"),
            ("jane@example", "contains an email address"),
            ("Jos\u00e9@example.com", "contains an email address"),
            ("Jane Doe <jane>", "angle brackets"),
            ("Jane Doe <jane@example.com> and team", "angle brackets"),
            ("Jane\u202eDoe", "control, private-use, or bidirectional formatting"),
            ("Jane\tDoe", "control, private-use, or bidirectional formatting"),
            ("Jane\u2066Doe", "control, private-use, or bidirectional formatting"),
            ("Jane\ue000Doe", "control, private-use, or bidirectional formatting"),
            ("---", "no letters or digits"),
            ("J" * 257, "longer than 256 characters"),
        ],
    )
    def test_malformed_or_unsafe_author_blocks_in_every_profile(self, tmp_path: Path, author: str, reason: str):
        """Malformed or unsafe values stay HIGH under both bundled profiles."""
        from skillevaluator.models.result import Severity
        from skillevaluator.validators.policy import load_profile

        skill_dir = self._write_skill(tmp_path, name="bad-author", author_line=f'"{author}"')

        for profile in ("default", "registry"):
            result = SchemaValidator(policy=load_profile(profile)).validate(skill_dir)
            findings = [f for f in result.findings if f.check_name == "author_format"]
            assert len(findings) == 1, (profile, [f.check_name for f in result.findings])
            assert findings[0].severity == Severity.HIGH
            assert reason in findings[0].message
            assert len(findings[0].line_content) <= len("author: ") + 256
            assert not result.passed

    def test_registry_profile_requires_contact_author(self, tmp_path: Path):
        """The registry profile blocks a missing author and a name without an email."""
        from skillevaluator.models.result import Severity
        from skillevaluator.validators.policy import load_profile

        policy = load_profile("registry")
        missing = self._write_skill(tmp_path, name="registry-missing", author_line=None)
        team = self._write_skill(tmp_path, name="registry-team", author_line="Example Team")

        missing_result = SchemaValidator(policy=policy).validate(missing)
        team_result = SchemaValidator(policy=policy).validate(team)

        assert not missing_result.passed
        assert not team_result.passed
        assert {(f.check_name, f.severity) for f in missing_result.findings if f.check_name.startswith("author")} == {
            ("author_missing", Severity.HIGH)
        }
        assert {(f.check_name, f.severity) for f in team_result.findings if f.check_name.startswith("author")} == {
            ("author_without_email", Severity.HIGH)
        }

    def test_domain_restriction_rejects_name_without_email(self, tmp_path: Path):
        """An author_email_regex needs an email in that domain, so a bare name fails it."""
        import re

        from skillevaluator.models.result import Severity
        from skillevaluator.validators.policy import ValidationPolicy, default_policy

        policy = ValidationPolicy(
            profile="org",
            author_email_regex=re.compile(r"@example\.org>$"),
            severity_overrides=dict(default_policy().severity_overrides),
        )
        team = self._write_skill(tmp_path, name="org-team", author_line="Example Team")
        member = self._write_skill(tmp_path, name="org-member", author_line="Jane Doe <jane@example.org>")

        team_result = SchemaValidator(policy=policy).validate(team)
        findings = [f for f in team_result.findings if f.check_name == "author_format"]
        assert len(findings) == 1
        assert findings[0].severity == Severity.HIGH
        assert "has no email address matching the required domain" in findings[0].message
        assert findings[0].metadata.get("shape_ok") is False
        assert SchemaValidator(policy=policy).validate(member).passed

    def test_domain_restriction_rejects_well_formed_author_outside_the_domain(self, tmp_path: Path):
        """The other half of the domain branch: a correct shape in the wrong domain still blocks.

        The domain check is the only blocking author rule an organization
        policy keeps, so a regression that accepted any domain has to fail a
        test rather than only the bare-name case above.
        """
        import re

        from skillevaluator.models.result import Severity
        from skillevaluator.validators.policy import ValidationPolicy, default_policy

        policy = ValidationPolicy(
            profile="org",
            author_email_regex=re.compile(r"@example\.org>$"),
            severity_overrides=dict(default_policy().severity_overrides),
        )
        outsider = self._write_skill(tmp_path, name="org-outsider", author_line="Jane Doe <jane@other.com>")

        result = SchemaValidator(policy=policy).validate(outsider)

        findings = [f for f in result.findings if f.check_name == "author_format"]
        assert len(findings) == 1
        assert findings[0].severity == Severity.HIGH
        assert "Author email does not match the required domain" in findings[0].message
        assert findings[0].metadata.get("shape_ok") is True
        assert findings[0].metadata.get("domain_ok") is False
        assert not result.passed

    @pytest.mark.parametrize(
        "author",
        [
            "Jane\u200cDoe",  # ZWNJ: required by Persian and Devanagari names
            "Jane\u200dDoe",  # ZWJ
            "Jane\u00adDoe",  # soft hyphen
            "Jane\u0530Doe",  # unassigned in Unicode 16, assigned in some later version
        ],
    )
    def test_joiners_and_unassigned_code_points_are_accepted_in_an_author(self, tmp_path: Path, author: str):
        """The character guard rejects spoofing characters, not every Unicode category C.

        ZWNJ/ZWJ are needed to render Persian, Arabic and Devanagari names, and
        an unassigned code point would make the verdict depend on the running
        interpreter's Unicode data rather than on the value.
        """
        from skillevaluator.validators.policy import load_profile

        skill_dir = self._write_skill(tmp_path, name="joiner-author", author_line=f'"{author} <jane@example.com>"')

        result = SchemaValidator(policy=load_profile("default")).validate(skill_dir)

        assert not [f for f in result.findings if f.check_name.startswith("author")], result.errors
        assert result.passed, result.errors

    @pytest.mark.parametrize(
        "author",
        ["Jane Doe (@jdoe)", "Team (@org-handle)", "@jdoe", "Platform Team @ Example"],
    )
    def test_social_handles_are_not_mistaken_for_a_bare_email(self, tmp_path: Path, author: str):
        """Widening the bare-email pattern must not turn a documented handle into a HIGH finding."""
        from skillevaluator.models.result import Severity
        from skillevaluator.validators.policy import load_profile

        skill_dir = self._write_skill(tmp_path, name="handle-author", author_line=f'"{author}"')

        result = SchemaValidator(policy=load_profile("default")).validate(skill_dir)

        assert result.passed, result.errors
        findings = [f for f in result.findings if f.check_name.startswith("author")]
        assert [(f.check_name, f.severity) for f in findings] == [("author_without_email", Severity.INFO)]

    def test_default_validator_uses_public_profile(self, tmp_path: Path):
        """Constructing SchemaValidator() without a policy accepts public domains."""

        skill_dir = self._write_skill(
            tmp_path,
            name="default-uses-public-profile",
            author_line="Jane Doe <jane@example.com>",
        )

        result = SchemaValidator().validate(skill_dir)

        assert result.passed


class TestMetadataIdContract:
    """Optional ``metadata.id``: validated when present, required only by policy."""

    @staticmethod
    def _write_skill(root: Path, name: str, id_line: str | None) -> Path:
        skill_dir = root / name
        skill_dir.mkdir(parents=True)
        id_block = f"  id: {id_line}\n" if id_line is not None else ""
        (skill_dir / "SKILL.md").write_text(
            f"""---
name: {name}
description: A skill used by the test suite to check the metadata.id contract
metadata:
  author: Jane Doe <jane@example.com>
{id_block}---

# {name}

## Instructions

1. Run.

## Examples

Example.
""",
            encoding="utf-8",
        )
        return skill_dir

    @staticmethod
    def _id_findings(result) -> list[tuple[str, str]]:
        return [(f.check_name, f.severity.value) for f in result.findings if f.check_name.startswith("metadata_id")]

    def test_valid_id_passes_and_does_not_claim_uniqueness(self, tmp_path: Path):
        from skillevaluator.validators.policy import load_profile

        skill_dir = self._write_skill(tmp_path, "id-ok", "id-ok-2")

        for profile in ("default", "registry"):
            result = SchemaValidator(policy=load_profile(profile)).validate(skill_dir)
            assert result.passed, result.errors
            assert self._id_findings(result) == []
            detail = next(d for d in result.success_details if d.check_name == "metadata_id_format")
            assert "Uniqueness is checked only across skills validated together" in detail.message
            assert result.metadata["metadata_id"] == "id-ok-2"

    def test_missing_id_follows_the_profile_requirement(self, tmp_path: Path):
        from skillevaluator.validators.policy import ValidationPolicy, load_profile

        skill_dir = self._write_skill(tmp_path, "no-id", None)

        default_result = SchemaValidator(policy=load_profile("default")).validate(skill_dir)
        recommend_result = SchemaValidator(policy=ValidationPolicy(metadata_id_requirement="recommend")).validate(
            skill_dir
        )
        registry_result = SchemaValidator(policy=load_profile("registry")).validate(skill_dir)

        assert default_result.passed
        assert self._id_findings(default_result) == []
        assert recommend_result.passed
        assert self._id_findings(recommend_result) == [("metadata_id_missing", "low")]
        assert not registry_result.passed
        assert self._id_findings(registry_result) == [("metadata_id_missing", "high")]
        assert "profile 'registry' requires" in registry_result.findings[-1].message

    @pytest.mark.parametrize(
        ("id_line", "reason"),
        [
            ('"My-Skill"', "lowercase ASCII"),
            ("my_skill", "lowercase ASCII"),
            ("-my-skill", "lowercase ASCII"),
            ("my--skill", "lowercase ASCII"),
            ('" my-skill"', "no surrounding whitespace"),
            ('"skill-é"', "lowercase ASCII"),
            ('"' + "a" * 65 + '"', "at most 64 characters (got 65)"),
            ('""', "must not be empty"),
            ("123", "must be a string (got int)"),
            ("true", "must be a string (got bool)"),
            ("null", "must be a string (got NoneType)"),
            ("[a, b]", "must be a string (got list)"),
        ],
    )
    def test_invalid_id_is_advisory_by_default_and_blocks_in_registry(self, tmp_path: Path, id_line: str, reason: str):
        from skillevaluator.validators.policy import load_profile

        skill_dir = self._write_skill(tmp_path, "bad-id", id_line)

        default_result = SchemaValidator(policy=load_profile("default")).validate(skill_dir)
        registry_result = SchemaValidator(policy=load_profile("registry")).validate(skill_dir)

        assert default_result.passed, default_result.errors
        assert self._id_findings(default_result) == [("metadata_id_format", "medium")]
        assert not registry_result.passed
        assert self._id_findings(registry_result) == [("metadata_id_format", "high")]
        finding = next(f for f in registry_result.findings if f.check_name == "metadata_id_format")
        assert reason in finding.message
        assert len(finding.line_content) <= len("id: ") + 81
        assert "metadata_id" not in registry_result.metadata

    def test_collection_reports_duplicate_ids(self, tmp_path: Path):
        from skillevaluator.validators.policy import load_profile

        skills = tmp_path / "skills"
        self._write_skill(skills, "first-skill", "shared-id")
        self._write_skill(skills, "second-skill", "shared-id")
        self._write_skill(skills, "third-skill", "third-skill")
        self._write_skill(skills, "fourth-skill", None)

        default_result = SchemaValidator(policy=load_profile("default")).validate(tmp_path)
        registry_result = SchemaValidator(policy=load_profile("registry")).validate(tmp_path)

        duplicates = [f for f in default_result.findings if f.check_name == "metadata_id_duplicate"]
        assert [(f.severity.value, f.metadata["skills"]) for f in duplicates] == [
            ("medium", ["skills/first-skill", "skills/second-skill"])
        ]
        assert "metadata.id 'shared-id' is declared by 2 skills" in duplicates[0].message
        assert ("metadata_id_duplicate", "high") in [(f.check_name, f.severity.value) for f in registry_result.findings]
        assert not registry_result.passed

    def test_collection_with_unique_ids_reports_scope(self, tmp_path: Path):
        skills = tmp_path / "skills"
        self._write_skill(skills, "first-skill", "first-skill")
        self._write_skill(skills, "second-skill", "second-skill")

        result = SchemaValidator().validate(tmp_path)

        assert result.passed, result.errors
        assert not [f for f in result.findings if f.check_name == "metadata_id_duplicate"]
        detail = next(d for d in result.success_details if d.check_name == "metadata_id_unique")
        assert detail.message == "2 declared metadata.id value(s) are unique within this collection"

    @pytest.mark.parametrize("include_failure", [False, True])
    def test_folder_api_preserves_child_advisories_once(self, tmp_path: Path, include_failure: bool):
        from skillevaluator import tier1

        skills = tmp_path / "skills"
        team = self._write_skill(skills, "team-skill", "INVALID_ID")
        manifest = team / "SKILL.md"
        manifest.write_text(manifest.read_text().replace("Jane Doe <jane@example.com>", "Example Team"))
        missing = self._write_skill(skills, "missing-author", "missing-author")
        manifest = missing / "SKILL.md"
        manifest.write_text(manifest.read_text().replace("  author: Jane Doe <jane@example.com>\n", ""))
        if include_failure:
            broken = self._write_skill(skills, "broken-author", "INVALID_ID")
            manifest = broken / "SKILL.md"
            manifest.write_text(manifest.read_text().replace("Jane Doe <jane@example.com>", "jane@example.com"))

        (result,) = tier1.run_validation(tmp_path, checks="schema")

        expected = [
            ("author_missing", "low"),
            ("author_without_email", "info"),
            ("metadata_id_format", "medium"),
        ]
        if include_failure:
            expected += [("author_format", "high"), ("metadata_id_format", "medium")]
        assert sorted((finding.check_name, finding.severity.value) for finding in result.findings) == sorted(expected)
        assert result.passed is not include_failure
        assert result.summary.warnings == len(result.warnings) == (4 if include_failure else 3)
        assert result.summary.errors == len(result.errors) == int(include_failure)
        assert result.summary.low_count == 1
        assert result.summary.medium_count == (2 if include_failure else 1)
        assert result.summary.high_count == int(include_failure)
        for name in ("team-skill", "missing-author"):
            summaries = [detail for detail in result.success_details if detail.check_name == name]
            assert len(summaries) == 1
            checks = summaries[0].metadata["checks"]
            assert len(checks) == summaries[0].metadata["total_checks"]
            assert len({(check["name"], check["description"]) for check in checks}) == len(checks)
            assert not any(detail.check_name.startswith(f"[{name}]") for detail in result.success_details)

    def test_folder_api_preserves_recommended_id_advisory(self, tmp_path: Path):
        from skillevaluator import tier1
        from skillevaluator.validators.policy import ValidationPolicy

        self._write_skill(tmp_path / "skills", "no-id", None)

        (result,) = tier1.run_validation(
            tmp_path, checks="schema", policy=ValidationPolicy(metadata_id_requirement="recommend")
        )

        assert result.passed
        assert self._id_findings(result) == [("metadata_id_missing", "low")]
        assert result.summary.warnings == len(result.warnings) == 1
        assert result.summary.low_count == 1


class TestCollectionAdvisoryIdentity:
    """A preserved advisory must say which skill in the collection produced it.

    Collection validation keeps the advisories of the skills that passed, but
    a consumer reading the collection result also has to be able to attribute
    each one, which a finding carried over unchanged does not allow.
    """

    @staticmethod
    def _write_skill(root: Path, name: str, *, author_line: str, id_line: str | None) -> Path:
        skill_dir = root / name
        skill_dir.mkdir(parents=True)
        id_block = f"  id: {id_line}\n" if id_line is not None else ""
        (skill_dir / "SKILL.md").write_text(
            f"""---
name: {name}
description: A skill used by the test suite to check collection-level advisory handling
metadata:
  author: {author_line}
{id_block}---

# {name}

## Instructions

1. Run.

## Examples

Example.
""",
            encoding="utf-8",
        )
        return skill_dir

    def test_carried_advisory_names_its_skill(self, tmp_path: Path):
        from skillevaluator.models.result import Severity
        from skillevaluator.validators.policy import load_profile

        skills = tmp_path / "skills"
        self._write_skill(skills, "advisory-skill", author_line="Example Team", id_line="INVALID_ID")

        skill_result = SchemaValidator(policy=load_profile("default")).validate(skills / "advisory-skill")
        collection = SchemaValidator(policy=load_profile("default")).validate(tmp_path)

        assert skill_result.passed, skill_result.errors
        carried = {f.check_name: f for f in collection.findings if f.metadata.get("collection_skill")}
        assert carried.keys() >= {"author_without_email", "metadata_id_format"}
        assert carried["author_without_email"].severity == Severity.INFO
        assert carried["metadata_id_format"].severity == Severity.MEDIUM
        for finding in carried.values():
            assert finding.metadata["collection_skill"] == "advisory-skill"
            assert finding.file_path.startswith("[advisory-skill] ")
        assert any("[advisory-skill] " in warning for warning in collection.warnings)
        # An advisory from a passing child stays advisory: it cannot fail the collection.
        assert collection.passed, collection.errors
        assert not collection.errors
        assert not [f for f in collection.findings if f.severity.is_error()]
        # The roll-up must not claim everything passed while advisories exist.
        rollup = next(d for d in collection.success_details if d.check_name == "advisory-skill")
        assert rollup.metadata["advisory_findings"] == len(skill_result.findings)
        assert "advisory finding(s)" in rollup.message

    def test_collection_of_clean_skills_carries_no_advisories(self, tmp_path: Path):
        """Negative case: nothing is invented for a skill with no findings of its own."""
        skills = tmp_path / "skills"
        self._write_skill(skills, "clean-skill", author_line="Jane Doe <jane@example.com>", id_line="clean-skill")

        collection = SchemaValidator().validate(tmp_path)

        assert collection.passed, collection.errors
        assert not [f for f in collection.findings if f.metadata.get("collection_skill")]
        rollup = next(d for d in collection.success_details if d.check_name == "clean-skill")
        assert rollup.message == "All schema and structure checks passed"
        assert rollup.metadata["advisory_findings"] == 0


class TestMetadataIdPreviewBounds:
    """A rejected ``metadata.id`` must never be rendered by expanding its value."""

    @staticmethod
    def _alias_bomb(levels: int) -> str:
        lines = ["a0: &a0 [x, x]"]
        lines += [f"a{i}: &a{i} [*a{i - 1}, *a{i - 1}]" for i in range(1, levels + 1)]
        lines.append(f"id: *a{levels}")
        return "\n".join(lines)

    def test_non_string_id_is_described_by_type_not_rendered(self):
        from skillevaluator.validators.schema import _display_metadata_id

        assert _display_metadata_id([1, 2, 3]) == "<list>"
        assert _display_metadata_id(123) == "<int>"
        assert _display_metadata_id(None) == "<NoneType>"
        assert _display_metadata_id(True) == "<bool>"

    def test_yaml_alias_graph_does_not_expand_into_the_finding(self):
        import yaml

        from skillevaluator.validators.schema import _display_metadata_id

        # 30 doubling aliases: repr() of this value is ~2**30 elements, which
        # would exhaust memory. The preview must not build it.
        value = yaml.safe_load(self._alias_bomb(30))["id"]

        preview = _display_metadata_id(value)

        assert preview == "<list>"
        assert len(preview) <= 16

    def test_alias_graph_id_is_rejected_with_a_bounded_finding(self, tmp_path: Path):
        skill_dir = tmp_path / "alias-id"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: alias-id\ndescription: A skill whose metadata.id is a YAML alias graph\nmetadata:\n"
            + "\n".join(f"  {line}" for line in self._alias_bomb(20).splitlines())
            + "\n---\n\n# Alias\n\n## Instructions\n\n1. Run.\n",
            encoding="utf-8",
        )

        result = SchemaValidator().validate(skill_dir)

        findings = [f for f in result.findings if f.check_name == "metadata_id_format"]
        assert len(findings) == 1
        assert "must be a string (got list)" in findings[0].message
        assert len(findings[0].line_content) <= 32

    @pytest.mark.parametrize("id_value", ["&recursive [*recursive]", "&recursive {self: *recursive}"])
    def test_recursive_id_is_rejected_with_bounded_evidence(self, tmp_path: Path, id_value: str):
        skill = TestMetadataIdContract._write_skill(tmp_path, "recursive-id", id_value)

        result = SchemaValidator().validate(skill)

        assert result.passed
        (finding,) = [finding for finding in result.findings if finding.check_name == "metadata_id_format"]
        assert "must be a string" in finding.message
        assert len(finding.line_content) <= 32


def _write_skill(
    root: Path, frontmatter: str, body: str = "# Portable\n\n## Instructions\n\n1. Step\n\n## Examples\n\nExample.\n"
) -> Path:
    skill_dir = root / "skills" / "portable-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\n{frontmatter}---\n\n{body}", encoding="utf-8")
    return skill_dir


_BASE_FM = (
    "name: portable-skill\n"
    "description: Use when checking harness portability of a skill.\n"
    "metadata:\n  author: Test User <testuser@example.com>\n"
)


def _checks(result, name: str) -> list:
    return [finding for finding in result.findings if finding.check_name == name]


class TestSchemaPortabilityChecks:
    """Spec-valid patterns that agent harnesses load differently (Tier 1, harness-neutral)."""

    def test_clean_skill_has_no_portability_findings(self, tmp_path: Path):
        result = SchemaValidator().validate(_write_skill(tmp_path, _BASE_FM))

        assert result.passed, result.errors
        names = {
            "duplicate_key",
            "manifest_bom",
            "manifest_symlink",
            "manifest_size",
            "nested_manifest",
            "description_utf16_length",
            "description_trigger",
            "unknown_field",
        }
        assert not [f for f in result.findings if f.check_name in names]

    def test_duplicate_top_level_key_is_high(self, tmp_path: Path):
        result = SchemaValidator().validate(_write_skill(tmp_path, _BASE_FM + "description: Second value.\n"))

        findings = _checks(result, "duplicate_key")
        assert not result.passed
        assert [f.metadata["key"] for f in findings] == ["description"]
        assert findings[0].severity == "high"
        assert findings[0].line_number == 6

    def test_duplicate_nested_key_is_reported_with_path(self, tmp_path: Path):
        fm = _BASE_FM + "  author: Other User <other@example.com>\n"
        result = SchemaValidator().validate(_write_skill(tmp_path, fm))

        assert [f.metadata["key"] for f in _checks(result, "duplicate_key")] == ["metadata.author"]

    def test_same_key_in_different_mappings_is_not_duplicate(self, tmp_path: Path):
        fm = _BASE_FM + "  tags:\n    - name: a\n    - name: b\n"
        result = SchemaValidator().validate(_write_skill(tmp_path, fm))

        assert not _checks(result, "duplicate_key")

    def test_bom_is_medium_and_does_not_fail(self, tmp_path: Path):
        skill_dir = _write_skill(tmp_path, _BASE_FM)
        manifest = skill_dir / "SKILL.md"
        manifest.write_bytes(b"\xef\xbb\xbf" + manifest.read_bytes())

        result = SchemaValidator().validate(skill_dir)

        assert result.passed, result.errors
        assert [f.severity for f in _checks(result, "manifest_bom")] == ["medium"]

    def test_portability_severity_can_be_overridden_by_policy(self, tmp_path: Path):
        from skillevaluator.validators.policy import ValidationPolicy

        skill_dir = _write_skill(tmp_path, _BASE_FM)
        manifest = skill_dir / "SKILL.md"
        manifest.write_bytes(b"\xef\xbb\xbf" + manifest.read_bytes())
        policy = ValidationPolicy(severity_overrides={"SCHEMA.manifest_bom": Severity.HIGH})
        result = SchemaValidator(policy=policy).validate(skill_dir)

        assert not result.passed
        assert [f.severity for f in _checks(result, "manifest_bom")] == ["high"]
