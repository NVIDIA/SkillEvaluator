# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for skillevaluator.validators.frontmatter_parser module."""

from pathlib import Path

import pytest
from pydantic import BaseModel
from tests.yaml_alias_fixtures import YAML_DIRECTIVE_VALUE_ERROR, alias_bomb_skill_md

from skillevaluator.validators.base import ValidationResult
from skillevaluator.validators.frontmatter_parser import parse_frontmatter, validate_pydantic_model


class TestParseFrontmatter:
    """Tests for parse_frontmatter utility."""

    def test_valid_frontmatter(self, tmp_path: Path):
        """Test parsing valid frontmatter."""
        test_file = tmp_path / "test.mdc"
        test_file.write_text("""---
title: Test
description: A test file
---

# Content here
""")
        parsed, result = parse_frontmatter(test_file)
        assert parsed is not None
        assert result.passed
        assert parsed.yaml_data["title"] == "Test"
        assert parsed.content.strip() == "# Content here"

    @pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
    def test_first_body_line_indentation_is_preserved(self, tmp_path: Path, newline: str):
        test_file = tmp_path / "indented-body.mdc"
        content = newline.join(["---", "title: Test", "---", "    [code](false.md)", ""])
        test_file.write_bytes(content.encode())

        parsed, result = parse_frontmatter(test_file)

        assert parsed is not None
        assert result.passed
        assert parsed.content == "    [code](false.md)\n"

    def test_utf8_bom_is_accepted(self, tmp_path: Path):
        """A valid frontmatter file that starts with a UTF-8 BOM still parses (#91)."""
        test_file = tmp_path / "bom.md"
        body = """---
title: Bom
description: A file whose only extra is a leading UTF-8 BOM
---

# Content
"""
        test_file.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))

        parsed, result = parse_frontmatter(test_file)

        assert parsed is not None
        assert result.passed
        assert parsed.yaml_data["title"] == "Bom"
        assert parsed.content.strip() == "# Content"

    def test_missing_frontmatter(self, tmp_path: Path):
        """Test file without frontmatter markers."""
        test_file = tmp_path / "test.mdc"
        test_file.write_text("# Just content, no frontmatter")

        parsed, result = parse_frontmatter(test_file)
        assert parsed is None
        assert not result.passed
        assert any("frontmatter" in err.lower() for err in result.errors)

    def test_invalid_yaml(self, tmp_path: Path):
        """Test file with invalid YAML in frontmatter."""
        test_file = tmp_path / "test.mdc"
        test_file.write_text("""---
title: [invalid yaml
  missing bracket
---

# Content
""")
        parsed, result = parse_frontmatter(test_file)
        assert parsed is None
        assert not result.passed
        assert any("yaml" in err.lower() for err in result.errors)

    def test_empty_frontmatter(self, tmp_path: Path):
        """Test file with empty frontmatter."""
        test_file = tmp_path / "test.mdc"
        test_file.write_text("""---
---

# Content
""")
        parsed, result = parse_frontmatter(test_file)
        assert parsed is None
        assert not result.passed

    def test_nonexistent_file(self, tmp_path: Path):
        """Test parsing nonexistent file."""
        test_file = tmp_path / "nonexistent.mdc"
        parsed, result = parse_frontmatter(test_file)
        assert parsed is None
        assert not result.passed


class TestParseFrontmatterBounds:
    """Frontmatter is read with the bounded YAML loader."""

    def test_nested_alias_graph_is_rejected_not_returned(self, tmp_path: Path):
        manifest = tmp_path / "SKILL.md"
        manifest.write_text(alias_bomb_skill_md(), encoding="utf-8")

        parsed, result = parse_frontmatter(manifest)

        assert parsed is None
        assert not result.passed
        assert any("complexity limits" in error for error in result.errors)

    def test_invalid_yaml_keeps_the_pyyaml_reason(self, tmp_path: Path):
        manifest = tmp_path / "SKILL.md"
        manifest.write_text("---\nname: [unclosed\n---\n# Body\n", encoding="utf-8")

        parsed, result = parse_frontmatter(manifest)

        assert parsed is None
        assert any(error.startswith("Invalid YAML in frontmatter: ") and "line" in error for error in result.errors)

    @pytest.mark.usefixtures("default_int_digit_limit")
    def test_a_yaml_directive_number_past_the_digit_limit_is_an_error_not_a_crash(self, tmp_path: Path):
        manifest = tmp_path / "SKILL.md"
        manifest.write_text(f"---\n{YAML_DIRECTIVE_VALUE_ERROR}\n---\n# Body\n", encoding="utf-8")

        parsed, result = parse_frontmatter(manifest)

        assert parsed is None
        assert any(error.startswith("Invalid YAML in frontmatter: ") for error in result.errors)

    def test_a_repeated_key_keeps_its_last_value(self, tmp_path: Path):
        manifest = tmp_path / "SKILL.md"
        manifest.write_text("---\nname: first\nname: second\n---\n# Body\n", encoding="utf-8")

        parsed, result = parse_frontmatter(manifest)

        assert result.passed
        assert parsed.yaml_data == {"name": "second"}


class TestValidatePydanticModel:
    """Tests for validate_pydantic_model utility."""

    def test_valid_data(self):
        """Test validation with valid data."""

        class TestModel(BaseModel):
            name: str
            count: int

        result = ValidationResult()
        model = validate_pydantic_model(TestModel, {"name": "test", "count": 5}, result)
        assert model is not None
        assert result.passed
        assert model.name == "test"

    def test_invalid_data(self):
        """Test validation with invalid data."""

        class TestModel(BaseModel):
            name: str
            count: int

        result = ValidationResult()
        model = validate_pydantic_model(TestModel, {"name": "test"}, result)  # Missing count
        assert model is None
        assert not result.passed
        assert any("count" in err.lower() for err in result.errors)


def test_nested_alias_frontmatter_is_an_error_not_a_parse(tmp_path: Path) -> None:

    test_file = tmp_path / "SKILL.md"
    test_file.write_text(alias_bomb_skill_md(), encoding="utf-8")

    parsed, result = parse_frontmatter(test_file)

    assert parsed is None
    assert any("complexity limits" in error for error in result.errors)
