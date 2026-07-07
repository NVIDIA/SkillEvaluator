# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for skillevaluator.validators.frontmatter_parser module."""

from pathlib import Path

from pydantic import BaseModel

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
