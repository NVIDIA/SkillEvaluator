# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for skillevaluator.validators.naming_utils module."""

from skillevaluator.validators.naming_utils import (
    NamingValidationConfig,
    validate_kebab_case_name,
    validate_line_count,
)


class TestValidateKebabCaseName:
    """Tests for validate_kebab_case_name utility."""

    def test_valid_name(self):
        """Test valid kebab-case name passes."""
        config = NamingValidationConfig(name_type="Test name")
        result = validate_kebab_case_name("my-valid-name", config)
        assert result.passed
        assert any("naming convention" in msg.lower() for msg in result.messages)

    def test_invalid_uppercase(self):
        """Test uppercase letters are rejected."""
        config = NamingValidationConfig(name_type="Test name")
        result = validate_kebab_case_name("MyName", config)
        assert not result.passed
        assert any("kebab-case" in err.lower() for err in result.errors)

    def test_consecutive_hyphens(self):
        """Test consecutive hyphens are rejected by kebab-case pattern."""
        config = NamingValidationConfig(name_type="Test name")
        result = validate_kebab_case_name("my--name", config)
        assert not result.passed
        # Caught by pattern match, not explicit check
        assert any("kebab-case" in err.lower() for err in result.errors)

    def test_trailing_hyphen(self):
        """Test trailing hyphen is rejected by kebab-case pattern."""
        config = NamingValidationConfig(name_type="Test name")
        result = validate_kebab_case_name("my-name-", config)
        assert not result.passed
        # Caught by pattern match, not explicit check
        assert any("kebab-case" in err.lower() for err in result.errors)

    def test_exceeds_max_length(self):
        """Test name exceeding max length is rejected."""
        config = NamingValidationConfig(name_type="Test name", max_length=10)
        result = validate_kebab_case_name("very-long-name-here", config)
        assert not result.passed
        assert any("limit" in err.lower() for err in result.errors)

    def test_use_warnings_instead_of_errors(self):
        """Test using warnings for non-strict validation."""
        config = NamingValidationConfig(name_type="Test name", use_errors=False)
        result = validate_kebab_case_name("InvalidName", config)
        assert result.passed  # Still passes when using warnings
        assert any("kebab-case" in warn.lower() for warn in result.warnings)


class TestValidateLineCount:
    """Tests for validate_line_count utility."""

    def test_within_limit(self):
        """Test line count within limit passes."""
        result = validate_line_count(100, 500, "Test file")
        assert result.passed
        assert any("100" in msg for msg in result.messages)

    def test_exceeds_limit(self):
        """Test line count exceeding limit warns."""
        result = validate_line_count(600, 500, "Test file")
        assert result.passed  # Line count only warns, doesn't fail
        assert any("600" in warn for warn in result.warnings)
        assert any("500" in warn for warn in result.warnings)
