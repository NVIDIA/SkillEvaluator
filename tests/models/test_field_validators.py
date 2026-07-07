# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for skillevaluator.models.field_validators module."""

import pytest
from pydantic import BaseModel

from skillevaluator.models.field_validators import ensure_string_list, parse_nested_model


class TestEnsureStringList:
    """Tests for ensure_string_list field validator."""

    def test_none_returns_none(self):
        """Test that None input returns None."""
        assert ensure_string_list(None) is None

    def test_string_returns_single_item_list(self):
        """Test that string input returns list with one item."""
        result = ensure_string_list("single-tag")
        assert result == ["single-tag"]

    def test_list_returns_stringified_list(self):
        """Test that list input returns list of strings."""
        result = ensure_string_list(["tag1", "tag2", 123])
        assert result == ["tag1", "tag2", "123"]

    def test_empty_list_returns_empty_list(self):
        """Test that empty list returns empty list."""
        result = ensure_string_list([])
        assert result == []


class TestParseNestedModel:
    """Tests for parse_nested_model utility."""

    def test_none_without_error_returns_none(self):
        """Test that None without error message returns None."""
        result = parse_nested_model(BaseModel, None)
        assert result is None

    def test_none_with_error_raises(self):
        """Test that None with error message raises ValueError."""
        with pytest.raises(ValueError, match="Field is required"):
            parse_nested_model(BaseModel, None, "Field is required")

    def test_dict_creates_model(self):
        """Test that dict input creates model instance."""

        class SimpleModel(BaseModel):
            name: str

        result = parse_nested_model(SimpleModel, {"name": "test"})
        assert isinstance(result, SimpleModel)
        assert result.name == "test"

    def test_already_model_returns_as_is(self):
        """Test that model instance is returned unchanged."""

        class SimpleModel(BaseModel):
            name: str

        model = SimpleModel(name="test")
        result = parse_nested_model(SimpleModel, model)
        assert result is model
