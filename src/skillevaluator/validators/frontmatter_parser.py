# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared frontmatter parsing utilities for validators.

Extracts and validates YAML frontmatter from .mdc and .md files,
eliminating duplication across RulesSchemaValidator and WorkflowsSchemaValidator.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from skillevaluator.validators.base import ValidationResult

# Regex pattern for extracting frontmatter between --- markers
FRONTMATTER_PATTERN = re.compile(
    r"^---[^\S\r\n]*\r?\n(.*?)\r?\n---[^\S\r\n]*(?=\r?\n|\Z)(?:\r?\n)?(.*)",
    re.DOTALL,
)


@dataclass
class ParsedFrontmatter:
    """Result of parsing a file's frontmatter."""

    yaml_data: dict[str, Any] | None
    content: str
    raw_yaml: str
    line_count: int


def parse_frontmatter(file_path: Path) -> tuple[ParsedFrontmatter | None, ValidationResult]:
    """Parse YAML frontmatter from a markdown file.

    Centralizes the frontmatter extraction logic used by multiple validators.

    Args:
        file_path: Path to the file to parse

    Returns:
        Tuple of (ParsedFrontmatter or None, ValidationResult with any errors)
    """
    result = ValidationResult()

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        result.add_error(f"Failed to read {file_path}: {e}")
        return None, result

    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        result.add_error(f"{file_path.name} must have YAML frontmatter between --- markers")
        return None, result

    frontmatter_yaml, markdown_content = match.groups()

    try:
        data = yaml.safe_load(frontmatter_yaml)
    except yaml.YAMLError as e:
        result.add_error(f"Invalid YAML in frontmatter: {e}")
        return None, result

    if not data or not isinstance(data, dict):
        result.add_error("Frontmatter must be a non-empty YAML dictionary")
        return None, result

    return ParsedFrontmatter(
        yaml_data=data,
        content=markdown_content,
        raw_yaml=frontmatter_yaml,
        line_count=len(content.splitlines()),
    ), result


def validate_pydantic_model(
    model_class: type,
    data: dict[str, Any],
    result: ValidationResult,
) -> Any | None:
    """Validate data against a Pydantic model, adding errors to result.

    Args:
        model_class: The Pydantic model class to validate against
        data: The data dict to validate
        result: ValidationResult to add errors to

    Returns:
        The validated model instance, or None if validation failed
    """
    from pydantic import ValidationError

    try:
        return model_class(**data)
    except ValidationError as e:
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            result.add_error(f"Field '{field}': {error['msg']}")
        return None
