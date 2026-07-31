# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for Rules (.mdc) schema validation.

Based on SkillEvaluator HOW_TO_CONTRIBUTE_WORKFLOW_RULES.md specification.
Rules use .mdc (Markdown Cursor) format with YAML frontmatter.

Key differences from Skills:
- Rules REQUIRE alwaysApply and allow globs (Skills FORBID these)
- Rules use title/description at top level (Skills use name/description)
- Rules are in team-rules/ folder (Skills in skills/ or team-skills/)
"""

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from skillevaluator.constants import (
    DESCRIPTION_MAX_LENGTH,
    DESCRIPTION_MIN_LENGTH,
    REQUIRED_RULES_FIELDS,
    TITLE_MAX_LENGTH,
    TITLE_MIN_LENGTH,
)
from skillevaluator.models.field_validators import ensure_string_list, parse_nested_model


class RulesMetadata(BaseModel):
    """Metadata object for Rules .mdc frontmatter.

    Contains SkillEvaluator-specific categorization fields nested under 'metadata'.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    # Categorization fields
    tags: list[str] | None = Field(
        default=None,
        description="List of searchable tags (include project, team, framework)",
    )
    language: str | None = Field(
        default=None,
        description="Primary programming language (e.g., python, typescript)",
    )
    framework: str | None = Field(
        default=None,
        description="Target framework (e.g., fastapi, react, django)",
    )
    library: str | None = Field(
        default=None,
        description="Specific library the rule targets (e.g., pydantic, sqlalchemy)",
    )
    version: str | None = Field(
        default=None,
        description="Library/framework version (e.g., '2.0', '>=3.9')",
    )
    project: str | None = Field(
        default=None,
        description="Project name (e.g., data-platform, developer-tools)",
    )
    team: str | None = Field(
        default=None,
        description="Team name (e.g., ipp, av-perception, infra)",
    )
    domain: str | None = Field(
        default=None,
        description="Domain category (e.g., backend, frontend, ml, devops)",
    )

    @field_validator("tags", mode="before")
    @classmethod
    def ensure_tags_list(cls, v: Any) -> list[str] | None:
        """Ensure tags field is properly formatted as a list."""
        return ensure_string_list(v)


class RulesFrontmatter(BaseModel):
    """Pydantic model for Rules .mdc frontmatter validation.

    Based on SkillEvaluator specification for team-rules:
    - Required top-level: alwaysApply, title, description
    - Optional top-level: globs
    - Optional nested: metadata (with tags, language, framework, etc.)

    Key difference from Skills: alwaysApply and globs are ALLOWED here.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    # Required Cursor-standard fields (top level)
    alwaysApply: bool = Field(  # noqa: N815
        ...,
        description="Whether to always apply this rule (typically false)",
    )
    title: str = Field(
        ...,
        min_length=TITLE_MIN_LENGTH,
        max_length=TITLE_MAX_LENGTH,
        description="Human-readable rule title",
    )
    description: str = Field(
        ...,
        min_length=DESCRIPTION_MIN_LENGTH,
        max_length=DESCRIPTION_MAX_LENGTH,
        description="Brief description of what the rule covers",
    )

    # Optional Cursor-standard fields
    globs: list[str] | None = Field(
        default=None,
        description="File patterns this rule applies to (e.g., '*.py', 'app/**/*.py')",
    )

    # SkillEvaluator-specific metadata (nested)
    metadata: RulesMetadata | None = Field(
        default=None,
        description="SkillEvaluator-specific metadata (tags, language, framework, etc.)",
    )

    @field_validator("globs", mode="before")
    @classmethod
    def ensure_globs_list(cls, v: Any) -> list[str] | None:
        """Ensure globs field is properly formatted as a list."""
        return ensure_string_list(v)

    @field_validator("metadata", mode="before")
    @classmethod
    def parse_metadata(cls, v: Any) -> RulesMetadata | None:
        """Parse metadata dict into RulesMetadata model."""
        return parse_nested_model(RulesMetadata, v)

    @model_validator(mode="before")
    @classmethod
    def check_required_fields(cls, data: Any) -> Any:
        """Verify all required fields are present."""
        if not isinstance(data, dict):
            return data

        missing = []
        for field in REQUIRED_RULES_FIELDS:
            if field not in data:
                missing.append(field)

        if missing:
            raise ValueError(
                f"Missing required fields for Rules: {', '.join(missing)}. "
                "Rules must have alwaysApply, title, and description."
            )

        return data


class RulesManifest(BaseModel):
    """Complete Rules manifest including frontmatter and content."""

    model_config = ConfigDict(extra="allow")

    frontmatter: RulesFrontmatter = Field(..., description="Parsed frontmatter")
    content: str = Field(default="", description="Markdown content after frontmatter")
    file_path: str | None = Field(default=None, description="Path to .mdc file")
    line_count: int | None = Field(default=None, description="Number of lines in file")

    @property
    def has_content(self) -> bool:
        """Check if rule has non-empty content."""
        return bool(self.content.strip())

    @property
    def title(self) -> str:
        """Get title from frontmatter."""
        return self.frontmatter.title

    @property
    def tags(self) -> list[str] | None:
        """Get tags from metadata if available."""
        if self.frontmatter.metadata:
            return self.frontmatter.metadata.tags
        return None

    @property
    def team(self) -> str | None:
        """Get team from metadata if available."""
        if self.frontmatter.metadata:
            return self.frontmatter.metadata.team
        return None

    @property
    def project(self) -> str | None:
        """Get project from metadata if available."""
        if self.frontmatter.metadata:
            return self.frontmatter.metadata.project
        return None

    def get_full_text(self) -> str:
        """Get complete rule text including frontmatter."""
        return f"---\n{self.frontmatter.model_dump_json()}\n---\n{self.content}"


def validate_rules_filename(filename: str) -> tuple[bool, str | None]:
    """Validate that a filename follows Rules naming conventions.

    Args:
        filename: The filename to validate (without path)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not filename.endswith(".mdc"):
        return False, f"Rules file '{filename}' must have .mdc extension"

    # Extract name without extension
    name = filename[:-4]  # Remove .mdc

    # Check kebab-case pattern
    pattern = r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$"
    if not re.match(pattern, name):
        return False, (
            f"Rules filename '{name}' should follow kebab-case convention "
            "(lowercase letters, numbers, and hyphens only)"
        )

    if "--" in name:
        return False, f"Rules filename '{name}' cannot contain consecutive hyphens"

    if name.endswith("-"):
        return False, f"Rules filename '{name}' cannot end with a hyphen"

    return True, None
