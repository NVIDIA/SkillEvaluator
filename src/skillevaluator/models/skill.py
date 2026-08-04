# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for skill schema validation.

Based on the SkillEvaluator HOW_TO_CONTRIBUTE_SKILLS.md specification
and agentskills.io standard.
"""

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from skillevaluator.constants import (
    COMPATIBILITY_MAX_LENGTH,
    DESCRIPTION_MAX_LENGTH,
    DESCRIPTION_MIN_LENGTH,
    FORBIDDEN_SKILL_FIELDS,
    KEBAB_CASE_PATTERN,
    NAME_MAX_LENGTH,
    NAME_MIN_LENGTH,
    RESERVED_SKILL_NAMES,
)

XML_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9_-]*(?:\s[^<>]*)?>|</?[A-Za-z][A-Za-z0-9_-]*\s[^<>]*$")

#: Strict, bounded ASCII semantic-version pattern for optional metadata.version labels.
_SEMVER_COMPONENT = r"(?:0|[1-9][0-9]{0,8})"
SEMVER_RE = re.compile(rf"^{_SEMVER_COMPONENT}\.{_SEMVER_COMPONENT}\.{_SEMVER_COMPONENT}$", re.ASCII)


class SkillMetadata(BaseModel):
    """Metadata object for SKILL.md frontmatter.

    Contains optional categorization and attribution fields.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    # Author field uses the portable Name <email@example.com> shape.
    author: str | None = Field(
        default=None,
        description="Author name and email (e.g., 'Your Name <your-email@example.com>')",
    )

    # Categorization fields
    tags: list[str] | None = Field(default=None, description="Categorization tags")
    languages: list[str] | None = Field(default=None, description="Programming languages")
    frameworks: list[str] | None = Field(default=None, description="Frameworks used")
    domain: str | None = Field(default=None, description="Domain category (e.g., backend, ai-ml)")

    @field_validator("tags", "languages", "frameworks", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> list[str] | None:
        """Ensure list fields are properly formatted."""
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(item) for item in v]
        return v

    @field_validator("author")
    @classmethod
    def validate_author_format(cls, v: str | None) -> str | None:
        """Validate author format (recommended: Name <email@example.com>)."""
        if v is None:
            return None
        # Warn if not in expected format but don't fail
        # Expected format: "Name <email@example.com>"
        return v


class SkillFrontmatter(BaseModel):
    """Pydantic model for SKILL.md frontmatter validation.

    Based on SkillEvaluator specification:
    - Required: name, description
    - Optional: license, compatibility, metadata
    - Forbidden: alwaysApply, globs

    Name constraints:
    - 1-64 characters
    - Lowercase only
    - Alphanumeric + hyphens
    - No leading/trailing/consecutive hyphens
    - Must match directory name
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True, populate_by_name=True)

    # Required fields
    name: str = Field(
        ...,
        min_length=NAME_MIN_LENGTH,
        max_length=NAME_MAX_LENGTH,
        description="Unique skill identifier in kebab-case (must match directory name)",
    )
    description: str = Field(
        ...,
        min_length=DESCRIPTION_MIN_LENGTH,
        max_length=DESCRIPTION_MAX_LENGTH,
        description="Clear description of what the skill does and when to use it",
    )

    # Optional fields
    license: str | None = Field(
        default=None,
        description="License name or reference (e.g., 'MIT', 'Apache-2.0')",
    )
    compatibility: str | None = Field(
        default=None,
        max_length=COMPATIBILITY_MAX_LENGTH,
        description="Environment requirements (max 500 chars)",
    )
    metadata: SkillMetadata | None = Field(
        default=None,
        description="Additional metadata (author, tags, languages, frameworks, domain)",
    )
    allowed_tools: str | None = Field(
        default=None,
        alias="allowed-tools",
        description="Space-separated pre-approved tools (experimental, per agentskills.io spec)",
    )

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, v: str) -> str:
        """Validate that name follows SkillEvaluator and agentskills.io naming conventions.

        Rules:
        - 1-64 characters
        - Lowercase letters, numbers, and hyphens only
        - Must start with a letter
        - No leading, trailing, or consecutive hyphens
        - Must not contain reserved words ("anthropic", "claude")
        - Must not contain XML tags
        """
        if not re.match(KEBAB_CASE_PATTERN, v):
            raise ValueError(
                f"Skill name '{v}' must be in kebab-case format "
                "(lowercase letters, numbers, and hyphens only, starting with a letter)"
            )

        if "--" in v:
            raise ValueError(f"Skill name '{v}' cannot contain consecutive hyphens")

        if v.endswith("-"):
            raise ValueError(f"Skill name '{v}' cannot end with a hyphen")

        if any(segment in RESERVED_SKILL_NAMES for segment in v.lower().split("-")):
            raise ValueError(f"Skill name '{v}' must not contain reserved words {RESERVED_SKILL_NAMES}")

        if XML_TAG_RE.search(v):
            raise ValueError(f"Skill name '{v}' must not contain XML tags")

        return v

    @field_validator("description")
    @classmethod
    def validate_description_content(cls, v: str) -> str:
        """Reject descriptions that contain XML tags."""
        if XML_TAG_RE.search(v):
            raise ValueError("Skill description must not contain XML tags")
        return v

    @field_validator("metadata", mode="before")
    @classmethod
    def parse_metadata(cls, v: Any) -> SkillMetadata | None:
        """Parse metadata dict into SkillMetadata model."""
        if v is None:
            return None
        if isinstance(v, dict):
            return SkillMetadata(**v)
        return v

    @model_validator(mode="before")
    @classmethod
    def check_forbidden_fields(cls, data: Any) -> Any:
        """Check for forbidden fields that should not be present."""
        if not isinstance(data, dict):
            return data

        found_forbidden = []
        for field in FORBIDDEN_SKILL_FIELDS:
            if field in data:
                found_forbidden.append(field)

        if found_forbidden:
            raise ValueError(
                f"Forbidden fields detected: {', '.join(found_forbidden)}. "
                "Skills should not use 'alwaysApply' or 'globs' fields."
            )

        return data


class SkillManifest(BaseModel):
    """Complete skill manifest including frontmatter and content."""

    model_config = ConfigDict(extra="allow")

    frontmatter: SkillFrontmatter = Field(..., description="Parsed frontmatter")
    content: str = Field(default="", description="Markdown content after frontmatter")
    file_path: str | None = Field(default=None, description="Path to SKILL.md file")
    line_count: int | None = Field(default=None, description="Number of lines in SKILL.md")

    @property
    def has_content(self) -> bool:
        """Check if skill has non-empty content."""
        return bool(self.content.strip())

    @property
    def author(self) -> str | None:
        """Get author from metadata if available."""
        if self.frontmatter.metadata:
            return self.frontmatter.metadata.author
        return None

    @property
    def tags(self) -> list[str] | None:
        """Get tags from metadata if available."""
        if self.frontmatter.metadata:
            return self.frontmatter.metadata.tags
        return None

    def get_full_text(self) -> str:
        """Get complete skill text including frontmatter."""
        return f"---\n{self.frontmatter.model_dump_json()}\n---\n{self.content}"
