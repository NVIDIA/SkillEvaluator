# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for Workflows schema validation.

Based on Skill Evaluator HOW_TO_CONTRIBUTE_WORKFLOW_RULES.md specification.
Workflows are directory-based structures containing:
- README.md (required)
- workflow-rules.mdc (required)
- references/ directory with .mdc files (required)
- references/scripts/ directory (optional)
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from skillevaluator.constants import (
    DESCRIPTION_MAX_LENGTH,
    DESCRIPTION_MIN_LENGTH,
    REQUIRED_WORKFLOWS_FIELDS,
    REQUIRED_WORKFLOWS_METADATA_FIELDS,
    TITLE_MAX_LENGTH,
    TITLE_MIN_LENGTH,
    WORKFLOWS_MANIFEST_FILE,
    WORKFLOWS_README_FILE,
    WORKFLOWS_REFERENCES_DIR,
)
from skillevaluator.models.field_validators import ensure_string_list, parse_nested_model


class WorkflowsMetadata(BaseModel):
    """Metadata object for Workflows workflow-rules.mdc frontmatter.

    Contains Skill Evaluator-specific categorization fields nested under 'metadata'.
    For workflows, 'author' is required.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    # Required for workflows
    author: str = Field(
        ...,
        description="Author name and email (e.g., 'Your Name <your-email@example.com>')",
    )

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
        description="Specific library the workflow targets",
    )
    version: str | None = Field(
        default=None,
        description="Library/framework version",
    )
    project: str | None = Field(
        default=None,
        description="Project name (required for team workflows)",
    )
    team: str | None = Field(
        default=None,
        description="Team name (required for team workflows)",
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


class WorkflowsFrontmatter(BaseModel):
    """Pydantic model for workflow-rules.mdc frontmatter validation.

    Based on Skill Evaluator specification for workflows:
    - Required top-level: alwaysApply, title, description
    - Optional top-level: globs
    - Required in metadata: author
    - Optional in metadata: tags, language, framework, etc.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    # Required Cursor-standard fields (top level)
    alwaysApply: bool = Field(  # noqa: N815
        ...,
        description="Whether to always apply this workflow (typically false)",
    )
    title: str = Field(
        ...,
        min_length=TITLE_MIN_LENGTH,
        max_length=TITLE_MAX_LENGTH,
        description="Human-readable workflow title",
    )
    description: str = Field(
        ...,
        min_length=DESCRIPTION_MIN_LENGTH,
        max_length=DESCRIPTION_MAX_LENGTH,
        description="Brief description of what the workflow covers",
    )

    # Optional Cursor-standard fields
    globs: list[str] | None = Field(
        default=None,
        description="File patterns this workflow applies to",
    )

    # Skill Evaluator-specific metadata (nested) - author is required for workflows
    metadata: WorkflowsMetadata = Field(
        ...,
        description="Skill Evaluator-specific metadata (author required, plus tags, language, etc.)",
    )

    @field_validator("globs", mode="before")
    @classmethod
    def ensure_globs_list(cls, v: Any) -> list[str] | None:
        """Ensure globs field is properly formatted as a list."""
        return ensure_string_list(v)

    @field_validator("metadata", mode="before")
    @classmethod
    def parse_metadata(cls, v: Any) -> WorkflowsMetadata | None:
        """Parse metadata dict into WorkflowsMetadata model."""
        return parse_nested_model(
            WorkflowsMetadata,
            v,
            error_message=(
                "Workflows require metadata with author field. "
                "Add metadata.author with format 'Name <email@example.com>'"
            ),
        )

    @model_validator(mode="before")
    @classmethod
    def check_required_fields(cls, data: Any) -> Any:
        """Verify all required fields are present."""
        if not isinstance(data, dict):
            return data

        missing = []
        for field in REQUIRED_WORKFLOWS_FIELDS:
            if field not in data:
                missing.append(field)

        if missing:
            raise ValueError(
                f"Missing required fields for Workflows: {', '.join(missing)}. "
                "Workflows must have alwaysApply, title, and description."
            )

        # Check for metadata and author
        metadata = data.get("metadata")
        if not metadata:
            raise ValueError(
                "Workflows require metadata with author field. "
                "Add metadata.author with format 'Name <email@example.com>'"
            )

        if isinstance(metadata, dict):
            for field in REQUIRED_WORKFLOWS_METADATA_FIELDS:
                if field not in metadata or not metadata[field]:
                    raise ValueError(
                        f"Workflows require metadata.{field}. Add author with format 'Name <email@example.com>'"
                    )

        return data


class ReferenceFrontmatter(BaseModel):
    """Pydantic model for reference .mdc file frontmatter.

    Reference files in references/ directory have simpler requirements.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    # Required fields
    alwaysApply: bool = Field(  # noqa: N815
        ...,
        description="Whether to always apply (typically false for references)",
    )
    title: str = Field(
        ...,
        min_length=TITLE_MIN_LENGTH,
        max_length=TITLE_MAX_LENGTH,
        description="Reference title",
    )
    description: str = Field(
        ...,
        min_length=DESCRIPTION_MIN_LENGTH,
        max_length=DESCRIPTION_MAX_LENGTH,
        description="What this reference covers",
    )

    # Optional fields specific to references
    parent_workflow: str | None = Field(
        default=None,
        description="Name of the parent workflow this reference belongs to",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Tags for categorization",
    )
    language: str | None = Field(
        default=None,
        description="Programming language",
    )
    project: str | None = Field(
        default=None,
        description="Project name",
    )
    globs: list[str] | None = Field(
        default=None,
        description="File patterns",
    )

    @field_validator("tags", "globs", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> list[str] | None:
        """Ensure list fields are properly formatted."""
        return ensure_string_list(v)


class WorkflowsManifest(BaseModel):
    """Complete Workflows manifest including all components.

    Represents an entire workflow directory structure.
    """

    model_config = ConfigDict(extra="allow")

    # Main workflow rules
    frontmatter: WorkflowsFrontmatter = Field(
        ...,
        description="Parsed frontmatter from workflow-rules.mdc",
    )
    content: str = Field(
        default="",
        description="Markdown content from workflow-rules.mdc",
    )

    # Directory information
    workflow_dir: str | None = Field(
        default=None,
        description="Path to workflow directory",
    )
    workflow_name: str | None = Field(
        default=None,
        description="Name of the workflow (directory name)",
    )

    # Component files
    has_readme: bool = Field(
        default=False,
        description="Whether README.md exists",
    )
    has_references: bool = Field(
        default=False,
        description="Whether references/ directory exists",
    )
    reference_files: list[str] = Field(
        default_factory=list,
        description="List of reference .mdc files found",
    )
    has_scripts: bool = Field(
        default=False,
        description="Whether references/scripts/ directory exists",
    )
    script_files: list[str] = Field(
        default_factory=list,
        description="List of script files found",
    )

    # Validation metadata
    line_count: int | None = Field(
        default=None,
        description="Number of lines in workflow-rules.mdc",
    )

    @property
    def title(self) -> str:
        """Get title from frontmatter."""
        return self.frontmatter.title

    @property
    def author(self) -> str:
        """Get author from metadata."""
        return self.frontmatter.metadata.author

    @property
    def team(self) -> str | None:
        """Get team from metadata if available."""
        return self.frontmatter.metadata.team

    @property
    def project(self) -> str | None:
        """Get project from metadata if available."""
        return self.frontmatter.metadata.project

    @property
    def is_complete(self) -> bool:
        """Check if workflow has all required components."""
        return self.has_readme and self.has_references and len(self.reference_files) > 0


class WorkflowsStructure(BaseModel):
    """Represents the expected structure of a workflow directory.

    Used for validation and documentation purposes.
    """

    model_config = ConfigDict(extra="allow")

    workflow_name: str = Field(..., description="Name of the workflow")
    required_files: list[str] = Field(
        default_factory=lambda: [WORKFLOWS_README_FILE, WORKFLOWS_MANIFEST_FILE],
        description="Required files in workflow root",
    )
    required_dirs: list[str] = Field(
        default_factory=lambda: [WORKFLOWS_REFERENCES_DIR],
        description="Required directories",
    )
    optional_dirs: list[str] = Field(
        default_factory=lambda: [f"{WORKFLOWS_REFERENCES_DIR}/scripts"],
        description="Optional directories",
    )

    @classmethod
    def from_path(cls, workflow_path: Path) -> "WorkflowsStructure":
        """Create structure representation from a workflow path."""
        return cls(workflow_name=workflow_path.name)

    def validate_structure(self, workflow_path: Path) -> tuple[bool, list[str]]:
        """Validate that a path contains required workflow structure.

        Args:
            workflow_path: Path to workflow directory

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check required files
        for required_file in self.required_files:
            if not (workflow_path / required_file).exists():
                errors.append(f"Missing required file: {required_file}")

        # Check required directories
        for required_dir in self.required_dirs:
            dir_path = workflow_path / required_dir
            if not dir_path.exists():
                errors.append(f"Missing required directory: {required_dir}/")
            elif not dir_path.is_dir():
                errors.append(f"{required_dir} must be a directory")

        # Check references has at least one .mdc file
        refs_dir = workflow_path / WORKFLOWS_REFERENCES_DIR
        if refs_dir.exists() and refs_dir.is_dir():
            mdc_files = list(refs_dir.glob("*.mdc"))
            if not mdc_files:
                errors.append("references/ directory must contain at least one .mdc file")

        return len(errors) == 0, errors
