# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for bundle-reference plugin manifest validation.

Plugins are a Tier 1 content type rooted by an ``agent_plugin.yaml`` (or
``agent_plugin.yml``) manifest. Unlike skills/rules/workflows -- which embed
their content -- a plugin *references* existing skills, rules, and MCP servers
("bundle reference" mode). These models mirror the style of
:mod:`skillevaluator.models.workflows` and align to the backend bundle contract.
"""

from __future__ import annotations

from typing import Any, Literal, Union, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


class PluginAuthor(BaseModel):
    """Author block for a plugin manifest.

    ``email`` is required and non-empty. Additional fields (e.g. ``name``)
    are permitted so authors can carry extra contact metadata.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    email: str = Field(..., min_length=1, description="Author contact email (required)")

    @field_validator("email")
    @classmethod
    def email_must_look_valid(cls, v: str) -> str:
        """Basic sanity check that the email contains an '@'."""
        if "@" not in v:
            raise ValueError("author.email must be a valid email address (missing '@')")
        return v


class PluginSelector(BaseModel):
    """A source-control selector pointing at a referenced resource.

    Selectors are the dict form of a dependency ref. They identify a resource
    by source system, repository, and in-repo path.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source: Literal["github", "git"] = Field(..., description="Source control system (github or git)")
    repo: str = Field(..., min_length=1, description="Repository identifier (must contain '/')")
    path: str = Field(..., min_length=1, description="In-repo path to the referenced resource")

    @field_validator("repo")
    @classmethod
    def repo_must_be_full_logical_name(cls, v: str) -> str:
        """Require ``repo`` to be a full repository name (e.g. 'owner/repository')."""
        if "/" not in v:
            raise ValueError(
                "repo must be a full repository name containing '/' (e.g. 'owner/repository'), not a shorthand name"
            )
        return v

    @field_validator("path")
    @classmethod
    def path_must_be_repo_relative(cls, v: str) -> str:
        """Require ``path`` to be a full repo-relative path (e.g. 'skills/foo')."""
        if "/" not in v:
            raise ValueError(
                "path must be a full repo-relative path containing '/' "
                "(e.g. 'skills/confluence-to-markdown'), not a shorthand name"
            )
        return v


# A dependency ref is either a canonical ID string (``<source>::<repo>::...``)
# or a selector dict. The before-validator on PluginDependencySection enforces
# the canonical-ID source+repo invariants (mirroring PluginSelector); dict
# entries are parsed into PluginSelector by the union.
PluginRef = Union[str, PluginSelector]


# Allowed source systems, derived from PluginSelector's own ``source`` Literal so
# the canonical-ID string form and the selector-dict form share a single source
# of truth and cannot drift apart.
_ALLOWED_SELECTOR_SOURCES: tuple[str, ...] = get_args(PluginSelector.model_fields["source"].annotation)


def _validate_canonical_ref(entry: str) -> None:
    """Validate a canonical-ID string ref against the selector source+repo rules.

    A canonical ID has the shape ``<source>::<repo>::<type>::<name>`` (e.g.
    ``github::owner/repository::skills::example``). To keep the two
    ref forms consistent we enforce only the confidently supported invariants:
    ``source`` (1st segment) is an allowed system, ``repo`` (2nd segment) is a
    full logical name containing '/', and no segment is empty. The exact segment
    count is intentionally not enforced.
    """
    segments = entry.split("::")
    if (
        len(segments) < 2
        or any(seg == "" for seg in segments)
        or segments[0] not in _ALLOWED_SELECTOR_SOURCES
        or "/" not in segments[1]
    ):
        allowed = ", ".join(_ALLOWED_SELECTOR_SOURCES)
        raise ValueError(
            f"canonical ref must be '<source>::<repo>::...' with source in "
            f"{{{allowed}}} and repo (2nd segment) containing '/'; got '{entry}'"
        )


class PluginDependencySection(BaseModel):
    """A ``skills`` or ``rules`` dependency section.

    ``refs`` is required when the section is present, but an empty list is
    permitted (the aggregate "at least one dependency" rule is enforced at the
    manifest level). Legacy ``include``/``exclude`` keys are tolerated only when
    empty (compatibility no-ops); populated values are rejected. ``extra="forbid"``
    rejects any other unknown key.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    refs: list[PluginRef] = Field(..., description="List of canonical '::' IDs and/or source selectors")
    include: list = Field(
        default_factory=list,
        description="Legacy compatibility filter; only an empty list is accepted",
    )
    exclude: list = Field(
        default_factory=list,
        description="Legacy compatibility filter; only an empty list is accepted",
    )

    @field_validator("include", "exclude")
    @classmethod
    def legacy_filters_must_be_empty(cls, v: Any, info: ValidationInfo) -> Any:
        """Tolerate empty ``include``/``exclude`` as no-ops; reject populated ones."""
        if v:
            raise ValueError(
                f"'{info.field_name}' is not supported; only an empty "
                f"'{info.field_name}: []' is allowed for compatibility"
            )
        return v

    @field_validator("refs", mode="before")
    @classmethod
    def validate_refs(cls, v: Any) -> Any:
        """Validate canonical-ID string refs; reject non-mapping/non-string entries."""
        if not isinstance(v, list):
            raise ValueError("refs must be a list of canonical IDs or selector objects")
        for entry in v:
            if isinstance(entry, str):
                _validate_canonical_ref(entry)
            elif not isinstance(entry, dict):
                raise ValueError(
                    f"each ref must be a canonical '::' ID string or a selector object (got {type(entry).__name__})"
                )
        return v


class PluginMcpEntry(BaseModel):
    """A single MCP server dependency entry.

    MCP entries must be mapping objects (not bare strings). ``name`` and
    ``provider`` are required and non-empty.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(..., min_length=1, description="MCP server name")
    provider: str = Field(..., min_length=1, description="MCP provider or transport identifier")


class PluginManifest(BaseModel):
    """Top-level bundle-reference plugin manifest (``agent_plugin.yaml``).

    Unknown top-level fields are rejected. ``name`` and ``author`` (with a
    non-empty ``author.email``) are required, and the manifest must declare at
    least one dependency across ``skills.refs``, ``rules.refs``, or ``mcp``.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(..., min_length=1, description="Plugin name (required, non-empty)")
    description: str | None = Field(default=None, description="Short plugin description")
    version: str | None = Field(default=None, description="Plugin version string")
    team: str | None = Field(default=None, description="Owning team")
    domain: str | None = Field(default=None, description="Domain category")
    tags: list[str] | None = Field(default=None, description="Searchable tags")
    metadata: dict[str, Any] | None = Field(default=None, description="Free-form metadata")
    author: PluginAuthor = Field(..., description="Author block (author.email required)")
    skills: PluginDependencySection | None = Field(default=None, description="Referenced skills (refs only)")
    rules: PluginDependencySection | None = Field(default=None, description="Referenced rules (refs only)")
    mcp: list[PluginMcpEntry] | None = Field(default=None, description="Referenced MCP server entries")

    @field_validator("version", mode="before")
    @classmethod
    def coerce_version_to_str(cls, v: Any) -> Any:
        """Coerce a YAML-parsed numeric version (e.g. ``1.0``) to a string."""
        if isinstance(v, (int, float)):
            return str(v)
        return v

    @model_validator(mode="after")
    def check_dependencies_and_mcp(self) -> PluginManifest:
        """Enforce at-least-one dependency and reject duplicate MCP pairs."""
        has_skill_refs = bool(self.skills and self.skills.refs)
        has_rule_refs = bool(self.rules and self.rules.refs)
        has_mcp = bool(self.mcp)
        if not (has_skill_refs or has_rule_refs or has_mcp):
            raise ValueError("Plugin must declare at least one dependency in 'skills.refs', 'rules.refs', or 'mcp'.")

        if self.mcp:
            seen: set[tuple[str, str]] = set()
            for entry in self.mcp:
                key = (entry.name, entry.provider)
                if key in seen:
                    raise ValueError(
                        f"Duplicate MCP entry (name='{entry.name}', provider='{entry.provider}'); "
                        "each (name, provider) pair must be unique."
                    )
                seen.add(key)

        return self
