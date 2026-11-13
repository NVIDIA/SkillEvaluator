# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema and Repository Governance Validator.

Validates SKILL.md frontmatter against schema, enforces folder hierarchy
and naming conventions per SkillEvaluator specification.
"""

import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from skillevaluator.constants import (
    DEFAULT_ALLOWED_SKILL_DIRS,
    DESCRIPTION_MAX_LENGTH,
    DESCRIPTION_TRIGGER_PATTERN,
    DESCRIPTION_TRIGGER_WINDOW,
    KEBAB_CASE_PATTERN,
    MAX_SKILL_MD_LINES,
    RECOMMENDED_BODY_SECTIONS,
    REQUIRED_BODY_SECTIONS,
    SCAN_EXCLUDED_FILES,
    SCHEMA_ALLOWED_DIRS_ENV,
    SKILL_MANIFEST_FILE,
    SKILL_MANIFEST_VARIANTS,
    SKILL_MD_SIZE_THRESHOLDS,
    SKILL_NOT_SHIPPED_DIRS,
    SPEC_SKILL_FIELDS,
)
from skillevaluator.logging_config import get_logger
from skillevaluator.models.result import Finding, Severity
from skillevaluator.models.skill import SkillFrontmatter, SkillManifest
from skillevaluator.utils.path_security import matches_filesystem_name
from skillevaluator.validators.base import ValidationResult, ValidatorBase
from skillevaluator.validators.frontmatter_parser import FRONTMATTER_PATTERN, format_validation_location
from skillevaluator.validators.policy import AUTHOR_SHAPE_PATTERN, ValidationPolicy, default_policy

logger = get_logger(__name__)

#: Upper bound for a ``metadata.author`` value; longer values are malformed.
AUTHOR_MAX_LENGTH = 256

#: An email address written outside the ``Name <email@host>`` shape. The
#: discriminator is ``\w@\w``: an address has a local part, so a word character
#: sits on each side of the ``@``, while a social handle such as
#: ``Jane Doe (@jdoe)`` or a bare ``@jdoe`` does not. The domain needs no dot,
#: so an intranet address such as ``root@buildhost`` is caught as well.
_BARE_EMAIL_PATTERN = re.compile(r"[^\s<>@]*\w@\w[^\s<>@]*")

#: Unicode general categories never usable in an author value: C0/C1 controls,
#: surrogates, and private-use code points.
_UNSAFE_AUTHOR_CATEGORIES = frozenset({"Cc", "Cs", "Co"})

#: Invisible format characters (category ``Cf``) that can make an author value
#: render as something other than what it says: the bidirectional marks,
#: embeddings, overrides and isolates, plus the byte-order mark. The rest of
#: ``Cf`` stays allowed, because ZWNJ (U+200C) and ZWJ (U+200D) are required
#: for Persian, Arabic and Devanagari names to render correctly.
_UNSAFE_AUTHOR_FORMAT_CHARS = frozenset(
    {
        chr(0x061C),  # ARABIC LETTER MARK
        chr(0x200E),  # LEFT-TO-RIGHT MARK
        chr(0x200F),  # RIGHT-TO-LEFT MARK
        *(chr(code) for code in range(0x202A, 0x202F)),  # embedding, pop, override
        *(chr(code) for code in range(0x2066, 0x206A)),  # isolate controls
        chr(0xFEFF),  # ZERO WIDTH NO-BREAK SPACE (byte-order mark)
    }
)

#: ``metadata.id`` contract: an exact (unnormalized) lowercase ASCII slug.
METADATA_ID_MAX_LENGTH = 64
METADATA_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.ASCII)

#: Longest raw ``metadata.id`` text repeated in a finding.
_METADATA_ID_DISPLAY_LIMIT = 80


def author_value_problem(author: str) -> str | None:
    """Return why *author* is malformed or unsafe, or ``None`` when it is usable.

    A usable value is either the ``Name <email@host>`` shape or plain
    attribution: a person, team, or organization name such as
    ``NVIDIA cuOpt Team`` or ``Jane Doe (@jdoe)``. Plain attribution may not
    contain angle brackets or an email address, because those mean an attempted
    but broken ``Name <email>`` value.

    The character guard rejects controls, surrogates, private-use code points
    and the bidirectional/BOM format characters. It deliberately does not
    reject all of Unicode category ``C``: that also covers the joiners real
    names need (U+200C/U+200D) and every code point unassigned in the running
    interpreter's Unicode data, which makes the verdict depend on the
    interpreter's version rather than on the value.
    """
    if len(author) > AUTHOR_MAX_LENGTH:
        return f"is longer than {AUTHOR_MAX_LENGTH} characters"
    if any(
        char in _UNSAFE_AUTHOR_FORMAT_CHARS or unicodedata.category(char) in _UNSAFE_AUTHOR_CATEGORIES
        for char in author
    ):
        return "contains control, private-use, or bidirectional formatting characters"
    if AUTHOR_SHAPE_PATTERN.fullmatch(author):
        return None
    if "<" in author or ">" in author:
        return "uses angle brackets but is not of the form 'Name <email@host>'"
    if _BARE_EMAIL_PATTERN.search(author):
        return "contains an email address that is not in the form 'Name <email@host>'"
    if not any(char.isalnum() for char in author):
        return "has no letters or digits"
    return None


def metadata_id_problem(value: Any) -> str | None:
    """Return why *value* breaks the ``metadata.id`` contract, or ``None``."""
    if not isinstance(value, str):
        return f"must be a string (got {type(value).__name__})"
    if not value:
        return "must not be empty"
    if len(value) > METADATA_ID_MAX_LENGTH:
        return f"must be at most {METADATA_ID_MAX_LENGTH} characters (got {len(value)})"
    if not METADATA_ID_PATTERN.fullmatch(value):
        return (
            "must be lowercase ASCII letters and digits, with single hyphens between them "
            "(for example 'my-skill'), and no surrounding whitespace"
        )
    return None


def _relative_label(skill_dir: Path, root: Path) -> str:
    try:
        return skill_dir.relative_to(root).as_posix()
    except ValueError:
        return skill_dir.as_posix()


def duplicate_metadata_ids(declared_ids: dict[str, list[Path]], root: Path) -> dict[str, list[str]]:
    """Map each ``metadata.id`` declared by more than one skill to sorted skill labels under *root*."""
    return {
        value: sorted(_relative_label(skill_dir, root) for skill_dir in dirs)
        for value, dirs in sorted(declared_ids.items())
        if len(dirs) > 1
    }


def declared_metadata_id(skill_dir: Path) -> str | None:
    """Return the valid ``metadata.id`` declared in *skill_dir*'s manifest, if any.

    Used by catalog runs that validate each skill separately, so they can
    still check uniqueness across the catalog. Unparseable manifests and
    invalid IDs return ``None``; the per-skill schema check reports those.
    """
    validator = SchemaValidator()
    manifest = validator._find_skill_manifest(skill_dir)
    if manifest is None:
        return None
    frontmatter = validator._validate_frontmatter(manifest).metadata.get("frontmatter")
    if frontmatter is None or frontmatter.metadata is None:
        return None
    value = (frontmatter.metadata.model_extra or {}).get("id")
    return value if metadata_id_problem(value) is None else None


def metadata_id_duplicate_severity(policy: ValidationPolicy) -> Severity:
    """Severity of a duplicate ``metadata.id`` under *policy* (HIGH only when IDs are required)."""
    default = Severity.HIGH if policy.metadata_id_requirement == "require" else Severity.MEDIUM
    return policy.severity_for("SCHEMA", "metadata_id_duplicate", default=default)


def _display_metadata_id(value: Any) -> str:
    """Render a bounded preview of a ``metadata.id`` value for a finding.

    A non-string value is described by its type alone and never rendered.
    ``metadata.id`` accepts arbitrary YAML before this check rejects a
    non-string, and ``repr`` expands a YAML alias graph eagerly: a 445-byte
    manifest with 20 doubling aliases builds a 14,680,060-character string,
    and a slightly deeper graph exhausts memory. Truncating afterwards is too
    late, because the whole string is built first.
    """
    if not isinstance(value, str):
        return f"<{type(value).__name__}>"
    if len(value) > _METADATA_ID_DISPLAY_LIMIT:
        return value[:_METADATA_ID_DISPLAY_LIMIT] + "…"
    return value


_UTF8_BOM = b"\xef\xbb\xbf"
_TRIGGER_RE = re.compile(DESCRIPTION_TRIGGER_PATTERN, re.IGNORECASE)


def _duplicate_frontmatter_keys(frontmatter_yaml: str) -> list[tuple[str, int]]:
    """Return ``(dotted_key, line)`` for every repeated mapping key (PyYAML keeps the last silently)."""
    try:
        root = yaml.compose(frontmatter_yaml, Loader=yaml.SafeLoader)
    except yaml.YAMLError:
        return []
    duplicates: list[tuple[str, int]] = []
    stack: list[tuple[yaml.Node, str]] = [(root, "")] if root is not None else []
    visited: set[int] = set()
    while stack:
        node, prefix = stack.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        if isinstance(node, yaml.MappingNode):
            seen: set[str] = set()
            for key_node, value_node in node.value:
                if not isinstance(key_node, yaml.ScalarNode) or key_node.value == "<<":
                    continue
                key = f"{prefix}{key_node.value}"
                if key_node.value in seen:
                    duplicates.append((key, key_node.start_mark.line + 2))
                seen.add(key_node.value)
                stack.append((value_node, f"{key}."))
        elif isinstance(node, yaml.SequenceNode):
            stack.extend((item, prefix) for item in node.value)
    return sorted(duplicates, key=lambda item: item[1])


class SchemaValidator(ValidatorBase):
    """Validates skill schema and repository structure per SkillEvaluator spec.

    Checks: frontmatter schema, folder hierarchy, naming conventions,
    directory-name consistency, required/forbidden fields.

    Author attribution and ``metadata.id`` are policy-driven: the active
    :class:`ValidationPolicy` decides whether a missing author, a name without
    an email, or a missing ID blocks, and may restrict the author email domain.
    Malformed or unsafe author values and invalid IDs are always reported.
    """

    def __init__(self, policy: ValidationPolicy | None = None) -> None:
        self.policy = policy or default_policy()

    @property
    def name(self) -> str:
        return "Schema & Repository Governance"

    @property
    def description(self) -> str:
        return "Validate SKILL.md frontmatter and repository structure"

    def validate(self, skill_path: Path) -> ValidationResult:
        """Validate skill(s) at path for schema and structure compliance."""
        if self._is_skill_directory(skill_path):
            return self._validate_single_skill(skill_path)

        # Folder-level validation with compliance checks
        skill_dirs = self._find_all_skills(skill_path)
        if not skill_dirs:
            result = ValidationResult()
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=Severity.HIGH,
                    check_name="skill_discovery",
                    message="No skills found in target directory",
                    file_path=str(skill_path),
                    suggestion="Ensure SKILL.md files exist in skill directories under skills/ or team-skills/",
                )
            )
            return result

        result = ValidationResult()
        result.summary.files_scanned = len(skill_dirs)
        result.add_success(
            check_name="skill_discovery",
            message=f"Found {len(skill_dirs)} skill(s) in folder",
            skill_count=len(skill_dirs),
        )
        result.merge(self._validate_folder_compliance(skill_path))

        declared_ids: dict[str, list[Path]] = {}
        for skill_dir in skill_dirs:
            skill_result = self._validate_single_skill(skill_dir)
            metadata_id = skill_result.metadata.get("metadata_id")
            if isinstance(metadata_id, str):
                declared_ids.setdefault(metadata_id, []).append(skill_dir)
            if skill_result.passed:
                # Collect detailed check information
                check_details = []
                if skill_result.success_details:
                    for detail in skill_result.success_details:
                        check_details.append(
                            {
                                "name": detail.check_name,
                                "description": detail.message,
                                "metadata": detail.metadata,
                            }
                        )

                advisories = len(skill_result.findings)
                result.add_success(
                    check_name=skill_dir.name,
                    message=(
                        "All schema and structure checks passed"
                        if not advisories
                        else f"No blocking schema or structure findings ({advisories} advisory finding(s))"
                    ),
                    checks=check_details,
                    total_checks=len(check_details),
                    advisory_findings=advisories,
                )
                # A passing child can still carry policy advisories. Keep them
                # named after the skill they came from, without merging its
                # successes a second time; the roll-up above only says that
                # nothing blocked.
                result.merge_advisories_with_prefix(skill_result, skill_dir.name)
            else:
                result.merge_with_prefix(skill_result, skill_dir.name)
                # merge_with_prefix copies errors/warnings, but not severity
                # counters. Count findings once in this collection's summary.
                for finding in skill_result.findings:
                    result.summary.tally_severity(finding.severity)

        result.merge(self._validate_metadata_id_uniqueness(declared_ids, skill_path))
        return result

    def _validate_metadata_id_uniqueness(self, declared_ids: dict[str, list[Path]], root: Path) -> ValidationResult:
        """Report ``metadata.id`` values declared by more than one skill in this collection.

        Uniqueness is evaluated only across the skills discovered together in
        one run. It does not consult a registry or a saved catalog, whose
        identity stays keyed by content type and relative path.
        """
        result = ValidationResult()
        if not declared_ids:
            return result
        policy = self.policy
        duplicates = duplicate_metadata_ids(declared_ids, root)
        for value, relative in duplicates.items():
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=metadata_id_duplicate_severity(policy),
                    check_name="metadata_id_duplicate",
                    message=(
                        f"metadata.id '{value}' is declared by {len(relative)} skills in this collection: "
                        f"{', '.join(relative)}"
                    ),
                    file_path=str(root),
                    suggestion="Give each skill its own metadata.id.",
                    metadata={
                        "metadata_id": value,
                        "skills": relative,
                        "profile": policy.profile,
                        "metadata_id_requirement": policy.metadata_id_requirement,
                    },
                )
            )
        if not duplicates:
            count = sum(len(dirs) for dirs in declared_ids.values())
            result.add_success(
                check_name="metadata_id_unique",
                message=f"{count} declared metadata.id value(s) are unique within this collection",
                metadata_id_count=count,
            )
        return result

    def _validate_single_skill(self, skill_path: Path) -> ValidationResult:
        """Run all schema validation checks on a single skill directory."""
        result = ValidationResult()
        skill_md = self._find_skill_manifest(skill_path)

        if not skill_md:
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=Severity.HIGH,
                    check_name="manifest_exists",
                    message="SKILL.md not found in skill directory",
                    file_path=str(skill_path),
                    suggestion="Create a SKILL.md file with valid YAML frontmatter",
                )
            )
            return result

        result.add_success(
            check_name="manifest_exists",
            message=f"Found skill manifest: {skill_md.name}",
            file_path=str(skill_md),
        )

        if self._is_lowercase_manifest(skill_path):
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=Severity.HIGH,
                    check_name="manifest_naming",
                    message=(
                        "Skill manifest uses non-canonical name 'skill.md' — "
                        "the agentskills.io spec requires 'SKILL.md' (uppercase). "
                        "Codex and Hermes never load it; Claude Code, OpenCode, Copilot CLI "
                        "and OpenClaw load it only on case-insensitive filesystems (macOS), "
                        "so it can work locally and be missing on Linux"
                    ),
                    file_path=str(skill_md),
                    suggestion="Rename 'skill.md' to 'SKILL.md' to comply with the agentskills.io spec",
                )
            )

        # Frontmatter validation is prerequisite for other checks
        frontmatter_result = self._validate_frontmatter(skill_md)
        result.merge(frontmatter_result)
        if not frontmatter_result.passed:
            return result

        # Run remaining validations
        result.merge(self._validate_folder_structure(skill_path))
        result.merge(self._validate_naming_conventions(skill_path))
        result.merge(self._validate_line_count(skill_md))
        result.merge(self._validate_body_content(skill_md))
        result.merge(self._validate_optional_files(skill_path))

        # Frontmatter-dependent validations
        if frontmatter := result.metadata.get("frontmatter"):
            result.merge(self._validate_name_matches_directory(skill_path, frontmatter))
            result.merge(self._validate_author(frontmatter, str(skill_md)))
            result.merge(self._validate_metadata_id(frontmatter, str(skill_md)))
            result.merge(self._validate_portability(skill_path, skill_md, frontmatter))

        return result

    def _validate_frontmatter(self, skill_md: Path) -> ValidationResult:
        """Parse SKILL.md and validate YAML frontmatter against Pydantic schema."""
        result = ValidationResult()
        file_path = str(skill_md)

        try:
            content = skill_md.read_text(encoding="utf-8-sig")
        except Exception as e:
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=Severity.HIGH,
                    check_name="file_readable",
                    message=f"Failed to read file: {e}",
                    file_path=file_path,
                    suggestion="Check file permissions and encoding",
                )
            )
            return result

        match = FRONTMATTER_PATTERN.match(content)
        if not match:
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=Severity.HIGH,
                    check_name="frontmatter_format",
                    message="SKILL.md must have YAML frontmatter between --- markers",
                    file_path=file_path,
                    line_number=1,
                    suggestion="Add frontmatter block: start with '---', add YAML content, end with '---'",
                )
            )
            return result

        frontmatter_yaml, markdown_content = match.groups()

        try:
            data = yaml.safe_load(frontmatter_yaml)
        except yaml.YAMLError as e:
            # Extract line number from YAML error if available
            line_num = getattr(e, "problem_mark", None)
            line_number = line_num.line + 2 if line_num else 2  # +2 for --- offset
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=Severity.HIGH,
                    check_name="yaml_syntax",
                    message=f"Invalid YAML syntax in frontmatter: {e}",
                    file_path=file_path,
                    line_number=line_number,
                    suggestion="Fix YAML syntax errors (check indentation, quotes, colons)",
                )
            )
            return result

        if not data or not isinstance(data, dict):
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=Severity.HIGH,
                    check_name="frontmatter_content",
                    message="Frontmatter must be a non-empty YAML dictionary",
                    file_path=file_path,
                    line_number=2,
                    suggestion="Add required fields: name, description, globs",
                )
            )
            return result

        for key, line_number in _duplicate_frontmatter_keys(frontmatter_yaml):
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=self.policy.severity_for("SCHEMA", "duplicate_key", default=Severity.HIGH),
                    check_name="duplicate_key",
                    message=(
                        f"Duplicate frontmatter key '{key}': YAML parsers disagree on it "
                        "(Codex, OpenCode, Copilot CLI and OpenClaw drop the skill; others keep the last value)"
                    ),
                    file_path=file_path,
                    line_number=line_number,
                    suggestion=f"Keep a single '{key}' entry",
                    metadata={"key": key},
                )
            )

        try:
            frontmatter = SkillFrontmatter.model_validate(data)
            result.add_success(
                check_name="frontmatter_valid",
                message=f"Valid frontmatter for skill '{frontmatter.name}'",
                description_length=len(frontmatter.description),
            )
            result.metadata["frontmatter"] = frontmatter
            result.metadata["manifest"] = SkillManifest(
                frontmatter=frontmatter,
                content=markdown_content,
                file_path=str(skill_md),
                line_count=len(content.splitlines()),
            )
        except ValidationError as e:
            for error in e.errors():
                field = format_validation_location(error)
                result.add_finding(
                    Finding(
                        category="SCHEMA",
                        severity=Severity.HIGH,
                        check_name="frontmatter_field",
                        message=f"Field '{field}': {error['msg']}",
                        file_path=file_path,
                        suggestion=f"Fix the '{field}' field in frontmatter according to schema requirements",
                        metadata={"field": field, "error_type": error.get("type", "unknown")},
                    )
                )

        return result

    def _validate_folder_structure(self, skill_path: Path) -> ValidationResult:
        """Verify skill is in valid folder hierarchy (skills/ or team-skills/)."""
        result = ValidationResult()
        parts = skill_path.parts
        file_path = str(skill_path)

        if "skills" in parts:
            idx = parts.index("skills")
            if idx == len(parts) - 2:
                result.add_success(
                    check_name="folder_hierarchy",
                    message=f"Valid general skill structure: skills/{parts[-1]}/",
                )
            else:
                result.add_finding(
                    Finding(
                        category="SCHEMA",
                        severity=Severity.MEDIUM,
                        check_name="folder_hierarchy",
                        message="Unexpected nesting depth for general skill",
                        file_path=file_path,
                        suggestion="General skills should be at: skills/<skill-name>/",
                    ),
                    fail_on_medium=False,
                )
        elif "team-skills" in parts:
            idx = parts.index("team-skills")
            depth = len(parts) - idx - 1
            if depth >= 2:
                result.add_success(
                    check_name="folder_hierarchy",
                    message=f"Valid team skill structure: team-skills/{parts[idx + 1]}/.../{parts[-1]}/",
                    team=parts[idx + 1],
                )
            else:
                result.add_finding(
                    Finding(
                        category="SCHEMA",
                        severity=Severity.HIGH,
                        check_name="folder_hierarchy",
                        message="Invalid team-skills structure",
                        file_path=file_path,
                        suggestion="Team skills must be at: team-skills/<team>/<skill-name>/",
                    )
                )
        else:
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=Severity.MEDIUM,
                    check_name="folder_hierarchy",
                    message="Skill not in standard location (skills/ or team-skills/)",
                    file_path=file_path,
                    suggestion="Move skill to skills/<skill-name>/ or team-skills/<team>/<skill-name>/",
                ),
                fail_on_medium=False,
            )

        # Verify SKILL.md exists (case-insensitive)
        if not self._find_skill_manifest(skill_path):
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=Severity.HIGH,
                    check_name="manifest_required",
                    message=f"Missing {SKILL_MANIFEST_FILE} in skill directory",
                    file_path=file_path,
                    suggestion=f"Create a {SKILL_MANIFEST_FILE} file with valid frontmatter",
                )
            )

        return result

    def _validate_naming_conventions(self, skill_path: Path) -> ValidationResult:
        """Verify folder name follows kebab-case convention (1-64 chars)."""
        result = ValidationResult()
        name = skill_path.name
        file_path = str(skill_path)

        if not re.match(KEBAB_CASE_PATTERN, name):
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=Severity.HIGH,
                    check_name="naming_convention",
                    message=f"Folder '{name}' must be kebab-case (lowercase, numbers, hyphens)",
                    file_path=file_path,
                    suggestion=f"Rename folder to: {name.lower().replace('_', '-').replace(' ', '-')}",
                )
            )
        elif "--" in name:
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=Severity.HIGH,
                    check_name="naming_convention",
                    message=f"Folder '{name}' cannot contain consecutive hyphens",
                    file_path=file_path,
                    suggestion=f"Rename folder to: {name.replace('--', '-')}",
                )
            )
        elif name.endswith("-"):
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=Severity.HIGH,
                    check_name="naming_convention",
                    message=f"Folder '{name}' cannot end with a hyphen",
                    file_path=file_path,
                    suggestion=f"Rename folder to: {name.rstrip('-')}",
                )
            )
        elif len(name) > 64:
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=Severity.HIGH,
                    check_name="naming_convention",
                    message=f"Folder '{name}' exceeds 64 char limit ({len(name)} chars)",
                    file_path=file_path,
                    suggestion="Shorten the folder name to 64 characters or less",
                    metadata={"current_length": len(name), "max_length": 64},
                )
            )
        else:
            result.add_success(
                check_name="naming_convention",
                message=f"Folder name '{name}' follows kebab-case convention",
                name_length=len(name),
            )

        return result

    def _validate_name_matches_directory(self, skill_path: Path, frontmatter: SkillFrontmatter) -> ValidationResult:
        """Ensure directory name matches frontmatter 'name' field."""
        result = ValidationResult()
        dir_name, fm_name = skill_path.name, frontmatter.name
        file_path = str(skill_path / SKILL_MANIFEST_FILE)

        if dir_name != fm_name:
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=Severity.HIGH,
                    check_name="name_consistency",
                    message=f"Directory name '{dir_name}' doesn't match frontmatter name '{fm_name}'",
                    file_path=file_path,
                    suggestion=f"Either rename directory to '{fm_name}' or update frontmatter name to '{dir_name}'",
                    metadata={"directory_name": dir_name, "frontmatter_name": fm_name},
                )
            )
        else:
            result.add_success(
                check_name="name_consistency",
                message=f"Directory name matches frontmatter: '{fm_name}'",
            )

        return result

    def _portability_finding(
        self,
        check_name: str,
        default: Severity,
        message: str,
        file_path: str,
        suggestion: str,
        **metadata: object,
    ) -> Finding:
        return Finding(
            category="SCHEMA",
            severity=self.policy.severity_for("SCHEMA", check_name, default=default),
            check_name=check_name,
            message=message,
            file_path=file_path,
            suggestion=suggestion,
            metadata=dict(metadata),
        )

    def _validate_portability(
        self, skill_path: Path, skill_md: Path, frontmatter: SkillFrontmatter
    ) -> ValidationResult:
        """Flag spec-valid patterns that common agent harnesses load differently or not at all."""
        result = ValidationResult()
        file_path = str(skill_md)
        findings: list[Finding] = []

        if skill_md.is_symlink():
            findings.append(
                self._portability_finding(
                    "manifest_symlink",
                    Severity.MEDIUM,
                    "SKILL.md is a symlink; Codex, Hermes and OpenClaw do not load symlinked manifests",
                    file_path,
                    "Replace the symlink with a regular SKILL.md file",
                )
            )

        try:
            raw = skill_md.read_bytes()
        except OSError:
            raw = b""
        if raw.startswith(_UTF8_BOM):
            findings.append(
                self._portability_finding(
                    "manifest_bom",
                    Severity.MEDIUM,
                    "SKILL.md starts with a UTF-8 byte-order mark; Codex drops the skill",
                    file_path,
                    "Save SKILL.md as UTF-8 without BOM",
                )
            )
        for limit, level in SKILL_MD_SIZE_THRESHOLDS:
            if len(raw) > limit:
                findings.append(
                    self._portability_finding(
                        "manifest_size",
                        Severity(level),
                        f"SKILL.md is {len(raw):,} bytes (over {limit:,}); harnesses cut, refuse or skip large "
                        "manifests (OpenClaw reads 16,000-64,000 chars and skips over 256,000 bytes; "
                        "OpenCode cuts at 50 KiB)",
                        file_path,
                        "Move detail into references/ files and keep SKILL.md short",
                        size_bytes=len(raw),
                        limit_bytes=limit,
                    )
                )
                break

        # Same folders the harness check skips: evals/ and versions/ are scanned like any other folder.
        nested = sorted(
            path
            for name in SKILL_MANIFEST_VARIANTS
            for path in skill_path.rglob(name)
            if path.parent != skill_path
            and not any(part in SKILL_NOT_SHIPPED_DIRS for part in path.relative_to(skill_path).parts)
        )
        for path in nested:
            findings.append(
                self._portability_finding(
                    "nested_manifest",
                    Severity.MEDIUM,
                    f"Nested skill manifest '{path.relative_to(skill_path).as_posix()}' inside this skill; "
                    "Codex, OpenCode and Hermes load it as a separate skill, others ignore it",
                    str(path),
                    "Rename the nested file or move it out of the skill folder",
                )
            )

        description = frontmatter.description
        utf16_units = len(description.encode("utf-16-le", "surrogatepass")) // 2
        if len(description) <= DESCRIPTION_MAX_LENGTH < utf16_units:
            findings.append(
                self._portability_finding(
                    "description_utf16_length",
                    Severity.MEDIUM,
                    f"Description is {len(description)} characters but {utf16_units} UTF-16 units; "
                    f"Copilot CLI counts UTF-16 units and drops descriptions over {DESCRIPTION_MAX_LENGTH}",
                    file_path,
                    "Shorten the description or replace emoji and other non-BMP characters",
                    utf16_units=utf16_units,
                )
            )

        trigger = _TRIGGER_RE.search(description)
        if trigger is None or trigger.end() > DESCRIPTION_TRIGGER_WINDOW:
            where = (
                "no 'Use when ...' trigger"
                if trigger is None
                else f"its trigger at characters {trigger.start()}-{trigger.end()}"
            )
            findings.append(
                self._portability_finding(
                    "description_trigger",
                    Severity.LOW,
                    f"Description has {where}; some harnesses show only the first "
                    f"~{DESCRIPTION_TRIGGER_WINDOW} characters in their skill list",
                    file_path,
                    f"State when to use the skill within the first {DESCRIPTION_TRIGGER_WINDOW} characters",
                )
            )

        unknown = sorted(set(frontmatter.model_extra or {}) - SPEC_SKILL_FIELDS)
        if unknown:
            findings.append(
                self._portability_finding(
                    "unknown_field",
                    Severity.LOW,
                    f"Frontmatter field(s) outside the agentskills.io spec: {', '.join(unknown)}; "
                    "agent harnesses do not pass them to the model",
                    file_path,
                    "Remove them or move them under 'metadata'",
                    fields=unknown,
                )
            )

        for finding in findings:
            result.add_finding(finding)
        return result

    def _validate_line_count(self, skill_md: Path) -> ValidationResult:
        """Warn if SKILL.md exceeds recommended line count."""
        result = ValidationResult()
        file_path = str(skill_md)

        try:
            line_count = len(skill_md.read_text(encoding="utf-8-sig").splitlines())
            if line_count > MAX_SKILL_MD_LINES:
                result.add_finding(
                    Finding(
                        category="SCHEMA",
                        severity=Severity.MEDIUM,
                        check_name="line_count",
                        message=f"SKILL.md has {line_count} lines (limit: {MAX_SKILL_MD_LINES})",
                        file_path=file_path,
                        suggestion="Consider moving content to reference files in references/ directory",
                        metadata={"line_count": line_count, "limit": MAX_SKILL_MD_LINES},
                    ),
                    fail_on_medium=False,
                )
            else:
                result.add_success(
                    check_name="line_count",
                    message=f"SKILL.md within line limit ({line_count}/{MAX_SKILL_MD_LINES})",
                    line_count=line_count,
                    limit=MAX_SKILL_MD_LINES,
                )
        except Exception as e:
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=Severity.LOW,
                    check_name="line_count",
                    message=f"Could not check line count: {e}",
                    file_path=file_path,
                ),
                fail_on_low=False,
            )

        return result

    _INSTRUCTIONS_ALTERNATIVES = ("## Instructions", "## Usage")

    def _validate_body_content(self, skill_md: Path) -> ValidationResult:
        """Validate SKILL.md body has required heading and sections.

        Per agentskills.io spec, the body has no format restrictions, so we
        only hard-fail on missing top-level title. Section headings are
        convention nudges:
        - At least one top-level heading (# Title) — HIGH if missing
        - Recommended sections: ## Instructions (or ## Usage), ## Examples
          — MEDIUM if missing
        """
        result = ValidationResult()
        file_path = str(skill_md)

        try:
            content = skill_md.read_text(encoding="utf-8-sig")
        except Exception:
            return result

        match = FRONTMATTER_PATTERN.match(content)
        body = match.group(2) if match else content

        if not re.search(r"^# .+", body, re.MULTILINE):
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=Severity.HIGH,
                    check_name="body_heading",
                    message="Missing top-level heading (# Title) in SKILL.md body",
                    file_path=file_path,
                    suggestion="Add a top-level heading after the frontmatter, e.g. '# My Skill'",
                )
            )
        else:
            result.add_success(
                check_name="body_heading",
                message="Body contains a top-level heading",
            )

        for section in REQUIRED_BODY_SECTIONS:
            if section not in body:
                result.add_finding(
                    Finding(
                        category="SCHEMA",
                        severity=Severity.HIGH,
                        check_name="body_required_section",
                        message=f"Missing required section: '{section}'",
                        file_path=file_path,
                        suggestion=f"Add a '{section}' section to the SKILL.md body",
                        metadata={"missing_section": section},
                    )
                )
            else:
                result.add_success(
                    check_name="body_required_section",
                    message=f"Found required section: '{section}'",
                )

        for section in RECOMMENDED_BODY_SECTIONS:
            if section == "## Instructions":
                if any(alt in body for alt in self._INSTRUCTIONS_ALTERNATIVES):
                    result.add_success(
                        check_name="body_recommended_section",
                        message="Found recommended section: '## Instructions' (or '## Usage')",
                    )
                else:
                    result.add_finding(
                        Finding(
                            category="SCHEMA",
                            severity=Severity.MEDIUM,
                            check_name="body_recommended_section",
                            message=f"Missing recommended section: '{section}'",
                            file_path=file_path,
                            suggestion=(
                                "Consider adding a '## Instructions' or '## Usage' section. "
                                "Per agentskills.io the body format is unrestricted, so this is "
                                "a convention nudge — it also gives the quality scorer a stable "
                                "anchor for instruction-quality heuristics."
                            ),
                            metadata={"missing_section": section},
                        )
                    )
            elif section not in body:
                result.add_finding(
                    Finding(
                        category="SCHEMA",
                        severity=Severity.MEDIUM,
                        check_name="body_recommended_section",
                        message=f"Missing recommended section: '{section}'",
                        file_path=file_path,
                        suggestion=(
                            f"Consider adding a '{section}' section. "
                            "If examples are already inline under instructions, "
                            "this can be skipped."
                        ),
                        metadata={"missing_section": section},
                    )
                )
            else:
                result.add_success(
                    check_name="body_recommended_section",
                    message=f"Found recommended section: '{section}'",
                )

        return result

    def _validate_author(self, frontmatter: SkillFrontmatter, file_path: str = "") -> ValidationResult:
        """Classify ``metadata.author`` and apply the active policy.

        Four outcomes, each with its own check name so a policy can set its
        severity independently:

        - ``author_missing``: no author (default policy: LOW, advisory).
        - ``author_without_email``: a person, team, or organization name with
          no email address (default policy: INFO).
        - ``author_format``: a malformed or unsafe value, such as a bare email,
          stray angle brackets, or control characters, or an email outside the
          policy's ``author_email_regex`` domain (HIGH, blocking).
        - success: ``Name <email@host>`` that satisfies any domain restriction.

        A policy with an ``author_email_regex`` needs an email in that domain,
        so a name without an email fails its domain check.
        """
        result = ValidationResult()
        policy = self.policy
        author = (frontmatter.metadata.author or "").strip() if frontmatter.metadata else ""

        if not author:
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=policy.severity_for("SCHEMA", "author_missing", default=Severity.LOW),
                    check_name="author_missing",
                    message="Author not specified in metadata",
                    file_path=file_path,
                    suggestion=(
                        "Add 'metadata.author' with a person, team, or organization name, "
                        "or 'Name <email@example.com>' to include a contact address"
                    ),
                    metadata={"profile": policy.profile},
                )
            )
            return result

        problem = author_value_problem(author)
        shape_ok = problem is None and policy.author_shape_regex.fullmatch(author) is not None
        domain_ok = shape_ok and policy.is_author_email_acceptable(author)
        finding_metadata = {
            "current_author": author[:AUTHOR_MAX_LENGTH],
            "profile": policy.profile,
            "shape_ok": shape_ok,
            "domain_ok": domain_ok,
        }

        if problem is not None:
            message = f"Author {problem}"
            suggestion = "Use a plain name such as 'Example Team', or 'Full Name <email@example.com>'"
        elif shape_ok and domain_ok:
            result.add_success(check_name="author_format", message=f"Valid author format: {author}")
            return result
        elif policy.author_email_regex is not None:
            domain_pattern = policy.author_email_regex.pattern
            subject = "Author email does not match" if shape_ok else "Author has no email address matching"
            message = f"{subject} the required domain (profile '{policy.profile}', pattern: {domain_pattern})"
            suggestion = (
                "Use 'Name <email>' with an email matching the active profile's author_email_regex, "
                "or review whether that domain restriction is appropriate for your repository."
            )
        else:
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=policy.severity_for("SCHEMA", "author_without_email", default=Severity.INFO),
                    check_name="author_without_email",
                    message=f"Author has no email address: {author}",
                    file_path=file_path,
                    line_content=f"author: {author}",
                    suggestion=(
                        "A name is accepted as attribution unless the profile requires a contact address "
                        "(such as 'registry'); then use 'Name <email@example.com>'"
                    ),
                    metadata=finding_metadata,
                )
            )
            return result

        result.add_finding(
            Finding(
                category="SCHEMA",
                severity=policy.severity_for("SCHEMA", "author_format", default=Severity.HIGH),
                check_name="author_format",
                message=message,
                file_path=file_path,
                line_content=f"author: {author[:AUTHOR_MAX_LENGTH]}",
                suggestion=suggestion,
                metadata=finding_metadata,
            )
        )
        return result

    def _validate_metadata_id(self, frontmatter: SkillFrontmatter, file_path: str = "") -> ValidationResult:
        """Validate an optional ``metadata.id`` against the stable-ID contract.

        A present ID must be an exact 1-64 character lowercase ASCII slug in
        every policy mode. The policy's ``metadata_id_requirement`` decides
        whether absence is ignored, recommended (LOW), or required (HIGH), and
        whether an invalid ID blocks (``require``) or is advisory (MEDIUM).
        A valid ID is stored in ``result.metadata['metadata_id']`` for the
        collection-level uniqueness check.
        """
        result = ValidationResult()
        policy = self.policy
        requirement = policy.metadata_id_requirement
        extra = frontmatter.metadata.model_extra if frontmatter.metadata is not None else None
        details = {"profile": policy.profile, "metadata_id_requirement": requirement}

        if not extra or "id" not in extra:
            if requirement == "ignore":
                return result
            default = Severity.HIGH if requirement == "require" else Severity.LOW
            verb = "requires" if requirement == "require" else "recommends"
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=policy.severity_for("SCHEMA", "metadata_id_missing", default=default),
                    check_name="metadata_id_missing",
                    message=f"metadata.id not specified; profile '{policy.profile}' {verb} a stable skill ID",
                    file_path=file_path,
                    suggestion="Add 'metadata.id' with a lowercase slug such as 'my-skill' that stays fixed across versions",
                    metadata=details,
                )
            )
            return result

        value = extra["id"]
        problem = metadata_id_problem(value)
        if problem is not None:
            default = Severity.HIGH if requirement == "require" else Severity.MEDIUM
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=policy.severity_for("SCHEMA", "metadata_id_format", default=default),
                    check_name="metadata_id_format",
                    message=f"metadata.id {problem}",
                    file_path=file_path,
                    line_content=f"id: {_display_metadata_id(value)}",
                    suggestion="Use a lowercase slug such as 'my-skill' (1-64 characters: a-z, 0-9, single hyphens)",
                    metadata=details,
                )
            )
            return result

        result.add_success(
            check_name="metadata_id_format",
            message=(
                f"Valid metadata.id '{value}'. Uniqueness is checked only across skills validated together "
                "in one collection run"
            ),
            metadata_id=value,
        )
        result.metadata["metadata_id"] = value
        return result

    def _validate_folder_compliance(self, folder_path: Path) -> ValidationResult:
        """Check if folder contains standard SkillEvaluator directories (skills/, team-skills/)."""
        result = ValidationResult()
        has_skills = (folder_path / "skills").exists()
        has_team_skills = (folder_path / "team-skills").exists()
        file_path = str(folder_path)

        if has_skills or has_team_skills:
            result.add_success(
                check_name="folder_structure",
                message="Folder structure compliant with SkillEvaluator guidelines",
            )
            if has_skills:
                count = len(list((folder_path / "skills").iterdir()))
                result.add_success(
                    check_name="skills_directory",
                    message=f"Found 'skills/' directory with {count} items",
                    item_count=count,
                )
            if has_team_skills:
                count = len(list((folder_path / "team-skills").iterdir()))
                result.add_success(
                    check_name="team_skills_directory",
                    message=f"Found 'team-skills/' directory with {count} teams",
                    team_count=count,
                )
        elif "skills" in folder_path.parts or "team-skills" in folder_path.parts:
            result.add_success(
                check_name="folder_structure",
                message="Validating skills within standard folder structure",
            )
        else:
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=Severity.MEDIUM,
                    check_name="folder_structure",
                    message="Folder doesn't follow SkillEvaluator structure (missing skills/ or team-skills/)",
                    file_path=file_path,
                    suggestion="Create skills/ and/or team-skills/ directories for organizing skills",
                ),
                fail_on_medium=False,
            )

        return result

    @staticmethod
    def _allowed_skill_dirs() -> set[str]:
        """Allowed skill-root subdirectories: built-in defaults plus per-repo extras.

        Recognised skill-root subdirectories (``DEFAULT_ALLOWED_SKILL_DIRS``):
        - ``agents/`` — recommended agent-facing UI metadata.
        - ``references/``, ``scripts/``, ``assets/`` — SkillEvaluator canonical content dirs.
        - ``evals/`` — canonical location for Tier 3 evaluation datasets
          (``evals/evals.json``, optional ``evals/EVAL.md``, and Harbor outputs
          under ``evals/results/``).
        - ``tests/`` — skill-local verification discovered by repository tooling.
        - ``tools/`` — agentskills.io canonical name for executable helpers
          (the skill_evaluator onboarding renames ``scripts/`` -> ``tools/``).
        - ``config/`` — data-driven runtime config (JSON/YAML the skill loads).

        Consumers can EXTEND (never shrink) this set per-repo via the
        ``SKILLEVALUATOR_SCHEMA_ALLOWED_DIRS`` env var (comma- or whitespace-separated),
        so a team with e.g. a ``data/`` directory can clear the otherwise
        unavoidable LOW finding without forking the validator.
        """
        dirs = set(DEFAULT_ALLOWED_SKILL_DIRS)
        extra = os.environ.get(SCHEMA_ALLOWED_DIRS_ENV, "")
        dirs.update(name for name in re.split(r"[,\s]+", extra) if name)
        return dirs

    def _validate_optional_files(self, skill_path: Path) -> ValidationResult:
        """Check for optional supporting files and warn about unexpected items."""
        result = ValidationResult()
        allowed_dirs = self._allowed_skill_dirs()
        optional = {"README.md", *allowed_dirs}
        expected = optional | {"SKILL.md", "skill.md"}

        found = [f for f in sorted(optional) if (skill_path / f).exists()]
        if found:
            result.add_success(
                check_name="optional_files",
                message=f"Found optional supporting files: {', '.join(found)}",
                files_found=found,
            )

        # Warn about unexpected files
        allowed_hint = ", ".join(f"{d}/" for d in sorted(allowed_dirs))
        unexpected_items = []
        for item in skill_path.iterdir():
            if matches_filesystem_name(item, SCAN_EXCLUDED_FILES):
                continue
            if item.name not in expected and not item.name.startswith("."):
                unexpected_items.append(item.name)
                result.add_finding(
                    Finding(
                        category="SCHEMA",
                        severity=Severity.LOW,
                        check_name="unexpected_file",
                        message=f"Unexpected '{item.name}' in skill root",
                        file_path=str(item),
                        suggestion=(
                            f"Consider moving to one of: {allowed_hint}. To allow "
                            f"additional directories, set ${SCHEMA_ALLOWED_DIRS_ENV}."
                        ),
                    ),
                    fail_on_low=False,
                )

        if not found and not unexpected_items:
            result.add_success(
                check_name="optional_files",
                message="Skill directory contains only expected files",
            )

        return result


def parse_skill_manifest(skill_md: Path) -> SkillManifest | None:
    """Parse SKILL.md file into SkillManifest (utility function)."""
    validator = SchemaValidator()
    result = validator._validate_frontmatter(skill_md)
    return result.metadata.get("manifest") if result.passed else None
