# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema and Repository Governance Validator.

Validates SKILL.md frontmatter against schema, enforces folder hierarchy
and naming conventions per Skill Evaluator specification.
"""

import os
import re
from pathlib import Path

import yaml
from pydantic import ValidationError

from skillevaluator.constants import (
    DEFAULT_ALLOWED_SKILL_DIRS,
    KEBAB_CASE_PATTERN,
    MAX_SKILL_MD_LINES,
    RECOMMENDED_BODY_SECTIONS,
    REQUIRED_BODY_SECTIONS,
    SCAN_EXCLUDED_FILES,
    SCHEMA_ALLOWED_DIRS_ENV,
    SKILL_MANIFEST_FILE,
)
from skillevaluator.logging_config import get_logger
from skillevaluator.models.result import Finding, Severity
from skillevaluator.models.skill import SkillFrontmatter, SkillManifest
from skillevaluator.validators.base import ValidationResult, ValidatorBase
from skillevaluator.validators.frontmatter_parser import FRONTMATTER_PATTERN
from skillevaluator.validators.policy import ValidationPolicy, default_policy

logger = get_logger(__name__)


class SchemaValidator(ValidatorBase):
    """Validates skill schema and repository structure per Skill Evaluator spec.

    Checks: frontmatter schema, folder hierarchy, naming conventions,
    directory-name consistency, required/forbidden fields.

    Author attribution is policy-driven: the ``Name <email>`` *shape* is always
    enforced, while an optional email-domain restriction can come from the
    active :class:`ValidationPolicy`.
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

        for skill_dir in skill_dirs:
            skill_result = self._validate_single_skill(skill_dir)
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

                result.add_success(
                    check_name=skill_dir.name,
                    message="All schema and structure checks passed",
                    checks=check_details,
                    total_checks=len(check_details),
                )
            else:
                result.merge_with_prefix(skill_result, skill_dir.name)

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
                        "On case-sensitive filesystems (Linux), this will not be "
                        "recognized by spec-compliant tooling"
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

        return result

    def _validate_frontmatter(self, skill_md: Path) -> ValidationResult:
        """Parse SKILL.md and validate YAML frontmatter against Pydantic schema."""
        result = ValidationResult()
        file_path = str(skill_md)

        try:
            content = skill_md.read_text(encoding="utf-8")
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

        try:
            frontmatter = SkillFrontmatter(**data)
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
                field = ".".join(str(loc) for loc in error["loc"])
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

    def _validate_line_count(self, skill_md: Path) -> ValidationResult:
        """Warn if SKILL.md exceeds recommended line count."""
        result = ValidationResult()
        file_path = str(skill_md)

        try:
            line_count = len(skill_md.read_text(encoding="utf-8").splitlines())
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
            content = skill_md.read_text(encoding="utf-8")
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
        """Require author field with name + email; domain restriction is policy-driven.

        Splits the historical single-regex check into two orthogonal pieces:

        1. *Shape* — every author value must match ``Name <email@host>``,
           regardless of profile.
        2. *Domain* — only enforced when the active policy supplies an
           ``author_email_regex`` (the ``internal`` profile pins authors to
        an organization-specific domain; the default public profile leaves it
        open).

        Severity for both ``author_missing`` and ``author_format`` comes from
        the active policy with internal-equivalent defaults.
        """
        result = ValidationResult()
        policy = self.policy

        if frontmatter.metadata and frontmatter.metadata.author:
            author = frontmatter.metadata.author.strip()
            shape_ok = policy.author_shape_regex.fullmatch(author) is not None
            domain_ok = policy.is_author_email_acceptable(author)

            if shape_ok and domain_ok:
                result.add_success(
                    check_name="author_format",
                    message=f"Valid author format: {author}",
                )
            else:
                if not shape_ok:
                    message = "Author must be of the form 'Name <email@host>'"
                    suggestion = "Use format: 'Full Name <email@example.com>'"
                else:
                    domain_pattern = policy.author_email_regex.pattern if policy.author_email_regex is not None else ""
                    message = (
                        f"Author email does not match required domain "
                        f"(profile '{policy.profile}', pattern: {domain_pattern})"
                    )
                    suggestion = (
                        "Use an email matching the active profile's author_email_regex, "
                        "or run with --profile external if this skill is for public publication."
                    )

                result.add_finding(
                    Finding(
                        category="SCHEMA",
                        severity=policy.severity_for("SCHEMA", "author_format", default=Severity.HIGH),
                        check_name="author_format",
                        message=message,
                        file_path=file_path,
                        line_content=f"author: {author}",
                        suggestion=suggestion,
                        metadata={
                            "current_author": author,
                            "profile": policy.profile,
                            "shape_ok": shape_ok,
                            "domain_ok": domain_ok,
                        },
                    )
                )
        else:
            result.add_finding(
                Finding(
                    category="SCHEMA",
                    severity=policy.severity_for("SCHEMA", "author_missing", default=Severity.HIGH),
                    check_name="author_missing",
                    message="Author not specified in metadata",
                    file_path=file_path,
                    suggestion="Add 'metadata.author' field with format 'Name <email@example.com>'",
                    metadata={"profile": policy.profile},
                )
            )

        return result

    def _validate_folder_compliance(self, folder_path: Path) -> ValidationResult:
        """Check if folder contains standard Skill Evaluator directories (skills/, team-skills/)."""
        result = ValidationResult()
        has_skills = (folder_path / "skills").exists()
        has_team_skills = (folder_path / "team-skills").exists()
        file_path = str(folder_path)

        if has_skills or has_team_skills:
            result.add_success(
                check_name="folder_structure",
                message="Folder structure compliant with Skill Evaluator guidelines",
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
                    message="Folder doesn't follow Skill Evaluator structure (missing skills/ or team-skills/)",
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
        - ``references/``, ``scripts/``, ``assets/`` — Skill Evaluator canonical content dirs.
        - ``evals/`` — canonical location for Tier 3 evaluation datasets
          (``evals/evals.json``, optional ``evals/EVAL.md``, and Harbor outputs
          under ``evals/results/``).
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
            if item.name.lower() in SCAN_EXCLUDED_FILES:
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
