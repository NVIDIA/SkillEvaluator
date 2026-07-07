# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rules Schema and Repository Governance Validator.

Validates .mdc rule files against schema, enforces folder hierarchy
and naming conventions per Skill Evaluator specification for Rules.

Key differences from Skills validator:
- Rules use .mdc extension (not SKILL.md)
- Rules REQUIRE alwaysApply and allow globs (Skills FORBID these)
- Rules are in team-rules/ folder
"""

from pathlib import Path

from skillevaluator.constants import MAX_RULES_MDC_LINES, RULES_FILE_EXTENSION
from skillevaluator.logging_config import get_logger
from skillevaluator.models.rules import RulesFrontmatter, RulesManifest
from skillevaluator.validators.base import ValidationResult, ValidatorBase
from skillevaluator.validators.frontmatter_parser import (
    parse_frontmatter,
    validate_pydantic_model,
)
from skillevaluator.validators.naming_utils import (
    NamingValidationConfig,
    validate_kebab_case_name,
    validate_line_count,
)

logger = get_logger(__name__)


class RulesSchemaValidator(ValidatorBase):
    """Validates Rules .mdc schema and repository structure per Skill Evaluator spec.

    Checks: frontmatter schema, folder hierarchy, naming conventions,
    required fields (alwaysApply, title, description).
    """

    @property
    def name(self) -> str:
        return "Rules Schema & Repository Governance"

    @property
    def description(self) -> str:
        return "Validate .mdc rule files and team-rules/ structure"

    def validate(self, rules_path: Path) -> ValidationResult:
        """Validate rule(s) at path for schema and structure compliance."""
        if self._is_rules_file(rules_path):
            return self._validate_single_rule(rules_path)

        if self._is_rules_directory(rules_path):
            return self._validate_rules_directory(rules_path)

        # Folder-level validation - find all .mdc files
        mdc_files = self._find_all_rules(rules_path)
        if not mdc_files:
            result = ValidationResult()
            result.add_error(f"No .mdc rule files found in {rules_path}. Expected .mdc files in team-rules/ structure.")
            return result

        result = ValidationResult()
        result.add_message(f"Found {len(mdc_files)} rule file(s) in folder")
        result.merge(self._validate_folder_compliance(rules_path))

        for mdc_file in mdc_files:
            rule_result = self._validate_single_rule(mdc_file)
            rule_name = mdc_file.stem  # filename without extension

            if rule_result.passed:
                result.add_success(
                    check_name=rule_name,
                    message="passed",
                )
            else:
                result.merge_with_prefix(rule_result, rule_name)

        return result

    def _is_rules_file(self, path: Path) -> bool:
        """Check if path is a .mdc rule file."""
        return path.is_file() and path.suffix == RULES_FILE_EXTENSION

    def _is_rules_directory(self, path: Path) -> bool:
        """Check if path is a directory containing .mdc files directly."""
        if not path.is_dir():
            return False
        return any(f.suffix == RULES_FILE_EXTENSION for f in path.iterdir() if f.is_file())

    def _find_all_rules(self, root_path: Path) -> list[Path]:
        """Recursively find all .mdc rule files."""
        return sorted(root_path.rglob(f"*{RULES_FILE_EXTENSION}"))

    def _validate_single_rule(self, rule_path: Path) -> ValidationResult:
        """Run all schema validation checks on a single .mdc file."""
        result = ValidationResult()

        if not rule_path.exists():
            result.add_error(f"Rule file not found: {rule_path}")
            return result

        if rule_path.suffix != RULES_FILE_EXTENSION:
            result.add_error(f"Rule file must have {RULES_FILE_EXTENSION} extension")
            return result

        result.add_message(f"Found rule file: {rule_path}")

        # Frontmatter validation is prerequisite for other checks
        frontmatter_result = self._validate_frontmatter(rule_path)
        result.merge(frontmatter_result)
        if not frontmatter_result.passed:
            return result

        # Run remaining validations
        result.merge(self._validate_folder_structure(rule_path))
        result.merge(self._validate_naming_conventions(rule_path))
        result.merge(self._validate_line_count(rule_path))

        # Frontmatter-dependent validations
        if frontmatter := result.metadata.get("frontmatter"):
            result.merge(self._validate_metadata_fields(frontmatter))

        return result

    def _validate_rules_directory(self, rules_dir: Path) -> ValidationResult:
        """Validate a directory containing .mdc files."""
        result = ValidationResult()
        mdc_files = [f for f in rules_dir.iterdir() if f.suffix == RULES_FILE_EXTENSION]

        result.add_message(f"Found {len(mdc_files)} rule file(s) in {rules_dir.name}/")

        for mdc_file in sorted(mdc_files):
            rule_result = self._validate_single_rule(mdc_file)
            rule_name = mdc_file.stem

            if rule_result.passed:
                result.add_success(
                    check_name=rule_name,
                    message="passed",
                )
            else:
                result.merge_with_prefix(rule_result, rule_name)

        return result

    def _validate_frontmatter(self, rule_path: Path) -> ValidationResult:
        """Parse .mdc file and validate YAML frontmatter against Pydantic schema."""
        parsed, result = parse_frontmatter(rule_path)
        if parsed is None:
            return result

        frontmatter = validate_pydantic_model(RulesFrontmatter, parsed.yaml_data, result)
        if frontmatter is None:
            return result

        result.add_message(f"Rule title: {frontmatter.title}")
        result.add_message(f"alwaysApply: {frontmatter.alwaysApply}")
        result.add_message(f"Description length: {len(frontmatter.description)} chars")

        result.metadata["frontmatter"] = frontmatter
        result.metadata["manifest"] = RulesManifest(
            frontmatter=frontmatter,
            content=parsed.content,
            file_path=str(rule_path),
            line_count=parsed.line_count,
        )

        return result

    def _validate_folder_structure(self, rule_path: Path) -> ValidationResult:
        """Verify rule is in valid folder hierarchy (team-rules/)."""
        result = ValidationResult()
        parts = rule_path.parts

        if "team-rules" in parts:
            idx = parts.index("team-rules")
            # team-rules/<team-name>/ or team-rules/<team-name>/<project>/
            depth = len(parts) - idx - 2  # -2 for team-rules and filename
            if depth >= 1:
                team_name = parts[idx + 1]
                result.add_message(f"Valid team rules structure: team-rules/{team_name}/...")
            else:
                result.add_error("Invalid team-rules structure. Expected: team-rules/<team-name>/<rule>.mdc")
        else:
            result.add_warning(f"Rule not in standard location (team-rules/): {rule_path.parent}")

        return result

    def _validate_naming_conventions(self, rule_path: Path) -> ValidationResult:
        """Verify filename follows kebab-case convention."""
        config = NamingValidationConfig(
            name_type="Filename",
            use_errors=False,  # Rules use warnings for naming issues
        )
        return validate_kebab_case_name(rule_path.stem, config)

    def _validate_line_count(self, rule_path: Path) -> ValidationResult:
        """Warn if rule file exceeds recommended line count."""
        try:
            line_count = len(rule_path.read_text(encoding="utf-8").splitlines())
            return validate_line_count(line_count, MAX_RULES_MDC_LINES, "Rule file")
        except Exception as e:
            result = ValidationResult()
            result.add_warning(f"Could not check line count: {e}")
            return result

    def _validate_metadata_fields(self, frontmatter: RulesFrontmatter) -> ValidationResult:
        """Check for recommended metadata fields."""
        result = ValidationResult()

        if frontmatter.metadata:
            meta = frontmatter.metadata
            if meta.tags:
                result.add_message(f"Tags: {', '.join(meta.tags[:5])}")
            else:
                result.add_warning("Consider adding tags for better discoverability")

            if meta.team:
                result.add_message(f"Team: {meta.team}")
            if meta.project:
                result.add_message(f"Project: {meta.project}")
            if meta.language:
                result.add_message(f"Language: {meta.language}")
        else:
            result.add_warning("Consider adding metadata (tags, language, team, project) for better categorization")

        return result

    def _validate_folder_compliance(self, folder_path: Path) -> ValidationResult:
        """Check if folder follows Skill Evaluator team-rules/ structure."""
        result = ValidationResult()
        has_team_rules = (folder_path / "team-rules").exists()

        if has_team_rules:
            result.add_message("Folder structure compliant with Skill Evaluator guidelines")
            team_rules_dir = folder_path / "team-rules"
            teams = [d for d in team_rules_dir.iterdir() if d.is_dir()]
            result.add_message(f"Found 'team-rules/' directory with {len(teams)} team(s)")
        elif "team-rules" in folder_path.parts:
            result.add_message("Validating rules within team-rules/ structure")
        else:
            result.add_warning("Folder doesn't follow Skill Evaluator structure (expected team-rules/ directory)")

        return result


def parse_rules_manifest(rule_path: Path) -> RulesManifest | None:
    """Parse .mdc rule file into RulesManifest (utility function)."""
    validator = RulesSchemaValidator()
    result = validator._validate_frontmatter(rule_path)
    return result.metadata.get("manifest") if result.passed else None
