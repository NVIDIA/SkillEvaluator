# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflows Schema and Structure Validator.

Validates workflow directories against Skill Evaluator specification.
Workflows are directory-based structures containing:
- README.md (required)
- workflow-rules.mdc (required)
- references/ directory with .mdc files (required)
- references/scripts/ directory (optional)
"""

import re
from pathlib import Path

from skillevaluator.constants import (
    MAX_WORKFLOWS_MDC_LINES,
    WORKFLOWS_MANIFEST_FILE,
    WORKFLOWS_README_FILE,
    WORKFLOWS_REFERENCES_DIR,
    WORKFLOWS_SCRIPTS_DIR,
)
from skillevaluator.logging_config import get_logger
from skillevaluator.models.workflows import (
    ReferenceFrontmatter,
    WorkflowsFrontmatter,
    WorkflowsManifest,
    WorkflowsStructure,
)
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


class WorkflowsSchemaValidator(ValidatorBase):
    """Validates Workflows directory structure and content per Skill Evaluator spec.

    Checks:
    - Directory structure (README.md, workflow-rules.mdc, references/)
    - Main workflow-rules.mdc frontmatter
    - Reference .mdc files in references/
    - Optional scripts in references/scripts/
    """

    @property
    def name(self) -> str:
        return "Workflows Schema & Structure"

    @property
    def description(self) -> str:
        return "Validate workflow directories and their components"

    def validate(self, workflows_path: Path) -> ValidationResult:
        """Validate workflow(s) at path for schema and structure compliance."""
        if self._is_workflow_directory(workflows_path):
            return self._validate_single_workflow(workflows_path)

        # Folder-level validation - find all workflow directories
        workflow_dirs = self._find_all_workflows(workflows_path)
        if not workflow_dirs:
            result = ValidationResult()
            result.add_error(
                f"No workflow directories found in {workflows_path}. "
                "Expected directories containing workflow-rules.mdc."
            )
            return result

        result = ValidationResult()
        result.add_message(f"Found {len(workflow_dirs)} workflow(s) in folder")
        result.merge(self._validate_folder_compliance(workflows_path))

        for workflow_dir in workflow_dirs:
            workflow_result = self._validate_single_workflow(workflow_dir)
            workflow_name = workflow_dir.name

            if workflow_result.passed:
                result.add_success(
                    check_name=workflow_name,
                    message="passed",
                )
            else:
                result.merge_with_prefix(workflow_result, workflow_name)

        return result

    def _is_workflow_directory(self, path: Path) -> bool:
        """Check if path is a workflow directory (contains workflow-rules.mdc)."""
        return path.is_dir() and (path / WORKFLOWS_MANIFEST_FILE).exists()

    def _find_all_workflows(self, root_path: Path) -> list[Path]:
        """Recursively find all workflow directories."""
        workflow_dirs: set[Path] = set()

        for manifest in root_path.rglob(WORKFLOWS_MANIFEST_FILE):
            workflow_dirs.add(manifest.parent)

        return sorted(workflow_dirs)

    def _validate_single_workflow(self, workflow_path: Path) -> ValidationResult:
        """Run all validation checks on a single workflow directory."""
        result = ValidationResult()

        if not workflow_path.is_dir():
            result.add_error(f"Workflow path must be a directory: {workflow_path}")
            return result

        result.add_message(f"Validating workflow: {workflow_path.name}")

        # Structure validation
        result.merge(self._validate_directory_structure(workflow_path))
        if not result.passed:
            return result  # Structure must pass before content validation

        # Validate main workflow-rules.mdc
        manifest_path = workflow_path / WORKFLOWS_MANIFEST_FILE
        result.merge(self._validate_workflow_manifest(manifest_path))

        # Validate folder location
        result.merge(self._validate_folder_structure(workflow_path))

        # Validate naming conventions
        result.merge(self._validate_naming_conventions(workflow_path))

        # Validate references
        refs_dir = workflow_path / WORKFLOWS_REFERENCES_DIR
        if refs_dir.exists():
            result.merge(self._validate_references(refs_dir, workflow_path.name))

        # Validate scripts (if present)
        scripts_dir = refs_dir / WORKFLOWS_SCRIPTS_DIR
        if scripts_dir.exists():
            result.merge(self._validate_scripts(scripts_dir))

        return result

    def _validate_directory_structure(self, workflow_path: Path) -> ValidationResult:
        """Verify workflow has required directory structure."""
        result = ValidationResult()
        structure = WorkflowsStructure.from_path(workflow_path)
        is_valid, errors = structure.validate_structure(workflow_path)

        if is_valid:
            result.add_message("Directory structure is valid")

            # Count components
            readme = workflow_path / WORKFLOWS_README_FILE
            refs_dir = workflow_path / WORKFLOWS_REFERENCES_DIR
            scripts_dir = refs_dir / WORKFLOWS_SCRIPTS_DIR

            if readme.exists():
                result.add_message(f"Found {WORKFLOWS_README_FILE}")
            if refs_dir.exists():
                ref_files = list(refs_dir.glob("*.mdc"))
                result.add_message(f"Found {len(ref_files)} reference file(s)")
            if scripts_dir.exists():
                script_files = list(scripts_dir.iterdir())
                result.add_message(f"Found {len(script_files)} script file(s)")
        else:
            for error in errors:
                result.add_error(error)

        return result

    def _validate_workflow_manifest(self, manifest_path: Path) -> ValidationResult:
        """Parse and validate workflow-rules.mdc frontmatter."""
        result = ValidationResult()

        if not manifest_path.exists():
            result.add_error(f"Missing {WORKFLOWS_MANIFEST_FILE}")
            return result

        parsed, parse_result = parse_frontmatter(manifest_path)
        result.merge(parse_result)
        if parsed is None:
            return result

        frontmatter = validate_pydantic_model(WorkflowsFrontmatter, parsed.yaml_data, result)
        if frontmatter is None:
            return result

        result.add_message(f"Workflow title: {frontmatter.title}")
        result.add_message(f"Author: {frontmatter.metadata.author}")
        result.add_message(f"alwaysApply: {frontmatter.alwaysApply}")

        # Check author email format
        author = frontmatter.metadata.author
        if not re.search(r"<[^>]+@[^>]+>", author):
            result.add_warning("Author format should be: 'Name <email@example.com>'")

        # Build and store manifest
        workflow_dir = manifest_path.parent
        refs_dir = workflow_dir / WORKFLOWS_REFERENCES_DIR
        scripts_dir = refs_dir / WORKFLOWS_SCRIPTS_DIR

        manifest = WorkflowsManifest(
            frontmatter=frontmatter,
            content=parsed.content,
            workflow_dir=str(workflow_dir),
            workflow_name=workflow_dir.name,
            has_readme=(workflow_dir / WORKFLOWS_README_FILE).exists(),
            has_references=refs_dir.exists(),
            reference_files=[f.name for f in refs_dir.glob("*.mdc")] if refs_dir.exists() else [],
            has_scripts=scripts_dir.exists(),
            script_files=[f.name for f in scripts_dir.iterdir()] if scripts_dir.exists() else [],
            line_count=parsed.line_count,
        )
        result.metadata["frontmatter"] = frontmatter
        result.metadata["manifest"] = manifest

        # Line count validation
        result.merge(
            validate_line_count(
                parsed.line_count,
                MAX_WORKFLOWS_MDC_LINES,
                WORKFLOWS_MANIFEST_FILE,
            )
        )

        return result

    def _validate_folder_structure(self, workflow_path: Path) -> ValidationResult:
        """Verify workflow is in valid folder hierarchy."""
        result = ValidationResult()
        parts = workflow_path.parts

        if "workflows" in parts:
            idx = parts.index("workflows")
            # workflows/<workflow-name>/
            if idx == len(parts) - 2:
                result.add_message(f"Valid general workflow structure: workflows/{parts[-1]}/")
            else:
                result.add_warning("Unexpected nesting depth for general workflow")
        elif "team-workflows" in parts:
            idx = parts.index("team-workflows")
            depth = len(parts) - idx - 1
            if depth >= 2:
                team_name = parts[idx + 1]
                result.add_message(f"Valid team workflow structure: team-workflows/{team_name}/.../{parts[-1]}/")
            else:
                result.add_error("Invalid team-workflows structure. Expected: team-workflows/<team>/<workflow-name>/")
        else:
            result.add_warning(f"Workflow not in standard location (workflows/ or team-workflows/): {workflow_path}")

        return result

    def _validate_naming_conventions(self, workflow_path: Path) -> ValidationResult:
        """Verify workflow directory name follows kebab-case convention."""
        config = NamingValidationConfig(
            name_type="Workflow name",
            max_length=64,
            use_errors=True,  # Workflows use errors for naming issues
        )
        return validate_kebab_case_name(workflow_path.name, config)

    def _validate_references(self, refs_dir: Path, workflow_name: str) -> ValidationResult:
        """Validate reference .mdc files in references/ directory."""
        result = ValidationResult()
        ref_files = list(refs_dir.glob("*.mdc"))

        if not ref_files:
            result.add_error("references/ directory must contain at least one .mdc file")
            return result

        result.add_message(f"Validating {len(ref_files)} reference file(s)")

        for ref_file in sorted(ref_files):
            ref_result = self._validate_reference_file(ref_file, workflow_name)
            ref_name = ref_file.stem

            if ref_result.passed:
                result.add_success(
                    check_name=ref_name,
                    message="valid",
                )
            else:
                result.merge_with_prefix(ref_result, f"reference/{ref_name}")

        return result

    def _validate_reference_file(self, ref_path: Path, workflow_name: str) -> ValidationResult:
        """Validate a single reference .mdc file."""
        parsed, result = parse_frontmatter(ref_path)
        if parsed is None:
            return result

        ref_frontmatter = validate_pydantic_model(ReferenceFrontmatter, parsed.yaml_data, result)
        if ref_frontmatter is None:
            return result

        # Check parent_workflow if present
        if ref_frontmatter.parent_workflow and ref_frontmatter.parent_workflow != workflow_name:
            result.add_warning(
                f"parent_workflow '{ref_frontmatter.parent_workflow}' doesn't match workflow name '{workflow_name}'"
            )

        return result

    def _validate_scripts(self, scripts_dir: Path) -> ValidationResult:
        """Validate scripts in references/scripts/ directory."""
        result = ValidationResult()
        script_files = list(scripts_dir.iterdir())

        if not script_files:
            result.add_message("scripts/ directory is empty")
            return result

        result.add_message(f"Found {len(script_files)} script(s)")

        for script_file in sorted(script_files):
            if script_file.is_file():
                # Check for shebang in shell scripts
                if script_file.suffix in (".sh", ".bash"):
                    try:
                        first_line = script_file.read_text(encoding="utf-8").split("\n")[0]
                        if not first_line.startswith("#!"):
                            result.add_warning(f"Script {script_file.name} should have a shebang line")
                    except Exception:
                        pass

                # Check for documentation comment
                try:
                    content = script_file.read_text(encoding="utf-8")
                    lines = content.split("\n")[:10]  # Check first 10 lines
                    has_description = any("Description:" in line or "Usage:" in line for line in lines)
                    if not has_description:
                        result.add_warning(f"Script {script_file.name} should include Description/Usage comments")
                except Exception:
                    pass

        return result

    def _validate_folder_compliance(self, folder_path: Path) -> ValidationResult:
        """Check if folder follows Skill Evaluator workflows structure."""
        result = ValidationResult()
        has_workflows = (folder_path / "workflows").exists()
        has_team_workflows = (folder_path / "team-workflows").exists()

        if has_workflows or has_team_workflows:
            result.add_message("Folder structure compliant with Skill Evaluator guidelines")
            if has_workflows:
                workflows_dir = folder_path / "workflows"
                count = len([d for d in workflows_dir.iterdir() if d.is_dir()])
                result.add_message(f"Found 'workflows/' directory with {count} workflow(s)")
            if has_team_workflows:
                team_workflows_dir = folder_path / "team-workflows"
                teams = [d for d in team_workflows_dir.iterdir() if d.is_dir()]
                result.add_message(f"Found 'team-workflows/' directory with {len(teams)} team(s)")
        elif "workflows" in folder_path.parts or "team-workflows" in folder_path.parts:
            result.add_message("Validating workflows within standard folder structure")
        else:
            result.add_warning(
                "Folder doesn't follow Skill Evaluator structure (missing workflows/ or team-workflows/)"
            )

        return result


def parse_workflows_manifest(workflow_path: Path) -> WorkflowsManifest | None:
    """Parse workflow directory into WorkflowsManifest (utility function)."""
    validator = WorkflowsSchemaValidator()
    manifest_path = workflow_path / WORKFLOWS_MANIFEST_FILE
    result = validator._validate_workflow_manifest(manifest_path)
    return result.metadata.get("manifest") if result.passed else None
