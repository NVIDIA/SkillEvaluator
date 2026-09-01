# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Script Lint Validator — AST-based advisory checks for skill scripts.

Ported from SkillEvaluator lint_skill_scripts.py. These are ADVISORY checks
for code style and maintainability — they produce warnings but never
fail validation.

Checks performed per .py file in scripts/ or tools/:
  - Flat script (no function definitions)
  - Deep nesting (> 6 levels of control flow)
  - Magic numbers (raw numeric constants)
  - Missing shebang line
  - Missing input validation
"""

from __future__ import annotations

import ast
from pathlib import Path

from skillevaluator.constants import EXECUTABLE_SKILL_DIRS, SCRIPT_LINT_MAX_NESTING, SCRIPT_LINT_SAFE_CONSTANTS
from skillevaluator.logging_config import get_logger
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.validators.base import ValidatorBase

logger = get_logger(__name__)

_NESTING_NODES = (
    ast.If,
    ast.For,
    ast.While,
    ast.Try,
    ast.With,
    ast.AsyncFor,
    ast.AsyncWith,
)


def _calculate_nesting_depth(tree: ast.AST) -> int:
    """Calculate maximum nesting depth of control flow statements."""
    max_depth = 0

    def _walk(node: ast.AST, depth: int) -> None:
        nonlocal max_depth
        if isinstance(node, _NESTING_NODES):
            depth += 1
            max_depth = max(max_depth, depth)
        for child in ast.iter_child_nodes(node):
            _walk(child, depth)

    _walk(tree, 0)
    return max_depth


def _has_magic_numbers(tree: ast.AST) -> bool:
    """Check if the AST contains magic number constants."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and node.value not in SCRIPT_LINT_SAFE_CONSTANTS
        ):
            return True
    return False


def _count_functions(tree: ast.AST) -> int:
    """Count function definitions in the AST."""
    return sum(1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))


def _has_input_validation(content: str) -> bool:
    """Heuristic check for input validation patterns."""
    return any(
        p in content
        for p in [
            "if len(sys.argv)",
            "argparse",
            "click",
            "typer",
            "ValueError",
            "raise",
        ]
    )


class ScriptLintValidator(ValidatorBase):
    """Advisory AST-based linter for skill scripts/ directories.

    All findings are MEDIUM/LOW severity — they produce warnings but
    never fail validation. Lint remains advisory and separate from quality
    scores.
    """

    @property
    def name(self) -> str:
        return "Script Lint (Advisory)"

    @property
    def description(self) -> str:
        return "AST-based code quality checks for skill scripts"

    def validate(self, skill_path: Path) -> ValidationResult:
        """Lint scripts in a skill directory or folder of skills."""
        if self._is_skill_directory(skill_path):
            return self._lint_skill(skill_path)
        return self._lint_folder(skill_path)

    def _lint_folder(self, root: Path) -> ValidationResult:
        skill_dirs = self._find_all_skills(root)
        result = ValidationResult(
            validator_name="SCRIPT_LINT",
            validator_description=self.description,
        )
        if not skill_dirs:
            result.add_success(check_name="lint", message="No skills found to lint")
            return result
        for sd in skill_dirs:
            sub = self._lint_skill(sd)
            result.merge_with_prefix(sub, sd.name)
        return result

    def _lint_skill(self, skill_path: Path) -> ValidationResult:
        """Lint all Python scripts in a skill's scripts/ or tools/ directory."""
        result = ValidationResult(
            validator_name="SCRIPT_LINT",
            validator_description=self.description,
        )
        py_files: list[Path] = []
        for dirname in EXECUTABLE_SKILL_DIRS:
            directory = skill_path / dirname
            if directory.is_dir():
                py_files.extend(sorted(directory.glob("*.py")))
        if not py_files:
            if not any((skill_path / dirname).is_dir() for dirname in EXECUTABLE_SKILL_DIRS):
                result.add_success(check_name="lint", message="No scripts/ or tools/ directory found")
            else:
                result.add_success(check_name="lint", message="No Python scripts found in scripts/ or tools/")
            return result

        for script in py_files:
            self._lint_script(script, result)

        if not result.findings:
            result.add_success(
                check_name="lint",
                message=f"All {len(py_files)} script(s) passed lint checks",
            )
        return result

    def _lint_script(self, script: Path, result: ValidationResult) -> None:
        """Run all lint checks on a single Python script."""
        try:
            content = script.read_text(encoding="utf-8")
        except Exception as e:
            result.add_finding(
                Finding(
                    category="SCRIPT_LINT",
                    severity=Severity.MEDIUM,
                    check_name="read_error",
                    message=f"Cannot read file: {e}",
                    file_path=str(script),
                )
            )
            return

        try:
            tree = ast.parse(content, filename=script.name)
        except SyntaxError:
            return

        fname = script.name

        if _count_functions(tree) == 0:
            result.add_finding(
                Finding(
                    category="SCRIPT_LINT",
                    severity=Severity.MEDIUM,
                    check_name="flat_script",
                    message=f"{fname} has no function definitions (flat script)",
                    file_path=str(script),
                    suggestion="Wrap logic in functions for maintainability and testability",
                )
            )

        depth = _calculate_nesting_depth(tree)
        if depth > SCRIPT_LINT_MAX_NESTING:
            result.add_finding(
                Finding(
                    category="SCRIPT_LINT",
                    severity=Severity.MEDIUM,
                    check_name="deep_nesting",
                    message=f"{fname} has deeply nested code (depth {depth}, max {SCRIPT_LINT_MAX_NESTING})",
                    file_path=str(script),
                    suggestion="Refactor into smaller functions to reduce complexity",
                )
            )

        if _has_magic_numbers(tree):
            result.add_finding(
                Finding(
                    category="SCRIPT_LINT",
                    severity=Severity.LOW,
                    check_name="magic_numbers",
                    message=f"{fname} contains magic numbers",
                    file_path=str(script),
                    suggestion="Extract magic numbers to named constants",
                )
            )

        if not content.startswith("#!"):
            result.add_finding(
                Finding(
                    category="SCRIPT_LINT",
                    severity=Severity.LOW,
                    check_name="missing_shebang",
                    message=f"{fname} missing shebang line",
                    file_path=str(script),
                    suggestion="Add: #!/usr/bin/env python3",
                )
            )

        if not _has_input_validation(content):
            result.add_finding(
                Finding(
                    category="SCRIPT_LINT",
                    severity=Severity.LOW,
                    check_name="no_input_validation",
                    message=f"{fname} may lack input validation",
                    file_path=str(script),
                    suggestion="Add argument checks and raise descriptive errors",
                )
            )
