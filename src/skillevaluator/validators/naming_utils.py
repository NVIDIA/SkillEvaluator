# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared naming convention validation utilities.

Provides consistent validation of kebab-case names across
Skills, Rules, and Workflows validators.
"""

import re
from dataclasses import dataclass

from skillevaluator.constants import KEBAB_CASE_PATTERN
from skillevaluator.validators.base import ValidationResult


@dataclass
class NamingValidationConfig:
    """Configuration for naming validation behavior."""

    name_type: str  # e.g., "Skill name", "Rule filename", "Workflow name"
    max_length: int = 64
    use_errors: bool = True  # If False, use warnings instead of errors


def validate_kebab_case_name(
    name: str,
    config: NamingValidationConfig,
) -> ValidationResult:
    """Validate that a name follows kebab-case convention.

    Centralizes the naming validation logic used across validators.

    Args:
        name: The name to validate
        config: Configuration for validation behavior

    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult()
    add_issue = result.add_error if config.use_errors else result.add_warning

    # KEBAB_CASE_PATTERN already rejects leading/trailing/consecutive hyphens, so
    # those cases fall through to the single kebab-case message below.
    if not re.match(KEBAB_CASE_PATTERN, name):
        add_issue(f"{config.name_type} '{name}' must be kebab-case (lowercase letters, numbers, and hyphens)")
    elif len(name) > config.max_length:
        add_issue(f"{config.name_type} '{name}' exceeds {config.max_length} char limit ({len(name)} chars)")
    else:
        result.add_message(f"{config.name_type} '{name}' follows naming convention")

    return result


def validate_line_count(
    line_count: int,
    max_lines: int,
    file_description: str,
) -> ValidationResult:
    """Validate that a file doesn't exceed recommended line count.

    Args:
        line_count: Actual number of lines
        max_lines: Maximum recommended lines
        file_description: Description for messages (e.g., "Rule file")

    Returns:
        ValidationResult with warning if exceeded
    """
    result = ValidationResult()

    if line_count > max_lines:
        result.add_warning(
            f"{file_description} has {line_count} lines (>{max_lines}). Consider splitting or using references."
        )
    else:
        result.add_message(f"{file_description}: {line_count} lines (limit: {max_lines})")

    return result
