# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Naming checks must validate the complete path component, including its final newline."""

from pathlib import Path

import pytest

from skillevaluator.validators.naming_utils import NamingValidationConfig, validate_kebab_case_name
from skillevaluator.validators.rules_schema import RulesSchemaValidator
from skillevaluator.validators.schema import SchemaValidator
from skillevaluator.validators.workflows_schema import WorkflowsSchemaValidator


@pytest.mark.parametrize("use_errors", [True, False])
def test_trailing_newline_is_not_a_valid_kebab_case_name(use_errors):
    result = validate_kebab_case_name("my-skill\n", NamingValidationConfig(name_type="Name", use_errors=use_errors))

    assert result.errors if use_errors else result.warnings
    assert not any("follows naming convention" in message for message in result.messages)


def test_skill_folder_naming_rejects_trailing_newline():
    result = SchemaValidator()._validate_naming_conventions(Path("skills/my-skill\n"))

    assert any(finding.check_name == "naming_convention" for finding in result.findings)


def test_rule_filename_naming_warns_for_trailing_newline():
    result = RulesSchemaValidator()._validate_naming_conventions(Path("team-rules/example/my-rule\n.mdc"))

    assert any("kebab-case" in warning for warning in result.warnings)


def test_workflow_folder_naming_rejects_trailing_newline():
    result = WorkflowsSchemaValidator()._validate_naming_conventions(Path("workflows/my-workflow\n"))

    assert any("kebab-case" in error for error in result.errors)
