# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validators for SkillEvaluator skill, rules, and workflows validation."""

from typing import TYPE_CHECKING

from skillevaluator.validators.base import (
    Finding,
    Severity,
    SuccessDetail,
    ValidationResult,
    ValidationSummary,
    ValidatorBase,
)
from skillevaluator.validators.code_risk import CodeRiskValidator
from skillevaluator.validators.dependencies import DependencySecurityValidator
from skillevaluator.validators.frontmatter_parser import ParsedFrontmatter, parse_frontmatter
from skillevaluator.validators.hygiene import HygieneValidator
from skillevaluator.validators.license import LicenseValidator
from skillevaluator.validators.naming_utils import NamingValidationConfig, validate_kebab_case_name
from skillevaluator.validators.plugin_schema import PluginSchemaValidator
from skillevaluator.validators.policy import (
    DEFAULT_PROFILE_NAME,
    ValidationPolicy,
    apply_policy,
    default_policy,
    load_profile,
    resolve_policy,
)
from skillevaluator.validators.quality_score import QualityScoreValidator
from skillevaluator.validators.rules_schema import RulesSchemaValidator
from skillevaluator.validators.schema import SchemaValidator
from skillevaluator.validators.script_lint import ScriptLintValidator
from skillevaluator.validators.secrets import SecretsValidator
from skillevaluator.validators.security import SecurityValidator
from skillevaluator.validators.unicode_smuggle import UnicodeSmuggleValidator
from skillevaluator.validators.version import VersionValidator
from skillevaluator.validators.workflows_schema import WorkflowsSchemaValidator

if TYPE_CHECKING:
    from skillevaluator.deduplication.intra_skill.intra_skill_validator import IntraSkillValidator
    from skillevaluator.validators.similarity import SimilarityValidator


def __getattr__(name: str) -> object:
    """Lazily expose validators that would otherwise create import cycles.

    ``IntraSkillValidator`` (deduplication) and ``SimilarityValidator`` (embedding)
    both depend on packages that in turn import ``validators.frontmatter_parser``.
    Importing them eagerly here makes leaf modules under ``deduplication`` /
    ``embedding`` impossible to import in isolation (e.g. under ``pytest -n auto``).
    """
    if name == "IntraSkillValidator":
        from skillevaluator.deduplication.intra_skill.intra_skill_validator import (
            IntraSkillValidator,
        )

        return IntraSkillValidator
    if name == "SimilarityValidator":
        from skillevaluator.validators.similarity import SimilarityValidator

        return SimilarityValidator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_PROFILE_NAME",
    "CodeRiskValidator",
    "DependencySecurityValidator",
    "Finding",
    "HygieneValidator",
    "IntraSkillValidator",
    "LicenseValidator",
    "NamingValidationConfig",
    "ParsedFrontmatter",
    "PluginSchemaValidator",
    "QualityScoreValidator",
    "RulesSchemaValidator",
    "SchemaValidator",
    "ScriptLintValidator",
    "SecretsValidator",
    "SecurityValidator",
    "Severity",
    "SimilarityValidator",
    "SuccessDetail",
    "UnicodeSmuggleValidator",
    "ValidationPolicy",
    "ValidationResult",
    "ValidationSummary",
    "ValidatorBase",
    "VersionValidator",
    "WorkflowsSchemaValidator",
    "apply_policy",
    "default_policy",
    "load_profile",
    "parse_frontmatter",
    "resolve_policy",
    "validate_kebab_case_name",
]
