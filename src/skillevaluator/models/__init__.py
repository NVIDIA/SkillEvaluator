# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models and data structures for SkillEvaluator."""

from skillevaluator.models.plugin import (
    PluginAuthor,
    PluginDependencySection,
    PluginManifest,
    PluginMcpEntry,
    PluginSelector,
)
from skillevaluator.models.quality import (
    QualityDimension,
    QualityIssue,
    QualityScoreResult,
    score_to_grade,
)
from skillevaluator.models.result import (
    Finding,
    Severity,
    SuccessDetail,
    ValidationResult,
    ValidationSummary,
)
from skillevaluator.models.rules import RulesFrontmatter, RulesManifest, RulesMetadata
from skillevaluator.models.skill import SkillFrontmatter, SkillManifest
from skillevaluator.models.workflows import (
    ReferenceFrontmatter,
    WorkflowsFrontmatter,
    WorkflowsManifest,
    WorkflowsMetadata,
    WorkflowsStructure,
)

__all__ = [
    "Finding",
    "PluginAuthor",
    "PluginDependencySection",
    "PluginManifest",
    "PluginMcpEntry",
    "PluginSelector",
    "QualityDimension",
    "QualityIssue",
    "QualityScoreResult",
    "ReferenceFrontmatter",
    "RulesFrontmatter",
    "RulesManifest",
    "RulesMetadata",
    "Severity",
    "SkillFrontmatter",
    "SkillManifest",
    "SuccessDetail",
    "ValidationResult",
    "ValidationSummary",
    "WorkflowsFrontmatter",
    "WorkflowsManifest",
    "WorkflowsMetadata",
    "WorkflowsStructure",
    "score_to_grade",
]
