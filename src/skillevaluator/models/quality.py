# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quality score models for SkillEvaluator-compatible 4-dimension skill evaluation.

Ported from SkillEvaluator SkillQualityAnalyzer scoring system. Provides:
- QualityDimension: individual dimension score (0-100) with weight and issues
- QualityScoreResult: composite result with overall score, grade, and metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field

from skillevaluator.constants import (
    QUALITY_GRADE_THRESHOLDS,
    QUALITY_RECOMMENDED_MAX_TOKENS,
    QUALITY_SCORE_WEIGHTS,
)


@dataclass
class QualityIssue:
    """A quality issue found during dimension analysis."""

    severity: str  # "error", "warning", "info"
    dimension: str  # "correctness", "discoverability", "reliability", "efficiency"
    message: str
    suggestion: str | None = None
    deduction: float = 0.0


@dataclass
class QualityDimension:
    """Score for a single quality dimension."""

    name: str
    score: float = 100.0
    weight: float = 0.0
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight

    @property
    def issues_count(self) -> int:
        return len(self.issues)

    def deduct(self, amount: float, severity: str, message: str, suggestion: str | None = None) -> None:
        """Deduct from score and record the issue."""
        self.issues.append(
            QualityIssue(
                severity=severity,
                dimension=self.name,
                message=message,
                suggestion=suggestion,
                deduction=amount,
            )
        )
        self.score = max(0.0, self.score - amount)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 1),
            "weight": self.weight,
            "issues_count": self.issues_count,
            "issues": [
                {
                    "severity": i.severity,
                    "message": i.message,
                    "suggestion": i.suggestion,
                    "deduction": i.deduction,
                }
                for i in self.issues
            ],
        }


def score_to_grade(score: float) -> str:
    """Convert numeric score to letter grade."""
    for threshold, grade in QUALITY_GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


@dataclass
class QualityScoreResult:
    """Complete quality score result for a skill.

    Carries the 4-dimension scores, overall composite, grade, skill type,
    and metadata. Designed to be serialized into result.metadata["quality_scores"]
    for consumption by all reporters.
    """

    skill_name: str = ""
    skill_type: str = "unknown"

    correctness: QualityDimension = field(
        default_factory=lambda: QualityDimension(name="correctness", weight=QUALITY_SCORE_WEIGHTS["correctness"])
    )
    discoverability: QualityDimension = field(
        default_factory=lambda: QualityDimension(
            name="discoverability", weight=QUALITY_SCORE_WEIGHTS["discoverability"]
        )
    )
    reliability: QualityDimension = field(
        default_factory=lambda: QualityDimension(name="reliability", weight=QUALITY_SCORE_WEIGHTS["reliability"])
    )
    efficiency: QualityDimension = field(
        default_factory=lambda: QualityDimension(name="efficiency", weight=QUALITY_SCORE_WEIGHTS["efficiency"])
    )

    # Metrics
    total_tokens: int = 0
    frontmatter_tokens: int = 0
    instructions_tokens: int = 0
    script_count: int = 0
    has_examples: bool = False
    has_error_handling: bool = False
    has_frontmatter: bool = False
    has_instructions: bool = False
    has_scripts: bool = False
    has_lib_module: bool = False

    @property
    def overall_score(self) -> float:
        return (
            self.correctness.weighted_score
            + self.discoverability.weighted_score
            + self.reliability.weighted_score
            + self.efficiency.weighted_score
        )

    @property
    def grade(self) -> str:
        return score_to_grade(self.overall_score)

    @property
    def all_issues(self) -> list[QualityIssue]:
        return self.correctness.issues + self.discoverability.issues + self.reliability.issues + self.efficiency.issues

    @property
    def dimensions(self) -> list[QualityDimension]:
        return [self.correctness, self.discoverability, self.reliability, self.efficiency]

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 1),
            "grade": self.grade,
            "skill_type": self.skill_type,
            "skill_name": self.skill_name,
            "dimensions": {
                "correctness": self.correctness.to_dict(),
                "discoverability": self.discoverability.to_dict(),
                "reliability": self.reliability.to_dict(),
                "efficiency": self.efficiency.to_dict(),
            },
            "metrics": {
                "total_tokens": self.total_tokens,
                "recommended_max_tokens": QUALITY_RECOMMENDED_MAX_TOKENS,
                "frontmatter_tokens": self.frontmatter_tokens,
                "instructions_tokens": self.instructions_tokens,
                "script_count": self.script_count,
                "has_examples": self.has_examples,
                "has_error_handling": self.has_error_handling,
                "has_frontmatter": self.has_frontmatter,
                "has_instructions": self.has_instructions,
                "has_scripts": self.has_scripts,
                "has_lib_module": self.has_lib_module,
            },
        }
