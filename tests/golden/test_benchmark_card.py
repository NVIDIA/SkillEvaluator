# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden regression guard for the BENCHMARK.md skill evaluation card.

The card content is a faithful Skill Evaluator 3.2.1 port and must not drift. If this
test fails after an intentional change, regenerate the golden and review the
diff. Timestamps are disabled so the snapshot is deterministic.
"""

from __future__ import annotations

from pathlib import Path

from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.reporting import BenchmarkReporter

GOLDEN = Path(__file__).resolve().parent / "benchmark_tier1.md"


def _deterministic_results() -> list[ValidationResult]:
    t1 = ValidationResult(
        validator_name="Schema & Repository Governance",
        validator_description="Validate SKILL.md frontmatter and repository structure",
    )
    t1.add_success(check_name="author_format", message="Valid author format: Dev One <dev@nvidia.com>")
    t1.metadata["policy"] = {"profile": "internal"}
    t1.metadata["quality_scores"] = {"skill_name": "demo-skill"}

    t2 = ValidationResult(
        validator_name="Context Deduplication",
        validator_description="Detect redundant content within one skill",
    )
    t2.add_finding(
        Finding(
            category="CONTENT_DEDUP",
            severity=Severity.LOW,
            check_name="partial_overlap",
            message="Partial overlap with another skill",
            file_path="SKILL.md",
        )
    )
    return [t1, t2]


def test_benchmark_card_matches_golden() -> None:
    rendered = BenchmarkReporter(include_timestamp=False).render_all(_deterministic_results())
    assert rendered == GOLDEN.read_text(encoding="utf-8"), (
        "BENCHMARK.md content drifted from the faithful Skill Evaluator golden. If intentional, "
        "regenerate tests/golden/benchmark_tier1.md and review the diff."
    )
