# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof L4: BENCHMARK.md \"Blocking Findings\" lists only what blocks, and says how many rows it did not show."""

from __future__ import annotations

import pytest
from _f90_gate_payloads import section

from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.reporting.benchmark import BenchmarkReporter

# ---------------------------------------------------------------------------
# L4: "Blocking Findings" lists only what blocks, and says how many it did not show
# ---------------------------------------------------------------------------


def _tier1_with_mixed_findings() -> ValidationResult:
    result = ValidationResult(validator_name="MCP Static Analysis")
    for index in range(7):
        result.add_finding(
            Finding("MCP", Severity.HIGH, "mcp_unpinned_package", f"server-{index} runs an unpinned package", "x.json")
        )
    result.add_finding(Finding("MCP", Severity.MEDIUM, "mcp_server_duplicate_name", "duplicate name", "x.json"))
    result.add_finding(Finding("MCP", Severity.LOW, "mcp_endpoint_note", "endpoint note", "x.json"))
    result.metadata["gating"] = {"tier": 1, "blocking": True}
    return result


@pytest.mark.parametrize("content_type", ["plugin", "skill"])
def test_blocking_findings_lists_only_blocking_rows_and_counts_the_rest(content_type: str) -> None:
    card = BenchmarkReporter(content_type=content_type, skill_name="gate-demo").render_all(
        [_tier1_with_mixed_findings()]
    )
    blocking = section(card, "Blocking Findings")
    rows = [line for line in blocking.splitlines() if line.startswith("- **")]

    assert len(rows) == 5
    assert all(row.startswith("- **HIGH**") for row in rows)
    assert "MEDIUM" not in blocking and "LOW" not in blocking
    assert "2 more blocking finding(s)" in blocking


def test_advisory_tier_findings_are_not_listed_as_blocking() -> None:
    tier1 = ValidationResult(validator_name="Plugin Schema & Bundle References")
    tier1.add_success("schema", "ok")
    tier1.metadata["gating"] = {"tier": 1, "blocking": True}
    tier2 = ValidationResult(validator_name="Context Deduplication")
    tier2.add_finding(Finding("DUPLICATE", Severity.HIGH, "whole_duplicate", "duplicate skill", "a.md"))
    tier2.metadata["gating"] = {"tier": 2, "blocking": False}

    card = BenchmarkReporter(content_type="plugin", skill_name="gate-demo").render_all([tier1, tier2])

    assert "## Blocking Findings" not in card


def test_policy_blocking_medium_is_listed() -> None:
    result = ValidationResult(validator_name="Dependency Audit")
    result.add_finding(
        Finding("DEPENDENCY", Severity.MEDIUM, "npm_moderate", "moderate advisory", "package.json"),
        fail_on_medium=True,
    )
    result.metadata["gating"] = {"tier": 1, "blocking": True}

    card = BenchmarkReporter(content_type="plugin", skill_name="gate-demo").render_all([result])

    assert "- **MEDIUM** DEPENDENCY/npm\\_moderate" in section(card, "Blocking Findings")
