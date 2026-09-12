# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted producer contracts for publication-certifying Tier 1 and Tier 2 evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.models.result import ValidationResult
from skillevaluator.publication_evidence import (
    PublicationEvidenceIdentity,
    publication_evidence_dict,
    result_publication_evidence,
    stamp_publication_evidence,
)
from skillevaluator.reporting import HTMLReporter, JSONReporter
from skillevaluator.reporting.base import assess_publication, result_has_execution_evidence
from skillevaluator.utils.tool_runner import ToolResult, Tools


def test_stamp_and_project_recognized_publication_evidence() -> None:
    result = ValidationResult(validator_name="SCHEMA")

    stamp_publication_evidence([result], tier=1, check_id="schema")

    expected = {
        "schema_version": 1,
        "producer": "skillevaluator.tier1",
        "tier": 1,
        "check_id": "schema",
    }
    assert result.metadata["publication_evidence"] == expected
    assert result_publication_evidence(result) == PublicationEvidenceIdentity(
        schema_version=1,
        producer="skillevaluator.tier1",
        tier=1,
        check_id="schema",
    )
    assert publication_evidence_dict(result.metadata["publication_evidence"]) == expected


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"producer": "skillevaluator.tier2"},
        {"tier": True},
        {"tier": 2},
        {"check_id": "unknown"},
        {"extra": "ambiguous"},
    ],
    ids=["bool-version", "version", "producer", "bool-tier", "tier", "check-id", "extra-field"],
)
def test_publication_evidence_parser_fails_closed(mutation: dict[str, object]) -> None:
    marker: dict[str, object] = {
        "schema_version": 1,
        "producer": "skillevaluator.tier1",
        "tier": 1,
        "check_id": "schema",
    }
    marker.update(mutation)
    result = ValidationResult(validator_name="SCHEMA")
    result.metadata["publication_evidence"] = marker

    assert result_publication_evidence(result) is None
    assert publication_evidence_dict(marker) is None


def test_stamp_rejects_unknown_check_identity() -> None:
    result = ValidationResult(validator_name="custom")

    with pytest.raises(ValueError, match="recognized publication check"):
        stamp_publication_evidence([result], tier=1, check_id="custom")

    assert "publication_evidence" not in result.metadata


def test_all_built_in_command_wrappers_stamp_canonical_check_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier1 import commands as tier1_commands
    from skillevaluator.tier2 import commands as tier2_commands

    skill = tmp_path / "demo"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: test\n---\n", encoding="utf-8")

    class SuccessfulValidator:
        name = "Built-in validator"
        description = "Built-in validation"

        def __init__(self, **_kwargs: object) -> None:
            pass

        @staticmethod
        def validate(_target: Path) -> ValidationResult:
            result = ValidationResult(validator_name="Built-in validator")
            result.add_success("built_in", "Built-in validation completed")
            return result

        validate_security_only = validate
        validate_pii_only = validate

    monkeypatch.setattr(tier1_commands, "_schema_validator_for", lambda *_args: SuccessfulValidator())
    for validator_name in (
        "SecurityValidator",
        "CodeRiskValidator",
        "SecretsValidator",
        "HygieneValidator",
        "UnicodeSmuggleValidator",
        "QualityScoreValidator",
        "ScriptLintValidator",
        "VersionValidator",
        "LicenseValidator",
        "DependencySecurityValidator",
        "RubricEvalValidator",
    ):
        monkeypatch.setattr(tier1_commands, validator_name, SuccessfulValidator)
    monkeypatch.setattr(tier2_commands, "SimilarityValidator", SuccessfulValidator)
    monkeypatch.setattr(tier2_commands, "IntraSkillValidator", SuccessfulValidator)

    tier1 = tier1_commands.run_validation(skill, checks=",".join(sorted(tier1_commands.RECOGNIZED_CHECKS)))
    rubric = tier1_commands.run_rubric_eval(skill)
    tier2 = [
        *tier2_commands.run_similarity_check(skill),
        *tier2_commands.run_context_optimization_check(skill),
    ]

    assert {result.metadata["publication_evidence"]["check_id"] for result in tier1} == set(
        tier1_commands.RECOGNIZED_CHECKS
    )
    assert rubric[0].metadata["publication_evidence"]["check_id"] == "rubric"
    assert {result.metadata["publication_evidence"]["check_id"] for result in tier2} == {
        "similarity",
        "context-optimization",
    }


def test_clean_actual_tier1_validators_produce_publication_execution_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier1.commands import run_validation

    skill = tmp_path / "demo"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        """---
name: demo
description: Summarize short user notes into clear action items.
license: MIT
metadata:
  author: Test Author <test-author@example.com>
---

# Demo

## Instructions

Read the user's note and return a concise list of action items.

## Examples

Input: "Follow up tomorrow."

Output: "Action item: Follow up tomorrow."
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(Tools.gitleaks, "_path", "/usr/bin/gitleaks")
    monkeypatch.setattr(
        Tools.gitleaks,
        "run",
        lambda *_args, **_kwargs: ToolResult(
            success=True,
            stdout="[]",
            stderr="",
            exit_code=0,
        ),
    )

    results = run_validation(
        skill,
        checks="schema,license,code-integrity,dependency",
        content_type="skill",
        on_check=lambda _check: None,
    )
    for result in results:
        result.metadata["benchmark_policy"] = {
            "tier2_required": False,
            "tier3_required": False,
        }

    assert all(result.passed for result in results)
    assert all(result_has_execution_evidence(result) for result in results)
    assessment = assess_publication(results, expected_skill_name="demo")
    assert assessment.status == "pass"
    assert not assessment.reasons


def test_dependency_publication_is_incomplete_when_one_requirements_scan_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One clean source must not hide another source's missing audit evidence."""
    from skillevaluator.tier1.commands import run_validation

    skill = tmp_path / "demo"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Test dependency audit publication evidence.\n---\n",
        encoding="utf-8",
    )
    (skill / "requirements-a.txt").write_text("requests==2.32.0\n", encoding="utf-8")
    (skill / "requirements-b.txt").write_text("urllib3==2.2.2\n", encoding="utf-8")

    tool_results = iter(
        [
            ToolResult(
                success=True,
                stdout='{"dependencies": [], "fixes": []}',
                stderr="",
                exit_code=0,
            ),
            ToolResult(
                success=False,
                stdout="",
                stderr="",
                exit_code=-1,
                error_message="pip-audit timed out",
            ),
        ]
    )
    monkeypatch.setattr(Tools.pip_audit, "_path", "/usr/bin/pip-audit")
    monkeypatch.setattr(Tools.pip_audit, "run", lambda *_args, **_kwargs: next(tool_results))
    monkeypatch.setattr(Tools.safety, "_path", None)

    results = run_validation(
        skill,
        checks="dependency",
        content_type="skill",
        on_check=lambda _check: None,
    )
    dependency = results[0]
    dependency.metadata["benchmark_policy"] = {
        "tier2_required": False,
        "tier3_required": False,
    }

    assert dependency.status == "incomplete"
    assert dependency.incomplete_scans == ["pip-audit"]
    assert len([detail for detail in dependency.success_details if detail.check_name == "pip_audit"]) == 1
    assessment = assess_publication(results, expected_skill_name="demo")
    assert assessment.status == "incomplete"
    assert "One or more validators reported incomplete scanner evidence." in assessment.reasons


def test_machine_reporters_preserve_only_valid_publication_evidence() -> None:
    recognized = ValidationResult(validator_name="SCHEMA")
    recognized.add_success("schema", "Schema passed")
    stamp_publication_evidence([recognized], tier=1, check_id="schema")
    malformed = ValidationResult(validator_name="SCHEMA")
    malformed.add_success("schema", "Schema passed")
    malformed.metadata["publication_evidence"] = {
        **recognized.metadata["publication_evidence"],
        "check_id": "custom",
    }

    json_results = json.loads(JSONReporter(include_timestamp=False).render_all([recognized, malformed]))["results"]
    html_results = HTMLReporter(include_timestamp=False)._results_to_dict([recognized, malformed])

    assert json_results[0]["publication_evidence"] == recognized.metadata["publication_evidence"]
    assert html_results[0]["publication_evidence"] == recognized.metadata["publication_evidence"]
    assert "publication_evidence" not in json_results[1]
    assert "publication_evidence" not in html_results[1]


@pytest.mark.parametrize(("tier", "check_id"), [(1, "schema"), (2, "similarity")])
def test_tier3_result_cannot_reuse_a_tier1_or_tier2_producer_marker(tier: int, check_id: str) -> None:
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.metadata["agent_eval"] = {"execution_status": "succeeded"}
    stamp_publication_evidence([result], tier=tier, check_id=check_id)

    assert result_publication_evidence(result) is None
