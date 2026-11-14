# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Completed harness evaluations must carry truthful execution evidence."""

from pathlib import Path

import pytest

from skillevaluator.models.result import ValidationResult
from skillevaluator.publication_evidence import tier1_required_checks
from skillevaluator.reporting.base import assess_publication, result_has_execution_evidence
from skillevaluator.tier1.commands import _as_result, run_validation
from skillevaluator.validators.harness import HarnessValidator


@pytest.fixture
def clean_profiles(tmp_path: Path) -> Path:
    profiles = tmp_path / "clean-profiles"
    profiles.mkdir()
    (profiles / "clean-fixture.yaml").write_text(
        "harness: clean-fixture\n"
        "listing: {cut: 1024}\n"
        "rules:\n"
        "  - id: FIXTURE-DESCRIPTION\n"
        "    when: description_missing\n"
        "    effect: not_loaded\n"
        "    message: A description is required.\n",
        encoding="utf-8",
    )
    return profiles


def _skill(root: Path) -> Path:
    skill = root / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Use when reviewing a project.\nlicense: MIT\n"
        "metadata:\n  author: Demo Author\n---\n# Demo\n\n## Instructions\n\nRead the project.\n"
        "\n## Examples\n\nExplain its workflow.\n",
        encoding="utf-8",
    )
    return skill


def test_clean_harness_records_completed_profile_evaluations(tmp_path: Path, clean_profiles: Path) -> None:
    skill = _skill(tmp_path)

    result = HarnessValidator(profiles_dir=clean_profiles).validate(skill)

    assert result.findings == []
    assert result_has_execution_evidence(result)
    detail = result.success_details[0]
    assert detail.check_name == "harness_profile_evaluation"
    assert detail.metadata == {
        "harnesses": ["clean-fixture"],
        "profiles_evaluated": 1,
        "skills_evaluated": 1,
        "evaluations_performed": 1,
    }


def test_completed_harness_evidence_keeps_advisory_findings(tmp_path: Path) -> None:
    skill = _skill(tmp_path / ".claude" / "skills")

    result = HarnessValidator("claude-code,codex").validate(skill)

    assert any(finding.check_name == "CX-ROOT" for finding in result.findings)
    assert result.success_details[0].metadata["harnesses"] == ["claude-code", "codex"]
    assert result.success_details[0].metadata["evaluations_performed"] == 2
    assert {row["harness"]: row["status"] for row in result.metadata["harness_compatibility"][str(skill)]} == {
        "claude-code": "loads",
        "codex": "not_loaded",
    }


@pytest.mark.parametrize("missing", [False, True], ids=["empty-directory", "missing-target"])
def test_no_skills_does_not_claim_completed_harness_evaluations(tmp_path: Path, missing: bool) -> None:
    target = tmp_path / "missing" if missing else tmp_path

    result = HarnessValidator().validate(target)

    assert result.success_details == []
    assert not result_has_execution_evidence(result)


def test_no_profiles_does_not_claim_completed_harness_evaluations(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir()

    result = HarnessValidator(profiles_dir=profiles).validate(skill)

    assert result.success_details == []
    assert not result_has_execution_evidence(result)


@pytest.mark.parametrize("actual_harness", [True, False], ids=["completed-harness", "empty-harness"])
def test_harness_publication_requires_actual_execution_evidence(
    tmp_path: Path, actual_harness: bool, clean_profiles: Path
) -> None:
    skill = _skill(tmp_path)

    def successful(_target: Path) -> ValidationResult:
        result = ValidationResult()
        result.add_success("completed", "Required check completed")
        return result

    required = [
        _as_result(check, check, successful, skill, publication_check_id=check)
        for check in sorted(tier1_required_checks(skill_like=True))
    ]
    for result in required:
        result.metadata["benchmark_policy"] = {"tier2_required": False, "tier3_required": False}
    assert assess_publication(required, expected_skill_name="demo").status == "pass"

    if actual_harness:
        harness = run_validation(
            skill, checks="harness", harness_profiles=clean_profiles, on_check=lambda _check: None
        )[0]
        assert harness.findings == []
        assert harness.success_details[0].metadata["harnesses"] == ["clean-fixture"]
        assert harness.success_details[0].metadata["evaluations_performed"] == 1
    else:
        harness = _as_result(
            "Harness Compatibility",
            "Harness",
            lambda _target: ValidationResult(),
            skill,
            publication_check_id="harness",
        )
    harness.metadata["benchmark_policy"] = {"tier2_required": False, "tier3_required": False}
    assessment = assess_publication([*required, harness], expected_skill_name="demo")

    assert assessment.status == ("pass" if actual_harness else "incomplete")
    assert assessment.reasons == (() if actual_harness else ("Tier 1 lacks trustworthy execution evidence.",))
