# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for LLM rubric evaluation validator."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from skillevaluator.constants import RUBRIC_CRITERIA
from skillevaluator.reporting import HTMLReporter
from skillevaluator.validators.rubric_eval import (
    RubricEvalValidator,
    RubricJudge,
    _collect_supplementary_content,
    _extract_json,
)


def _write_skill(tmp_path: Path, name: str = "my-skill") -> Path:
    skill_dir = tmp_path / name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test\n---\n# Test",
        encoding="utf-8",
    )
    return skill_dir


def _complete_checks(
    *,
    scores: dict[str, int | float] | None = None,
    passes: dict[str, bool] | None = None,
) -> list[dict]:
    scores = scores or {}
    passes = passes or {}
    return [
        {
            "id": criterion["id"],
            "criterion": criterion["criterion"],
            "pass": passes.get(criterion["id"], True),
            "score": scores.get(criterion["id"], 9),
            "notes": f"Evidence for {criterion['id']}",
        }
        for criterion in RUBRIC_CRITERIA
    ]


class TestExtractJson:
    def test_plain_json(self):
        result = _extract_json('{"score": 85, "overall_pass": true}')
        assert result["score"] == 85

    def test_json_in_code_fence(self):
        result = _extract_json('```json\n{"score": 90}\n```')
        assert result["score"] == 90

    def test_json_with_preamble(self):
        result = _extract_json('Here is the result:\n{"score": 75, "summary": "ok"}')
        assert result["score"] == 75

    def test_empty_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("")

    def test_no_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("This is just text with no JSON")


class TestRubricJudge:
    def test_system_prompt_not_empty(self):
        judge = RubricJudge()
        assert "criterion" in judge.get_system_prompt().lower()

    def test_user_prompt_contains_skill_content(self):
        judge = RubricJudge()
        prompt = judge.create_user_prompt(
            skill_name="test-skill",
            skill_content="# Test\nThis is a test skill.",
        )
        assert "test-skill" in prompt
        assert "This is a test skill" in prompt

    def test_user_prompt_with_supplementary(self):
        judge = RubricJudge()
        prompt = judge.create_user_prompt(
            skill_name="test-skill",
            skill_content="# Test",
            supplementary_content={
                "scripts": {"main.py": "print('hello')"},
                "references": {},
            },
        )
        assert "main.py" in prompt
        assert "print('hello')" in prompt

    def test_fallback_response(self):
        judge = RubricJudge()
        fallback = judge.get_fallback_response()
        assert fallback["overall_pass"] is False
        assert fallback["score"] == 0
        assert fallback["checks"] == []

    def test_parse_response_valid(self):
        judge = RubricJudge()
        result = judge.parse_response('{"score": 80, "overall_pass": true, "checks": []}')
        assert result["score"] == 80


class TestCollectSupplementaryContent:
    def test_empty_dir(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Test")
        result = _collect_supplementary_content(skill_dir)
        assert result == {"scripts": {}, "references": {}}

    def test_collects_scripts(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        scripts = skill_dir / "scripts"
        scripts.mkdir()
        (scripts / "main.py").write_text("print('hello')")
        result = _collect_supplementary_content(skill_dir)
        assert "main.py" in result["scripts"]

    def test_collects_references(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        refs = skill_dir / "references"
        refs.mkdir()
        (refs / "guide.md").write_text("# Guide")
        result = _collect_supplementary_content(skill_dir)
        assert "guide.md" in result["references"]

    def test_truncates_large_files(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        scripts = skill_dir / "scripts"
        scripts.mkdir()
        (scripts / "big.py").write_text("x" * 10000)
        result = _collect_supplementary_content(skill_dir, max_bytes_per_file=100)
        assert len(result["scripts"]["big.py"]) < 200


class TestRubricEvalValidator:
    def test_no_skills_found_records_failed_execution_status(self, tmp_path):
        validator = RubricEvalValidator()
        result = validator.validate(tmp_path)

        assert not result.passed
        assert any("No skills found" in f.message for f in result.findings)
        rubric = result.metadata["rubric_eval"]
        assert rubric["execution_status"] == "failed"
        assert rubric["overall_score"] is None
        assert rubric["overall_pass"] is False
        assert rubric["summary"] == "No skills found in target directory"
        assert rubric["checks"] == []

    def test_missing_skill_md_records_failed_execution_status(self, tmp_path):
        skill_dir = tmp_path / "bad-skill"
        skill_dir.mkdir()

        result = RubricEvalValidator()._validate_single_skill(skill_dir)

        assert not result.passed
        assert any("SKILL.md not found" in f.message for f in result.findings)
        rubric = result.metadata["rubric_eval"]
        assert rubric["execution_status"] == "failed"
        assert rubric["overall_score"] is None
        assert rubric["overall_pass"] is False
        assert rubric["summary"] == "SKILL.md not found"
        assert rubric["skill_name"] == "bad-skill"
        assert rubric["checks"] == []

    def test_symlinked_manifest_is_rejected_before_llm_request(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        outside = tmp_path / "outside.md"
        outside.write_text("SECRET OUTSIDE CONTENT", encoding="utf-8")
        (skill_dir / "SKILL.md").symlink_to(outside)

        with patch.object(RubricJudge, "process") as judge:
            result = RubricEvalValidator().validate(skill_dir)

        judge.assert_not_called()
        assert not result.passed
        assert any(f.check_name == "unsafe_skill_input" for f in result.findings)

    @pytest.mark.parametrize("unsafe_entry", ["file_symlink", "directory_symlink", "non_regular"])
    def test_unsafe_supplementary_entry_is_rejected_before_llm_request(self, tmp_path, unsafe_entry):
        skill_dir = _write_skill(tmp_path)
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "leak.py").write_text("SECRET OUTSIDE CONTENT", encoding="utf-8")

        if unsafe_entry == "directory_symlink":
            (skill_dir / "scripts").symlink_to(outside, target_is_directory=True)
        else:
            scripts = skill_dir / "scripts"
            scripts.mkdir()
            if unsafe_entry == "file_symlink":
                (scripts / "leak.py").symlink_to(outside / "leak.py")
            else:
                if not hasattr(os, "mkfifo"):
                    pytest.skip("FIFO creation is unavailable")
                os.mkfifo(scripts / "blocked.py")

        with patch.object(RubricJudge, "process") as judge:
            result = RubricEvalValidator().validate(skill_dir)

        judge.assert_not_called()
        assert not result.passed
        assert any(f.check_name == "unsafe_skill_input" for f in result.findings)

    def test_oversized_manifest_is_rejected_before_llm_request(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_bytes(b"x" * (1024 * 1024 + 1))

        with patch.object(RubricJudge, "process") as judge:
            result = RubricEvalValidator().validate(skill_dir)

        judge.assert_not_called()
        assert not result.passed
        assert any("maximum size" in f.message.lower() for f in result.findings)

    def test_llm_fallback_when_no_key(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: test\n---\n# Test")

        from skillevaluator.inference.types import LLMClientError

        with patch.object(RubricJudge, "completions", side_effect=LLMClientError("no provider configured")):
            validator = RubricEvalValidator()
            result = validator.validate(skill_dir)
        assert not result.passed
        assert any("unavailable" in f.message.lower() for f in result.findings)
        assert any(f.check_name == "llm_unavailable" and f.severity.value == "high" for f in result.findings)
        assert result.metadata["rubric_eval"]["execution_status"] == "failed"
        assert "unavailable" in result.metadata["rubric_eval"]["judge_summary"].lower()

    def test_successful_evaluation(self, tmp_path):
        skill_dir = _write_skill(tmp_path)

        mock_report = {
            "overall_pass": True,
            "score": 85,
            "summary": "Good skill",
            "checks": [
                {
                    "id": criterion["id"],
                    "criterion": criterion["criterion"],
                    "pass": True,
                    "score": 9,
                    "notes": "Clear and specific",
                }
                for criterion in RUBRIC_CRITERIA
            ],
        }

        with patch.object(RubricJudge, "process", return_value=mock_report):
            validator = RubricEvalValidator()
            result = validator.validate(skill_dir)

        assert result.metadata.get("rubric_eval")
        assert result.metadata["rubric_eval"]["overall_score"] == 90.0
        assert result.metadata["rubric_eval"]["overall_pass"] is True
        assert result.metadata["rubric_eval"]["judge_score"] == 85
        assert result.metadata["rubric_eval"]["judge_overall_pass"] is True
        assert len(result.findings) == len(RUBRIC_CRITERIA)

    @pytest.mark.parametrize(
        ("judge_score", "judge_pass", "criterion_score", "expected_score", "expected_pass"),
        [
            (0, False, 8, 80.0, True),
            (100, True, 6, 60.0, False),
        ],
    )
    def test_model_totals_cannot_override_criterion_evidence(
        self,
        tmp_path,
        judge_score,
        judge_pass,
        criterion_score,
        expected_score,
        expected_pass,
    ):
        skill_dir = _write_skill(tmp_path)
        report = {
            "score": judge_score,
            "overall_pass": judge_pass,
            "summary": "Contradictory model aggregate",
            "checks": _complete_checks(scores={criterion["id"]: criterion_score for criterion in RUBRIC_CRITERIA}),
        }

        with patch.object(RubricJudge, "process", return_value=report):
            result = RubricEvalValidator(min_score=70).validate(skill_dir)

        rubric = result.metadata["rubric_eval"]
        assert rubric["overall_score"] == expected_score
        assert rubric["overall_pass"] is expected_pass
        assert rubric["judge_score"] == judge_score
        assert rubric["judge_overall_pass"] is judge_pass
        assert result.passed is expected_pass

    def test_criterion_pass_is_derived_from_seven_point_threshold(self, tmp_path):
        skill_dir = _write_skill(tmp_path)
        zero_score_id = RUBRIC_CRITERIA[0]["id"]
        report = {
            "score": 100,
            "overall_pass": True,
            "summary": "Contradictory criterion evidence",
            "checks": _complete_checks(
                scores={zero_score_id: 0},
                passes={zero_score_id: True},
            ),
        }

        with patch.object(RubricJudge, "process", return_value=report):
            result = RubricEvalValidator(min_score=70).validate(skill_dir)

        rubric = result.metadata["rubric_eval"]
        zero_score_check = next(check for check in rubric["checks"] if check["id"] == zero_score_id)
        assert zero_score_check["pass"] is False
        assert zero_score_check["judge_pass"] is True
        assert rubric["aggregation"]["criterion_pass_score"] == 7
        assert rubric["overall_pass"] is False
        assert not result.passed

    def test_failed_local_verdict_adds_blocking_aggregate_finding(self, tmp_path):
        skill_dir = _write_skill(tmp_path)
        report = {
            "score": 100,
            "overall_pass": True,
            "summary": "Contradictory aggregate",
            "checks": _complete_checks(scores={criterion["id"]: 6 for criterion in RUBRIC_CRITERIA}),
        }

        with patch.object(RubricJudge, "process", return_value=report):
            result = RubricEvalValidator(min_score=70).validate(skill_dir)

        assert not result.passed
        assert result.summary.errors == 1
        assert any(
            finding.check_name == "rubric_overall" and finding.severity.value == "high" for finding in result.findings
        )
        html = HTMLReporter(include_timestamp=False).render_all([result])
        assert '"would_block": true' in html

    def test_empty_checks_use_local_incomplete_diagnostic(self, tmp_path):
        skill_dir = _write_skill(tmp_path)
        judge_summary = "All rubric criteria passed"
        report = {"score": 100, "overall_pass": True, "summary": judge_summary, "checks": []}

        with patch.object(RubricJudge, "process", return_value=report):
            result = RubricEvalValidator(min_score=70).validate(skill_dir)

        rubric = result.metadata["rubric_eval"]
        assert not result.passed
        assert rubric["judge_summary"] == judge_summary
        assert judge_summary not in rubric["summary"]
        assert "incomplete" in rubric["summary"].lower()
        assert any(finding.check_name == "llm_invalid_response" for finding in result.findings)

    def test_local_score_overrides_false_model_criterion_pass(self, tmp_path):
        skill_dir = _write_skill(tmp_path)
        failed_id = RUBRIC_CRITERIA[0]["id"]
        report = {
            "score": 100,
            "overall_pass": True,
            "summary": "Contradictory pass",
            "checks": _complete_checks(
                scores={criterion["id"]: 10 for criterion in RUBRIC_CRITERIA},
                passes={failed_id: False},
            ),
        }

        with patch.object(RubricJudge, "process", return_value=report):
            result = RubricEvalValidator(min_score=70).validate(skill_dir)

        assert result.passed
        assert result.metadata["rubric_eval"]["overall_score"] == 100.0
        assert result.metadata["rubric_eval"]["overall_pass"] is True
        criterion_finding = next(finding for finding in result.findings if finding.metadata.get("id") == failed_id)
        assert criterion_finding.metadata["pass"] is True
        assert criterion_finding.metadata["judge_pass"] is False

    def test_overall_fields_are_optional_diagnostics(self, tmp_path):
        skill_dir = _write_skill(tmp_path)
        report = {
            "summary": "Complete criterion evidence",
            "checks": _complete_checks(scores={criterion["id"]: 7 for criterion in RUBRIC_CRITERIA}),
        }

        with patch.object(RubricJudge, "process", return_value=report):
            result = RubricEvalValidator(min_score=70).validate(skill_dir)

        assert result.passed
        assert result.metadata["rubric_eval"]["overall_score"] == 70.0
        assert result.metadata["rubric_eval"]["overall_pass"] is True
        assert result.metadata["rubric_eval"]["judge_score"] is None
        assert result.metadata["rubric_eval"]["judge_overall_pass"] is None

    def test_importance_weights_drive_local_score(self, tmp_path):
        skill_dir = _write_skill(tmp_path)
        scores_by_importance = {"high": 10, "medium": 5, "low": 0}
        scores = {criterion["id"]: scores_by_importance[criterion["importance"]] for criterion in RUBRIC_CRITERIA}
        report = {
            "score": 1,
            "overall_pass": False,
            "summary": "Model aggregate is ignored",
            "checks": _complete_checks(scores=scores),
        }

        with patch.object(RubricJudge, "process", return_value=report):
            result = RubricEvalValidator(min_score=70).validate(skill_dir)

        # Current schema weights high/medium/low as 3/2/1: 180 / 22 * 10.
        assert result.metadata["rubric_eval"]["overall_score"] == 81.8
        assert result.metadata["rubric_eval"]["overall_pass"] is False
        assert not result.passed

    @pytest.mark.parametrize("case", ["missing", "duplicate", "unexpected"])
    def test_missing_duplicate_or_unexpected_criteria_fail_closed(self, tmp_path, case):
        skill_dir = _write_skill(tmp_path)
        checks = _complete_checks()
        if case == "missing":
            checks.pop()
        elif case == "duplicate":
            checks[-1]["id"] = checks[0]["id"]
        else:
            checks.append(
                {
                    "id": "unexpected_criterion",
                    "criterion": "Unexpected",
                    "pass": True,
                    "score": 10,
                    "notes": "Not in the schema",
                }
            )

        with patch.object(
            RubricJudge,
            "process",
            return_value={"score": 100, "overall_pass": True, "summary": "invalid", "checks": checks},
        ):
            result = RubricEvalValidator(min_score=70).validate(skill_dir)

        rubric = result.metadata["rubric_eval"]
        assert not result.passed
        assert rubric["execution_status"] == "failed"
        assert rubric["overall_score"] is None
        assert rubric["overall_pass"] is False
        assert case in rubric["summary"].lower()
        assert any(finding.check_name == "llm_invalid_response" for finding in result.findings)

    @pytest.mark.parametrize(
        "case",
        [
            "missing_score",
            "boolean_score",
            "score_below_range",
            "score_above_range",
            "oversized_integer_score",
            "nonfinite_score",
            "missing_pass",
            "nonboolean_pass",
            "non_object_check",
        ],
    )
    def test_malformed_criterion_evidence_fails_closed(self, tmp_path, case):
        skill_dir = _write_skill(tmp_path)
        checks = _complete_checks()
        criterion_id = checks[0]["id"]
        if case == "missing_score":
            checks[0].pop("score")
        elif case == "boolean_score":
            checks[0]["score"] = True
        elif case == "score_below_range":
            checks[0]["score"] = -0.1
        elif case == "score_above_range":
            checks[0]["score"] = 10.1
        elif case == "oversized_integer_score":
            checks[0]["score"] = 10**1000
        elif case == "nonfinite_score":
            checks[0]["score"] = float("nan")
        elif case == "missing_pass":
            checks[0].pop("pass")
        elif case == "nonboolean_pass":
            checks[0]["pass"] = "true"
        else:
            checks[0] = "not an object"

        with patch.object(
            RubricJudge,
            "process",
            return_value={"score": 100, "overall_pass": True, "summary": "invalid", "checks": checks},
        ):
            result = RubricEvalValidator(min_score=70).validate(skill_dir)

        rubric = result.metadata["rubric_eval"]
        assert not result.passed
        assert rubric["execution_status"] == "failed"
        assert rubric["overall_score"] is None
        assert rubric["overall_pass"] is False
        assert rubric["checks"] == []
        if case != "non_object_check":
            assert criterion_id in rubric["summary"]
        assert any(finding.check_name == "llm_invalid_response" for finding in result.findings)

    def test_unavailable_judging_invalidates_folder_aggregate(self, tmp_path):
        _write_skill(tmp_path, "skill-a")
        _write_skill(tmp_path, "skill-b")
        complete_report = {
            "score": 100,
            "overall_pass": True,
            "summary": "complete",
            "checks": _complete_checks(scores={criterion["id"]: 10 for criterion in RUBRIC_CRITERIA}),
        }
        unavailable_report = RubricJudge().get_fallback_response()

        with patch.object(RubricJudge, "process", side_effect=[complete_report, unavailable_report]):
            result = RubricEvalValidator(min_score=70).validate(tmp_path)

        rubric = result.metadata["rubric_eval"]
        assert not result.passed
        assert rubric["execution_status"] == "failed"
        assert rubric["overall_score"] is None
        assert rubric["overall_pass"] is False
        assert any(finding.check_name == "llm_unavailable" for finding in result.findings)

    def test_incomplete_rubric_response_fails_closed(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: test\n---\n# Test")
        mock_report = {
            "overall_pass": True,
            "score": 100,
            "summary": "Looks complete",
            "checks": [
                {
                    "id": RUBRIC_CRITERIA[0]["id"],
                    "criterion": RUBRIC_CRITERIA[0]["criterion"],
                    "pass": True,
                    "score": 10,
                    "notes": "Only one criterion was returned",
                }
            ],
        }

        with patch.object(RubricJudge, "process", return_value=mock_report):
            result = RubricEvalValidator(min_score=70).validate(skill_dir)

        assert not result.passed
        assert result.metadata["rubric_eval"]["execution_status"] == "failed"
        assert any(finding.check_name == "llm_invalid_response" for finding in result.findings)

    def test_min_score_threshold(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: test\n---\n# Test")

        mock_report = {
            "overall_pass": False,
            "score": 45,
            "summary": "Needs work",
            "checks": _complete_checks(scores={criterion["id"]: 4 for criterion in RUBRIC_CRITERIA}),
        }

        with patch.object(RubricJudge, "process", return_value=mock_report):
            validator = RubricEvalValidator(min_score=60)
            result = validator.validate(skill_dir)

        assert not result.passed
        assert result.metadata["rubric_eval"]["execution_status"] == "succeeded"
        assert result.metadata["rubric_eval"]["overall_score"] == 40.0

    def test_local_scores_override_explicit_judge_failure(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: test\n---\n# Test")
        mock_report = {
            "overall_pass": False,
            "score": 85,
            "summary": "A required criterion failed",
            "checks": [
                {
                    "id": criterion["id"],
                    "criterion": criterion["criterion"],
                    "pass": index != 0,
                    "score": 8,
                    "notes": "failed" if index == 0 else "passed",
                }
                for index, criterion in enumerate(RUBRIC_CRITERIA)
            ],
        }

        with patch.object(RubricJudge, "process", return_value=mock_report):
            result = RubricEvalValidator(min_score=70).validate(skill_dir)

        assert result.passed
        assert result.metadata["rubric_eval"]["execution_status"] == "succeeded"
        first_check = result.metadata["rubric_eval"]["checks"][0]
        assert first_check["pass"] is True
        assert first_check["judge_pass"] is False

    def test_folder_validation_fails_when_judging_is_unavailable(self, tmp_path):
        for name in ["skill-a", "skill-b"]:
            d = tmp_path / name
            d.mkdir()
            (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: test\n---\n# {name}")

        mock_report = {
            "overall_pass": True,
            "score": 80,
            "summary": "ok",
            "checks": [],
        }

        with patch.object(RubricJudge, "process", return_value=mock_report):
            validator = RubricEvalValidator()
            result = validator.validate(tmp_path)

        assert not result.passed
        assert result.metadata["rubric_eval"]["execution_status"] == "failed"
        assert result.metadata["rubric_eval"]["overall_score"] is None
        assert result.metadata["rubric_eval"]["overall_pass"] is False
