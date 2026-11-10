# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

from skillevaluator.evaluation.tier3_findings import failure_reason
from skillevaluator.tier3.eval_core.declarative_checks import (
    combine_behavior_check,
    evaluate_declarative_checks,
    mark_no_skill_script_checks,
    validate_declarative_checks,
)
from skillevaluator.tier3.harbor.adapter import _copy_verifier
from skillevaluator.tier3.harbor.report import _collect_fail_reasons, _collect_pass_reasons


def _evaluate(check, *, calls=None, response="", roots=None):
    return evaluate_declarative_checks(
        [check],
        tool_calls=calls or [],
        final_response=response,
        workspace_roots=roots or [],
    )[0]


def _call(command, *, observation="", is_error=None):
    call = {"action": "Bash", "action_input": {"command": command}, "observation": observation}
    if is_error is not None:
        call["is_error"] = is_error
    return call


def test_script_invoked_checks_command_trajectory_only():
    check = {"type": "script_invoked", "script": "scripts/x.py"}

    assert _evaluate(check, calls=[_call("python scripts/x.py")])["passed"] is True
    assert (
        _evaluate(
            check,
            calls=[{"action": "exec_command", "action_input": {"command": "python scripts/x.py"}}],
        )["passed"]
        is True
    )
    assert _evaluate(check, calls=[_call("cat scripts/x.py")])["passed"] is False


def test_command_succeeded_requires_invocation_and_recorded_success():
    check = {"type": "command_succeeded", "script": "scripts/x.py"}

    assert _evaluate(check, calls=[_call("python scripts/x.py", observation="Exit code: 0")])["passed"] is True
    failed = _evaluate(check, calls=[_call("python scripts/x.py", observation="Exit code: 2")])
    assert failed["passed"] is False
    assert failed["reason"] == "Expected scripts/x.py to exit successfully; it was invoked but exited 2"
    assert _evaluate(check, calls=[_call("python scripts/x.py")])["passed"] is None
    assert _evaluate(check, calls=[_call("python other.py", observation="Exit code: 0")])["passed"] is False


def test_command_succeeded_uses_anchored_shared_exit_status():
    check = {"type": "command_succeeded", "script": "scripts/x.py"}

    failed = _evaluate(check, calls=[_call("python scripts/x.py", observation="Process exited with code 2")])
    passed = _evaluate(check, calls=[_call("python scripts/x.py", observation="Process exited with code 0")])
    claude_success = _evaluate(
        check,
        calls=[
            _call(
                "python scripts/x.py",
                observation="pytest exit code: 1 (1 failed)",
                is_error=False,
            )
        ],
    )

    assert failed["passed"] is False
    assert passed["passed"] is True
    assert claude_success["passed"] is True


@pytest.mark.parametrize(
    ("command", "observation"),
    [
        ("python scripts/x.py | tail -20", "Process exited with code 0"),
        ("python scripts/x.py || true", "Process exited with code 0"),
        ("python scripts/x.py; echo done", "Process exited with code 0"),
        ("bash -c 'python scripts/x.py; echo done'", "Process exited with code 0"),
        ("python scripts/x.py && pytest -q", "Process exited with code 1"),
    ],
)
def test_command_succeeded_does_not_attribute_shell_tail_status(command, observation):
    result = _evaluate(
        {"type": "command_succeeded", "script": "scripts/x.py"},
        calls=[_call(command, observation=observation)],
    )

    assert result["passed"] is None
    assert "could not be attributed" in result["reason"]


def test_zero_exit_from_pure_and_chain_proves_script_success():
    result = _evaluate(
        {"type": "command_succeeded", "script": "scripts/x.py"},
        calls=[_call("python scripts/x.py && pytest -q", observation="Process exited with code 0")],
    )

    assert result["passed"] is True


def test_file_exists_uses_workspace_not_response_mentions(tmp_path: Path):
    check = {"type": "file_exists", "path": "out/report.md"}
    output = tmp_path / "out" / "report.md"
    output.parent.mkdir()
    output.write_text("done", encoding="utf-8")

    assert _evaluate(check, roots=[tmp_path])["passed"] is True
    output.unlink()
    mentioned = _evaluate(check, response="Created out/report.md", roots=[tmp_path])
    assert mentioned["passed"] is False
    assert _evaluate(check, roots=[tmp_path / "unavailable"])["passed"] is None


def test_final_response_checks_match_contains_and_regex():
    assert _evaluate({"type": "output_contains", "text": "Report ready"}, response="Report ready.")["passed"] is True
    assert _evaluate({"type": "output_contains", "text": "Report ready"}, response="Still working")["passed"] is False
    assert _evaluate({"type": "output_matches", "pattern": r"Report \d+"}, response="Report 17")["passed"] is True
    assert _evaluate({"type": "output_matches", "pattern": r"Report \d+"}, response="No report")["passed"] is False


def test_runtime_invalid_regex_and_unknown_check_are_not_checked():
    invalid_regex = _evaluate({"type": "output_matches", "pattern": "["}, response="ready")
    unknown = _evaluate({"type": "made_up", "value": "x"})
    malformed_type = _evaluate({"type": []})

    assert invalid_regex["passed"] is None
    assert "not a valid regular expression" in invalid_regex["reason"]
    assert unknown["passed"] is None
    assert "Unknown check type" in unknown["reason"]
    assert malformed_type["passed"] is None
    assert "Unknown check type" in malformed_type["reason"]


def test_skill_script_checks_are_not_applicable_without_the_skill():
    checks = [{"type": "script_invoked", "script": "scripts/x.py"}]
    results = [{"type": "script_invoked", "passed": False, "reason": "not invoked"}]

    marked = mark_no_skill_script_checks(
        checks,
        results,
        has_skill=False,
        skill_metrics_on=False,
        evaluated_skill="demo",
    )

    assert marked[0]["passed"] is None
    assert marked[0]["reason"] == "skill script; not applicable without the skill"


def test_not_checked_items_are_excluded_and_coverage_is_reported():
    result = combine_behavior_check(
        {"score": 0.5, "status": "accepted", "results": [{"step": 1, "passed": True}]},
        [{"type": "command_succeeded", "passed": None, "reason": "status unavailable"}],
        judged_count=2,
    )

    assert result["score"] == 0.5
    assert result["declarative_checks"]["summary"] == "Checked directly: 0, judged: 2, not checked: 1"


def test_mixed_judged_and_direct_results_are_averaged():
    result = combine_behavior_check(
        {
            "score": 0.5,
            "status": "accepted",
            "results": [{"step": 1, "passed": True}, {"step": 2, "passed": False}],
        },
        [
            {"type": "output_contains", "passed": True, "reason": "found"},
            {"type": "output_matches", "passed": False, "reason": "missing"},
        ],
        judged_count=2,
    )

    assert result["score"] == 0.5
    assert result["declarative_checks"]["summary"] == "Checked directly: 2, judged: 2, not checked: 0"
    assert result["results"][-1]["reason"] == "missing"


def test_checks_only_compute_behavior_score_without_a_judge_result():
    result = combine_behavior_check(
        {"score": None, "status": "not_applicable", "reason": "N/A", "results": []},
        [{"type": "output_contains", "passed": True, "reason": "found"}],
        judged_count=0,
    )

    assert result["score"] == 1.0
    assert result["status"] == "accepted"
    assert result["reason"] == "Checked directly: 1, judged: 0, not checked: 0"


def test_not_applicable_judge_reason_is_dropped_when_checks_are_scored():
    result = combine_behavior_check(
        {"score": None, "status": "not_applicable", "reason": "No expected behaviors", "results": []},
        [{"type": "output_contains", "passed": True, "reason": "found"}],
        judged_count=0,
    )

    assert result["reason"] == "Checked directly: 1, judged: 0, not checked: 0"


def test_behavior_coverage_and_failure_reasons_reach_report_details():
    detail = {
        "reason": "Checked directly: 1, judged: 0, not checked: 0",
        "results": [{"type": "output_contains", "passed": False, "reason": "Expected final response text"}],
        "declarative_checks": {
            "summary": "Checked directly: 1, judged: 0, not checked: 0",
        },
    }
    trials = [{"detail": detail}]

    assert "Checked directly: 1, judged: 0, not checked: 0" in _collect_fail_reasons("behavior_check", trials)
    assert "Expected final response text" in _collect_fail_reasons("behavior_check", trials)
    assert "Checked directly: 1, judged: 0, not checked: 0" in _collect_pass_reasons("behavior_check", trials)
    assert "Checked directly: 1, judged: 0, not checked: 0" in failure_reason(detail)


def test_pass_reasons_count_only_judged_behavior_results():
    detail = {
        "reason": "Checked directly: 1, judged: 1, not checked: 0",
        "results": [
            {"type": "output_contains", "passed": True, "reason": "found"},
            {"step": 1, "passed": False, "reason": "not observed"},
        ],
    }

    reasons = _collect_pass_reasons("behavior_check", [{"detail": detail}])

    assert "0/1 expected behaviors observed" in reasons
    assert not any("expected behaviors observed" in reason and "0/1" not in reason for reason in reasons)


def test_unknown_check_type_lists_supported_types():
    errors = validate_declarative_checks([{"type": "made_up", "value": "x"}])

    assert len(errors) == 1
    assert "made_up" in errors[0]
    assert "script_invoked" in errors[0]
    assert "output_contains" in errors[0]


def test_dataset_validation_rejects_unknown_check_type(tmp_path):
    import json

    from skillevaluator.tier3.evals_spec import validate_skillevaluators

    skill = tmp_path / "skill"
    evals = skill / "evals"
    evals.mkdir(parents=True)
    (evals / "evals.json").write_text(
        json.dumps(
            {
                "skill_name": "skill",
                "evals": [
                    {
                        "id": "case-1",
                        "prompt": "Run it",
                        "expected_output": "Done",
                        "checks": [{"type": "made_up", "value": "x"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    errors = [result.message for result in validate_skillevaluators(skill) if result.status == "error"]

    assert any("made_up" in error and "script_invoked" in error for error in errors)


def test_check_validation_rejects_unsafe_paths_and_invalid_patterns():
    errors = validate_declarative_checks(
        [
            {"type": "file_exists", "path": "../outside"},
            {"type": "output_matches", "pattern": "["},
        ]
    )

    assert any("cannot traverse parent directories" in error for error in errors)
    assert any("valid regular expression" in error for error in errors)


def test_template_uses_shared_declarative_check_logic(harbor_eval_template):
    calls = [_call("python scripts/x.py", observation="Exit code: 0")]
    check = {"type": "command_succeeded", "script": "scripts/x.py"}

    host = evaluate_declarative_checks([check], tool_calls=calls, final_response="", workspace_roots=[])
    template = harbor_eval_template._shared_declarative_checks.evaluate_declarative_checks(
        [check], tool_calls=calls, final_response="", workspace_roots=[]
    )

    assert template == host


def test_standalone_bundle_stages_and_imports_the_shared_helper(tmp_path: Path, monkeypatch):
    _copy_verifier(tmp_path)
    helper_path = tmp_path / "tests/skill_evaluator/declarative_checks.py"
    monkeypatch.syspath_prepend(str(helper_path.parent))
    spec = importlib.util.spec_from_file_location("staged_declarative_checks", helper_path)
    assert spec is not None and spec.loader is not None
    staged = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(staged)

    result = staged.evaluate_declarative_checks(
        [{"type": "output_contains", "text": "ready"}],
        tool_calls=[],
        final_response="ready",
        workspace_roots=[],
    )

    assert helper_path.is_file()
    assert result[0]["passed"] is True
