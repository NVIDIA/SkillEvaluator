# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.tier3 import grader_validation as gv


@pytest.fixture
def suite_dir(tmp_path):
    import shutil

    destination = tmp_path / "suite"
    shutil.copytree(gv.REFERENCE_SUITE, destination)
    return destination


def write_suite(directory, suite):
    (directory / "suite.json").write_text(json.dumps(suite))


def test_reference_behavior_labels_distinguish_format_from_correctness():
    suite = gv.load_suite(gv.REFERENCE_SUITE)
    for case in suite["cases"]:
        assert case["label"]["reviewed_on"] == "2026-11-06"
        if case["trajectory"] is None:
            continue
        trajectory = json.loads((gv.REFERENCE_SUITE / case["trajectory"]).read_text())
        behaviors = case["entry"]["expected_behavior"]
        if "Return only the numeric answer." in behaviors:
            index = behaviors.index("Return only the numeric answer.")
            numeric_only = re.fullmatch(r"-?\d+(?:\.\d+)?", gv.get_final_response(trajectory)) is not None
            assert case["expected"]["metrics"]["behavior_check"][index] is numeric_only, case["id"]
        if case["id"] in {"wrong-answer", "tuning-wrong"}:
            assert case["expected"]["metrics"]["behavior_check"] == [True, True]
            assert case["expected"]["metrics"]["accuracy"] == "fail"
            assert case["expected"]["metrics"]["goal_accuracy"] == "fail"
            assert case["expected"]["overall"] == "fail"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda s: s.update(extra=True), "unknown fields"),
        (lambda s: s.update(schema="v2"), "schema must"),
        (lambda s: s["cases"].append(copy.deepcopy(s["cases"][0])), "duplicate"),
        (lambda s: s["cases"][0].update(category="other"), "unknown category"),
        (lambda s: s["cases"][0].update(category=[]), "unknown category"),
        (lambda s: s["cases"][0].update(trajectory="../outside.json"), "inside suite"),
        (lambda s: s["cases"][0].update(trajectory="absent.json"), "not found"),
        (lambda s: s["cases"][0]["expected"]["metrics"].update(behavior_check=[True]), "behavior label length"),
        (lambda s: s["cases"][0]["entry"].update(unknown="x"), "unknown fields"),
        (lambda s: s.update(thresholds={"accuracy": 2}), "expected number"),
        (lambda s: s["cases"][0]["label"].update(checks=[]), "require checks"),
        (lambda s: s["cases"][0]["label"]["checks"][0].update(value="not present"), "check failed"),
        (lambda s: s["cases"][0]["label"]["checks"][0].update(type="unknown"), "unknown executable"),
    ],
)
def test_invalid_suite(suite_dir, mutation, message):
    suite = json.loads((suite_dir / "suite.json").read_text())
    mutation(suite)
    write_suite(suite_dir, suite)
    with pytest.raises(ValueError, match=message):
        gv.load_suite(suite_dir)


def test_symlink_escape(suite_dir, tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text('{"steps":[]}')
    (suite_dir / "link.json").symlink_to(outside)
    suite = json.loads((suite_dir / "suite.json").read_text())
    suite["cases"][0]["trajectory"] = "link.json"
    write_suite(suite_dir, suite)
    with pytest.raises(ValueError, match="inside suite"):
        gv.load_suite(suite_dir)


def test_executable_checks():
    trajectory = {
        "steps": [
            {
                "source": "agent",
                "tool_calls": [
                    {"function_name": "Write", "arguments": {"file_path": "/workspace/a", "content": "done"}}
                ],
                "observation": {"results": [{"content": "ok"}]},
            },
            {"source": "agent", "message": "finished"},
        ]
    }
    checks = [
        {"type": "final_response_contains", "value": "finished"},
        {"type": "final_response_not_contains", "value": "failed"},
        {"type": "tool_result_contains", "value": "ok"},
        {"type": "file_written_contains", "path": "/workspace/a", "value": "done"},
    ]
    gv.verify_checks({"method": "executable", "checks": checks}, trajectory, "test")
    checks[-1]["path"] = "/workspace/b"
    with pytest.raises(ValueError, match="check failed"):
        gv.verify_checks({"method": "executable", "checks": checks}, trajectory, "test")


def test_tool_result_checks_match_recorded_content_text_only():
    trajectory = {
        "steps": [
            {
                "observation": {
                    "results": [
                        None,
                        {"source_call_id": "call-3", "content": "Result: 14\nProcess exited with code 0"},
                    ]
                }
            }
        ]
    }
    label = {
        "method": "executable",
        "checks": [{"type": "tool_result_contains", "value": "Result: 14\nProcess exited"}],
    }
    gv.verify_checks(label, trajectory, "test")
    label["checks"] = [{"type": "tool_result_contains", "value": "source_call_id"}]
    with pytest.raises(ValueError, match="check failed"):
        gv.verify_checks(label, trajectory, "test")


@pytest.mark.parametrize(
    "tool_calls",
    [
        {"function_name": "Write", "arguments": {"file_path": "/workspace/a", "content": "done"}},
        [{"function_name": None, "arguments": {"file_path": "/workspace/a", "content": "done"}}],
        [{"function_name": "Write", "arguments": "not an object"}],
        [{"function_name": "Write", "arguments": {"file_path": "/workspace/a", "content": None}}],
    ],
)
def test_malformed_file_evidence_is_a_named_suite_error(suite_dir, tool_calls):
    suite = gv.load_suite(suite_dir)
    case = next(case for case in suite["cases"] if case["trajectory"] is not None)
    trajectory_path = suite_dir / case["trajectory"]
    trajectory = json.loads(trajectory_path.read_text())
    trajectory["steps"] = [{"source": "agent", "tool_calls": tool_calls}]
    trajectory_path.write_text(json.dumps(trajectory))
    case["label"]["method"] = "executable"
    case["label"]["checks"] = [{"type": "file_written_contains", "path": "/workspace/a", "value": "done"}]
    write_suite(suite_dir, suite)
    result = CliRunner().invoke(
        cli,
        ["tier3", "validate-grader", str(suite_dir), "--check-suite", "--case", case["id"]],
    )
    assert result.exit_code == 2, result.output
    assert case["id"] in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("reward", "metric", "expected"),
    [
        ({"accuracy": 0.8}, "accuracy", "pass"),
        ({"accuracy": 0.79}, "accuracy", "fail"),
        ({"error": "No trajectory or reconstructible agent log", "overall": 0}, "overall", "insufficient_evidence"),
        ({"details": {"accuracy": {"status": "error"}}}, "accuracy", "error"),
        ({"details": {"skill_execution": {"status": "not_applicable"}}}, "skill_execution", "not_applicable"),
        ({"overall": 0, "evaluation_status": "failed"}, "overall", "error"),
        ({}, "accuracy", "error"),
        ({"accuracy": float("nan")}, "accuracy", "error"),
    ],
)
def test_classification(reward, metric, expected):
    assert gv.classify(reward, metric, gv.DEFAULT_THRESHOLDS) == expected


def test_math_and_variability():
    rows = [
        {"judgment": "a:overall", "expected": "fail", "observed": v, "score": s}
        for v, s in [("pass", 0.8), ("fail", 0.2), ("error", None), ("insufficient_evidence", None)]
    ] + [{"judgment": "b:overall", "expected": "pass", "observed": "fail", "score": 0.1}]
    result = gv.summarize(rows)
    assert result["false_pass_rate"]["numerator"] == 1
    assert result["false_pass_rate"]["denominator"] == 2
    assert result["false_fail_rate"]["rate"] == 1
    assert result["unjudgeable_rate"]["numerator"] == 2
    assert result["unjudgeable_rate"]["denominator"] == 5
    assert result["variability"]["rate"] == 1
    assert result["variability"]["denominator"] == 1
    assert result["score_spread_by_judgment"]["a:overall"] == pytest.approx(0.6)
    assert result["cases"] == result["judgments"] == 2
    low, high = result["false_pass_rate"]["wilson_95"]
    assert low == pytest.approx(0.0945312)
    assert high == pytest.approx(0.9054688)
    assert gv.rate(0, 0)["wilson_95"] is None
    assert gv.summarize([])["attempts"] == 0
    assert gv.summarize([])["variability"]["rate"] is None


def test_behavior_pool():
    case = {"id": "mixed", "expected": {"overall": "fail", "metrics": {"behavior_check": [True, False]}}}
    reward = {
        "overall": 0.6,
        "behavior_check": 0.5,
        "details": {"behavior_check": {"results": [{"step": 2, "passed": False}, {"step": 1, "passed": True}]}},
    }
    rows = gv.observations(case, reward, gv.DEFAULT_THRESHOLDS)
    assert [r["observed"] for r in rows] == ["pass", "fail", "pass", "fail"]
    reward["details"]["behavior_check"]["status"] = "error"
    assert all(r["observed"] == "error" for r in gv.observations(case, reward, gv.DEFAULT_THRESHOLDS)[1:])


@pytest.fixture
def unchecked_behavior():
    case = {"id": "baseline", "expected": {"overall": "fail", "metrics": {"behavior_check": [False, True]}}}
    reward = {
        "overall": 0.4,
        "behavior_check": 0.5,
        "details": {
            "behavior_check": {
                "score": 0.5,
                "reason": "The numeric answer was returned, but instruction reading cannot be verified.",
                "results": [
                    {"step": 1, "passed": None, "reason": "Tool execution details are not provided."},
                    {"step": 2, "passed": True, "reason": "The final response was only the number 14."},
                ],
                "expectations": [
                    {
                        "step": 1,
                        "requirement": "Read calculator instructions before running calc.py.",
                        "status": "not_checked",
                        "reason": "Tool execution details are not provided.",
                    },
                    {
                        "step": 2,
                        "requirement": "Return only the numeric answer.",
                        "status": "met",
                        "reason": "The final response was only the number 14.",
                    },
                ],
            }
        },
    }
    return case, reward


@pytest.mark.parametrize("reverse_expectations", [False, True])
def test_unchecked_behavior_step_projects_insufficient_evidence(unchecked_behavior, reverse_expectations):
    case, reward = unchecked_behavior
    if reverse_expectations:
        reward["details"]["behavior_check"]["expectations"].reverse()
    original = copy.deepcopy(reward)
    rows = gv.observations(case, reward, gv.DEFAULT_THRESHOLDS)
    assert [(r["metric"], r["expected"], r["observed"], r["score"]) for r in rows] == [
        ("overall", "fail", "fail", 0.4),
        ("behavior_check", "fail", "fail", 0.5),
        ("behavior_steps", "fail", "insufficient_evidence", None),
        ("behavior_steps", "pass", "pass", 1),
    ]
    assert reward == original
    summary = gv.summarize([r for r in rows if r["metric"] == "behavior_steps"])
    assert summary["errors"] == 0
    assert summary["insufficient_evidence"] == 1
    assert summary["confusion_matrix"]["fail"]["insufficient_evidence"] == 1
    assert summary["unjudgeable_rate"]["numerator"] == 1
    assert summary["unjudgeable_rate"]["denominator"] == 2
    without_status = copy.deepcopy(reward)
    without_status["details"]["behavior_check"].pop("expectations")
    error_summary = gv.summarize(
        [r for r in gv.observations(case, without_status, gv.DEFAULT_THRESHOLDS) if r["metric"] == "behavior_steps"]
    )
    assert error_summary["errors"] == 1
    assert error_summary["insufficient_evidence"] == 0
    assert summary["unjudgeable_rate"] == error_summary["unjudgeable_rate"]


@pytest.mark.parametrize(
    "expectations",
    [
        None,
        {},
        [],
        [None],
        [{"status": "not_checked"}],
        [{"step": 2, "status": "not_checked"}],
        [{"step": "1", "status": "not_checked"}],
        [{"step": True, "status": "not_checked"}],
        [{"step": 1.0, "status": "not_checked"}],
        [{"step": 1}],
        *[{"step": 1, "status": status} for status in (None, "met", "unmet", "error", "unknown", [], {})],
    ],
)
def test_null_behavior_step_without_valid_matching_status_is_error(unchecked_behavior, expectations):
    case, reward = unchecked_behavior
    reward["details"]["behavior_check"]["expectations"] = expectations
    steps = [r for r in gv.observations(case, reward, gv.DEFAULT_THRESHOLDS) if r["metric"] == "behavior_steps"]
    assert [(r["observed"], r["score"]) for r in steps] == [("error", None), ("pass", 1)]


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"step": 1},
        {"step": 2, "passed": None},
        {"step": "1", "passed": None},
        {"step": True, "passed": None},
        {"step": 1.0, "passed": None},
        *[{"step": 1, "passed": passed} for passed in ("true", 1, 0, {}, [])],
    ],
)
def test_malformed_behavior_result_cannot_abstain(unchecked_behavior, result):
    case, reward = unchecked_behavior
    reward["details"]["behavior_check"]["results"][0] = result
    first = gv.observations(case, reward, gv.DEFAULT_THRESHOLDS)[2]
    assert (first["observed"], first["score"]) == ("error", None)


@pytest.mark.parametrize("passed", [False, True])
def test_boolean_behavior_result_keeps_verdict_with_not_checked_status(unchecked_behavior, passed):
    case, reward = unchecked_behavior
    reward["details"]["behavior_check"]["results"][0]["passed"] = passed
    first = gv.observations(case, reward, gv.DEFAULT_THRESHOLDS)[2]
    assert (first["observed"], first["score"]) == ("pass" if passed else "fail", int(passed))


@pytest.mark.parametrize("failure", ["status", "process", "score"])
def test_aggregate_behavior_error_overrides_not_checked_step(unchecked_behavior, failure):
    case, reward = unchecked_behavior
    if failure == "status":
        reward["details"]["behavior_check"]["status"] = "error"
    elif failure == "process":
        reward["_process_error"] = True
    else:
        reward.pop("behavior_check")
    rows = gv.observations(case, reward, gv.DEFAULT_THRESHOLDS)
    assert [r["observed"] for r in rows[1:]] == ["error", "error", "error"]


def test_behavior_expected_class_uses_suite_threshold():
    case = {"id": "mixed", "expected": {"overall": "fail", "metrics": {"behavior_check": [True, False]}}}
    reward = {
        "behavior_check": 0.5,
        "details": {"behavior_check": {"results": [{"step": 1, "passed": True}, {"step": 2, "passed": False}]}},
    }
    thresholds = gv.DEFAULT_THRESHOLDS | {"behavior_check": 0.5}
    rows = gv.observations(case, reward, thresholds)
    behavior = next(row for row in rows if row["metric"] == "behavior_check")
    steps = [row for row in rows if row["metric"] == "behavior_steps"]
    assert behavior["expected"] == behavior["observed"] == "pass"
    assert [(row["expected"], row["observed"]) for row in steps] == [("pass", "pass"), ("fail", "fail")]


def test_bundled_check_suite_and_usage(monkeypatch):
    monkeypatch.setattr(gv, "resolve_llm_provider", lambda: pytest.fail("provider accessed"))
    runner = CliRunner()
    result = runner.invoke(cli, ["tier3", "validate-grader", "--check-suite"])
    assert result.exit_code == 0, result.output
    assert "14 assessment cases" in result.output
    tuning = runner.invoke(cli, ["tier3", "validate-grader", "--check-suite", "--split", "tuning", "--json"])
    assert json.loads(tuning.output)["cases"] == 2
    for args in (["--case", "absent"], ["--repeats", "0"], ["--timeout", "0"], ["--max-false-pass-rate", "2"]):
        assert runner.invoke(cli, ["tier3", "validate-grader", "--check-suite", *args]).exit_code == 2


def test_suite_default_is_portable_and_documented():
    argument = next(param for param in gv.validate_grader.params if param.name == "suite_dir")
    assert argument.default is None
    assert argument.required is False
    runner = CliRunner()
    help_result = runner.invoke(cli, ["tier3", "validate-grader", "--help"])
    assert help_result.exit_code == 0
    assert "SUITE_DIR defaults to the bundled reference suite when omitted." in help_result.output
    implicit = runner.invoke(cli, ["tier3", "validate-grader", "--check-suite", "--json"])
    explicit = runner.invoke(cli, ["tier3", "validate-grader", str(gv.REFERENCE_SUITE), "--check-suite", "--json"])
    assert implicit.exit_code == explicit.exit_code == 0
    assert json.loads(implicit.output) == json.loads(explicit.output)


def test_missing_provider(monkeypatch):
    from skillevaluator.provider_config import ProviderConfigurationError

    def missing():
        raise ProviderConfigurationError("No judge provider configured")

    monkeypatch.setattr(gv, "resolve_llm_provider", missing)
    result = CliRunner().invoke(cli, ["tier3", "validate-grader"])
    assert result.exit_code == 2
    assert "No judge provider configured" in result.output


@pytest.fixture
def judge_server(monkeypatch):
    state = {"fail": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if state["fail"]:
                self.send_response(400)
                self.end_headers()
                return
            prompt = json.dumps(request["messages"])
            if "ACTION_CORRECT" in prompt:
                reply = {
                    "criteria": dict.fromkeys(
                        ["ACTION_CORRECT", "FACTUALLY_ACCURATE", "TASK_ADDRESSED", "ACTIONABLE", "FORMAT_FOLLOWED"],
                        True,
                    ),
                    "reason": "evidence matches",
                }
            elif "achieved" in prompt:
                reply = {"user_goal": "compute", "end_state": "14", "achieved": True, "reason": "met"}
            else:
                reply = {
                    "results": [
                        {"step": 1, "passed": True, "reason": "read"},
                        {"step": 2, "passed": True, "reason": "numeric"},
                    ],
                    "summary": "met",
                }
            body = json.dumps({"choices": [{"message": {"content": json.dumps(reply)}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", "qualification-test")
    monkeypatch.setenv("SKILL_EVAL_LLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("SKILL_EVAL_LLM_API_KEY", "test-secret-never-in-provenance")
    monkeypatch.setenv("SKILL_EVAL_LLM_MAX_RETRIES", "0")
    yield state
    server.shutdown()
    thread.join()
    server.server_close()


def test_real_template_cli(judge_server, tmp_path):
    output = tmp_path / "results"
    result = CliRunner().invoke(
        cli,
        [
            "tier3",
            "validate-grader",
            "--case",
            "arithmetic",
            "--case",
            "missing",
            "--case",
            "baseline",
            "--repeats",
            "2",
            "--concurrency",
            "2",
            "-o",
            str(output),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["coverage"]["attempts"] == 6
    assert report["metrics"]["overall"]["expected_insufficient"]["correctly_abstained"] == 2
    assert report["metrics"]["skill_execution"]["confusion_matrix"]["not_applicable"]["not_applicable"] == 2
    reward = report["attempts"][0]["reward"]
    assert reward["security"] == 1
    assert reward["skill_execution"] == 1
    assert reward["accuracy"] == 1
    assert report["provenance"]["contracts"]["judge_contract"]
    assert report["provenance"]["judge_models_used"] == ["qualification-test"]
    assert "test-secret-never-in-provenance" not in json.dumps(report)
    assert (output / "arithmetic/attempt-1/judge_attempts.json").exists()
    assert json.loads((output / "grader-validation.json").read_text()) == report


def test_attempt_uses_resolved_provider_environment(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.delenv("SKILL_EVAL_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-openai-key")
    monkeypatch.setenv("NVIDIA_API_KEY", "synthetic-nvidia-key")

    def run(_command, *, env, **_kwargs):
        captured.update(env)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(gv.subprocess, "run", run)
    case = {
        "id": "provider",
        "entry": {},
        "trajectory": None,
        "expected": {"overall": "fail", "metrics": {}},
    }
    gv._attempt(
        gv.REFERENCE_SUITE,
        case,
        1,
        tmp_path,
        gv.DEFAULT_THRESHOLDS,
        1,
        {
            "SKILL_EVAL_LLM_PROVIDER": "openai",
            "SKILL_EVAL_LLM_MODEL": "resolved-model",
            "OPENAI_API_KEY": "synthetic-openai-key",
        },
    )
    assert captured["SKILL_EVAL_LLM_PROVIDER"] == "openai"
    assert captured["SKILL_EVAL_LLM_MODEL"] == "resolved-model"
    assert captured["NVIDIA_API_KEY"] == "synthetic-nvidia-key"


def test_provenance_records_resolved_judge_controls_and_verifier_files(tmp_path):
    from skillevaluator.tier3.harbor.adapter import _VERIFIER_FILES
    from skillevaluator.tier3.harbor.runner import _judge_model_config

    provider = SimpleNamespace(provider="openai", model="chat-model")
    provider_env = {
        "LLM_JUDGE_MODEL": "override-judge",
        "LLM_JUDGE_FALLBACK_MODELS": "fallback-judge",
        "LLM_JUDGE_FALLBACK_POLICY": "replace",
        "LLM_JUDGE_RETRY_POLICY": "strict",
        "SKILL_EVAL_JUDGE_CHARACTER_FACTS": "1",
        "SKILL_EVAL_LLM_MAX_RETRIES": "2",
        "SKILL_EVAL_JUDGE_FALLBACK_MODEL": "unused",
        "SKILL_EVAL_JUDGE_REASONING_EFFORT": "unused",
        "SKILL_EVAL_GOAL_ACCURACY_METHOD": "unused",
        "OPENAI_API_KEY": "synthetic-secret",
    }
    suite = gv.load_suite(gv.REFERENCE_SUITE)
    output = tmp_path / "provenance"
    output.mkdir()
    result = gv.qualify(
        gv.REFERENCE_SUITE,
        suite,
        [],
        output,
        1,
        1,
        1,
        "assessment",
        provider,
        provider_env,
        _judge_model_config(provider, provider_env, "default"),
    )
    provenance = result["provenance"]
    assert provenance["judge_config"]["model"] == "override-judge"
    assert provenance["judge_environment"]["LLM_JUDGE_FALLBACK_MODELS"] == "fallback-judge"
    assert provenance["judge_environment"]["LLM_JUDGE_RETRY_POLICY"] == "strict"
    assert provenance["judge_environment"]["SKILL_EVAL_LLM_MAX_RETRIES"] == "2"
    assert "SKILL_EVAL_JUDGE_FALLBACK_MODEL" not in provenance["judge_environment"]
    assert "SKILL_EVAL_JUDGE_REASONING_EFFORT" not in provenance["judge_environment"]
    assert "SKILL_EVAL_GOAL_ACCURACY_METHOD" not in provenance["judge_environment"]
    assert "OPENAI_API_KEY" not in provenance["judge_environment"]
    assert provenance["skillevaluator_version"] == gv.__version__
    digests = provenance["verifier_file_sha256"]
    assert len(digests) == len(_VERIFIER_FILES)
    assert any(name.endswith("/eval.py") for name in digests)
    assert any(name.endswith("/checks.py") for name in digests)
    assert any(name.endswith("/evidence.py") for name in digests)
    assert len(provenance["verifier_sha256"]) == 64


@pytest.mark.parametrize("package_root", ["src", "site-packages"])
@pytest.mark.parametrize("use_alias", [False, True], ids=["canonical", "symlink-alias"])
def test_verifier_provenance_preserves_relative_hashes_through_source_alias(
    monkeypatch, tmp_path, package_root, use_alias
):
    from skillevaluator.tier3.harbor import adapter

    source = tmp_path / package_root
    module = source / "skillevaluator/tier3/grader_validation.py"
    module.parent.mkdir(parents=True)
    module.touch()
    contents = {
        "skillevaluator/tier3/eval_core/checks.py": b"checks\n",
        "skillevaluator/tier3/harbor/templates/eval.py": b"verifier\n",
    }
    for name, content in contents.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    imported_root = source
    if use_alias:
        imported_root = tmp_path / "source-alias"
        imported_root.symlink_to(source, target_is_directory=True)
    monkeypatch.setattr(gv, "__file__", str(imported_root / module.relative_to(source)))
    monkeypatch.setattr(adapter, "_VERIFIER_FILES", [imported_root / name for name in reversed(contents)])
    output = tmp_path / "qualification"
    output.mkdir()
    report = gv.qualify(
        gv.REFERENCE_SUITE,
        gv.load_suite(gv.REFERENCE_SUITE),
        [],
        output,
        1,
        1,
        1,
        "assessment",
        SimpleNamespace(provider="openai", model="test-model"),
        {},
        {},
    )
    expected_digest = hashlib.sha256()
    for name, content in contents.items():
        expected_digest.update(name.encode() + b"\0" + len(content).to_bytes(8, "big") + content)
    assert report["provenance"]["verifier_file_sha256"] == {
        name: hashlib.sha256(content).hexdigest() for name, content in contents.items()
    }
    assert report["provenance"]["verifier_sha256"] == expected_digest.hexdigest()
    assert report["provenance"]["verifier_git_commit"] is None
    assert report["provenance"]["verifier_git_dirty"] is None
    assert json.loads((output / "grader-validation.json").read_text())["provenance"] == report["provenance"]


@pytest.mark.parametrize("symlink_escape", [False, True], ids=["outside-file", "symlink-escape"])
def test_verifier_provenance_rejects_files_outside_source_root(monkeypatch, tmp_path, symlink_escape):
    from skillevaluator.tier3.harbor import adapter

    source = tmp_path / "site-packages"
    module = source / "skillevaluator/tier3/grader_validation.py"
    module.parent.mkdir(parents=True)
    module.touch()
    outside = tmp_path / "outside.py"
    outside.write_text("outside source boundary\n")
    verifier = outside
    if symlink_escape:
        verifier = module.parent / "escaped.py"
        verifier.symlink_to(outside)
    monkeypatch.setattr(gv, "__file__", str(module))
    monkeypatch.setattr(adapter, "_VERIFIER_FILES", [verifier])
    output = tmp_path / "qualification"
    output.mkdir()
    with pytest.raises(ValueError, match="not in the subpath"):
        gv.qualify(
            gv.REFERENCE_SUITE,
            gv.load_suite(gv.REFERENCE_SUITE),
            [],
            output,
            1,
            1,
            1,
            "assessment",
            SimpleNamespace(provider="openai", model="test-model"),
            {},
            {},
        )
    assert not (output / "grader-validation.json").exists()


def test_split_separation_and_gates(judge_server, tmp_path):
    result = CliRunner().invoke(
        cli,
        [
            "tier3",
            "validate-grader",
            "--split",
            "all",
            "--case",
            "arithmetic",
            "--case",
            "tuning-wrong",
            "--repeats",
            "1",
            "-o",
            str(tmp_path / "all"),
            "--json",
            "--max-false-pass-rate",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["metrics"] == {}
    assert report["by_split"]["assessment"]["overall"]["cases"] == 1
    assert report["by_split"]["tuning"]["overall"]["false_pass_rate"]["rate"] == 1
    assert gv.gate_failures(report, {"false_pass_rate": 0}) == []
    gated = CliRunner().invoke(
        cli,
        [
            "tier3",
            "validate-grader",
            "--split",
            "tuning",
            "--case",
            "tuning-wrong",
            "--repeats",
            "1",
            "-o",
            str(tmp_path / "gated"),
            "--json",
            "--max-false-pass-rate",
            "0",
        ],
    )
    assert gated.exit_code == 1, gated.output
    assert json.loads(gated.output)["metrics"]["overall"]["false_pass_rate"]["rate"] == 1
    assert (
        gv.gate_failures({"metrics": {"overall": {"false_pass_rate": gv.rate(1, 2)}}}, {"false_pass_rate": 0.5}) == []
    )
    text = CliRunner().invoke(
        cli,
        [
            "tier3",
            "validate-grader",
            "--split",
            "all",
            "--case",
            "arithmetic",
            "--case",
            "tuning-wrong",
            "--repeats",
            "1",
            "-o",
            str(tmp_path / "all-text"),
        ],
    )
    assert text.exit_code == 0, text.output
    assert "Assessment grader qualification" in text.output
    assert "Tuning grader qualification" in text.output


def test_timeout_is_unscored(monkeypatch, tmp_path):
    import subprocess

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("verifier", 0.01)

    monkeypatch.setattr(gv.subprocess, "run", timeout)
    suite = gv.load_suite(gv.REFERENCE_SUITE)
    result = gv._attempt(gv.REFERENCE_SUITE, suite["cases"][0], 1, tmp_path, gv.DEFAULT_THRESHOLDS, 0.01)
    assert result["returncode"] == -1
    assert all(row["observed"] == "error" for row in result["observations"])


def test_real_judge_failure_is_unscored(judge_server, tmp_path):
    judge_server["fail"] = True
    result = CliRunner().invoke(
        cli,
        [
            "tier3",
            "validate-grader",
            "--case",
            "arithmetic",
            "--repeats",
            "1",
            "-o",
            str(tmp_path / "failed"),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["attempts"][0]["returncode"] == 1
    assert report["attempts"][0]["reward"]["evaluation_status"] == "failed"
    assert report["metrics"]["accuracy"]["errors"] == 1
    assert report["metrics"]["accuracy"]["false_fail_rate"]["denominator"] == 0
    assert report["metrics"]["overall"]["errors"] == 1
    assert report["metrics"]["security"]["confusion_matrix"]["pass"]["pass"] == 1


@pytest.fixture
def qualification_policy(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "schema": "skillevaluator.grader-qualification-policy/v1",
                "policy_id": "test-overall-v1",
                "required_metrics": ["overall"],
                "min_class_cases": {"pass": 1, "fail": 1},
                "min_category_cases": {},
                "max_rates": {"false_pass_rate": 0.1, "false_fail_rate": 0.1, "unjudgeable_rate": 0.05},
            }
        )
    )
    return path


def test_policy_offline_readiness_no_provider_or_output(monkeypatch, qualification_policy, tmp_path):
    monkeypatch.setattr(gv, "resolve_llm_provider", lambda: pytest.fail("provider accessed"))
    output = tmp_path / "not-written"
    runner = CliRunner()
    ready = runner.invoke(
        cli,
        [
            "tier3",
            "validate-grader",
            "--qualification-policy",
            str(qualification_policy),
            "--check-suite",
            "--json",
            "-o",
            str(output),
        ],
    )
    assert ready.exit_code == 0, ready.output
    assert json.loads(ready.output)["qualification"]["status"] == "ready"
    insufficient = runner.invoke(
        cli,
        [
            "tier3",
            "validate-grader",
            "--qualification-policy",
            str(qualification_policy),
            "--check-suite",
            "--case",
            "arithmetic",
            "--json",
            "-o",
            str(output),
        ],
    )
    assert insufficient.exit_code == 1, insufficient.output
    assert json.loads(insufficient.output)["qualification"]["status"] == "insufficient_evidence"
    assert not output.exists()


def test_policy_insufficient_preflight_retains_not_run_receipt(monkeypatch, qualification_policy, tmp_path):
    monkeypatch.setattr(gv, "resolve_llm_provider", lambda: pytest.fail("provider accessed"))
    output = tmp_path / "preflight"
    args = [
        "tier3",
        "validate-grader",
        "--qualification-policy",
        str(qualification_policy),
        "--case",
        "arithmetic",
        "--json",
        "-o",
        str(output),
    ]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1, result.output
    report = json.loads(result.output)
    assert report["attempts"] == []
    assert report["execution_status"] == "not_run"
    assert report["qualification"]["phase"] == "preflight"
    assert report["qualification"]["status"] == "insufficient_evidence"
    assert report["provenance"]["case_ids"] == ["arithmetic"]
    assert report["provenance"]["provider"] is None
    assert json.loads((output / "grader-validation.json").read_text()) == report
    assert len(list(output.iterdir())) == 1
    assert CliRunner().invoke(cli, args).exit_code == 2


def test_policy_human_coverage_reason(monkeypatch, qualification_policy, tmp_path):
    monkeypatch.setattr(gv, "resolve_llm_provider", lambda: pytest.fail("provider accessed"))
    result = CliRunner().invoke(
        cli,
        [
            "tier3",
            "validate-grader",
            "--qualification-policy",
            str(qualification_policy),
            "--case",
            "arithmetic",
            "-o",
            str(tmp_path / "human"),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "overall" in result.output and "fail" in result.output and "coverage" in result.output


@pytest.mark.parametrize("args", [["--split", "tuning"], ["--split", "all"]])
def test_policy_assessment_only(monkeypatch, qualification_policy, args):
    monkeypatch.setattr(gv, "resolve_llm_provider", lambda: pytest.fail("provider accessed"))
    result = CliRunner().invoke(
        cli, ["tier3", "validate-grader", "--qualification-policy", str(qualification_policy), "--check-suite", *args]
    )
    assert result.exit_code == 2
    assert "assessment" in result.output


def test_malformed_policy_before_provider(monkeypatch, qualification_policy, tmp_path):
    monkeypatch.setattr(gv, "resolve_llm_provider", lambda: pytest.fail("provider accessed"))
    qualification_policy.write_text('{"schema":"wrong"}')
    output = tmp_path / "never-written"
    result = CliRunner().invoke(
        cli, ["tier3", "validate-grader", "--qualification-policy", str(qualification_policy), "-o", str(output)]
    )
    assert result.exit_code == 2
    assert not output.exists()


def test_policy_real_template_decision_equals_receipt(judge_server, qualification_policy, tmp_path):
    output = tmp_path / "policy-live"
    result = CliRunner().invoke(
        cli,
        [
            "tier3",
            "validate-grader",
            "--qualification-policy",
            str(qualification_policy),
            "--case",
            "arithmetic",
            "--case",
            "wrong-answer",
            "--repeats",
            "1",
            "--json",
            "-o",
            str(output),
        ],
    )
    assert result.exit_code == 1, result.output
    report = json.loads(result.output)
    assert report["qualification"]["phase"] == "evaluated"
    assert report["qualification"]["status"] == "failed"
    assert report["qualification"]["metrics"]["overall"]["scored_class_counts"] == {"pass": 1, "fail": 1}
    assert report["qualification"]["metrics"]["overall"]["rate_checks"]["false_pass_rate"]["measurement"]["rate"] == 1
    assert report == json.loads((output / "grader-validation.json").read_text())
    assert report["provenance"]["verifier_git_commit"] is not None
    assert report["provenance"]["verifier_git_dirty"] is False


def test_verifier_git_identity_requires_tracked_files(tmp_path):
    import subprocess

    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repository), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init")
    tracked = repository / "verifier.py"
    tracked.write_text("original\n")
    git("add", "verifier.py")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "test",
    )
    expected = git("rev-parse", "HEAD")
    assert gv._verifier_git_provenance([tracked]) == {"verifier_git_commit": expected, "verifier_git_dirty": False}
    tracked.write_text("changed\n")
    assert gv._verifier_git_provenance([tracked]) == {"verifier_git_commit": expected, "verifier_git_dirty": True}
    untracked = repository / "site-packages" / "verifier.py"
    untracked.parent.mkdir()
    untracked.write_text("installed\n")
    unknown = {"verifier_git_commit": None, "verifier_git_dirty": None}
    assert gv._verifier_git_provenance([untracked]) == unknown
    assert gv._verifier_git_provenance([tracked, untracked]) == unknown
    assert gv._verifier_git_provenance([tmp_path / "outside.py"]) == unknown


@pytest.mark.parametrize("judge_error", [False, True])
def test_policy_live_pass_or_insufficient(judge_server, qualification_policy, tmp_path, judge_error):
    value = json.loads(qualification_policy.read_text())
    value["max_rates"] = dict.fromkeys(value["max_rates"], 1.0)
    qualification_policy.write_text(json.dumps(value))
    judge_server["fail"] = judge_error
    output = tmp_path / "live"
    result = CliRunner().invoke(
        cli,
        [
            "tier3",
            "validate-grader",
            "--qualification-policy",
            str(qualification_policy),
            "--case",
            "arithmetic",
            "--case",
            "wrong-answer",
            "--repeats",
            "1",
            "--json",
            "-o",
            str(output),
        ],
    )
    assert result.exit_code == int(judge_error), result.output
    report = json.loads(result.output)
    assert report["qualification"]["status"] == ("insufficient_evidence" if judge_error else "passed")
    assert report == json.loads((output / "grader-validation.json").read_text())


def test_verifier_git_ignores_unrelated_dirty_files_and_git_environment(tmp_path, monkeypatch):
    import subprocess

    repository = tmp_path / "source"
    repository.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repository), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init")
    verifier = repository / "verifier.py"
    verifier.write_text("source\n")
    unrelated = repository / "other.py"
    unrelated.write_text("clean\n")
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "test",
    )
    expected = git("rev-parse", "HEAD")
    unrelated.write_text("dirty\n")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "does-not-exist"))
    assert gv._verifier_git_provenance([verifier]) == {"verifier_git_commit": expected, "verifier_git_dirty": False}


@pytest.mark.parametrize(
    "args",
    [
        ["--timeout", "nan"],
        ["--timeout", "inf"],
        ["--max-false-pass-rate", "nan"],
        ["--max-false-fail-rate", "nan"],
        ["--max-unjudgeable-rate", "nan"],
    ],
)
def test_nonfinite_options_rejected_before_provider(monkeypatch, args):
    monkeypatch.setattr(gv, "resolve_llm_provider", lambda: pytest.fail("provider accessed"))
    result = CliRunner().invoke(cli, ["tier3", "validate-grader", *args])
    assert result.exit_code == 2, result.output
    assert "finite" in result.output


def test_policy_cli_without_platform_nonblocking_flag(monkeypatch, qualification_policy):
    monkeypatch.delattr(gv.os, "O_NONBLOCK", raising=False)
    monkeypatch.setattr(gv, "resolve_llm_provider", lambda: pytest.fail("provider accessed"))
    result = CliRunner().invoke(
        cli,
        ["tier3", "validate-grader", "--qualification-policy", str(qualification_policy), "--check-suite", "--json"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["qualification"]["status"] == "ready"


def test_live_correct_unavailable_case_meets_observed_category_only(judge_server, qualification_policy, tmp_path):
    value = json.loads(qualification_policy.read_text())
    value["min_category_cases"] = {"unavailable_evidence": 1}
    value["max_rates"] = dict.fromkeys(value["max_rates"], 1)
    qualification_policy.write_text(json.dumps(value))
    output = tmp_path / "observed-category"
    result = CliRunner().invoke(
        cli,
        [
            "tier3",
            "validate-grader",
            "--qualification-policy",
            str(qualification_policy),
            "--case",
            "arithmetic",
            "--case",
            "wrong-answer",
            "--case",
            "missing",
            "--repeats",
            "1",
            "--json",
            "-o",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    receipt = json.loads(result.output)
    decision = receipt["qualification"]
    assert decision["status"] == "passed"
    assert decision["categories"]["unavailable_evidence"]["observed_cases"] == 1
    assert decision["categories"]["unavailable_evidence"]["observed_met"] is True
    assert decision["metrics"]["overall"]["scored_class_counts"] == {"pass": 1, "fail": 1}
    assert receipt == json.loads((output / "grader-validation.json").read_text())


@pytest.mark.parametrize("flag", ["--max-false-pass-rate", "--max-false-fail-rate", "--max-unjudgeable-rate"])
@pytest.mark.parametrize("offline", [True, False])
def test_policy_rejects_legacy_gate_flags_before_provider(monkeypatch, qualification_policy, tmp_path, flag, offline):
    monkeypatch.setattr(gv, "resolve_llm_provider", lambda: pytest.fail("provider accessed"))
    output = tmp_path / "not-written"
    args = [
        "tier3",
        "validate-grader",
        "--qualification-policy",
        str(qualification_policy),
        flag,
        "0",
        "-o",
        str(output),
    ]
    if offline:
        args.append("--check-suite")
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 2, result.output
    assert "cannot combine" in result.output
    assert not output.exists()
