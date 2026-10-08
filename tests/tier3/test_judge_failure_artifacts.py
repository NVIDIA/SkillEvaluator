# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed artifact regressions for required Tier 3 LLM judges."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from skillevaluator.tier3.harbor import collector, report
from skillevaluator.tier3.harbor.adapter import _EVALUATOR_TESTS_SUBDIR, _write_test_sh
from skillevaluator.tier3.harbor.metrics import (
    DEFAULT_METRIC_SET,
    MAX_CUSTOM_METRIC_NAME_BYTES,
    MAX_CUSTOM_METRICS,
    RESERVED_METRIC_NAMES,
    custom_metric_name_is_publishable,
    extract_custom_metrics,
    metric_set_for_reward,
    not_applicable_metrics,
    overall_score,
)
from skillevaluator.tier3.harbor.templates import custom_grader_runner

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_TEMPLATE = _REPO_ROOT / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
_CUSTOM_RUNNER_TEMPLATE = (
    _REPO_ROOT / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "custom_grader_runner.py"
)


def _load_verifier(tmp_path: Path) -> ModuleType:
    """Load and initialize the Harbor verifier template module in a temporary workspace."""
    module_name = f"harbor_eval_failure_artifacts_{tmp_path.name}"
    spec = importlib.util.spec_from_file_location(module_name, _EVAL_TEMPLATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    logs_dir = tmp_path / "logs"
    agent_dir = logs_dir / "agent"
    verifier_dir = logs_dir / "verifier"
    tests_dir = tmp_path / "tests"
    agent_dir.mkdir(parents=True)
    verifier_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)

    module.LOGS_DIR = logs_dir
    module.AGENT_LOGS_DIR = agent_dir
    module.VERIFIER_DIR = verifier_dir
    module.TESTS_DIR = tests_dir
    module.ATIF_PATH = agent_dir / "trajectory.json"
    module.ENTRY_PATH = tests_dir / "entry.json"
    module.REWARD_JSON = verifier_dir / "reward.json"
    module.REWARD_TXT = verifier_dir / "reward.txt"
    module.SKILL_EVALUATOR_REWARD_JSON = verifier_dir / "skill_evaluator_reward.json"

    module.ATIF_PATH.write_text(
        json.dumps(
            {
                "steps": [
                    {"source": "user", "message": "Complete the task."},
                    {"source": "agent", "message": "The task is complete."},
                ]
            }
        ),
        encoding="utf-8",
    )
    module.ENTRY_PATH.write_text(
        json.dumps(
            {
                "id": "judge-artifact-case",
                "question": "Complete the task.",
                "ground_truth": "The task is complete.",
                "expected_behavior": ["Complete the task"],
                "should_trigger": False,
                "evaluated_skill": "demo",
                "has_skill": True,
            }
        ),
        encoding="utf-8",
    )
    return module


def test_verifier_main_fails_closed_after_collecting_every_required_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify verifier main fails closed with exit code 1 after collecting every required judge error."""
    verifier = _load_verifier(tmp_path)
    credential = "dummy-secret-credential-DO-NOT-RETAIN"
    monkeypatch.setenv("ANTHROPIC_API_KEY", credential)
    calls: list[str] = []

    def accuracy(*_args, **_kwargs):
        calls.append("accuracy")
        return {"score": 0.0, "status": "error", "reason": f"HTTP 401 echoed {credential}"}

    def goal_accuracy(*_args, **_kwargs):
        calls.append("goal_accuracy")
        return {"score": True, "reason": "boolean is not a score"}

    def behavior_check(*_args, **_kwargs):
        calls.append("behavior_check")
        return {"score": math.inf, "reason": "non-finite score " + ("x" * 800)}

    monkeypatch.setattr(verifier, "judge_accuracy", accuracy)
    monkeypatch.setattr(verifier, "judge_goal_accuracy", goal_accuracy)
    monkeypatch.setattr(verifier, "judge_behavior_check", behavior_check)

    with pytest.raises(SystemExit) as exc_info:
        verifier.main()

    assert exc_info.value.code == 1
    assert calls == ["accuracy", "goal_accuracy", "behavior_check"]

    rich = json.loads(verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    numeric = json.loads(verifier.REWARD_JSON.read_text(encoding="utf-8"))
    assert rich["accuracy"] is None
    assert rich["goal_accuracy"] is None
    assert rich["behavior_check"] is None
    assert rich["evaluation_status"] == "failed"
    assert rich["details"]["accuracy"]["status"] == "error"
    assert rich["details"]["goal_accuracy"]["status"] == "error"
    assert rich["details"]["behavior_check"]["status"] == "error"
    assert set(rich["evaluation_errors"]) == {"accuracy", "goal_accuracy", "behavior_check"}
    assert all(0 < len(reason) <= 512 for reason in rich["evaluation_errors"].values())

    assert numeric == {
        "security": 1.0,
        "skill_execution": 1.0,
        "skill_efficiency": 1.0,
        "overall": 0.0,
    }
    assert verifier.REWARD_TXT.read_text(encoding="utf-8") == "0.0"

    artifact_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            verifier.SKILL_EVALUATOR_REWARD_JSON,
            verifier.REWARD_JSON,
            verifier.REWARD_TXT,
        )
    )
    assert credential not in artifact_text
    assert "[REDACTED]" in artifact_text

    # Harbor may retain only reward.json. Its canonical deterministic metrics
    # must still identify an incomplete default reward without the sidecar.
    verifier.SKILL_EVALUATOR_REWARD_JSON.unlink()
    assert metric_set_for_reward(numeric)[0] == DEFAULT_METRIC_SET
    assert overall_score(numeric) is None


def test_verifier_retries_leave_time_to_write_failure_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure slow transient calls finish before Harbor verifier timeout kills artifact writes."""
    verifier = _load_verifier(tmp_path)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-fake")
    monkeypatch.setenv("SKILL_EVAL_LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("SKILL_EVAL_LLM_RETRY_BASE_DELAY", "0")
    monkeypatch.setenv("SKILL_EVAL_LLM_RETRY_MAX_DELAY", "0")
    monkeypatch.setenv("LLM_JUDGE_FALLBACK_MODELS", "")
    monkeypatch.setattr(verifier, "_ragas_goal_accuracy_enabled", lambda: False)

    elapsed = [0.0]

    class FakeTime:
        """Simulate time progression for slow network timeouts."""

        def monotonic(self) -> float:
            return elapsed[0]

        def sleep(self, seconds: float) -> None:
            elapsed[0] += seconds

        def __getattr__(self, name: str):
            return getattr(time, name)

    def slow_timeout(_request, timeout=90):
        elapsed[0] += timeout
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr(verifier, "time", FakeTime())
    monkeypatch.setattr(verifier.urllib.request, "urlopen", slow_timeout)

    with pytest.raises(SystemExit) as exc_info:
        verifier.main()

    assert exc_info.value.code == 1
    assert elapsed[0] <= 540.0
    rich = json.loads(verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    numeric = json.loads(verifier.REWARD_JSON.read_text(encoding="utf-8"))
    assert rich["evaluation_status"] == "failed"
    assert "accuracy" in rich["evaluation_errors"]
    assert numeric["overall"] == 0.0
    assert overall_score(numeric) is None


def test_required_judge_deadline_interrupts_a_stalled_response_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interrupt stalled provider response body when required judge deadline expires."""
    if not hasattr(signal, "setitimer") or signal.getitimer(signal.ITIMER_REAL)[0] > 0:
        pytest.skip("free POSIX interval timer required")
    verifier = _load_verifier(tmp_path)
    monkeypatch.setattr(verifier, "_JUDGE_WALL_TIME_BUDGET_SEC", 0.05)

    class SlowResponse:
        """Simulate a trickling or stalled response body."""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            time.sleep(0.5)
            return b"late success"

    monkeypatch.setattr(verifier.urllib.request, "urlopen", lambda *_args, **_kwargs: SlowResponse())
    previous_handler = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    result = verifier._call_required_judge("accuracy", lambda: verifier._urlopen_with_retry("test"))

    assert time.monotonic() - started < 0.4
    assert result["status"] == "error"
    assert result["score"] is None
    assert signal.getsignal(signal.SIGALRM) == previous_handler
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


def test_required_judge_reports_an_exhausted_budget_as_a_timeout(tmp_path: Path) -> None:
    verifier = _load_verifier(tmp_path)

    def judge():
        raise verifier._JudgeBudgetExhausted("LLM judge time budget exhausted")

    result = verifier._call_required_judge("accuracy", judge)

    assert result["status"] == "error"
    assert result["reason"] == "Required accuracy judge raised TimeoutError: LLM judge time budget exhausted"


def test_required_judge_restores_alarm_handler_after_teardown_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restore previous SIGALRM handler when judge deadline teardown is interrupted."""
    if not hasattr(signal, "setitimer") or signal.getitimer(signal.ITIMER_REAL)[0] > 0:
        pytest.skip("free POSIX interval timer required")
    verifier = _load_verifier(tmp_path)
    original_setitimer = signal.setitimer
    previous_handler = signal.getsignal(signal.SIGALRM)

    def interrupted_setitimer(timer, seconds, interval=0):
        previous = original_setitimer(timer, seconds, interval)
        if seconds == 0:
            raise TimeoutError("interrupted during deadline teardown")
        return previous

    monkeypatch.setattr(verifier.signal, "setitimer", interrupted_setitimer)
    result = verifier._call_required_judge("accuracy", lambda: {"score": 1.0, "reason": "ok"})

    assert result["status"] == "error"
    assert signal.getsignal(signal.SIGALRM) == previous_handler
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


def test_ragas_goal_judge_obeys_the_required_judge_time_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure Ragas goal scorer returns before its Harbor budget expires."""
    verifier = _load_verifier(tmp_path)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-fake")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
    monkeypatch.setattr(verifier, "_ragas_goal_accuracy_enabled", lambda: True)
    monkeypatch.setattr(verifier, "_JUDGE_WALL_TIME_BUDGET_SEC", 0.01)

    class FakeMessage:
        """Mock message object for Ragas sample input."""

        def __init__(self, content: str):
            self.content = content

    class FakeMetric:
        """Mock metric object for Ragas async evaluation."""

        def __init__(self, llm):
            self.llm = llm

        async def ascore(self, _sample):
            await asyncio.sleep(0.05)
            return SimpleNamespace(value=1.0)

    fake_ragas = ModuleType("ragas")
    fake_ragas.SingleTurnSample = lambda **_kwargs: object()
    fake_ragas_llms = ModuleType("ragas.llms")
    fake_ragas_llms_base = ModuleType("ragas.llms.base")
    fake_ragas_llms_base.llm_factory = lambda *_args, **_kwargs: object()
    fake_ragas_messages = ModuleType("ragas.messages")
    fake_ragas_messages.AIMessage = FakeMessage
    fake_ragas_messages.HumanMessage = FakeMessage
    fake_ragas_metrics = ModuleType("ragas.metrics")
    fake_ragas_collections = ModuleType("ragas.metrics.collections")
    fake_ragas_collections.AgentGoalAccuracyWithReference = FakeMetric
    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = lambda **_kwargs: object()
    for name, module in {
        "ragas": fake_ragas,
        "ragas.llms": fake_ragas_llms,
        "ragas.llms.base": fake_ragas_llms_base,
        "ragas.messages": fake_ragas_messages,
        "ragas.metrics": fake_ragas_metrics,
        "ragas.metrics.collections": fake_ragas_collections,
        "openai": fake_openai,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        verifier.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("custom fallback must stop before another HTTP request"),
    )

    result = verifier._call_required_judge(
        "goal_accuracy", verifier.judge_goal_accuracy, "question", "ground truth", "agent response"
    )

    assert result["status"] == "error"
    assert result["score"] is None


def test_verifier_main_keeps_genuine_zero_judge_verdicts_scoreable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain genuine zero-score verdicts from successful judge evaluations as scoreable metrics."""
    verifier = _load_verifier(tmp_path)
    calls: list[str] = []

    def valid_zero(metric: str):
        def judge(*_args, **_kwargs):
            calls.append(metric)
            return {"score": 0.0, "reason": "valid model verdict"}

        return judge

    monkeypatch.setattr(verifier, "judge_accuracy", valid_zero("accuracy"))
    monkeypatch.setattr(verifier, "judge_goal_accuracy", valid_zero("goal_accuracy"))
    monkeypatch.setattr(verifier, "judge_behavior_check", valid_zero("behavior_check"))

    verifier.main()

    assert calls == ["accuracy", "goal_accuracy", "behavior_check"]
    rich = json.loads(verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    numeric = json.loads(verifier.REWARD_JSON.read_text(encoding="utf-8"))
    assert "evaluation_status" not in rich
    assert "evaluation_errors" not in rich
    assert {metric: numeric[metric] for metric in verifier.DISPLAY_METRICS} == {
        "security": 1.0,
        "skill_execution": 1.0,
        "skill_efficiency": 1.0,
        "accuracy": 0.0,
        "goal_accuracy": 0.0,
        "behavior_check": 0.0,
    }
    assert numeric["overall"] == 0.5
    assert overall_score(numeric) == 0.5


def test_verifier_main_recovers_malformed_accuracy_and_goal_judges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover malformed accuracy and goal judge responses on retry and record overall score."""
    verifier = _load_verifier(tmp_path)
    monkeypatch.setattr(verifier, "_ragas_goal_accuracy_enabled", lambda: False)
    pair_calls: list[tuple[str, dict]] = []
    goal_calls: list[tuple[str, dict]] = []
    pair_responses = [
        ("not-json", None),
        (
            json.dumps(
                {
                    "criteria": {
                        "SKILL_IDENTIFIED": True,
                        "ACTION_CORRECT": True,
                        "FACTUALLY_ACCURATE": True,
                        "TASK_ADDRESSED": True,
                        "ACTIONABLE": True,
                    },
                    "score": 1.0,
                    "reason": "accuracy recovered",
                }
            ),
            None,
        ),
        (json.dumps({"results": [{"step": 1, "passed": True}], "score": 1.0}), None),
    ]
    goal_responses = [
        ("not-json", None, {"provider": "nv_build", "model": "first-model"}),
        (
            json.dumps({"achieved": True, "score": 1.0, "reason": "goal recovered"}),
            None,
            {"provider": "nv_build", "model": "retry-model"},
        ),
    ]

    def pair_call(prompt: str, **kwargs):
        pair_calls.append((prompt, kwargs))
        return pair_responses[len(pair_calls) - 1]

    def goal_call(prompt: str, **kwargs):
        goal_calls.append((prompt, kwargs))
        return goal_responses[len(goal_calls) - 1]

    monkeypatch.setattr(verifier, "call_public_llm", pair_call)
    monkeypatch.setattr(verifier, "_call_public_llm_with_provenance", goal_call)

    verifier.main()

    rich = json.loads(verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    numeric = json.loads(verifier.REWARD_JSON.read_text(encoding="utf-8"))
    assert "evaluation_status" not in rich
    assert "evaluation_errors" not in rich
    assert rich["details"]["accuracy"]["score"] == 1.0
    assert rich["details"]["goal_accuracy"]["score"] == 1.0
    assert rich["details"]["goal_accuracy"]["model"] == "retry-model"
    assert numeric["accuracy"] == numeric["goal_accuracy"] == numeric["behavior_check"] == 1.0
    assert [kwargs["max_tokens"] for _, kwargs in pair_calls] == [4096, 4096, 4096]
    assert [kwargs["max_tokens"] for _, kwargs in goal_calls] == [4096, 4096]
    assert "previous reply could not be parsed or validated" in pair_calls[1][0]
    assert "previous reply could not be parsed or validated" in goal_calls[1][0]


def test_verifier_retries_non_string_judge_text_before_collector_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry non-string judge text payloads before collector and report generation."""
    verifier = _load_verifier(tmp_path)
    monkeypatch.setattr(verifier, "_ragas_goal_accuracy_enabled", lambda: False)
    accuracy_calls: list[str] = []
    goal_calls: list[str] = []

    def pair_call(prompt: str, **_kwargs):
        if "SKILL_IDENTIFIED" not in prompt:
            return json.dumps({"results": [{"step": 1, "passed": True}], "score": 1.0}), None
        accuracy_calls.append(prompt)
        if len(accuracy_calls) == 1:
            return json.dumps({"score": 1.0, "reason": {"nested": "accuracy"}}), None
        return json.dumps({"score": 1.0, "reason": "accuracy recovered"}), None

    def goal_call(prompt: str, **_kwargs):
        goal_calls.append(prompt)
        if len(goal_calls) == 1:
            return (
                json.dumps(
                    {
                        "achieved": True,
                        "score": 1.0,
                        "reason": ["nested", "goal"],
                        "user_goal": {"nested": "goal"},
                        "end_state": ["nested", "state"],
                    }
                ),
                None,
                {"provider": "nv_build", "model": "first-model"},
            )
        return (
            json.dumps(
                {
                    "achieved": True,
                    "score": 1.0,
                    "reason": "goal recovered",
                    "user_goal": "complete the task",
                    "end_state": "task completed",
                }
            ),
            None,
            {"provider": "nv_build", "model": "retry-model"},
        )

    monkeypatch.setattr(verifier, "call_public_llm", pair_call)
    monkeypatch.setattr(verifier, "_call_public_llm_with_provenance", goal_call)

    verifier.main()

    collected = json.loads(verifier.REWARD_JSON.read_text(encoding="utf-8"))
    collector._merge_reward_sidecars(collected, verifier.VERIFIER_DIR)
    findings = report._extract_findings([collected])

    assert len(accuracy_calls) == 2
    assert len(goal_calls) == 2
    assert collected["details"]["accuracy"]["reason"] == "accuracy recovered"
    assert collected["details"]["goal_accuracy"]["reason"] == "goal recovered"
    assert all(isinstance(reason, str) for finding in findings for reason in finding["reasons"])


@pytest.mark.parametrize(
    ("metric", "score", "detail"),
    [
        pytest.param("accuracy", 1.0, {"reason": {"nested": "a" * 600}}, id="accuracy-pass"),
        pytest.param("accuracy", 0.0, {"reason": ["nested", "accuracy"]}, id="accuracy-fail"),
        pytest.param(
            "goal_accuracy",
            1.0,
            {"reason": ["nested", "goal"], "end_state": {"nested": "e" * 600}},
            id="goal-pass",
        ),
        pytest.param("goal_accuracy", 0.0, {"reason": {"nested": "goal"}}, id="goal-fail"),
        pytest.param(
            "behavior_check",
            1.0,
            {"reason": {"nested": "summary"}, "results": [{"passed": True, "reason": "ok"}]},
            id="behavior-pass",
        ),
        pytest.param(
            "behavior_check",
            0.0,
            {"reason": "failed", "results": [{"passed": False, "reason": {"nested": "step"}}]},
            id="behavior-fail",
        ),
    ],
)
def test_report_coerces_and_bounds_non_string_reasons_from_existing_artifacts(
    metric: str,
    score: float,
    detail: dict,
) -> None:
    """Coerce and bound non-string reason fields from legacy judge artifacts."""
    reward = {
        "entry_id": "legacy-judge-artifact",
        metric: score,
        "details": {metric: detail},
    }

    findings = report._extract_findings([reward])

    finding = next(item for item in findings if item["metric"] == metric)
    assert finding["reasons"]
    assert all(isinstance(reason, str) for reason in finding["reasons"])
    assert all(len(reason) <= 512 for reason in finding["reasons"])


def test_report_redacts_configured_secret_before_bounding_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """Redact configured secrets before truncating long judge reason strings in report."""
    credential = "SECRET-ABCDEFGHIJKLMNOPQRSTUVWXYZ-0123456789"
    prefix = "x" * 490
    monkeypatch.setenv("OPENAI_API_KEY", credential)
    reward = {
        "entry_id": "credential-boundary-artifact",
        "accuracy": 1.0,
        "details": {"accuracy": {"reason": prefix + credential}},
    }

    findings = report._extract_findings([reward])

    accuracy = next(item for item in findings if item["metric"] == "accuracy")
    assert accuracy["reasons"] == [prefix + "[REDACTED]"]
    assert credential not in accuracy["reasons"][0]
    assert "SECRET-" not in accuracy["reasons"][0]


def test_verifier_main_keeps_accuracy_fail_closed_after_retry_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep accuracy metric fail-closed when verifier retries are exhausted."""
    verifier = _load_verifier(tmp_path)
    credential = "dummy-verifier-retry-secret-DO-NOT-RETAIN"
    monkeypatch.setenv("NVIDIA_API_KEY", credential)
    monkeypatch.setattr(verifier, "_ragas_goal_accuracy_enabled", lambda: False)
    pair_calls: list[tuple[str, dict]] = []
    pair_responses = [
        (f"not-json containing {credential}", None),
        (f"still-not-json containing {credential}", None),
        (json.dumps({"results": [{"step": 1, "passed": True}], "score": 1.0}), None),
    ]
    goal_calls: list[tuple[str, dict]] = []

    def pair_call(prompt: str, **kwargs):
        pair_calls.append((prompt, kwargs))
        return pair_responses[len(pair_calls) - 1]

    def goal_call(prompt: str, **kwargs):
        goal_calls.append((prompt, kwargs))
        return (
            json.dumps({"achieved": True, "score": 1.0, "reason": "goal valid"}),
            None,
            {"provider": "nv_build", "model": "goal-model"},
        )

    monkeypatch.setattr(verifier, "call_public_llm", pair_call)
    monkeypatch.setattr(verifier, "_call_public_llm_with_provenance", goal_call)

    with pytest.raises(SystemExit) as exc_info:
        verifier.main()

    assert exc_info.value.code == 1
    assert len(pair_calls) == 3
    accuracy_attempts = [call for call in pair_calls if "SKILL_IDENTIFIED" in call[0]]
    assert len(accuracy_attempts) == 2
    assert [kwargs["max_tokens"] for _, kwargs in accuracy_attempts] == [4096, 4096]
    assert len(goal_calls) == 1
    assert "previous reply could not be parsed or validated" in pair_calls[1][0]
    assert "previous reply could not be parsed or validated" not in pair_calls[2][0]
    rich = json.loads(verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    numeric = json.loads(verifier.REWARD_JSON.read_text(encoding="utf-8"))
    assert rich["evaluation_status"] == "failed"
    assert rich["accuracy"] is None
    assert rich["details"]["accuracy"]["status"] == "error"
    assert len(rich["evaluation_errors"]["accuracy"]) <= 512
    assert credential not in json.dumps(rich)
    assert "accuracy" not in numeric


_JUDGED_METRICS = ("accuracy", "goal_accuracy", "behavior_check")


def _rewrite_entry(verifier: ModuleType, **updates) -> None:
    entry = json.loads(verifier.ENTRY_PATH.read_text(encoding="utf-8"))
    entry.update(updates)
    verifier.ENTRY_PATH.write_text(json.dumps(entry), encoding="utf-8")


def test_verifier_main_records_judges_without_reference_as_not_applicable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep documented neutral judge skips scoreable with default passing scores."""
    verifier = _load_verifier(tmp_path)
    _rewrite_entry(verifier, ground_truth="", expected_behavior=[])

    def no_llm(*_args, **_kwargs):
        raise AssertionError("N/A judges must not call the LLM")

    monkeypatch.setattr(verifier, "call_public_llm", no_llm)
    monkeypatch.setattr(verifier, "_call_public_llm_with_provenance", no_llm)

    verifier.main()

    rich = json.loads(verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    numeric = json.loads(verifier.REWARD_JSON.read_text(encoding="utf-8"))
    assert "evaluation_status" not in rich
    for metric in _JUDGED_METRICS:
        assert rich[metric] is None
        assert rich["details"][metric]["score"] is None
        assert rich["details"][metric]["status"] == "not_applicable"
        assert rich["details"][metric]["reason"].startswith("N/A: no ")
    # reward.json stays numeric-only so Harbor's VerifierResult accepts it; the
    # every-judge-N/A overall is the mean of the deterministic metrics.
    assert numeric == {"security": 1.0, "skill_execution": 1.0, "skill_efficiency": 1.0, "overall": 1.0}
    assert verifier.REWARD_TXT.read_text(encoding="utf-8") == "1.0"
    harbor_result = pytest.importorskip("harbor.models.verifier.result")
    assert harbor_result.VerifierResult(rewards=numeric).rewards == numeric

    # Without the sidecar the numeric reward is incomplete, never silently N/A.
    assert overall_score(numeric) is None
    collected = dict(numeric)
    collector._merge_reward_sidecars(collected, verifier.VERIFIER_DIR)
    assert overall_score(collected) == 1.0
    assert not_applicable_metrics([collected]) == list(_JUDGED_METRICS)


def test_verifier_main_excludes_only_the_not_applicable_metric_from_overall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_verifier(tmp_path)
    _rewrite_entry(verifier, expected_behavior=[])

    def zero(*_args, **_kwargs):
        return {"score": 0.0, "reason": "valid model verdict"}

    monkeypatch.setattr(verifier, "judge_accuracy", zero)
    monkeypatch.setattr(verifier, "judge_goal_accuracy", zero)

    verifier.main()

    rich = json.loads(verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    numeric = json.loads(verifier.REWARD_JSON.read_text(encoding="utf-8"))
    assert rich["behavior_check"] is None
    assert rich["details"]["behavior_check"]["status"] == "not_applicable"
    assert "behavior_check" not in numeric
    assert numeric["accuracy"] == numeric["goal_accuracy"] == 0.0
    # mean(1, 1, 1, 0, 0): N/A is neither a fabricated 1.0 (0.667) nor a 0.0 (0.5).
    assert numeric["overall"] == 0.6
    collected = dict(numeric)
    collector._merge_reward_sidecars(collected, verifier.VERIFIER_DIR)
    assert overall_score(collected) == pytest.approx(0.6)


@pytest.mark.parametrize("metric", _JUDGED_METRICS)
def test_verifier_main_rejects_not_applicable_when_the_case_has_a_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric: str,
) -> None:
    verifier = _load_verifier(tmp_path)

    def scored(*_args, **_kwargs):
        return {"score": 1.0, "reason": "valid"}

    def claims_not_applicable(*_args, **_kwargs):
        return {"score": None, "status": "not_applicable", "reason": "N/A"}

    for name in _JUDGED_METRICS:
        monkeypatch.setattr(
            verifier,
            f"judge_{name}",
            claims_not_applicable if name == metric else scored,
        )

    with pytest.raises(SystemExit) as exc_info:
        verifier.main()

    assert exc_info.value.code == 1
    rich = json.loads(verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    assert rich["evaluation_status"] == "failed"
    assert rich["details"][metric]["status"] == "error"
    assert set(rich["evaluation_errors"]) == {metric}


@pytest.mark.parametrize("failure_kind", ["missing-score", "exception"])
def test_verifier_main_normalizes_malformed_or_raised_judge_failures_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    """Normalize malformed or raised judge failures and continue evaluating remaining judges."""
    verifier = _load_verifier(tmp_path)
    calls: list[str] = []

    def accuracy(*_args, **_kwargs):
        calls.append("accuracy")
        if failure_kind == "exception":
            raise RuntimeError("judge transport crashed")
        return {"reason": "judge omitted its score"}

    def successful(metric: str):
        def judge(*_args, **_kwargs):
            calls.append(metric)
            return {"score": 1.0, "reason": "valid verdict"}

        return judge

    monkeypatch.setattr(verifier, "judge_accuracy", accuracy)
    monkeypatch.setattr(verifier, "judge_goal_accuracy", successful("goal_accuracy"))
    monkeypatch.setattr(verifier, "judge_behavior_check", successful("behavior_check"))

    with pytest.raises(SystemExit) as exc_info:
        verifier.main()

    assert exc_info.value.code == 1
    assert calls == ["accuracy", "goal_accuracy", "behavior_check"]
    rich = json.loads(verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    assert rich["accuracy"] is None
    assert rich["goal_accuracy"] == 1.0
    assert rich["behavior_check"] == 1.0
    assert rich["details"]["accuracy"]["status"] == "error"
    assert set(rich["evaluation_errors"]) == {"accuracy"}
    numeric = json.loads(verifier.REWARD_JSON.read_text(encoding="utf-8"))
    assert numeric["goal_accuracy"] == 1.0
    assert numeric["behavior_check"] == 1.0
    assert "accuracy" not in numeric
    assert overall_score(numeric) is None


def test_numeric_reward_payload_excludes_boolean_and_non_finite_values(tmp_path: Path) -> None:
    """Exclude boolean, non-finite, and infinite values from numeric reward payloads."""
    verifier = _load_verifier(tmp_path)

    payload = verifier._numeric_reward_payload(
        {
            "finite_int": 1,
            "finite_float": 0.25,
            "boolean": True,
            "nan": math.nan,
            "positive_infinity": math.inf,
            "negative_infinity": -math.inf,
            "huge_integer": 10**1000,
        },
        0.0,
    )

    assert payload == {"finite_int": 1.0, "finite_float": 0.25, "overall": 0.0}
    assert all(math.isfinite(value) and not isinstance(value, bool) for value in payload.values())


def test_evaluation_failure_fields_are_reserved_metadata() -> None:
    """Confirm evaluation failure fields are reserved metadata in custom grader runner."""
    expected = {"evaluation_status", "evaluation_errors"}

    assert expected <= RESERVED_METRIC_NAMES
    assert custom_grader_runner.RESERVED == RESERVED_METRIC_NAMES
    assert custom_grader_runner._extract_custom_metrics(
        {"evaluation_status": 1.0, "evaluation_errors": 0.5, "domain_score": 0.75}
    ) == {"domain_score": 0.75}
    with pytest.raises(RuntimeError, match="collides with reserved"):
        custom_grader_runner._extract_custom_metrics({"custom_metrics": {"evaluation_status": 0.5}})


@pytest.mark.parametrize("malformed", [None, 0.5, "quality", [0.5]])
def test_custom_grader_runner_rejects_malformed_custom_metrics_container(malformed: object) -> None:
    with pytest.raises(RuntimeError, match=r"container.*JSON object"):
        custom_grader_runner._extract_custom_metrics({"custom_metrics": malformed, "quality": 0.8, "overall": 0.8})


@pytest.mark.parametrize(
    "invalid_score",
    [math.nan, math.inf, -math.inf, -0.01, 1.01, 1e308, 10**400],
    ids=[
        "nan",
        "positive-infinity",
        "negative-infinity",
        "negative",
        "above-one",
        "huge-finite",
        "unrepresentable-integer",
    ],
)
def test_custom_grader_runner_rejects_invalid_overall_and_custom_scores(
    invalid_score: float | int,
) -> None:
    assert custom_grader_runner._score_from_reward({"overall": invalid_score}) is None
    assert custom_grader_runner._extract_custom_metrics(
        {"custom_metrics": {"invalid": invalid_score, "valid": 0.5}}
    ) == {"valid": 0.5}
    assert custom_grader_runner._numeric_reward_payload(
        {"invalid": invalid_score, "valid": 0.5},
        overall=invalid_score,
    ) == {"valid": 0.5}


def test_custom_grader_runner_enforces_metric_publication_contract_before_artifacts() -> None:
    assert custom_grader_runner.MAX_CUSTOM_METRICS == MAX_CUSTOM_METRICS
    assert custom_grader_runner.MAX_CUSTOM_METRIC_NAME_BYTES == MAX_CUSTOM_METRIC_NAME_BYTES
    assert custom_grader_runner._extract_custom_metrics(
        {
            "custom_metrics": {
                "quality": 0.8,
                "sk-abcdefghijk": 0.7,
                "quality_sk-abcdefghijk": 0.6,
                "api_key_quality": 0.5,
                "secret_handling": 0.9,
            }
        }
    ) == {"quality": 0.8, "secret_handling": 0.9}

    with pytest.raises(RuntimeError, match="name exceeds"):
        custom_grader_runner._extract_custom_metrics(
            {"custom_metrics": {"x" * (MAX_CUSTOM_METRIC_NAME_BYTES + 1): 0.5}}
        )
    with pytest.raises(RuntimeError, match="count exceeds"):
        custom_grader_runner._extract_custom_metrics(
            {"custom_metrics": {f"metric_{index:03d}": 0.5 for index in range(MAX_CUSTOM_METRICS + 1)}}
        )


@pytest.mark.parametrize(
    "name",
    [
        "quality",
        "secret_handling",
        "token_efficiency",
        "token_count",
        "tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "max_tokens",
        "last_token_usage",
        "api_key_quality",
        "passwords_quality",
        "tokens_quality",
        "privatekey_quality",
        "sessiontoken_quality",
        "sk-abcdefghijk",
        "quality_sk-abcdefghijk",
        "quality_nvapi-abcdefghijk",
        "quality_crsr_0123456789abcdef",
        "quality_SK-ABCDEFGHIJK",
        "quality_NVAPI-ABCDEFGHIJK",
        "quality_CRSR_0123456789ABCDEF",
        "quality_ghp_" + ("a" * 36),
        "quality_gho_" + ("a" * 36),
        "quality_ghu_" + ("a" * 36),
        "quality_ghs_" + ("a" * 36),
        "quality_ghs_" + ("a" * 18) + ".-_" + ("b" * 18),
        "quality_ghr_" + ("a" * 36),
        "qualityghp_" + ("a" * 36),
        "qualitygho_" + ("a" * 36) + "suffix",
        "quality_github_pat_" + ("a" * 30),
        "quality_" + "".join(("xoxb-", "1234567890-abcdefghijklmnopqrstuvwx")),  # noqa: FLY002
        "quality_" + "AIza" + ("A" * 35),
        "quality_AIzA" + ("A" * 35),
        "quality_glpat-" + ("a" * 20),
        "https://user:pass@example.com",
        " x",
        "x" * (MAX_CUSTOM_METRIC_NAME_BYTES + 1),
    ],
)
def test_custom_grader_runner_metric_name_policy_matches_collector(name: str) -> None:
    assert custom_grader_runner._metric_name_is_publishable(name) is custom_metric_name_is_publishable(name)


def test_custom_grader_runner_sanitizer_removes_unsafe_dict_valued_implicit_metric() -> None:
    credential_metrics = [
        "api_key_quality",
        "ghp_" + ("a" * 36),
        "gho_" + ("a" * 36),
        "ghu_" + ("a" * 36),
        "ghs_" + ("a" * 36),
        "ghs_" + ("a" * 18) + ".-_" + ("b" * 18),
        "ghr_" + ("a" * 36),
        "qualityghp_" + ("a" * 36),
        "qualitygho_" + ("a" * 36) + "suffix",
        "github_pat_" + ("a" * 30),
        "".join(("xoxb-", "1234567890-abcdefghijklmnopqrstuvwx")),  # noqa: FLY002
        "AIza" + ("A" * 35),
        "glpat-" + ("a" * 20),
    ]
    unsafe_metrics = {name: {"score": 0.6, "reason": "must not survive"} for name in credential_metrics}
    reward = {
        "overall": 0.75,
        **unsafe_metrics,
        "quality": {"score": 0.8, "reason": "bounded evidence"},
    }
    custom_metrics = custom_grader_runner._extract_custom_metrics(reward)

    assert custom_metrics == {"quality": 0.8}
    assert custom_grader_runner._sanitized_custom_reward(reward, custom_metrics) == {
        "overall": 0.75,
        "quality": {"score": 0.8, "reason": "bounded evidence"},
    }


def test_custom_grader_runner_sanitizer_limits_both_detail_containers_to_validated_metrics() -> None:
    reward = {
        "overall": 0.75,
        "custom_metrics": {"quality": 0.8, "api_key_quality": 0.6},
        "details": {
            "quality": {"reason": "bounded evidence"},
            "api_key_quality": {"reason": "must not survive"},
        },
        "custom_details": {
            "quality": {"report": "bounded evidence"},
            "api_key_quality": {"report": "must not survive"},
        },
    }
    custom_metrics = custom_grader_runner._extract_custom_metrics(reward)

    assert custom_metrics == {"quality": 0.8}
    assert custom_grader_runner._sanitized_custom_reward(reward, custom_metrics) == {
        "overall": 0.75,
        "custom_metrics": {"quality": 0.8},
        "details": {"quality": {"reason": "bounded evidence"}},
        "custom_details": {"quality": {"report": "bounded evidence"}},
    }


@pytest.mark.parametrize(
    "rejected_value",
    [{"reason": "nonnumeric"}, "nonnumeric", None],
    ids=("dict", "string", "null"),
)
def test_custom_grader_runner_sanitizer_removes_nonnumeric_credential_shaped_keys(
    rejected_value: object,
) -> None:
    rejected_name = "quality_sk-abcdefghijk"
    reward = {
        "overall": 0.75,
        "custom_metrics": {"quality": 0.8},
        rejected_name: rejected_value,
    }
    custom_metrics = custom_grader_runner._extract_custom_metrics(reward)

    assert custom_metrics == {"quality": 0.8}
    assert rejected_name not in custom_grader_runner._sanitized_custom_reward(reward, custom_metrics)


@pytest.mark.parametrize(
    "invalid_score",
    ["nan", "inf", "-inf", "-0.01", "1.01", "1e308", str(10**400)],
)
def test_custom_grader_runner_rejects_invalid_text_overall_scores(invalid_score: str) -> None:
    assert custom_grader_runner._score_from_text(invalid_score) is None


def _run_generated_test_sh(task_dir: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Execute the generated test.sh script with the specified environment variables."""
    return subprocess.run(
        ["bash", str(task_dir / "tests" / "test.sh")],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, **env},
    )


@pytest.mark.parametrize("grading_mode", ["default", "default_plus_custom"])
def test_generated_standard_grading_scripts_stop_after_evaluator_failure(
    tmp_path: Path,
    grading_mode: str,
) -> None:
    """Halt generated grading scripts immediately when standard evaluator fails."""
    task_dir = tmp_path / grading_mode
    _write_test_sh(task_dir, grading_mode=grading_mode, custom_grader=grading_mode == "default_plus_custom")
    tests_dir = task_dir / "tests"
    evaluator_dir = tests_dir / _EVALUATOR_TESTS_SUBDIR
    evaluator_dir.mkdir()
    (evaluator_dir / "eval.py").write_text("raise SystemExit(7)\n", encoding="utf-8")
    marker = task_dir / "custom-ran"
    (evaluator_dir / "custom_grader_runner.py").write_text(
        "from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('ran')\n",
        encoding="utf-8",
    )

    completed = _run_generated_test_sh(task_dir, {"HARBOR_TESTS_DIR": str(tests_dir)})

    assert completed.returncode == 7
    assert not marker.exists()


def test_generated_custom_only_script_accepts_overall_only_custom_reward(tmp_path: Path) -> None:
    """Accept overall-only reward payloads from custom grading scripts."""
    task_dir = tmp_path / "custom-only"
    _write_test_sh(task_dir, grading_mode="custom_only", custom_grader=True)
    tests_dir = task_dir / "tests"
    evaluator_dir = tests_dir / _EVALUATOR_TESTS_SUBDIR
    evaluator_dir.mkdir()
    shutil.copy2(_CUSTOM_RUNNER_TEMPLATE, evaluator_dir / "custom_grader_runner.py")
    marker = task_dir / "custom-ran"
    (tests_dir / "custom_helper.py").write_text("OVERALL = 0.75\n", encoding="utf-8")
    (tests_dir / "grader.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "from custom_helper import OVERALL\n"
        f"Path({str(marker)!r}).write_text('ran')\n"
        "Path(os.environ['HARBOR_REWARD_JSON']).write_text(json.dumps({'overall': OVERALL}))\n",
        encoding="utf-8",
    )
    verifier_dir = task_dir / "verifier"
    verifier_dir.mkdir()
    reward_json = verifier_dir / "reward.json"
    reward_txt = verifier_dir / "reward.txt"

    completed = _run_generated_test_sh(
        task_dir,
        {
            "HARBOR_TESTS_DIR": str(tests_dir),
            "HARBOR_VERIFIER_DIR": str(verifier_dir),
            "HARBOR_REWARD_JSON": str(reward_json),
            "HARBOR_REWARD_TXT": str(reward_txt),
            "HARBOR_CUSTOM_REWARD_JSON": str(verifier_dir / "custom_reward.json"),
            "HARBOR_GRADER": str(tests_dir / "grader.py"),
            "HARBOR_GRADER_SH": str(tests_dir / "grader.sh"),
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert marker.read_text(encoding="utf-8") == "ran"
    assert json.loads(reward_json.read_text(encoding="utf-8")) == {"overall": 0.75}
    assert reward_txt.read_text(encoding="utf-8") == "0.75"


@pytest.mark.parametrize("mode", ["default_plus_custom", "custom_only"])
def test_generated_custom_grader_rejects_metric_overflow_before_retaining_raw_names(
    tmp_path: Path,
    mode: str,
) -> None:
    tests_dir = tmp_path / "tests"
    verifier_dir = tmp_path / "verifier"
    tests_dir.mkdir()
    verifier_dir.mkdir()
    raw_names = [f"metric_{index:03d}" for index in range(MAX_CUSTOM_METRICS + 1)]
    (tests_dir / "grader.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        f"payload = {{'overall': 0.75, 'custom_metrics': {dict.fromkeys(raw_names, 0.5)!r}}}\n"
        "Path(os.environ['HARBOR_REWARD_JSON']).write_text(json.dumps(payload))\n",
        encoding="utf-8",
    )
    reward_json = verifier_dir / "reward.json"
    reward_txt = verifier_dir / "reward.txt"
    skill_evaluator_reward = verifier_dir / "skill_evaluator_reward.json"
    custom_reward = verifier_dir / "custom_reward.json"
    if mode == "default_plus_custom":
        skill_evaluator_reward.write_text(
            json.dumps(
                {
                    "metric_set": DEFAULT_METRIC_SET,
                    **dict.fromkeys(RESERVED_METRIC_NAMES & set(custom_grader_runner.DEFAULT_METRICS), 1.0),
                    "overall": 1.0,
                }
            ),
            encoding="utf-8",
        )
        reward_txt.write_text("1.0", encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(_CUSTOM_RUNNER_TEMPLATE), "--mode", mode],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HARBOR_TESTS_DIR": str(tests_dir),
            "HARBOR_VERIFIER_DIR": str(verifier_dir),
            "HARBOR_REWARD_JSON": str(reward_json),
            "HARBOR_REWARD_TXT": str(reward_txt),
            "HARBOR_SKILL_EVALUATOR_REWARD_JSON": str(skill_evaluator_reward),
            "HARBOR_CUSTOM_REWARD_JSON": str(custom_reward),
            "HARBOR_GRADER": str(tests_dir / "grader.py"),
            "HARBOR_GRADER_SH": str(tests_dir / "grader.sh"),
        },
    )

    assert completed.returncode != 0
    retained_text = custom_reward.read_text(encoding="utf-8")
    retained = json.loads(retained_text)
    assert retained["overall"] == 0.0
    assert "count exceeds" in retained["error"]
    assert all(name not in retained_text for name in raw_names)
    assert json.loads(reward_json.read_text(encoding="utf-8")) == {"overall": 0.0}


@pytest.mark.parametrize("mode", ["default_plus_custom", "custom_only"])
def test_generated_custom_grader_omits_unsafe_dict_valued_metric_from_every_artifact(
    tmp_path: Path,
    mode: str,
) -> None:
    tests_dir = tmp_path / "tests"
    verifier_dir = tmp_path / "verifier"
    tests_dir.mkdir()
    verifier_dir.mkdir()
    credential_metrics = [
        "api_key_quality",
        "ghp_" + ("a" * 36),
        "gho_" + ("a" * 36),
        "ghu_" + ("a" * 36),
        "ghs_" + ("a" * 36),
        "ghs_" + ("a" * 18) + ".-_" + ("b" * 18),
        "ghr_" + ("a" * 36),
        "qualityghp_" + ("a" * 36),
        "qualitygho_" + ("a" * 36) + "suffix",
        "github_pat_" + ("a" * 30),
        "".join(("xoxb-", "1234567890-abcdefghijklmnopqrstuvwx")),  # noqa: FLY002
        "AIza" + ("A" * 35),
        "glpat-" + ("a" * 20),
    ]
    unsafe_metrics = {name: {"score": 0.6, "reason": "must not survive"} for name in credential_metrics}
    detail_only_name = "detail_only_not_a_metric"
    custom_details = {
        "quality": {"report": "bounded evidence"},
        credential_metrics[0]: {"report": "must not survive"},
        detail_only_name: {"report": "must not survive"},
    }
    (tests_dir / "grader.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "payload = {'overall': 0.75, "
        f"**{unsafe_metrics!r}, "
        "'quality': {'score': 0.8, 'reason': 'bounded evidence'}, "
        f"'custom_details': {custom_details!r}}}\n"
        "Path(os.environ['HARBOR_REWARD_JSON']).write_text(json.dumps(payload))\n",
        encoding="utf-8",
    )
    reward_json = verifier_dir / "reward.json"
    reward_txt = verifier_dir / "reward.txt"
    skill_evaluator_reward = verifier_dir / "skill_evaluator_reward.json"
    custom_reward = verifier_dir / "custom_reward.json"
    if mode == "default_plus_custom":
        skill_evaluator_reward.write_text(
            json.dumps(
                {
                    "metric_set": DEFAULT_METRIC_SET,
                    **dict.fromkeys(custom_grader_runner.DEFAULT_METRICS, 1.0),
                    "overall": 1.0,
                }
            ),
            encoding="utf-8",
        )
        reward_txt.write_text("1.0", encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(_CUSTOM_RUNNER_TEMPLATE), "--mode", mode],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HARBOR_TESTS_DIR": str(tests_dir),
            "HARBOR_VERIFIER_DIR": str(verifier_dir),
            "HARBOR_REWARD_JSON": str(reward_json),
            "HARBOR_REWARD_TXT": str(reward_txt),
            "HARBOR_SKILL_EVALUATOR_REWARD_JSON": str(skill_evaluator_reward),
            "HARBOR_CUSTOM_REWARD_JSON": str(custom_reward),
            "HARBOR_GRADER": str(tests_dir / "grader.py"),
            "HARBOR_GRADER_SH": str(tests_dir / "grader.sh"),
        },
    )

    assert completed.returncode == 0, completed.stderr
    for artifact in (reward_json, custom_reward, skill_evaluator_reward):
        if not artifact.exists():
            continue
        text = artifact.read_text(encoding="utf-8")
        assert all(name not in text for name in credential_metrics)
        assert detail_only_name not in text
        assert "must not survive" not in text
    assert json.loads(reward_json.read_text(encoding="utf-8"))["quality"] == 0.8
    custom_payload = json.loads(custom_reward.read_text(encoding="utf-8"))
    assert custom_payload["quality"]["score"] == 0.8
    assert custom_payload["custom_details"] == {"quality": {"report": "bounded evidence"}}


@pytest.mark.parametrize("mode", ["default_plus_custom", "custom_only"])
def test_generated_custom_grader_mixed_metric_surfaces_match_collector_without_credential_drift(
    tmp_path: Path,
    mode: str,
) -> None:
    tests_dir = tmp_path / "tests"
    verifier_dir = tmp_path / "verifier"
    tests_dir.mkdir()
    verifier_dir.mkdir()
    credential_metrics = [
        "api_key_quality",
        "ghp_" + ("a" * 36),
        "gho_" + ("a" * 36),
        "ghu_" + ("a" * 36),
        "ghs_" + ("a" * 36),
        "ghs_" + ("a" * 18) + ".-_" + ("b" * 18),
        "ghr_" + ("a" * 36),
        "qualityghp_" + ("a" * 36),
        "qualitygho_" + ("a" * 36) + "suffix",
        "github_pat_" + ("a" * 30),
        "".join(("xoxb-", "1234567890-abcdefghijklmnopqrstuvwx")),  # noqa: FLY002
        "AIza" + ("A" * 35),
        "glpat-" + ("a" * 20),
        "quality_sk-abcdefghijk_nonnumeric_dict",
        "quality_nvapi-abcdefghijk_nonnumeric_string",
        "quality_crsr_0123456789abcdef_nonnumeric_null",
    ]
    unsafe_metrics = {name: {"score": 0.6, "reason": "must not survive"} for name in credential_metrics}
    unsafe_metrics.update(
        {
            "quality_sk-abcdefghijk_nonnumeric_dict": {"reason": "must not survive"},
            "quality_nvapi-abcdefghijk_nonnumeric_string": "must not survive",
            "quality_crsr_0123456789abcdef_nonnumeric_null": None,
        }
    )
    (tests_dir / "grader.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "payload = {'overall': 0.75, 'custom_metrics': {'quality': 0.8}, "
        "'domain_score': {'score': 0.7, 'reason': 'bounded evidence'}, "
        f"**{unsafe_metrics!r}}}\n"
        "Path(os.environ['HARBOR_REWARD_JSON']).write_text(json.dumps(payload))\n",
        encoding="utf-8",
    )
    reward_json = verifier_dir / "reward.json"
    reward_txt = verifier_dir / "reward.txt"
    skill_evaluator_reward = verifier_dir / "skill_evaluator_reward.json"
    custom_reward = verifier_dir / "custom_reward.json"
    if mode == "default_plus_custom":
        skill_evaluator_reward.write_text(
            json.dumps(
                {
                    "metric_set": DEFAULT_METRIC_SET,
                    **dict.fromkeys(custom_grader_runner.DEFAULT_METRICS, 1.0),
                    "overall": 1.0,
                }
            ),
            encoding="utf-8",
        )
        reward_txt.write_text("1.0", encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(_CUSTOM_RUNNER_TEMPLATE), "--mode", mode],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HARBOR_TESTS_DIR": str(tests_dir),
            "HARBOR_VERIFIER_DIR": str(verifier_dir),
            "HARBOR_REWARD_JSON": str(reward_json),
            "HARBOR_REWARD_TXT": str(reward_txt),
            "HARBOR_SKILL_EVALUATOR_REWARD_JSON": str(skill_evaluator_reward),
            "HARBOR_CUSTOM_REWARD_JSON": str(custom_reward),
            "HARBOR_GRADER": str(tests_dir / "grader.py"),
            "HARBOR_GRADER_SH": str(tests_dir / "grader.sh"),
        },
    )

    assert completed.returncode == 0, completed.stderr
    custom_payload = json.loads(custom_reward.read_text(encoding="utf-8"))
    harbor_payload = json.loads(reward_json.read_text(encoding="utf-8"))
    expected = {"domain_score": 0.7, "quality": 0.8}
    assert extract_custom_metrics(custom_payload) == expected
    assert extract_custom_metrics(harbor_payload) == expected
    for artifact in (reward_json, custom_reward, skill_evaluator_reward):
        if artifact.exists():
            text = artifact.read_text(encoding="utf-8")
            assert all(name not in text for name in credential_metrics)


@pytest.mark.parametrize("source", ["reward_json", "reward_txt"])
@pytest.mark.parametrize("invalid_score", ["nan", "inf", "-inf", "-0.01", "1.01", "1e308"])
def test_generated_custom_only_script_rejects_invalid_overall_score(
    tmp_path: Path,
    source: str,
    invalid_score: str,
) -> None:
    task_dir = tmp_path / f"custom-only-{source}"
    _write_test_sh(task_dir, grading_mode="custom_only", custom_grader=True)
    tests_dir = task_dir / "tests"
    evaluator_dir = tests_dir / _EVALUATOR_TESTS_SUBDIR
    evaluator_dir.mkdir()
    shutil.copy2(_CUSTOM_RUNNER_TEMPLATE, evaluator_dir / "custom_grader_runner.py")
    grader_lines = [
        "import json, os",
        "from pathlib import Path",
        "reward_json = Path(os.environ['HARBOR_REWARD_JSON'])",
        "reward_txt = Path(os.environ['HARBOR_REWARD_TXT'])",
    ]
    if source == "reward_json":
        grader_lines.append(f"reward_json.write_text(json.dumps({{'overall': float({invalid_score!r})}}))")
    else:
        grader_lines.extend(
            [
                "reward_json.write_text('{}')",
                f"reward_txt.write_text({invalid_score!r})",
            ]
        )
    (tests_dir / "grader.py").write_text("\n".join(grader_lines) + "\n", encoding="utf-8")
    verifier_dir = task_dir / "verifier"
    verifier_dir.mkdir()
    reward_json = verifier_dir / "reward.json"
    reward_txt = verifier_dir / "reward.txt"
    custom_reward_json = verifier_dir / "custom_reward.json"

    completed = _run_generated_test_sh(
        task_dir,
        {
            "HARBOR_TESTS_DIR": str(tests_dir),
            "HARBOR_VERIFIER_DIR": str(verifier_dir),
            "HARBOR_REWARD_JSON": str(reward_json),
            "HARBOR_REWARD_TXT": str(reward_txt),
            "HARBOR_CUSTOM_REWARD_JSON": str(custom_reward_json),
            "HARBOR_GRADER": str(tests_dir / "grader.py"),
            "HARBOR_GRADER_SH": str(tests_dir / "grader.sh"),
        },
    )

    assert completed.returncode != 0
    failure = json.loads(custom_reward_json.read_text(encoding="utf-8"))
    assert failure["overall"] == 0.0
    assert "between 0.0 and 1.0" in failure["error"]
    assert json.loads(reward_json.read_text(encoding="utf-8")) == {"overall": 0.0}
    assert reward_txt.read_text(encoding="utf-8") == "0.0"
