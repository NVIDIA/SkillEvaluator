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
from skillevaluator.tier3.harbor.adapter import _write_test_sh
from skillevaluator.tier3.harbor.metrics import (
    DEFAULT_METRIC_SET,
    RESERVED_METRIC_NAMES,
    metric_set_for_reward,
    overall_score,
)
from skillevaluator.tier3.harbor.templates import custom_grader_runner

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_TEMPLATE = _REPO_ROOT / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
_CUSTOM_RUNNER_TEMPLATE = (
    _REPO_ROOT / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "custom_grader_runner.py"
)


def _load_verifier(tmp_path: Path) -> ModuleType:
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
    """Slow transient calls must finish before Harbor's verifier timeout kills artifact writes."""
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
    """A provider that trickles a response must not keep the verifier alive indefinitely."""
    if not hasattr(signal, "setitimer") or signal.getitimer(signal.ITIMER_REAL)[0] > 0:
        pytest.skip("free POSIX interval timer required")
    verifier = _load_verifier(tmp_path)
    monkeypatch.setattr(verifier, "_JUDGE_WALL_TIME_BUDGET_SEC", 0.05)

    class SlowResponse:
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


def test_required_judge_restores_alarm_handler_after_teardown_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    """The default OpenAI goal scorer must return before its Harbor budget expires."""
    verifier = _load_verifier(tmp_path)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-fake")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
    monkeypatch.setattr(verifier, "_ragas_goal_accuracy_enabled", lambda: True)
    monkeypatch.setattr(verifier, "_JUDGE_WALL_TIME_BUDGET_SEC", 0.01)

    class FakeMessage:
        def __init__(self, content: str):
            self.content = content

    class FakeMetric:
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


def test_verifier_main_keeps_documented_neutral_judge_skips_scoreable(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier(tmp_path)
    entry = json.loads(verifier.ENTRY_PATH.read_text(encoding="utf-8"))
    entry["ground_truth"] = ""
    entry["expected_behavior"] = []
    verifier.ENTRY_PATH.write_text(json.dumps(entry), encoding="utf-8")

    verifier.main()

    rich = json.loads(verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    numeric = json.loads(verifier.REWARD_JSON.read_text(encoding="utf-8"))
    assert "evaluation_status" not in rich
    assert {metric: numeric[metric] for metric in ("accuracy", "goal_accuracy", "behavior_check")} == {
        "accuracy": 1.0,
        "goal_accuracy": 1.0,
        "behavior_check": 1.0,
    }
    assert overall_score(numeric) == 1.0


@pytest.mark.parametrize("failure_kind", ["missing-score", "exception"])
def test_verifier_main_normalizes_malformed_or_raised_judge_failures_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
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
    expected = {"evaluation_status", "evaluation_errors"}

    assert expected <= RESERVED_METRIC_NAMES
    assert expected <= custom_grader_runner.RESERVED
    assert custom_grader_runner._extract_custom_metrics(
        {"evaluation_status": 1.0, "evaluation_errors": 0.5, "domain_score": 0.75}
    ) == {"domain_score": 0.75}
    with pytest.raises(RuntimeError, match="collides with reserved"):
        custom_grader_runner._extract_custom_metrics({"custom_metrics": {"evaluation_status": 0.5}})


def _run_generated_test_sh(task_dir: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
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
    task_dir = tmp_path / grading_mode
    _write_test_sh(task_dir, grading_mode=grading_mode, custom_grader=grading_mode == "default_plus_custom")
    tests_dir = task_dir / "tests"
    (tests_dir / "eval.py").write_text("raise SystemExit(7)\n", encoding="utf-8")
    marker = task_dir / "custom-ran"
    (tests_dir / "custom_grader_runner.py").write_text(
        "from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('ran')\n",
        encoding="utf-8",
    )

    completed = _run_generated_test_sh(task_dir, {"HARBOR_TESTS_DIR": str(tests_dir)})

    assert completed.returncode == 7
    assert not marker.exists()


def test_generated_custom_only_script_accepts_overall_only_custom_reward(tmp_path: Path) -> None:
    task_dir = tmp_path / "custom-only"
    _write_test_sh(task_dir, grading_mode="custom_only", custom_grader=True)
    tests_dir = task_dir / "tests"
    shutil.copy2(_CUSTOM_RUNNER_TEMPLATE, tests_dir / "custom_grader_runner.py")
    marker = task_dir / "custom-ran"
    (tests_dir / "grader.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('ran')\n"
        "Path(os.environ['HARBOR_REWARD_JSON']).write_text(json.dumps({'overall': 0.75}))\n",
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
