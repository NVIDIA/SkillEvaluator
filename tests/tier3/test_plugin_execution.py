# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private plugin evaluation lifecycle over the existing engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from skillevaluator.evaluation.options import EvaluationOptions
from skillevaluator.tier3 import plugin_execution as px

OK = {"execution_status": "succeeded"}
SETUP = {"execution_status": "failed", "error": ["claude-code with-skill: AgentSetupNetworkError: dns"]}
AGENT = {"execution_status": "failed", "error": ["claude-code with-skill: NonZeroAgentExitCodeError: 1"]}


def _options(tmp_path: Path) -> EvaluationOptions:
    return EvaluationOptions(skill_path=tmp_path)


def _sequence(*results):
    calls: list[object] = []
    items = list(results)

    def evaluate(options):
        calls.append(px.active_runtime())
        item = items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return dict(item)

    return evaluate, calls


def test_success_publishes_before_cleanup_and_activates_hooks_only_inside_the_run(tmp_path: Path) -> None:
    order: list[str] = []
    hooks = object()
    evaluate, calls = _sequence(OK)
    outcome = px.run_private_evaluation(
        _options(tmp_path),
        hooks=hooks,  # type: ignore[arg-type]
        provenance={"plugin": "demo"},
        evaluate=evaluate,
        publish=lambda _: order.append("publish"),
        cleanup=lambda: order.append("cleanup"),
    )
    assert outcome.status == "succeeded" and outcome.attempts == 1
    assert order == ["publish", "cleanup"]
    assert calls == [hooks] and px.active_runtime() is None


def test_setup_only_failure_is_retried_within_the_bound(tmp_path: Path) -> None:
    evaluate, _ = _sequence(SETUP, SETUP, OK)
    outcome = px.run_private_evaluation(_options(tmp_path), hooks=None, provenance={}, evaluate=evaluate)
    assert outcome.status == "succeeded" and outcome.attempts == 3 and len(outcome.retry_reasons) == 2
    evaluate, _ = _sequence(SETUP, SETUP, SETUP)
    outcome = px.run_private_evaluation(_options(tmp_path), hooks=None, provenance={}, evaluate=evaluate)
    assert outcome.status == "failed" and outcome.attempts == 3


@pytest.mark.parametrize(
    "result",
    [AGENT, {"execution_status": "failed", "error": ["AgentSetupNetworkError", "verifier timed out"]}, {}],
)
def test_agent_verifier_and_ambiguous_failures_are_never_retried(tmp_path: Path, result: dict) -> None:
    evaluate, calls = _sequence(result, OK)
    outcome = px.run_private_evaluation(_options(tmp_path), hooks=None, provenance={}, evaluate=evaluate)
    assert outcome.attempts == 1 and outcome.status == "failed" and len(calls) == 1


def test_cancellation_is_not_retried_and_still_cleans_up(tmp_path: Path) -> None:
    cleaned: list[bool] = []
    evaluate, calls = _sequence(KeyboardInterrupt(), OK)
    with pytest.raises(KeyboardInterrupt):
        px.run_private_evaluation(
            _options(tmp_path),
            hooks=object(),  # type: ignore[arg-type]
            provenance={},
            evaluate=evaluate,
            cleanup=lambda: cleaned.append(True),
        )
    assert len(calls) == 1 and cleaned == [True] and px.active_runtime() is None


def test_publication_and_cleanup_failures_keep_the_run_status(tmp_path: Path) -> None:
    def boom(*_):
        raise OSError("disk full")

    evaluate, _ = _sequence(AGENT)
    outcome = px.run_private_evaluation(
        _options(tmp_path), hooks=None, provenance={}, evaluate=evaluate, publish=boom, cleanup=boom
    )
    assert outcome.status == "failed"
    assert outcome.publication_error == "OSError: disk full" == outcome.cleanup_error


def test_retry_bound_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_setup_retries"):
        px.run_private_evaluation(_options(tmp_path), hooks=None, provenance={}, evaluate=dict, max_setup_retries=9)


def test_ordinary_runs_have_no_active_runtime() -> None:
    from skillevaluator.tier3.harbor import runner

    assert runner._private_runtime_hooks() is None
    assert runner._private_with_agent_import_path("claude-code", None) is None
