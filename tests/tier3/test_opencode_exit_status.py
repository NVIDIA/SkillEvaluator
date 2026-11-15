# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native OpenCode bash events preserve exit evidence through Harbor's converter."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from harbor.models.agent.context import AgentContext

from skillevaluator.tier3.eval_core.atif_helpers import extract_tool_calls_as_dicts
from skillevaluator.tier3.eval_core.checks import check_script_execution, check_tool_efficiency
from skillevaluator.tier3.eval_core.declarative_checks import evaluate_declarative_checks
from skillevaluator.tier3.eval_core.log_converters import load_trajectory_with_fallback
from skillevaluator.tier3.eval_core.plugin_signals import compute_plugin_signals
from skillevaluator.tier3.harbor.installed_agents import SkillEvaluatorOpenCode
from skillevaluator.tier3.harbor.local_agents import SkillEvaluatorGatewayOpenCode, SkillEvaluatorLocalOpenCode


@pytest.fixture
def native_events():
    path = Path(__file__).parent / "fixtures/opencode-bash-exit.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _convert(agent_class, events, tmp_path):
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")
    trajectory = agent._convert_events_to_trajectory(events)
    assert trajectory is not None
    return extract_tool_calls_as_dicts(trajectory.to_json_dict())


def _succeeded(calls):
    return evaluate_declarative_checks(
        [{"type": "command_succeeded", "script": "scripts/validate_input.py"}],
        tool_calls=calls,
        final_response="",
        workspace_roots=[],
    )[0]


@pytest.mark.parametrize(
    "agent_class", [SkillEvaluatorOpenCode, SkillEvaluatorLocalOpenCode, SkillEvaluatorGatewayOpenCode]
)
@pytest.mark.parametrize("exit_code", [0, 2, 17, -9])
def test_native_bash_exit_is_attributable(agent_class, exit_code, native_events, tmp_path):
    native_events[1]["part"]["state"]["metadata"]["exit"] = exit_code
    original = copy.deepcopy(native_events)
    calls = _convert(agent_class, native_events, tmp_path)

    assert native_events == original
    assert len(calls) == 1
    assert f"Exit code: {exit_code}\n" in calls[0]["observation"]
    result = _succeeded(calls)
    assert result["passed"] is (exit_code == 0)
    if exit_code:
        assert result["reason"].endswith(f"it was invoked but exited {exit_code}")
    assert check_script_execution(calls, "validate_input.py")["score"] == 1.0
    assert check_tool_efficiency(calls, expected_script="validate_input.py")["score"] == 1.0


@pytest.mark.parametrize("metadata", [None, {}, {"exit": None}, {"exit": True}, {"exit": 0.5}, {"exit": "unknown"}])
def test_unavailable_exit_stays_not_checked(metadata, native_events, tmp_path):
    native_events[1]["part"]["state"]["metadata"] = metadata
    calls = _convert(SkillEvaluatorOpenCode, native_events, tmp_path)
    assert _succeeded(calls)["passed"] is None
    assert "exit status was not recorded" in _succeeded(calls)["reason"]


@pytest.mark.parametrize("tool,status", [("read", "completed"), ("bash", "running"), ("bash", "pending")])
def test_only_finished_bash_calls_gain_exit_evidence(tool, status, native_events, tmp_path):
    native_events[1]["part"]["tool"] = tool
    native_events[1]["part"]["state"]["status"] = status
    calls = _convert(SkillEvaluatorOpenCode, native_events, tmp_path)
    assert "Exit code:" not in calls[0]["observation"]


def test_empty_events_remain_empty(tmp_path):
    agent = SkillEvaluatorOpenCode(logs_dir=tmp_path, model_name="openai/test-model")
    assert agent._convert_events_to_trajectory([]) is None


@pytest.mark.parametrize("status", ["completed", "error"])
@pytest.mark.parametrize("output", [None, ""])
def test_exit_without_output_is_retained(status, output, native_events, tmp_path):
    state = native_events[1]["part"]["state"]
    state.update(status=status, output=output)
    calls = _convert(SkillEvaluatorOpenCode, native_events, tmp_path)
    assert _succeeded(calls)["passed"] is False
    assert "exited 2" in _succeeded(calls)["reason"]


def test_grader_loads_generated_trajectory(native_events, tmp_path):
    agent = SkillEvaluatorOpenCode(logs_dir=tmp_path, model_name="openai/test-model")
    trajectory_path = tmp_path / "trajectory.json"
    (tmp_path / "opencode.txt").write_text("\n".join(json.dumps(event) for event in native_events))
    agent.populate_context_post_run(AgentContext())
    loaded, metadata = load_trajectory_with_fallback(trajectory_path)
    assert metadata["source"] == "trajectory.json"
    result = _succeeded(extract_tool_calls_as_dicts(loaded))
    assert result["passed"] is False
    assert "exited 2" in result["reason"]


def test_status_is_paired_per_call(native_events, tmp_path):
    second = copy.deepcopy(native_events[1])
    second["part"]["callID"] = "call-second"
    second["part"]["state"]["metadata"]["exit"] = 0
    native_events.insert(2, second)
    calls = _convert(SkillEvaluatorOpenCode, native_events, tmp_path)
    assert "Exit code: 2" in calls[0]["observation"]
    assert "Exit code: 0" in calls[1]["observation"]


def test_shell_tail_status_is_not_script_success(native_events, tmp_path):
    state = native_events[1]["part"]["state"]
    state["input"]["command"] += "; echo finished"
    state["metadata"]["exit"] = 0
    calls = _convert(SkillEvaluatorOpenCode, native_events, tmp_path)
    result = _succeeded(calls)
    assert result["passed"] is None
    assert "could not be attributed" in result["reason"]


@pytest.mark.parametrize(
    "agent_class", [SkillEvaluatorOpenCode, SkillEvaluatorLocalOpenCode, SkillEvaluatorGatewayOpenCode]
)
def test_failed_bash_with_leading_error_stays_failed_in_plugin_signals(agent_class, native_events, tmp_path):
    state = native_events[1]["part"]["state"]
    state["output"] = "bash: helper: command not found\nRun setup before retrying.\n"
    state["metadata"].update(output=state["output"], exit=127)
    trajectory = agent_class(logs_dir=tmp_path, model_name="openai/test-model")._convert_events_to_trajectory(
        native_events
    )
    assert trajectory is not None

    case = {"tool_arguments": [{"tool": "bash", "contains": {"command": "validate_input"}}]}
    arguments = compute_plugin_signals(trajectory.to_json_dict(), case, declared={})["arguments"]

    assert (arguments["checked"], arguments["passed"]) == (1, 0)
    assert [failure["rule"] for failure in arguments["failures"]] == ["call_failed"]


def test_zero_exit_bash_with_error_text_still_passes_in_plugin_signals(native_events, tmp_path):
    state = native_events[1]["part"]["state"]
    state["output"] = "bash: helper: command not found\nRun setup before retrying.\n"
    state["metadata"].update(output=state["output"], exit=0)
    trajectory = SkillEvaluatorOpenCode(
        logs_dir=tmp_path, model_name="openai/test-model"
    )._convert_events_to_trajectory(native_events)
    assert trajectory is not None

    case = {"tool_arguments": [{"tool": "bash", "contains": {"command": "validate_input"}}]}
    arguments = compute_plugin_signals(trajectory.to_json_dict(), case, declared={})["arguments"]

    assert (arguments["checked"], arguments["passed"]) == (1, 1)
