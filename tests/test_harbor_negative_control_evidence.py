# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted negative-control invocation evidence collected from Harbor ATIF."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.tier3.harbor.collector import (
    REWARD_DIAGNOSTIC_STRING_MAX_CHARS,
    collect_harbor_results,
)

STANDARD_REWARD = {
    "security": 0.9,
    "skill_execution": 0.4,
    "skill_efficiency": 0.8,
    "accuracy": 0.7,
    "goal_accuracy": 0.6,
    "behavior_check": 0.5,
}


def _nested_shell_command(command: str, depth: int) -> str:
    for _ in range(depth):
        command = shlex.join(["bash", "-c", command])
    return command


def _trajectory(*, skill: str | None = None) -> dict[str, Any]:
    tool_calls = []
    if skill is not None:
        tool_calls.append(
            {
                "tool_call_id": "call-1",
                "function_name": "Read",
                "arguments": {"file_path": f"/workspace/skills/{skill}/SKILL.md"},
            }
        )
    return {
        "schema_version": "ATIF-v1.2",
        "steps": [
            {
                "step_id": 1,
                "source": "agent",
                "message": "done",
                "tool_calls": tool_calls,
                "observation": {"results": []},
            }
        ],
    }


def _write_job_result(job_dir: Path, trial_names: list[str]) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": len(trial_names),
                "stats": {
                    "n_trials": len(trial_names),
                    "n_errors": 0,
                    "evals": {
                        "agent__model___harbor-tasks": {
                            "n_trials": len(trial_names),
                            "n_errors": 0,
                            "reward_stats": {"reward": {"0.1": trial_names}},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _write_trial(
    jobs_dir: Path,
    *,
    variant: str,
    trial_name: str,
    reward: dict[str, Any],
    trajectory: dict[str, Any] | str | None,
) -> Path:
    trial_dir = jobs_dir / f"demo-opencode-{variant}" / trial_name
    verifier_dir = trial_dir / "verifier"
    verifier_dir.mkdir(parents=True)
    (verifier_dir / "reward.json").write_text(json.dumps(reward), encoding="utf-8")
    if trajectory is not None:
        agent_dir = trial_dir / "agent"
        agent_dir.mkdir()
        serialized = trajectory if isinstance(trajectory, str) else json.dumps(trajectory)
        (agent_dir / "trajectory.json").write_text(serialized, encoding="utf-8")
    return trial_dir


def _collect(tmp_path: Path, *, skip_baseline: bool = True, expected_trials: int = 1) -> dict[str, Any]:
    return collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        skip_baseline=skip_baseline,
        expected_cases=expected_trials,
        expected_trials=expected_trials,
    )


def _persisted_reward(tmp_path: Path, variant: str, trial_name: str) -> dict[str, Any]:
    condition = "with-skill" if variant == "with" else "without-skill"
    path = tmp_path / "results" / "opencode" / condition / "trials" / trial_name / "reward.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_collect_persists_target_invocation_for_both_single_step_variants(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    for variant, observed_skill in (("with", "demo"), ("without", "unrelated")):
        trial_name = f"case-{variant}_attempt001"
        trial_dir = _write_trial(
            jobs_dir,
            variant=variant,
            trial_name=trial_name,
            reward=dict(STANDARD_REWARD),
            trajectory=_trajectory(skill=observed_skill),
        )
        (trial_dir / "result.json").write_text(
            json.dumps(
                {
                    "trial_name": trial_name,
                    "task_name": f"case-{variant}",
                    "verifier_result": {"rewards": dict(STANDARD_REWARD)},
                    "step_results": None,
                }
            ),
            encoding="utf-8",
        )
        _write_job_result(jobs_dir / f"demo-opencode-{variant}", [trial_name])

    result = _collect(tmp_path, skip_baseline=False)

    with_reward = _persisted_reward(tmp_path, "with", "case-with_attempt001")
    without_reward = _persisted_reward(tmp_path, "without", "case-without_attempt001")
    assert with_reward["skill_invoked"] is True
    assert without_reward["skill_invoked"] is False
    assert with_reward["invocation_evidence_source"] == "trajectory"
    assert without_reward["invocation_evidence_source"] == "trajectory"
    assert "routing_passed" not in with_reward
    assert "routing_passed" not in without_reward
    assert result["agents"]["opencode"]["with_skill"]["skill_execution"] == 0.4
    assert result["agents"]["opencode"]["without_skill"]["skill_execution"] == 0.4
    for variant, trial_name in (("with-skill", "case-with_attempt001"), ("without-skill", "case-without_attempt001")):
        trial_dir = tmp_path / "results" / "opencode" / variant / "trials" / trial_name
        assert {path.name for path in trial_dir.iterdir()} == {"result.json", "reward.json", "trajectory.json"}


def test_collect_redacts_successful_standard_reward_without_diagnostic_truncation(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    trial_name = "successful-redaction_attempt001"
    long_note = "x" * (REWARD_DIAGNOSTIC_STRING_MAX_CHARS + 1)
    reward = {
        **STANDARD_REWARD,
        "details": {
            "api_key": "plain-text-test-secret",
            "password": 123456789,
            "message": "Authorization: Bearer secret-token-value",
            "safe_note": long_note,
        },
        "custom_metrics": {
            "secret_safety": 0.91,
            "api_key": 135792468,
            "api_token": 135792468,
            "private_token": 135792468,
            "client_token": 135792468,
        },
        "metrics": {
            "password": 97531,
            "github_token": 97531,
            "gitlab_token": 97531,
            "ssh_key": 97531,
            "signing_key": 97531,
        },
        "evaluation_errors": {},
        "api_key": 246813579,
    }
    _write_trial(
        jobs_dir,
        variant="with",
        trial_name=trial_name,
        reward=reward,
        trajectory=_trajectory(skill="demo"),
    )
    _write_job_result(jobs_dir / "demo-opencode-with", [trial_name])

    _collect(tmp_path)

    persisted = _persisted_reward(tmp_path, "with", trial_name)
    assert persisted["skill_invoked"] is True
    assert persisted["invocation_evidence_source"] == "trajectory"
    assert persisted["details"]["api_key"] == "<redacted>"
    assert "secret-token-value" not in persisted["details"]["message"]
    assert "<redacted>" in persisted["details"]["message"]
    assert persisted["details"]["safe_note"] == long_note
    assert persisted["details"]["password"] == "<redacted>"
    assert persisted["custom_metrics"]["secret_safety"] == 0.91
    for name in ("api_key", "api_token", "private_token", "client_token"):
        assert persisted["custom_metrics"][name] == "<redacted>"
    for name in ("password", "github_token", "gitlab_token", "ssh_key", "signing_key"):
        assert persisted["metrics"][name] == "<redacted>"
    assert persisted["api_key"] == "<redacted>"


def test_collect_persists_native_skill_tool_invocation(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    trial_name = "native-skill_attempt001"
    trajectory = _trajectory()
    trajectory["steps"][0]["tool_calls"] = [
        {
            "tool_call_id": "call-1",
            "function_name": "Skill",
            "arguments": {"skill": "demo"},
        }
    ]
    _write_trial(
        jobs_dir,
        variant="with",
        trial_name=trial_name,
        reward=dict(STANDARD_REWARD),
        trajectory=trajectory,
    )
    _write_job_result(jobs_dir / "demo-opencode-with", [trial_name])

    _collect(tmp_path)

    reward = _persisted_reward(tmp_path, "with", trial_name)
    assert reward["skill_invoked"] is True
    assert reward["invocation_evidence_source"] == "trajectory"


@pytest.mark.parametrize("variant", ("with", "without"))
@pytest.mark.parametrize(
    ("fragments", "expected_invoked"),
    [
        (
            (
                ("prepare", _trajectory(skill="demo")),
                ("finish", _trajectory(skill="unrelated")),
            ),
            True,
        ),
        (
            (
                ("prepare", _trajectory(skill="unrelated")),
                ("finish", _trajectory()),
            ),
            False,
        ),
        (
            (
                ("prepare", _trajectory(skill="unrelated")),
                ("missing", None),
                ("malformed", "{not-json"),
            ),
            None,
        ),
    ],
    ids=("any-step-invoked", "all-readable-not-invoked", "missing-and-malformed-unknown"),
)
def test_collect_derives_authoritative_multistep_invocation_for_both_variants(
    tmp_path: Path,
    variant: str,
    fragments: tuple[tuple[str, dict[str, Any] | str | None], ...],
    expected_invoked: bool | None,
) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / f"demo-opencode-{variant}"
    trial_name = f"case-multi-{variant}_attempt001"
    trial_dir = job_dir / trial_name
    step_results = []
    for step_name, trajectory in fragments:
        if trajectory is not None:
            agent_dir = trial_dir / "steps" / step_name / "agent"
            agent_dir.mkdir(parents=True)
            serialized = trajectory if isinstance(trajectory, str) else json.dumps(trajectory)
            (agent_dir / "trajectory.json").write_text(serialized, encoding="utf-8")
        step_results.append(
            {
                "step_name": step_name,
                "verifier_result": {"rewards": dict(STANDARD_REWARD)},
            }
        )
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_name": trial_name,
                "task_name": "case-multi",
                "verifier_result": {"rewards": dict(STANDARD_REWARD)},
                "step_results": step_results,
            }
        ),
        encoding="utf-8",
    )
    _write_job_result(job_dir, [trial_name])

    _collect(tmp_path, skip_baseline=variant == "with")

    reward = _persisted_reward(tmp_path, variant, trial_name)
    if expected_invoked is None:
        assert "skill_invoked" not in reward
        assert "invocation_evidence_source" not in reward
    else:
        assert reward["skill_invoked"] is expected_invoked
        assert reward["invocation_evidence_source"] == "trajectory"


@pytest.mark.parametrize(
    "trial_result",
    [None, "{not-json", "{}", '{"step_results": null}'],
    ids=("missing-result", "malformed-result", "no-step-results", "null-with-steps-layout"),
)
def test_collect_does_not_certify_root_non_invocation_when_step_layout_is_ambiguous(
    tmp_path: Path,
    trial_result: str | None,
) -> None:
    jobs_dir = tmp_path / "jobs"
    trial_name = "ambiguous-steps_attempt001"
    trial_dir = _write_trial(
        jobs_dir,
        variant="with",
        trial_name=trial_name,
        reward=dict(STANDARD_REWARD),
        trajectory=_trajectory(),
    )
    step_agent_dir = trial_dir / "steps" / "invoke" / "agent"
    step_agent_dir.mkdir(parents=True)
    (step_agent_dir / "trajectory.json").write_text(json.dumps(_trajectory(skill="demo")), encoding="utf-8")
    if trial_result is not None:
        (trial_dir / "result.json").write_text(trial_result, encoding="utf-8")
    _write_job_result(jobs_dir / "demo-opencode-with", [trial_name])

    _collect(tmp_path)

    persisted = _persisted_reward(tmp_path, "with", trial_name)
    assert "skill_invoked" not in persisted
    assert "invocation_evidence_source" not in persisted


def test_collect_rejects_null_step_results_with_a_physical_steps_layout(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    trial_name = "null-with-steps_attempt001"
    trial_dir = _write_trial(
        jobs_dir,
        variant="with",
        trial_name=trial_name,
        reward=dict(STANDARD_REWARD),
        trajectory=_trajectory(skill="demo"),
    )
    step_agent_dir = trial_dir / "steps" / "unexpected" / "agent"
    step_agent_dir.mkdir(parents=True)
    (step_agent_dir / "trajectory.json").write_text(json.dumps(_trajectory(skill="demo")), encoding="utf-8")
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_name": trial_name,
                "task_name": "null-with-steps",
                "verifier_result": {"rewards": dict(STANDARD_REWARD)},
                "step_results": None,
            }
        ),
        encoding="utf-8",
    )
    _write_job_result(jobs_dir / "demo-opencode-with", [trial_name])

    result = _collect(tmp_path)

    condition = result["agents"]["opencode"]["conditions"]["with_skill"]
    assert result["execution_status"] == "failed"
    assert condition["execution_status"] == "failed"
    assert condition["scored_attempts"] == 0
    assert "constituent" in " ".join(condition["execution_errors"]).casefold()


def test_collect_replaces_spoofed_standard_evidence_and_leaves_unreadable_unknown(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    spoofed = {
        **STANDARD_REWARD,
        "skill_invoked": True,
        "routing_passed": True,
        "invocation_evidence_source": "reward",
    }
    trajectories: tuple[tuple[str, dict[str, Any] | str | None], ...] = (
        ("readable_attempt001", _trajectory(skill="unrelated")),
        ("missing_attempt001", None),
        ("malformed_attempt001", "{not-json"),
    )
    for trial_name, trajectory in trajectories:
        _write_trial(
            jobs_dir,
            variant="with",
            trial_name=trial_name,
            reward=dict(spoofed),
            trajectory=trajectory,
        )
    _write_job_result(jobs_dir / "demo-opencode-with", [name for name, _ in trajectories])

    result = _collect(tmp_path, expected_trials=3)

    readable = _persisted_reward(tmp_path, "with", "readable_attempt001")
    assert readable["skill_invoked"] is False
    assert readable["invocation_evidence_source"] == "trajectory"
    assert "routing_passed" not in readable
    for trial_name in ("missing_attempt001", "malformed_attempt001"):
        reward = _persisted_reward(tmp_path, "with", trial_name)
        assert "skill_invoked" not in reward
        assert "routing_passed" not in reward
        assert "invocation_evidence_source" not in reward
    assert result["agents"]["opencode"]["with_skill"]["skill_execution"] == 0.4
    assert result["agents"]["opencode"]["num_trials_with"] == 3


@pytest.mark.parametrize("metric_set", ("skill-evaluator-default-v1", "skill-evaluator-default-v2"))
def test_collect_replaces_spoofed_evidence_for_nested_standard_metrics(tmp_path: Path, metric_set: str) -> None:
    jobs_dir = tmp_path / "jobs"
    trial_name = "nested-standard_attempt001"
    nested_reward = {
        "metric_set": metric_set,
        "metrics": {name: {"score": score} for name, score in STANDARD_REWARD.items()},
        "skill_invoked": True,
        "routing_passed": True,
        "invocation_evidence_source": "reward",
    }
    _write_trial(
        jobs_dir,
        variant="with",
        trial_name=trial_name,
        reward=nested_reward,
        trajectory=_trajectory(skill="unrelated"),
    )
    _write_job_result(jobs_dir / "demo-opencode-with", [trial_name])

    result = _collect(tmp_path)

    persisted = _persisted_reward(tmp_path, "with", trial_name)
    assert persisted["metrics"] == nested_reward["metrics"]
    assert persisted["skill_invoked"] is False
    assert persisted["invocation_evidence_source"] == "trajectory"
    assert "routing_passed" not in persisted
    assert result["agents"]["opencode"]["with_skill"]["skill_execution"] == 0.4


@pytest.mark.parametrize(
    ("command", "expected_invoked"),
    [
        ("cat /workspace/skills/demo/SKILL.md", True),
        ("cat /workspace/skills/demo/./SKILL.md", True),
        ("cat /workspace/skills/demo//SKILL.md", True),
        ("cat /workspace/skills/'demo'/SKILL.md", True),
        (r"cat 'C:\skills\demo\SKILL.md'", True),
        ("python /workspace/skills/demo/scripts/run.py", True),
        ("cat /workspace/skills/demo-helper/SKILL.md", False),
        ("printf demo", False),
        ("echo /workspace/skills/demo/SKILL.md", False),
        ("printf /workspace/skills/demo/SKILL.md", False),
        ("echo /workspace/skills/demo/scripts/run.py", False),
        ("printf ignored\ncat /workspace/skills/demo/SKILL.md", True),
        ("echo ignored\npython /workspace/skills/demo/scripts/run.py", True),
        ("/usr/bin/env python /workspace/skills/demo/scripts/run.py", True),
        ("bash -c 'python /workspace/skills/demo/scripts/run.py'", True),
        ("dash -c 'cat /workspace/skills/demo/SKILL.md'", True),
        ("sh -c 'cat /workspace/skills/demo/SKILL.md'", True),
        ("zsh -c 'command cat /workspace/skills/demo/SKILL.md'", True),
        ("command cat /workspace/skills/demo/SKILL.md", True),
        ("command env MODE=test cat /workspace/skills/demo/SKILL.md", True),
        ("env MODE=test cat /workspace/skills/demo/SKILL.md", True),
        ("env -i python /workspace/skills/demo/scripts/run.py", True),
        ("source /workspace/skills/demo/scripts/run.sh", True),
        (". /workspace/skills/demo/scripts/run.sh", True),
        ("exec /workspace/skills/demo/scripts/run.sh", True),
        ("exec python /workspace/skills/demo/scripts/run.py", True),
        ("CONTENT=$(cat /workspace/skills/demo/SKILL.md)", True),
        ("timeout 10 cat /workspace/skills/demo/SKILL.md", True),
        ("nice python /workspace/skills/demo/scripts/run.py", True),
        ("echo prep & cat /workspace/skills/demo/SKILL.md", True),
        ('echo "$(cat /workspace/skills/demo/SKILL.md)"', True),
        ("echo $(cat /workspace/skills/demo/SKILL.md)", True),
        ('printf %s "$(cat /workspace/skills/demo/SKILL.md)"', True),
        ("echo <(cat /workspace/skills/demo/SKILL.md)", True),
        ("echo `cat /workspace/skills/demo/SKILL.md`", True),
        ("P=/workspace/skills/demo; cat $P/SKILL.md", True),
        ('P=/workspace/skills/demo; cat "${P}/SKILL.md"', True),
        ("cd /workspace/skills/demo && cat SKILL.md", True),
        ("sh -c -- 'cat /workspace/skills/demo/SKILL.md'", True),
        ("bash -c -- 'python /workspace/skills/demo/scripts/run.py'", True),
        ("bash -c 'echo /workspace/skills/demo/SKILL.md'", False),
        ("dash -c 'echo /workspace/skills/demo/SKILL.md'", False),
        ("bash -c 'echo /workspace/skills/demo/scripts/run.py'", False),
        ("env -- echo /workspace/skills/demo/scripts/run.py", False),
        ("env echo /workspace/skills/demo/scripts/run.py", False),
        ("echo 'note\ncat /workspace/skills/demo/SKILL.md'", False),
        ("echo /workspace/skills/demo/SKILL.md\nprintf /workspace/skills/demo/scripts/run.py", False),
        ("CONTENT=$(printf /workspace/skills/demo/SKILL.md)", False),
        ("timeout 10 echo /workspace/skills/demo/SKILL.md", False),
        ("nice echo /workspace/skills/demo/scripts/run.py", False),
        ("echo '$(cat /workspace/skills/demo/SKILL.md)'", False),
        ("echo '`cat /workspace/skills/demo/SKILL.md`'", False),
        ("P=/workspace/skills/demo; echo $P/SKILL.md", False),
        ("cd /workspace/skills/demo && echo SKILL.md", False),
    ],
)
def test_collect_shell_invocation_requires_exact_target_artifact_path(
    tmp_path: Path,
    command: str,
    expected_invoked: bool,
) -> None:
    jobs_dir = tmp_path / "jobs"
    trial_name = "shell_attempt001"
    trajectory = _trajectory()
    trajectory["steps"][0]["tool_calls"] = [
        {
            "tool_call_id": "call-1",
            "function_name": "exec_command",
            "arguments": {"cmd": command},
        }
    ]
    _write_trial(
        jobs_dir,
        variant="with",
        trial_name=trial_name,
        reward=dict(STANDARD_REWARD),
        trajectory=trajectory,
    )
    _write_job_result(jobs_dir / "demo-opencode-with", [trial_name])

    _collect(tmp_path)

    persisted = _persisted_reward(tmp_path, "with", trial_name)
    assert persisted["skill_invoked"] is expected_invoked
    assert persisted["invocation_evidence_source"] == "trajectory"


@pytest.mark.parametrize(
    "command",
    [
        "echo " + " ".join(f"padding-{index}" for index in range(255)) + "; cat /skills/demo/SKILL.md",
        "echo " + ("x" * 32_768) + "\ncat /skills/demo/SKILL.md",
        _nested_shell_command("cat /skills/demo/SKILL.md", 3),
        "mystery-wrapper cat /skills/demo/SKILL.md",
        "cat /workspace/skills/dem?/SKILL.md",
        "find /workspace/skills/demo -name SKILL.md -exec cat {} ';'",
        "python -c \"print(open('/workspace/skills/demo/SKILL.md').read())\"",
    ],
    ids=(
        "token-cap",
        "character-cap",
        "recursion-cap",
        "unsupported-wrapper",
        "glob-expansion",
        "find-expansion",
        "python-inline-open",
    ),
)
def test_collect_omits_invocation_evidence_when_shell_parser_hits_a_bound(tmp_path: Path, command: str) -> None:
    jobs_dir = tmp_path / "jobs"
    trial_name = "bounded-shell_attempt001"
    trajectory = _trajectory()
    trajectory["steps"][0]["tool_calls"] = [
        {
            "tool_call_id": "call-1",
            "function_name": "exec_command",
            "arguments": {"cmd": command},
        }
    ]
    _write_trial(
        jobs_dir,
        variant="with",
        trial_name=trial_name,
        reward=dict(STANDARD_REWARD),
        trajectory=trajectory,
    )
    _write_job_result(jobs_dir / "demo-opencode-with", [trial_name])

    _collect(tmp_path)

    persisted = _persisted_reward(tmp_path, "with", trial_name)
    assert "skill_invoked" not in persisted
    assert "invocation_evidence_source" not in persisted


@pytest.mark.parametrize(
    ("trajectory", "expected_invoked"),
    [
        ({"steps": []}, None),
        ({"steps": [{"source": "user", "message": "request"}]}, None),
        ({"steps": ["malformed"]}, None),
        ({"steps": [{"source": "agent"}]}, None),
        ({"steps": [{"source": "agent", "tool_calls": {}}]}, None),
        ({"steps": [{"source": "agent", "tool_calls": [None]}]}, None),
        ({"steps": [{"source": "agent", "tool_calls": [{}]}]}, None),
        ({"steps": [{"source": "agent", "tool_calls": [{"function_name": "  ", "arguments": {}}]}]}, None),
        (
            {
                "steps": [
                    {
                        "source": "agent",
                        "tool_calls": [{"function_name": "Read", "arguments": "/skills/demo/SKILL.md"}],
                    }
                ]
            },
            None,
        ),
        (
            {
                "steps": [
                    {"source": "agent", "tool_calls": []},
                    {"source": "agent", "tool_calls": [{"function_name": "Read", "arguments": []}]},
                ]
            },
            None,
        ),
        (
            {
                "steps": [
                    {"source": "user", "message": "request"},
                    {"source": "agent", "message": "done", "tool_calls": []},
                ]
            },
            False,
        ),
        ({"steps": [{"source": "agent", "message": "done", "tool_calls": []}]}, False),
    ],
    ids=(
        "empty",
        "user-only",
        "malformed-step",
        "missing-tool-calls",
        "mapping-tool-calls",
        "non-mapping-call",
        "empty-call",
        "blank-function-name",
        "string-arguments",
        "later-malformed-agent-call",
        "valid-user-and-agent",
        "valid-agent-no-tools",
    ),
)
def test_collect_requires_structurally_valid_agent_trajectory_before_certifying_non_invocation(
    tmp_path: Path,
    trajectory: dict[str, Any],
    expected_invoked: bool | None,
) -> None:
    jobs_dir = tmp_path / "jobs"
    trial_name = "trajectory-shape_attempt001"
    _write_trial(
        jobs_dir,
        variant="with",
        trial_name=trial_name,
        reward=dict(STANDARD_REWARD),
        trajectory=trajectory,
    )
    _write_job_result(jobs_dir / "demo-opencode-with", [trial_name])

    _collect(tmp_path)

    persisted = _persisted_reward(tmp_path, "with", trial_name)
    if expected_invoked is None:
        assert "skill_invoked" not in persisted
        assert "invocation_evidence_source" not in persisted
    else:
        assert persisted["skill_invoked"] is expected_invoked
        assert persisted["invocation_evidence_source"] == "trajectory"


@pytest.mark.parametrize(
    ("function_name", "arguments", "expected_invoked"),
    [
        ("open_file", {"file_path": "/workspace/skills/demo/SKILL.md"}, True),
        ("mcp__filesystem__open_file", {"path": "/workspace/skills/demo/SKILL.md"}, True),
        ("Grep", {"path": "/workspace/skills/demo/SKILL.md", "pattern": "name"}, True),
        ("Read", {"filename": "/workspace/skills/demo/SKILL.md"}, True),
        ("Read", {}, None),
    ],
)
def test_collect_native_file_tool_evidence_is_exact_or_unknown(
    tmp_path: Path,
    function_name: str,
    arguments: dict[str, Any],
    expected_invoked: bool | None,
) -> None:
    jobs_dir = tmp_path / "jobs"
    trial_name = "native-file-tool_attempt001"
    trajectory = _trajectory()
    trajectory["steps"][0]["tool_calls"] = [
        {
            "tool_call_id": "call-1",
            "function_name": function_name,
            "arguments": arguments,
        }
    ]
    _write_trial(
        jobs_dir,
        variant="with",
        trial_name=trial_name,
        reward=dict(STANDARD_REWARD),
        trajectory=trajectory,
    )
    _write_job_result(jobs_dir / "demo-opencode-with", [trial_name])

    _collect(tmp_path)

    persisted = _persisted_reward(tmp_path, "with", trial_name)
    if expected_invoked is None:
        assert "skill_invoked" not in persisted
        assert "invocation_evidence_source" not in persisted
    else:
        assert persisted["skill_invoked"] is expected_invoked
        assert persisted["invocation_evidence_source"] == "trajectory"


def test_collect_ignores_non_agent_skill_tool_calls(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    trial_name = "non-agent-skill-call_attempt001"
    trajectory = {
        "schema_version": "ATIF-v1.2",
        "steps": [
            {
                "step_id": 1,
                "source": "user",
                "message": "Use demo",
                "tool_calls": [
                    {
                        "tool_call_id": "call-user",
                        "function_name": "Skill",
                        "arguments": {"skill": "demo"},
                    }
                ],
            },
            {
                "step_id": 2,
                "source": "agent",
                "message": "Done",
                "tool_calls": [],
            },
        ],
    }
    _write_trial(
        jobs_dir,
        variant="with",
        trial_name=trial_name,
        reward=dict(STANDARD_REWARD),
        trajectory=trajectory,
    )
    _write_job_result(jobs_dir / "demo-opencode-with", [trial_name])

    _collect(tmp_path)

    persisted = _persisted_reward(tmp_path, "with", trial_name)
    assert persisted["skill_invoked"] is False
    assert persisted["invocation_evidence_source"] == "trajectory"


def test_collect_leaves_custom_only_reward_and_artifacts_unchanged(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    trial_name = "custom_attempt001"
    custom_reward = {
        "entry_id": "custom",
        "overall": 0.75,
        "custom_metrics": {"skill_execution": 0.25, "quality": 0.8},
        "details": {"skill_execution": {"note": "custom data"}},
        "skill_invoked": "verifier-owned",
        "routing_passed": "verifier-owned",
        "invocation_evidence_source": "custom-verifier",
    }
    _write_trial(
        jobs_dir,
        variant="with",
        trial_name=trial_name,
        reward=custom_reward,
        trajectory=_trajectory(skill="demo"),
    )
    _write_job_result(jobs_dir / "demo-opencode-with", [trial_name])

    result = _collect(tmp_path)

    persisted = _persisted_reward(tmp_path, "with", trial_name)
    for key in ("custom_metrics", "details", "skill_invoked", "routing_passed", "invocation_evidence_source"):
        assert persisted[key] == custom_reward[key]
    agent = result["agents"]["opencode"]
    assert agent["with_skill"] == {}
    assert agent["custom_with_skill"] == {"quality": 0.8}
    summary = json.loads(
        (tmp_path / "results" / "opencode" / "with-skill" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["overall_score"] == 0.75
    trial_dir = tmp_path / "results" / "opencode" / "with-skill" / "trials" / trial_name
    assert {path.name for path in trial_dir.iterdir()} == {"reward.json", "trajectory.json"}


@pytest.mark.parametrize(
    "custom_score_fields",
    [
        {"skill_execution": 0.25, "quality": 0.8},
        {"custom_metrics": {"skill_execution": 0.25, "quality": 0.8}},
    ],
    ids=("top-level", "nested"),
)
def test_collect_respects_explicit_custom_metric_set_with_standard_named_metric(
    tmp_path: Path,
    custom_score_fields: dict[str, Any],
) -> None:
    jobs_dir = tmp_path / "jobs"
    trial_name = "explicit_custom_attempt001"
    custom_reward = {
        "entry_id": "custom",
        "metric_set": "custom-only",
        "overall": 0.75,
        **custom_score_fields,
        "skill_invoked": "custom-verifier-value",
        "routing_passed": "custom-verifier-value",
        "invocation_evidence_source": "custom-verifier",
    }
    _write_trial(
        jobs_dir,
        variant="with",
        trial_name=trial_name,
        reward=custom_reward,
        trajectory=_trajectory(),
    )
    _write_job_result(jobs_dir / "demo-opencode-with", [trial_name])

    _collect(tmp_path)

    persisted = _persisted_reward(tmp_path, "with", trial_name)
    for key, expected in custom_reward.items():
        assert persisted[key] == expected
