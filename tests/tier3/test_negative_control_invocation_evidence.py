# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted invocation evidence for Tier 3 negative-control cases."""

from __future__ import annotations

import importlib.util
import json
import shlex
from pathlib import Path

import pytest

from skillevaluator.tier3.eval_core import checks as shared_checks


def _load_template():
    template_path = Path(__file__).parents[2] / "src/skillevaluator/tier3/harbor/templates/eval.py"
    spec = importlib.util.spec_from_file_location("harbor_eval_negative_control", template_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TEMPLATE = _load_template()


def _nested_shell_command(command: str, depth: int) -> str:
    for _ in range(depth):
        command = shlex.join(["bash", "-c", command])
    return command


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ({"expected_skill": "target", "should_trigger": False}, False),
        ({"expected_skill": None, "should_trigger": True}, True),
        ({"expected_skill": "target"}, True),
        ({"expected_skill": None}, False),
        ({}, None),
    ],
)
def test_routing_tri_state_preserves_explicit_precedence_and_legacy_unlabeled(entry, expected) -> None:
    assert shared_checks.resolve_should_trigger(entry) is expected
    assert TEMPLATE.resolve_should_trigger(entry) is expected


TARGET_READ = [
    {
        "action": "Read",
        "action_input": {"file_path": "/workspace/skills/trusted-target/SKILL.md"},
        "observation": "# Trusted target",
    }
]
OTHER_READ = [
    {
        "action": "Read",
        "action_input": {"file_path": "/workspace/skills/other/SKILL.md"},
        "observation": "# Other",
    }
]


@pytest.mark.parametrize(
    ("tool_calls", "skill_tool_names", "expected_passed"),
    [
        ([], ["demo"], False),
        ([], ["DEMO"], False),
        ([], ["demo-helper"], True),
        (
            [
                {
                    "action": "Read",
                    "action_input": {"file_path": "/workspace/skills/demo/SKILL.md"},
                    "observation": "",
                }
            ],
            None,
            False,
        ),
        (
            [
                {
                    "action": "Read",
                    "action_input": {"file_path": "/workspace/skills/demo-helper/SKILL.md"},
                    "observation": "",
                }
            ],
            None,
            True,
        ),
        (
            [
                {
                    "action": "Read",
                    "action_input": {"file_path": "/workspace/skills/demo/subdir/SKILL.md"},
                    "observation": "",
                }
            ],
            None,
            True,
        ),
        (
            [
                {
                    "action": "Read",
                    "action_input": {"file_path": "/workspace/skills/demo/SKILL.md/notes"},
                    "observation": "",
                }
            ],
            None,
            True,
        ),
        (
            [
                {
                    "action": "Read",
                    "action_input": {"file_path": "/workspace/skills/other/../demo/./SKILL.md"},
                    "observation": "",
                }
            ],
            None,
            False,
        ),
        (
            [
                {
                    "action": "exec_command",
                    "action_input": {"cmd": "cat /workspace/skills/demo/SKILL.md"},
                    "observation": "",
                }
            ],
            None,
            False,
        ),
        (
            [
                {
                    "action": "exec_command",
                    "action_input": {"cmd": "cat /workspace/skills/demo/./SKILL.md"},
                    "observation": "",
                }
            ],
            None,
            False,
        ),
        (
            [
                {
                    "action": "exec_command",
                    "action_input": {"cmd": "cat /workspace/skills/demo//SKILL.md"},
                    "observation": "",
                }
            ],
            None,
            False,
        ),
        (
            [
                {
                    "action": "exec_command",
                    "action_input": {"cmd": "cat /workspace/skills/'demo'/SKILL.md"},
                    "observation": "",
                }
            ],
            None,
            False,
        ),
        (
            [
                {
                    "action": "exec_command",
                    "action_input": {"cmd": r"cat 'C:\skills\demo\SKILL.md'"},
                    "observation": "",
                }
            ],
            None,
            False,
        ),
        (
            [
                {
                    "action": "exec_command",
                    "action_input": {"cmd": "python /workspace/skills/demo/scripts/run.py"},
                    "observation": "",
                }
            ],
            None,
            False,
        ),
        (
            [
                {
                    "action": "exec_command",
                    "action_input": {"cmd": "echo /workspace/skills/demo/SKILL.md"},
                    "observation": "",
                }
            ],
            None,
            True,
        ),
        (
            [
                {
                    "action": "exec_command",
                    "action_input": {"cmd": "printf /workspace/skills/demo/SKILL.md"},
                    "observation": "",
                }
            ],
            None,
            True,
        ),
        (
            [
                {
                    "action": "exec_command",
                    "action_input": {"cmd": "echo /workspace/skills/demo/scripts/run.py"},
                    "observation": "",
                }
            ],
            None,
            True,
        ),
        (
            [
                {
                    "action": "exec_command",
                    "action_input": {"cmd": "cat /workspace/skills/demo-helper/SKILL.md"},
                    "observation": "",
                }
            ],
            None,
            True,
        ),
        (
            [{"action": "exec_command", "action_input": {"cmd": "printf demo"}, "observation": ""}],
            None,
            True,
        ),
    ],
)
def test_negative_case_requires_exact_target_reference(tool_calls, skill_tool_names, expected_passed) -> None:
    shared = shared_checks.check_negative_case(tool_calls, "demo", skill_tool_names=skill_tool_names)
    template = TEMPLATE.check_negative_case(tool_calls, "demo", skill_tool_names=skill_tool_names)

    assert shared == template
    assert shared["passed"] is expected_passed


@pytest.mark.parametrize(
    ("action", "arguments", "expected_passed"),
    [
        ("open_file", {"file_path": "/workspace/skills/demo/SKILL.md"}, False),
        ("mcp__filesystem__open_file", {"path": "/workspace/skills/demo/SKILL.md"}, False),
        ("Grep", {"path": "/workspace/skills/demo/SKILL.md", "pattern": "name"}, False),
        ("Read", {"filename": "/workspace/skills/demo/SKILL.md"}, False),
        ("Read", {}, None),
        ("Read", {"file_path": "/workspace/skills/other/SKILL.md"}, True),
    ],
)
def test_negative_case_native_file_tools_are_exact_or_unknown(action, arguments, expected_passed) -> None:
    tool_calls = [{"action": action, "action_input": arguments, "observation": ""}]

    shared = shared_checks.check_negative_case(tool_calls, "demo")
    template = TEMPLATE.check_negative_case(tool_calls, "demo")

    assert shared == template
    assert shared["passed"] is expected_passed


@pytest.mark.parametrize(
    ("command", "expected_passed"),
    [
        ("printf ignored\ncat /workspace/skills/demo/SKILL.md", False),
        ("echo ignored\npython /workspace/skills/demo/scripts/run.py", False),
        ("/usr/bin/env python /workspace/skills/demo/scripts/run.py", False),
        ("bash -c 'python /workspace/skills/demo/scripts/run.py'", False),
        ("dash -c 'cat /workspace/skills/demo/SKILL.md'", False),
        ("sh -c 'cat /workspace/skills/demo/SKILL.md'", False),
        ("zsh -c 'command cat /workspace/skills/demo/SKILL.md'", False),
        ("command cat /workspace/skills/demo/SKILL.md", False),
        ("command env MODE=test cat /workspace/skills/demo/SKILL.md", False),
        ("env MODE=test cat /workspace/skills/demo/SKILL.md", False),
        ("env -i python /workspace/skills/demo/scripts/run.py", False),
        ("source /workspace/skills/demo/scripts/run.sh", False),
        (". /workspace/skills/demo/scripts/run.sh", False),
        ("exec /workspace/skills/demo/scripts/run.sh", False),
        ("exec python /workspace/skills/demo/scripts/run.py", False),
        ("CONTENT=$(cat /workspace/skills/demo/SKILL.md)", False),
        ("timeout 10 cat /workspace/skills/demo/SKILL.md", False),
        ("nice python /workspace/skills/demo/scripts/run.py", False),
        ("echo prep & cat /workspace/skills/demo/SKILL.md", False),
        ('echo "$(cat /workspace/skills/demo/SKILL.md)"', False),
        ("echo $(cat /workspace/skills/demo/SKILL.md)", False),
        ('printf %s "$(cat /workspace/skills/demo/SKILL.md)"', False),
        ("echo <(cat /workspace/skills/demo/SKILL.md)", False),
        ("echo `cat /workspace/skills/demo/SKILL.md`", False),
        ("P=/workspace/skills/demo; cat $P/SKILL.md", False),
        ('P=/workspace/skills/demo; cat "${P}/SKILL.md"', False),
        ("cd /workspace/skills/demo && cat SKILL.md", False),
        ("sh -c -- 'cat /workspace/skills/demo/SKILL.md'", False),
        ("bash -c -- 'python /workspace/skills/demo/scripts/run.py'", False),
        ("bash -c 'echo /workspace/skills/demo/SKILL.md'", True),
        ("dash -c 'echo /workspace/skills/demo/SKILL.md'", True),
        ("bash -c 'echo /workspace/skills/demo/scripts/run.py'", True),
        ("env -- echo /workspace/skills/demo/scripts/run.py", True),
        ("env echo /workspace/skills/demo/scripts/run.py", True),
        ("echo 'note\ncat /workspace/skills/demo/SKILL.md'", True),
        ("echo /workspace/skills/demo/SKILL.md\nprintf /workspace/skills/demo/scripts/run.py", True),
        ("CONTENT=$(printf /workspace/skills/demo/SKILL.md)", True),
        ("timeout 10 echo /workspace/skills/demo/SKILL.md", True),
        ("nice echo /workspace/skills/demo/scripts/run.py", True),
        ("echo '$(cat /workspace/skills/demo/SKILL.md)'", True),
        ("echo '`cat /workspace/skills/demo/SKILL.md`'", True),
        ("P=/workspace/skills/demo; echo $P/SKILL.md", True),
        ("cd /workspace/skills/demo && echo SKILL.md", True),
    ],
)
def test_negative_case_shell_parser_handles_wrappers_and_multiline_commands(command, expected_passed) -> None:
    tool_calls = [{"action": "exec_command", "action_input": {"cmd": command}, "observation": ""}]

    shared = shared_checks.check_negative_case(tool_calls, "demo")
    template = TEMPLATE.check_negative_case(tool_calls, "demo")

    assert shared == template
    assert shared["passed"] is expected_passed


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
def test_negative_case_shell_parser_caps_are_unknown(command: str) -> None:
    tool_calls = [{"action": "exec_command", "action_input": {"cmd": command}, "observation": ""}]

    shared = shared_checks.check_negative_case(tool_calls, "demo")
    template = TEMPLATE.check_negative_case(tool_calls, "demo")

    assert shared == template
    assert shared["passed"] is None
    assert shared["score"] == 0.0


@pytest.mark.parametrize(
    ("tool_calls", "expected_score"),
    [(TARGET_READ, 0.0), (OTHER_READ, 1.0), ([], 1.0)],
)
def test_explicit_negative_scores_only_trusted_target_invocation(tool_calls, expected_score) -> None:
    kwargs = {
        "tool_calls": tool_calls,
        "expected_skill": "authored-spoof",
        "should_trigger": False,
        "evaluated_skill": "trusted-target",
    }

    shared = shared_checks.score_skill_execution(**kwargs)
    template = TEMPLATE.score_skill_execution(**kwargs)

    assert shared == template
    assert shared["score"] == expected_score


def test_explicit_negative_without_trusted_identity_fails_closed() -> None:
    kwargs = {
        "tool_calls": OTHER_READ,
        "expected_skill": "",
        "should_trigger": False,
        "evaluated_skill": "",
        "require_evaluated_skill": True,
    }

    shared = shared_checks.score_skill_execution(**kwargs)
    template = TEMPLATE.score_skill_execution(**kwargs)

    assert shared == template
    assert shared["score"] == 0.0
    assert "trusted evaluated_skill" in shared["details"]["message"]


def test_legacy_direct_negative_uses_expected_skill_compatibly() -> None:
    kwargs = {
        "tool_calls": [],
        "expected_skill": "demo",
        "should_trigger": False,
    }

    shared = shared_checks.score_skill_execution(**kwargs)
    template = TEMPLATE.score_skill_execution(**kwargs)

    assert shared == template
    assert shared["score"] == 1.0
    assert shared["details"]["negative_check"]["score"] == 1.0


def test_legacy_unlabeled_case_retains_skip_pass_behavior() -> None:
    kwargs = {
        "tool_calls": TARGET_READ,
        "expected_skill": "",
        "should_trigger": None,
        "evaluated_skill": "",
    }

    shared = shared_checks.score_skill_execution(**kwargs)
    template = TEMPLATE.score_skill_execution(**kwargs)

    assert shared == template
    assert shared["score"] == 1.0
    assert "skipped" in shared["details"]["message"].lower()


def test_positive_scoring_ignores_evaluated_skill_and_remains_unchanged() -> None:
    kwargs = {
        "tool_calls": TARGET_READ,
        "expected_skill": "trusted-target",
        "should_trigger": True,
    }

    baseline = shared_checks.score_skill_execution(**kwargs)
    with_unrelated_identity = shared_checks.score_skill_execution(**kwargs, evaluated_skill="different-target")
    template = TEMPLATE.score_skill_execution(**kwargs, evaluated_skill="different-target")

    assert with_unrelated_identity == baseline
    assert template == baseline
    assert baseline["score"] == 1.0


@pytest.mark.parametrize("include_trusted_identity", (True, False), ids=("trusted", "missing-fail-closed"))
def test_template_main_scores_standard_negative_fail_closed(tmp_path, monkeypatch, include_trusted_identity) -> None:
    entry_path = tmp_path / "entry.json"
    trajectory_path = tmp_path / "trajectory.json"
    reward_json = tmp_path / "reward.json"
    rich_reward_json = tmp_path / "skill_evaluator_reward.json"
    reward_txt = tmp_path / "reward.txt"
    entry = {
        "id": "negative-001",
        "question": "Answer without using the target skill",
        "expected_skill": None,
    }
    if include_trusted_identity:
        entry["evaluated_skill"] = "trusted-target"
    entry_path.write_text(json.dumps(entry), encoding="utf-8")
    trajectory_path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "source": "agent",
                        "message": "Done",
                        "tool_calls": [
                            {
                                "tool_call_id": "read-1",
                                "function_name": "Read",
                                "arguments": {"file_path": "/workspace/skills/trusted-target/SKILL.md"},
                            }
                        ],
                        "observation": {"results": [{"source_call_id": "read-1", "content": "# Trusted target"}]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(TEMPLATE, "ENTRY_PATH", entry_path)
    monkeypatch.setattr(TEMPLATE, "ATIF_PATH", trajectory_path)
    monkeypatch.setattr(TEMPLATE, "REWARD_JSON", reward_json)
    monkeypatch.setattr(TEMPLATE, "SKILL_EVALUATOR_REWARD_JSON", rich_reward_json)
    monkeypatch.setattr(TEMPLATE, "REWARD_TXT", reward_txt)

    def passing_judge(*_args, **_kwargs):
        return {"score": 1.0}

    monkeypatch.setattr(TEMPLATE, "judge_accuracy", passing_judge)
    monkeypatch.setattr(TEMPLATE, "judge_goal_accuracy", passing_judge)
    monkeypatch.setattr(TEMPLATE, "judge_behavior_check", passing_judge)

    TEMPLATE.main()

    result = json.loads(rich_reward_json.read_text(encoding="utf-8"))
    harbor_reward = json.loads(reward_json.read_text(encoding="utf-8"))
    expected_reward_keys = {*TEMPLATE.DISPLAY_METRICS, "overall"}
    expected_overall = round(
        sum(float(result.get(metric, 0.0) or 0.0) for metric in TEMPLATE.DISPLAY_METRICS)
        / len(TEMPLATE.DISPLAY_METRICS),
        4,
    )

    assert result["skill_execution"] == 0.0
    if include_trusted_identity:
        assert result["details"]["skill_execution"]["negative_check"]["score"] == 0.0
    else:
        assert "trusted evaluated_skill" in result["details"]["skill_execution"]["message"]
    assert set(harbor_reward) == expected_reward_keys
    assert all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in harbor_reward.values())
    assert harbor_reward["skill_execution"] == 0.0
    assert harbor_reward["overall"] == expected_overall
    assert float(reward_txt.read_text(encoding="utf-8")) == expected_overall
