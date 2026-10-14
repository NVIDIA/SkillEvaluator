# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failed tool results cannot prove a successful SKILL.md read."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from skillevaluator.tier3.eval_core import atif_helpers, checks
from skillevaluator.tier3.eval_core.log_converters import synthetic_trajectory_from_claude_stream_jsonl


@pytest.fixture(params=("host", "verifier"))
def grader(request: pytest.FixtureRequest, harbor_eval_template: ModuleType) -> tuple[ModuleType, ModuleType]:
    if request.param == "host":
        return atif_helpers, checks
    return harbor_eval_template, harbor_eval_template


def _claude_tool_result(*, name: str, arguments: dict, is_error: bool, content: str) -> dict:
    events = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "read-1",
                        "name": name,
                        "input": arguments,
                    }
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "read-1",
                        "is_error": is_error,
                        "content": content,
                    }
                ]
            },
        },
    ]
    trajectory = synthetic_trajectory_from_claude_stream_jsonl("\n".join(json.dumps(event) for event in events))
    assert trajectory is not None
    return trajectory


def _claude_read(*, is_error: bool, content: str) -> dict:
    return _claude_tool_result(
        name="Read",
        arguments={"file_path": "/skills/demo-skill/SKILL.md"},
        is_error=is_error,
        content=content,
    )


def test_claude_failed_read_cannot_receive_perfect_execution_credit(grader: tuple[ModuleType, ModuleType]) -> None:
    extractor, scorer = grader
    trajectory = _claude_read(is_error=True, content="Error: File does not exist: /skills/demo-skill/SKILL.md")
    calls = extractor.extract_tool_calls_as_dicts(trajectory)

    assert calls[0]["is_error"] is True
    execution = scorer.score_skill_execution(calls, "demo-skill")
    assert execution["details"]["activation"]["score"] == 0.0
    assert execution["details"]["workflow_order"]["score"] == 0.0
    assert execution["score"] < 1.0
    assert scorer.check_routing(calls, "demo-skill")["score"] == 0.0


def test_successful_skill_content_can_discuss_errors(grader: tuple[ModuleType, ModuleType]) -> None:
    extractor, scorer = grader
    calls = extractor.extract_tool_calls_as_dicts(
        _claude_read(is_error=False, content="# Demo skill\nError: File does not exist. Handle this by retrying.")
    )

    assert scorer.score_skill_execution(calls, "demo-skill")["score"] == 1.0
    assert scorer.check_routing(calls, "demo-skill")["score"] == 1.0


@pytest.mark.parametrize("action", ("Read", "Bash"))
def test_failed_read_does_not_satisfy_workflow_before_execution(
    grader: tuple[ModuleType, ModuleType],
    action: str,
) -> None:
    _, scorer = grader
    read = {
        "action": action,
        "action_input": {"file_path": "/skills/demo-skill/SKILL.md"}
        if action == "Read"
        else {"command": "cat /skills/demo-skill/SKILL.md"},
        "observation": "Unable to load /skills/demo-skill/SKILL.md",
        "is_error": True,
    }
    execute = {"action": "Bash", "action_input": {"command": "python scripts/run.py"}, "observation": "ok"}

    assert scorer.check_activation([read], "demo-skill")["score"] == 0.0
    assert scorer.check_routing([read], "demo-skill")["score"] == 0.0
    assert scorer.check_workflow_order([read, execute], expected_skill="demo-skill")["score"] == 0.0
    retry = {**read, "observation": "# Demo skill", "is_error": False}
    assert scorer.check_workflow_order([read, retry, execute], expected_skill="demo-skill")["score"] == 1.0
    assert scorer.check_routing([read, retry], "demo-skill")["score"] == 1.0


@pytest.mark.parametrize(
    "command",
    (
        "cat /skills/demo-skill/SKILL.md && ls /skills/demo-skill/scripts/",
        "cat /skills/demo-skill/SKILL.md /skills/demo-skill/reference.md",
        "cat /skills/demo-skill/SKILL.md -",
    ),
)
def test_failed_compound_shell_read_keeps_credit(grader, command) -> None:
    _, scorer = grader
    read = {"action": "Bash", "action_input": {"command": command}, "observation": "# Demo skill", "is_error": True}
    execute = {"action": "Bash", "action_input": {"command": "python scripts/run.py"}}
    assert scorer.check_activation([read], "demo-skill")["score"] == 0.75
    assert scorer.check_workflow_order([read, execute], expected_skill="demo-skill")["score"] == 1.0
    assert scorer.check_routing([read], "demo-skill")["score"] == 1.0


def test_failed_native_skill_call_is_not_activation(grader) -> None:
    extractor, scorer = grader
    trajectory = {
        "steps": [
            {
                "source": "agent",
                "message": "",
                "tool_calls": [{"tool_call_id": "s1", "function_name": "Skill", "arguments": {"skill": "demo-skill"}}],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "s1",
                            "content": "Skill not found",
                            "extra": {"tool_result_is_error": True},
                        }
                    ]
                },
            }
        ]
    }
    calls = extractor.extract_tool_calls_as_dicts(trajectory)
    assert extractor.get_skill_tool_calls(trajectory) == []
    assert scorer.check_activation(calls, "demo-skill", skill_tool_names=[])["score"] == 0.0
    assert scorer.check_workflow_order(calls, skill_tool_names=[], expected_skill="demo-skill")["score"] == 0.0
    assert scorer.check_routing(calls, "demo-skill", skill_tool_names=[])["score"] == 0.0


def test_opencode_failed_read_is_marked_as_error(grader) -> None:
    from skillevaluator.tier3.eval_core.log_converters import synthetic_trajectory_from_opencode_json

    extractor, scorer = grader
    event = {
        "type": "tool_use",
        "part": {
            "tool": "read",
            "callID": "failed",
            "state": {
                "status": "error",
                "input": {"filePath": "/skills/demo-skill/SKILL.md"},
                "error": {"message": "File not found"},
            },
        },
    }
    trajectory = synthetic_trajectory_from_opencode_json(json.dumps(event))
    assert trajectory is not None
    calls = extractor.extract_tool_calls_as_dicts(trajectory)
    assert calls[0]["is_error"] is True
    assert scorer.check_activation(calls, "demo-skill")["score"] == 0.0
    assert scorer.check_routing(calls, "demo-skill")["score"] == 0.0


def test_native_atif_error_is_matched_to_its_own_call(grader: tuple[ModuleType, ModuleType]) -> None:
    from harbor.models.trajectories.trajectory import Trajectory

    extractor, scorer = grader
    trajectory = Trajectory.model_validate(
        {
            "schema_version": "ATIF-v1.7",
            "agent": {"name": "claude-code", "version": "test"},
            "steps": [
                {
                    "step_id": 1,
                    "source": "agent",
                    "message": "",
                    "tool_calls": [
                        {
                            "tool_call_id": "wrong",
                            "function_name": "Read",
                            "arguments": {"file_path": "/skills/wrong/SKILL.md"},
                        },
                        {
                            "tool_call_id": "right",
                            "function_name": "Read",
                            "arguments": {"file_path": "/skills/demo-skill/SKILL.md"},
                        },
                    ],
                    "observation": {
                        "results": [
                            {
                                "source_call_id": "right",
                                "content": "# Demo skill",
                                "extra": {"tool_result_is_error": False},
                            },
                            {
                                "source_call_id": "wrong",
                                "content": "Missing file",
                                "extra": {"tool_result_is_error": True},
                            },
                        ]
                    },
                }
            ],
        }
    ).to_json_dict()
    calls = extractor.extract_tool_calls_as_dicts(trajectory)

    assert calls[0]["is_error"] is True
    assert not calls[1].get("is_error")
    assert scorer.score_skill_execution(calls, "demo-skill")["score"] == 1.0
    assert scorer.check_routing(calls, "demo-skill")["score"] == 1.0


@pytest.mark.parametrize("action", ("Read", "Skill"))
def test_unrelated_error_does_not_cancel_successful_skill_read(
    grader: tuple[ModuleType, ModuleType], action: str
) -> None:
    from harbor.models.trajectories.trajectory import Trajectory

    extractor, scorer = grader
    trajectory = Trajectory.model_validate(
        {
            "schema_version": "ATIF-v1.7",
            "agent": {"name": "claude-code", "version": "test"},
            "steps": [
                {
                    "step_id": 1,
                    "source": "agent",
                    "message": "",
                    "tool_calls": [
                        {
                            "tool_call_id": "read-1",
                            "function_name": action,
                            "arguments": {"file_path": "/skills/demo-skill/SKILL.md"}
                            if action == "Read"
                            else {"skill": "demo-skill"},
                        }
                    ],
                    "observation": {
                        "results": [
                            {
                                "source_call_id": "read-1",
                                "content": "# Demo skill",
                                "extra": {"tool_result_is_error": False},
                            },
                            {
                                "source_call_id": None,
                                "content": "Unrelated system action failed",
                                "extra": {"tool_result_is_error": True},
                            },
                        ]
                    },
                }
            ],
        }
    ).to_json_dict()
    calls = extractor.extract_tool_calls_as_dicts(trajectory)
    names = extractor.get_skill_tool_calls(trajectory)

    assert not calls[0].get("is_error")
    assert names == (["demo-skill"] if action == "Skill" else [])
    assert scorer.score_skill_execution(calls, "demo-skill", skill_tool_names=names)["score"] == 1.0
    assert scorer.check_routing(calls, "demo-skill", skill_tool_names=names)["score"] == 1.0


@pytest.mark.parametrize("delimiter", ("--", "'--'"))
def test_failed_cat_with_dash_prefixed_second_file_keeps_read_credit(
    grader: tuple[ModuleType, ModuleType], tmp_path: Path, delimiter: str
) -> None:
    extractor, scorer = grader
    skill = tmp_path / "demo-skill" / "SKILL.md"
    skill.parent.mkdir()
    skill.write_text("# Demo skill\nActual instructions\n")
    command = f"cat {delimiter} {shlex.quote(str(skill))} -missing-reference.md"
    result = subprocess.run(["bash", "-c", command], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1
    assert result.stdout == skill.read_text()
    assert "-missing-reference.md" in result.stderr
    trajectory = _claude_tool_result(
        name="Bash",
        arguments={"command": command},
        is_error=True,
        content=result.stdout + result.stderr,
    )
    calls = extractor.extract_tool_calls_as_dicts(trajectory)

    assert calls[0]["is_error"] is True
    assert scorer.check_activation(calls, "demo-skill")["score"] == 0.75
    assert scorer.check_workflow_order(calls, expected_skill="demo-skill")["score"] == 1.0
    assert scorer.check_routing(calls, "demo-skill")["score"] == 1.0


def test_failed_brace_expanded_cat_keeps_read_credit(grader: tuple[ModuleType, ModuleType], tmp_path: Path) -> None:
    extractor, scorer = grader
    skill = tmp_path / "skills" / "demo-skill" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: demo-skill\n---\n# Demo skill\n", encoding="utf-8")
    command = f"cat {shlex.quote(str(tmp_path / 'skills'))}/{{demo-skill,missing}}/SKILL.md"
    result = subprocess.run(["bash", "-c", command], cwd=tmp_path, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 1
    assert result.stdout == skill.read_text(encoding="utf-8")
    assert "missing/SKILL.md" in result.stderr
    trajectory = _claude_tool_result(
        name="Bash",
        arguments={"command": command},
        is_error=True,
        content=result.stdout + result.stderr,
    )
    calls = extractor.extract_tool_calls_as_dicts(trajectory)

    assert calls[0]["is_error"] is True
    assert scorer.check_activation(calls, "demo-skill")["score"] == 0.75
    assert scorer.check_workflow_order(calls, expected_skill="demo-skill")["score"] == 1.0
    assert scorer.check_routing(calls, "demo-skill")["score"] == 1.0


@pytest.mark.parametrize(
    "command",
    (
        "cat /skills/*/SKILL.md",
        "cat /skills/[a-z]*/SKILL.md",
        "cat /skills/demo-skill?/SKILL.md",
        "cat $SKILLS_DIR/SKILL.md",
        "cat `pwd`/SKILL.md",
    ),
)
def test_failed_cat_with_expanding_operand_keeps_read_credit(grader, command: str) -> None:
    _, scorer = grader
    read = {
        "action": "Bash",
        "action_input": {"command": command},
        "observation": "---\nname: demo-skill\n---\n# Demo skill\ncat: /skills/other/SKILL.md: Is a directory",
        "is_error": True,
    }
    execute = {"action": "Bash", "action_input": {"command": "python scripts/run.py"}}
    assert scorer.check_activation([read], "demo-skill")["score"] == 0.75
    assert scorer.check_workflow_order([read, execute], expected_skill="demo-skill")["score"] == 1.0
    assert scorer.check_routing([read], "demo-skill")["score"] == 1.0


def test_failed_cat_of_home_relative_skill_loses_credit(grader) -> None:
    # Tilde expansion always yields one path, so this is still an isolated read.
    _, scorer = grader
    read = {
        "action": "Bash",
        "action_input": {"command": "cat ~/.claude/skills/demo-skill/SKILL.md"},
        "observation": "cat: /root/.claude/skills/demo-skill/SKILL.md: No such file or directory",
        "is_error": True,
    }
    assert scorer.check_activation([read], "demo-skill")["score"] == 0.0
    assert scorer.check_routing([read], "demo-skill")["score"] == 0.0


@pytest.mark.parametrize("options", ("--", "'--'", "-n", "-n --"))
@pytest.mark.parametrize("is_error", (False, True))
def test_single_cat_read_with_options(grader: tuple[ModuleType, ModuleType], options: str, is_error: bool) -> None:
    extractor, scorer = grader
    trajectory = _claude_tool_result(
        name="Bash",
        arguments={"command": f"cat {options} /skills/demo-skill/SKILL.md"},
        is_error=is_error,
        content="File not found" if is_error else "# Demo skill",
    )
    calls = extractor.extract_tool_calls_as_dicts(trajectory)

    assert scorer.check_activation(calls, "demo-skill")["score"] == (0.0 if is_error else 0.75)
    assert scorer.check_workflow_order(calls, expected_skill="demo-skill")["score"] == (0.0 if is_error else 1.0)
    assert scorer.check_routing(calls, "demo-skill")["score"] == (0.0 if is_error else 1.0)
