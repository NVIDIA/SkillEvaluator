# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM and tool time from Harbor ATIF trajectories."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.tier3.harbor.collector import _merged_step_trajectory, _redacted_trajectory_data
from skillevaluator.tier3.harbor.trajectory_timing import trajectory_timing

FIXTURE = Path(__file__).parent / "fixtures" / "harbor_0_24_codex_trajectory.json"


def _codex_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _inferred(llm: float, tool: float) -> dict[str, Any]:
    return {"llm_seconds": llm, "tool_seconds": tool, "source": "step_timestamps"}


def _tool_step(second: int, durations: list[float]) -> dict[str, Any]:
    return {
        "source": "agent",
        "timestamp": f"2026-01-01T00:00:{second:02d}+00:00",
        "tool_calls": [{"tool_call_id": f"call-{i}", "function_name": "exec_command"} for i in range(len(durations))],
        "observation": {
            "results": [
                {"source_call_id": f"call-{i}", "content": f"Wall time: {duration} seconds\nOutput:\nok"}
                for i, duration in enumerate(durations)
            ]
        },
    }


def _user(second: int = 0) -> dict[str, Any]:
    return {"source": "user", "timestamp": f"2026-01-01T00:00:{second:02d}+00:00"}


def _agent(second: int) -> dict[str, Any]:
    return {"source": "agent", "timestamp": f"2026-01-01T00:00:{second:02d}+00:00"}


def test_codex_like_trajectory_infers_llm_from_step_timestamps() -> None:
    trajectory = {
        "steps": [
            {"source": "user", "timestamp": "2026-10-21T22:09:21.777Z"},
            {
                "source": "agent",
                "timestamp": "2026-10-21T22:09:25.199Z",
                "tool_calls": [{"function_name": "exec_command", "arguments": {}}],
                "observation": {
                    "results": [{"content": "Wall time: 1.5000 seconds\nProcess exited with code 0\nOutput:\nok\n"}]
                },
            },
            {"source": "agent", "timestamp": "2026-10-21T22:09:30.143Z"},
        ]
    }
    # 3.422 s before the tool step, then 4.944 s minus 1.5 s of tool time.
    assert trajectory_timing(trajectory) == _inferred(6.866, 1.5)


@pytest.mark.parametrize("wall_time", ["9" * 400, "86400.5"])
def test_unbounded_wall_time_is_rejected(wall_time: str) -> None:
    trajectory = {
        "steps": [
            {"source": "user", "timestamp": "2026-10-21T22:09:21.777Z"},
            {
                "source": "agent",
                "timestamp": "2026-10-21T22:09:25.199Z",
                "tool_calls": [{"function_name": "exec_command"}],
                "observation": {"results": [{"content": f"Wall time: {wall_time} seconds\nOutput:\n"}]},
            },
            {"source": "agent", "timestamp": "2026-10-23T22:09:25.199Z"},
        ]
    }
    assert trajectory_timing(trajectory) is None


def test_claude_like_trajectory_without_wall_time_is_not_inferred() -> None:
    trajectory = {
        "steps": [
            {"source": "user", "timestamp": "2026-10-20T17:33:50.301Z"},
            {
                "source": "agent",
                "timestamp": "2026-10-20T17:34:05.177Z",
                "tool_calls": [{"function_name": "Bash", "tool_call_id": "call-1", "arguments": {}}],
                "observation": {"results": [{"source_call_id": "call-1", "content": "done\n"}]},
            },
            {"source": "agent", "timestamp": "2026-10-20T17:34:09.000Z"},
        ]
    }
    assert trajectory_timing(trajectory) is None


def test_wall_time_in_output_is_not_counted() -> None:
    trajectory = {
        "steps": [
            _user(),
            {
                "source": "agent",
                "timestamp": "2026-01-01T00:00:03+00:00",
                "tool_calls": [{"function_name": "exec_command", "arguments": {}}],
                "observation": {
                    "results": [
                        {
                            "content": (
                                "Wall time: 0.5000 seconds\nOutput:\n"
                                + "\n".join(f"iteration {i}: Wall time: 30 seconds" for i in range(5))
                                + "\nWall time: 9 seconds\n"
                            )
                        }
                    ]
                },
            },
            _agent(10),
        ]
    }
    assert trajectory_timing(trajectory) == _inferred(9.5, 0.5)


def test_partial_native_agent_llm_does_not_report_subtotal() -> None:
    trajectory = {
        "steps": [
            {
                "source": "agent",
                "extra": {"llm_duration_ms": 1200},
                "tool_calls": [{"extra": {"duration_ms": 500}}],
            },
            {"source": "agent"},
        ]
    }
    assert trajectory_timing(trajectory) == {"llm_seconds": None, "tool_seconds": 0.5, "source": "atif"}


def test_partial_native_tool_timing_does_not_report_subtotal() -> None:
    trajectory = {
        "steps": [
            {
                "source": "agent",
                "extra": {"llm_duration_ms": 1200},
                "tool_calls": [{"extra": {"duration_ms": 500}}, {"function_name": "Bash"}],
            }
        ]
    }
    assert trajectory_timing(trajectory) == {"llm_seconds": 1.2, "tool_seconds": None, "source": "atif"}


def test_native_atif_extras_win_over_inference() -> None:
    trajectory = {
        "steps": [
            _user(),
            {"source": "agent", "timestamp": "2026-01-01T00:00:10+00:00", "extra": {"llm_duration_ms": 500}},
        ]
    }
    # Inference alone would report 10 s of LLM time.
    assert trajectory_timing(trajectory) == {"llm_seconds": 0.5, "tool_seconds": 0.0, "source": "atif"}


def test_native_parallel_tool_intervals_are_counted_once() -> None:
    trajectory = {
        "steps": [
            {
                "source": "agent",
                "extra": {"llm_duration_ms": 250},
                "tool_calls": [
                    {"extra": {"started_at": "2026-01-01T00:00:00Z", "ended_at": "2026-01-01T00:00:02Z"}},
                    {"extra": {"started_at": "2026-01-01T00:00:01Z", "ended_at": "2026-01-01T00:00:03Z"}},
                    {"extra": {"duration_ms": 400}},
                ],
            }
        ]
    }
    assert trajectory_timing(trajectory) == {"llm_seconds": 0.25, "tool_seconds": 3.4, "source": "atif"}


@pytest.mark.parametrize(
    "extra",
    [
        {"started_at": "2026-01-01T00:00:00", "ended_at": "2026-01-01T00:00:02"},
        {"started_at": "2026-01-01T00:00:02Z", "ended_at": "2026-01-01T00:00:01Z"},
        {"duration_ms": float("nan")},
        {"duration_ms": True},
        {"duration_ms": -1},
    ],
)
def test_invalid_native_tool_timing_is_not_a_measurement(extra: dict[str, Any]) -> None:
    trajectory = {"steps": [{"source": "agent", "tool_calls": [{"extra": extra}]}]}
    assert trajectory_timing(trajectory) is None


def test_consecutive_tool_steps_count_each_gap_once() -> None:
    trajectory = {"steps": [_user(), _tool_step(3, [1.0]), _tool_step(7, [2.0]), _agent(12)]}
    assert trajectory_timing(trajectory) == _inferred(9.0, 3.0)


def test_parallel_timestamp_tools_use_longest_wall_time() -> None:
    trajectory = {"steps": [_user(), _tool_step(3, [1.0, 2.0]), _agent(10)]}
    assert trajectory_timing(trajectory) == _inferred(8.0, 2.0)


@pytest.mark.parametrize(
    "final_timestamp",
    [None, "2026-01-01T00:00:02+00:00", "2026-01-01T00:00:04+00:00", "2026-01-01T00:00:12", "not a time"],
)
def test_inconsistent_or_missing_timestamps_do_not_fabricate_latency(final_timestamp: str | None) -> None:
    # Missing, earlier, too short for the 2 s tool call, naive, or unparsable.
    trajectory = {
        "steps": [_user(), _tool_step(3, [2.0]), {"source": "agent", "timestamp": final_timestamp}],
    }
    assert trajectory_timing(trajectory) is None


def test_trailing_tool_step_without_a_later_timestamp_is_rejected() -> None:
    assert trajectory_timing({"steps": [_user(), _tool_step(3, [1.0])]}) is None


def test_measured_zero_tool_duration_is_preserved() -> None:
    trajectory = {"steps": [_user(), _tool_step(3, [0.0]), _agent(10)]}
    assert trajectory_timing(trajectory) == _inferred(10.0, 0.0)


def test_trajectory_without_tool_calls_measures_zero_tool_time() -> None:
    assert trajectory_timing({"steps": [_user(), _agent(4), _agent(9)]}) == _inferred(9.0, 0.0)


def test_non_codex_tool_output_cannot_supply_timing_metadata() -> None:
    tool_step = _tool_step(3, [1.0])
    tool_step["tool_calls"][0]["function_name"] = "Bash"
    trajectory = {"steps": [_user(), tool_step, _agent(10)]}
    assert trajectory_timing(trajectory) is None


@pytest.mark.parametrize("name", ["web_search_call", "update_plan", "search_docs", "exec", "view_image"])
def test_unmeasured_codex_tools_leave_the_trial_unmeasured(name: str) -> None:
    # Even with a Wall time header, only Codex's own shell and apply_patch output is trusted.
    unmeasured = _tool_step(5, [0.2])
    unmeasured["tool_calls"][0]["function_name"] = name
    trajectory = {"steps": [_user(), _tool_step(3, [1.0]), unmeasured, _agent(10)]}
    assert trajectory_timing(trajectory) is None


def test_apply_patch_header_counts_as_tool_time() -> None:
    patch = _tool_step(5, [0.0])
    patch["tool_calls"][0]["function_name"] = "apply_patch"
    patch["observation"]["results"][0]["content"] = (
        "Exit code: 0\nWall time: 0.2 seconds\nOutput:\nSuccess. Updated the following files:\nM a.py\n"
    )
    trajectory = {"steps": [_user(), _tool_step(3, [1.0]), patch, _agent(10)]}
    assert trajectory_timing(trajectory) == _inferred(8.8, 1.2)
    # Older Codex releases return the bare result, which has no duration.
    patch["observation"]["results"][0]["content"] = "Success. Updated the following files:\nM a.py\n"
    assert trajectory_timing(trajectory) is None


def test_namespaced_codex_shell_tools_are_measured() -> None:
    step = _tool_step(3, [1.0, 0.5])
    step["tool_calls"][0]["function_name"] = "functions.shell_command"
    step["tool_calls"][1]["function_name"] = "wait_command"
    assert trajectory_timing({"steps": [_user(), step, _agent(10)]}) == _inferred(9.0, 1.0)


def test_duplicate_call_ids_are_rejected() -> None:
    step = _tool_step(3, [1.0, 2.0])
    step["tool_calls"][1]["tool_call_id"] = "call-0"
    assert trajectory_timing({"steps": [_user(), step, _agent(10)]}) is None
    step = _tool_step(3, [1.0, 2.0])
    step["observation"]["results"][1]["source_call_id"] = "call-0"
    assert trajectory_timing({"steps": [_user(), step, _agent(10)]}) is None


def test_header_with_two_wall_time_lines_is_rejected() -> None:
    step = _tool_step(3, [1.0])
    step["observation"]["results"][0]["content"] = "Wall time: 1 seconds\nWall time: 2 seconds\nOutput:\nok"
    assert trajectory_timing({"steps": [_user(), step, _agent(10)]}) is None


def _write_stdin_trajectory(content: str) -> dict[str, Any]:
    write_stdin_step = _tool_step(7, [0.0])
    write_stdin_step["tool_calls"][0]["function_name"] = "write_stdin"
    write_stdin_step["observation"]["results"][0]["content"] = content
    return {"steps": [_user(), _tool_step(3, [1.0]), write_stdin_step, _agent(15)]}


def test_codex_write_stdin_with_exec_header_counts_as_tool_time() -> None:
    trajectory = _write_stdin_trajectory(
        "Chunk ID: 638ae2\nWall time: 5.0000 seconds\nProcess exited with code 0\nOriginal token count: 4\nOutput:\nok\n"
    )
    assert trajectory_timing(trajectory) == _inferred(9.0, 6.0)


@pytest.mark.parametrize(
    "content",
    [
        "write_stdin failed: stdin is closed for this session",
        "Chunk ID: 638ae2\nProcess exited with code 0\nOutput:\nWall time: 5.0000 seconds\n",
        "Chunk ID: 638ae2\nWall time: 5.0000 seconds\nProcess exited with code 0\n",
    ],
)
def test_codex_write_stdin_without_exec_header_is_rejected(content: str) -> None:
    assert trajectory_timing(_write_stdin_trajectory(content)) is None


def test_single_call_tool_metadata_supplies_duration() -> None:
    step = _tool_step(3, [0.0])
    step["tool_calls"][0]["function_name"] = "shell"
    step["observation"]["results"][0]["content"] = "README.md\n"
    step["extra"] = {"tool_metadata": {"exit_code": 0, "duration_seconds": 1.25}}
    assert trajectory_timing({"steps": [_user(), step, _agent(10)]}) == _inferred(8.75, 1.25)
    step["extra"]["tool_metadata"]["duration_seconds"] = float("inf")
    assert trajectory_timing({"steps": [_user(), step, _agent(10)]}) is None


def test_harbor_codex_trajectory_fixture_is_timed_after_redaction() -> None:
    trajectory = _codex_fixture()
    redacted = _redacted_trajectory_data(trajectory)
    assert redacted is not None
    # 3.06 s to the first model output, then each tool step's gap minus its tool time:
    # exec_command 1.4 s, parallel exec_command calls max(0.7, 1.1) s, shell metadata 0.8 s,
    # apply_patch 0.1 s.
    assert trajectory_timing(trajectory) == _inferred(10.56, 3.4)
    assert trajectory_timing(redacted) == trajectory_timing(trajectory)


def test_harbor_codex_fixture_with_an_unmeasured_call_stays_unmeasured() -> None:
    trajectory = _codex_fixture()
    del trajectory["steps"][5]["extra"]["tool_call_details"]["call_4"]["metadata"]
    assert trajectory_timing(trajectory) is None


def _write_multistep_trial(trial_root: Path, trajectories: list[tuple[str, dict[str, Any]]]) -> None:
    (trial_root / "result.json").write_text(
        json.dumps({"config": {"agent": {}}, "step_results": [{"step_name": name} for name, _ in trajectories]}),
        encoding="utf-8",
    )
    for name, trajectory in trajectories:
        agent_dir = trial_root / "steps" / name / "agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")


def test_merged_multistep_trajectory_excludes_the_gap_between_harbor_steps(tmp_path: Path) -> None:
    first = _codex_fixture()
    second = copy.deepcopy(first)
    second["session_id"] = "00000000-0000-4000-8000-000000000002"
    # The second Harbor step starts ten minutes later, after verifier and setup time.
    for step in second["steps"]:
        step["timestamp"] = step["timestamp"].replace("T12:00:", "T12:10:")
    _write_multistep_trial(tmp_path, [("prepare", first), ("finish", second)])
    merged = _merged_step_trajectory(tmp_path)
    assert merged is not None
    assert {step["extra"]["harbor_step_name"] for step in merged["steps"]} == {"prepare", "finish"}
    assert trajectory_timing(merged) == _inferred(21.12, 6.8)


def _named(name: str, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**step, "extra": {"harbor_step_name": name}} for step in steps]


def test_continuation_step_does_not_count_the_gap_before_it() -> None:
    first = [_user(), _tool_step(3, [1.0]), _agent(10)]
    # A resumed Harbor step can start with an agent step 30 s after the previous one ended.
    second = [_agent(40), _agent(45)]
    merged = [*_named("prepare", first), *_named("finish", second)]
    assert trajectory_timing({"steps": merged}) == _inferred(14.0, 1.0)
    # Read as one Harbor step, the 30 s between the steps would count as LLM time.
    assert trajectory_timing({"steps": [*first, *second]}) == _inferred(44.0, 1.0)


def test_merged_multistep_trajectory_needs_every_harbor_step_timed() -> None:
    first = [_user(), _tool_step(3, [1.0]), _agent(10)]
    second = [_user(40), _tool_step(43, [1.0])]
    assert trajectory_timing({"steps": [*_named("prepare", first), *_named("finish", second)]}) is None


def _copied(step: dict[str, Any]) -> dict[str, Any]:
    return {**step, "is_copied_context": True}


def _exec_step(timestamp: str, call_id: str) -> dict[str, Any]:
    return {
        "source": "agent",
        "timestamp": timestamp,
        "tool_calls": [{"tool_call_id": call_id, "function_name": "exec_command"}],
        "observation": {"results": [{"source_call_id": call_id, "content": "Wall time: 1.0 seconds\nOutput:\nok"}]},
    }


def test_copied_context_steps_are_not_timed_as_new_work() -> None:
    trajectory = {
        "steps": [
            _copied(_exec_step("2026-01-01T11:00:02Z", "call-old")),
            {"source": "user", "timestamp": "2026-01-01T12:00:00Z"},
            _exec_step("2026-01-01T12:00:02Z", "call-new"),
            {"source": "agent", "timestamp": "2026-01-01T12:00:05Z"},
        ]
    }
    # Timing the copied step would count its 1 s again and the hour after it as LLM time.
    assert trajectory_timing(trajectory) == _inferred(4.0, 1.0)


def test_copied_context_native_timing_is_not_counted_twice() -> None:
    copied_extra = {"llm_duration_ms": 1000}
    trajectory = {
        "steps": [
            _copied({"source": "agent", "extra": copied_extra, "tool_calls": [{"extra": {"duration_ms": 300}}]}),
            {"source": "user"},
            {"source": "agent", "extra": {"llm_duration_ms": 2000}, "tool_calls": [{"extra": {"duration_ms": 500}}]},
        ]
    }
    assert trajectory_timing(trajectory) == {"llm_seconds": 2.0, "tool_seconds": 0.5, "source": "atif"}


def test_native_extras_on_copied_context_only_do_not_block_inference() -> None:
    copied = _copied({"source": "agent", "timestamp": "2026-01-01T00:00:00+00:00", "extra": {"llm_duration_ms": 9}})
    assert trajectory_timing({"steps": [copied, _user(1), _agent(4)]}) == _inferred(3.0, 0.0)


def test_each_harbor_step_drops_its_own_copied_context_prefix() -> None:
    first = [_user(), _tool_step(3, [1.0]), _agent(10)]
    # An untrusted continuation keeps the copied prefix in the merged trajectory.
    second = [*(_copied(step) for step in first), _user(40), _tool_step(43, [2.0]), _agent(50)]
    merged = [*_named("prepare", first), *_named("finish", second)]
    assert trajectory_timing({"steps": merged}) == _inferred(17.0, 3.0)


def _after_copied_context(first_new_step: dict[str, Any] | None) -> list[dict[str, Any]]:
    copied = [
        _copied({"source": "user", "timestamp": "2026-01-01T11:00:00Z"}),
        _copied({"source": "user", "timestamp": "2026-01-01T12:00:00Z"}),
    ]
    new_work = [
        _exec_step("2026-01-01T12:00:10Z", "call-new"),
        {"source": "agent", "timestamp": "2026-01-01T12:00:15Z"},
    ]
    return [*copied, *([first_new_step] if first_new_step else []), *new_work]


def test_new_work_that_starts_with_an_agent_step_after_copied_context_is_unmeasured() -> None:
    # The copied timestamps do not show when the first new response started, so timing from it would
    # report 4.0 LLM seconds and drop the 10 s spent producing the first tool call.
    assert trajectory_timing({"steps": _after_copied_context(None)}) is None
    first = [_user(), _tool_step(3, [1.0]), _agent(10)]
    second = _after_copied_context(None)
    assert trajectory_timing({"steps": [*_named("prepare", first), *_named("finish", second)]}) is None


@pytest.mark.parametrize("source", ["user", "system"])
def test_new_user_or_system_step_after_copied_context_starts_the_timing(source: str) -> None:
    steps = _after_copied_context({"source": source, "timestamp": "2026-01-01T12:00:00Z"})
    assert trajectory_timing({"steps": steps}) == _inferred(14.0, 1.0)


def test_native_timing_after_copied_context_needs_no_new_user_step() -> None:
    trajectory = {
        "steps": [
            _copied({"source": "user"}),
            {"source": "agent", "extra": {"llm_duration_ms": 2000}, "tool_calls": [{"extra": {"duration_ms": 500}}]},
        ]
    }
    assert trajectory_timing(trajectory) == {"llm_seconds": 2.0, "tool_seconds": 0.5, "source": "atif"}


@pytest.mark.parametrize(
    "steps",
    [
        [_user(), _tool_step(3, [1.0]), _copied(_agent(5)), _agent(10)],
        [_copied(_user()), _copied(_agent(5))],
    ],
    ids=["copied-after-new-step", "only-copied-steps"],
)
def test_copied_context_that_is_not_a_prefix_of_new_work_is_unmeasured(steps: list[dict[str, Any]]) -> None:
    assert trajectory_timing({"steps": steps}) is None


@pytest.mark.parametrize("value", [None, [], {}, {"steps": "x"}, {"steps": [None]}, {"steps": []}])
def test_malformed_trajectories_have_no_timing(value: Any) -> None:
    assert trajectory_timing(value) is None
