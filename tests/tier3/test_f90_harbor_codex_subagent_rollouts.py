# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof H4 (Codex subagent shape): the trajectory must come from the main Codex thread.

When a Codex agent calls ``spawn_agent``, the child thread writes its own rollout
next to the main one. Harbor 0.24's Codex converter turns only the newest rollout
into ``agent/trajectory.json``, and the newest one is the child. The main agent's
calls, its final answer and its first-turn tokens then vanish, and the judge grades
the child's last message. The saved ``rk-review-draft-008`` trials (``j6Tnytt``,
``pGWMr6p`` with ``fork_context``; ``cbyJUoc`` without) all have this shape.

These tests write rollouts shaped like those saved ones, run Harbor's own post-run
conversion through every Codex class Tier 3 can load, and read the result the way
the verifier, the collector and the plugin signals do. No model, network, Docker or
sandbox is used.
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("harbor")

from harbor.agents.installed.codex import Codex
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories.trajectory import Trajectory

from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.eval_core import plugin_signals as ps
from skillevaluator.tier3.eval_core.atif_helpers import get_final_response
from skillevaluator.tier3.harbor.collector import _codex_mcp_calls, _trial_usage, _with_harness_statuses
from skillevaluator.tier3.harbor.runner import _agent_import_path, build_harbor_run_command
from skillevaluator.tier3.plugin_native import native_agent_import_path

_MAIN_ID = "01a1066b-8191-7f53-9a08-f41298ef1c72"
_CHILD_ID = "01a1066b-c47c-75f1-aac2-4223c3d589a0"
_MAIN_TURN = "01a1066b-8203-7fb2-acd9-0e3a31110941"
_CHILD_TURN = "01a1066b-c4aa-7e13-a841-5f5480fd8c99"
_MAIN_ROLLOUT = f"rollout-2026-10-04T10-17-51-{_MAIN_ID}.jsonl"
_CHILD_ROLLOUT = f"rollout-2026-10-04T10-18-08-{_CHILD_ID}.jsonl"

_PROMPT = (
    "Review the draft release notes in input/RELEASE_NOTES.draft.md. Run the release-kit release check on that "
    "file and have the release reviewer look at it too.\n"
)
_REVIEW_PROMPT = "Review `/workspace/input/RELEASE_NOTES.draft.md` as a release reviewer. Do not edit anything."
_MAIN_FINAL = "I reviewed the draft and got a reviewer pass too.\n\n**Problems Found**\n- the version is a build tag."
_CHILD_FINAL = "- `atlas 1.5.0+rk.23b59e` looks like a draft/build version, not a final release version."

# ``last_token_usage`` (input, cached input, output) of each model request, as in the saved trial.
_MAIN_REQUESTS = [
    (13149, 12928, 98),  # read the draft
    (13357, 12928, 89),  # web search
    (13504, 12928, 146),  # compute the version through the plugin's MCP server
    (13696, 13440, 64),  # spawn the reviewer
    (14583, 13440, 58),  # wait for it
    (18284, 13440, 173),  # final answer
]
_CHILD_REQUESTS = [
    (12034, 11904, 100),  # read the draft
    (12988, 12800, 119),  # publish (forbidden for the reviewer)
    (13520, 12928, 140),  # final review
]


def _sum(requests: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    return (sum(r[0] for r in requests), sum(r[1] for r in requests), sum(r[2] for r in requests))


_MAIN_TOTAL = _sum(_MAIN_REQUESTS)
_CHILD_OWN_USAGE = _sum(_CHILD_REQUESTS)
_INHERITED_AT_SPAWN = _sum(_MAIN_REQUESTS[:4])


def _ts(minute: int, second: float) -> str:
    return f"2026-10-04T10:{minute:02d}:{second:06.3f}Z"


def _token_count(when: str, total: tuple[int, int, int] | None, last: tuple[int, int, int] | None) -> dict[str, Any]:
    if total is None or last is None:
        info = None
    else:
        info = {
            "total_token_usage": {
                "input_tokens": total[0],
                "cached_input_tokens": total[1],
                "output_tokens": total[2],
                "reasoning_output_tokens": 0,
                "total_tokens": total[0] + total[2],
            },
            "last_token_usage": {
                "input_tokens": last[0],
                "cached_input_tokens": last[1],
                "output_tokens": last[2],
                "reasoning_output_tokens": 0,
                "total_tokens": last[0] + last[2],
            },
            "model_context_window": 258400,
        }
    return {"timestamp": when, "type": "event_msg", "payload": {"type": "token_count", "info": info}}


class _Usage:
    """Codex's running ``total_token_usage``: each closed model request adds its ``last_token_usage``."""

    def __init__(self, start: tuple[int, int, int] = (0, 0, 0)) -> None:
        self.total = start

    def close(self, when: str, last: tuple[int, int, int]) -> list[dict[str, Any]]:
        self.total = (self.total[0] + last[0], self.total[1] + last[1], self.total[2] + last[2])
        # Codex repeats the same token_count right after the tool output.
        return [_token_count(when, self.total, last), _token_count(when, self.total, last)]


def _event(when: str, kind: str, **payload: Any) -> dict[str, Any]:
    return {"timestamp": when, "type": "event_msg", "payload": {"type": kind, **payload}}


def _message(when: str, role: str, text: str) -> dict[str, Any]:
    part = "output_text" if role == "assistant" else "input_text"
    return {
        "timestamp": when,
        "type": "response_item",
        "payload": {"type": "message", "role": role, "content": [{"type": part, "text": text}]},
    }


def _call(when: str, call_id: str, name: str, arguments: dict[str, Any], namespace: str | None = None):
    payload: dict[str, Any] = {
        "type": "function_call",
        "name": name,
        "arguments": json.dumps(arguments),
        "call_id": call_id,
    }
    if namespace:
        payload["namespace"] = namespace
    return {"timestamp": when, "type": "response_item", "payload": payload}


def _output(when: str, call_id: str, output: str) -> dict[str, Any]:
    return {
        "timestamp": when,
        "type": "response_item",
        "payload": {"type": "function_call_output", "call_id": call_id, "output": output},
    }


def _main_meta(when: str) -> dict[str, Any]:
    return {
        "timestamp": when,
        "type": "session_meta",
        "payload": {
            "id": _MAIN_ID,
            "forked_from_id": None,
            "timestamp": _ts(17, 51.346),
            "cwd": "/workspace",
            "originator": "codex_exec",
            "cli_version": "0.124.0",
            "source": "exec",
            "model_provider": "openai",
        },
    }


def _turn_preamble(when: str, turn_id: str, prompt: str) -> list[dict[str, Any]]:
    return [
        _event(when, "task_started", turn_id=turn_id, model_context_window=258400),
        _message(when, "developer", "<permissions instructions>sandbox</permissions>"),
        _message(when, "user", "<environment_context>\n  <cwd>/workspace</cwd>\n</environment_context>"),
        {
            "timestamp": when,
            "type": "turn_context",
            "payload": {"turn_id": turn_id, "cwd": "/workspace", "model": "gpt-test", "approval_policy": "never"},
        },
        _message(when, "user", prompt),
        _event(when, "user_message", message=prompt),
        _token_count(when, None, None),
    ]


_READ_DRAFT = {"cmd": "sed -n '1,220p' input/RELEASE_NOTES.draft.md"}
_DRAFT_OUTPUT = "Process exited with code 0\nOutput:\n# atlas 1.5.0+rk.23b59e"


def _main_rollout(*, final_at: float = 23.561) -> list[dict[str, Any]]:
    """The main thread of the saved ``j6Tnytt`` trial, cut down to one call of each kind."""
    usage = _Usage()
    requests = iter(_MAIN_REQUESTS)
    events = [_main_meta(_ts(17, 51.373))]
    events += _turn_preamble(_ts(17, 51.377), _MAIN_TURN, _PROMPT)
    events.append(_message(_ts(17, 52.388), "assistant", "I'll inspect the draft, then ask the reviewer agent."))
    events.append(_call(_ts(17, 52.609), "call_main_read", "exec_command", _READ_DRAFT))
    events += usage.close(_ts(17, 52.700), next(requests))
    events.append(_output(_ts(17, 52.705), "call_main_read", _DRAFT_OUTPUT))
    events.append(
        {
            "timestamp": _ts(18, 2.564),
            "type": "response_item",
            "payload": {
                "type": "web_search_call",
                "id": "ws_main_1",
                "status": "completed",
                "action": {"type": "search", "query": "release-kit release check"},
            },
        }
    )
    events.append(_message(_ts(18, 2.727), "assistant", "No local release-kit install; I'll use the plugin tools."))
    events += usage.close(_ts(18, 2.800), next(requests))
    events.append(
        _call(_ts(18, 4.100), "call_main_compute", "compute_version", {"current": "1.4.2"}, "mcp__reltools__")
    )
    events += usage.close(_ts(18, 4.200), next(requests))
    events.append(_output(_ts(18, 4.300), "call_main_compute", json.dumps([{"type": "text", "text": "1.5.0"}])))
    events.append(
        _call(_ts(18, 8.230), "call_main_spawn", "spawn_agent", {"fork_context": True, "message": _REVIEW_PROMPT})
    )
    events += usage.close(_ts(18, 8.300), next(requests))
    events.append(
        _output(_ts(18, 8.472), "call_main_spawn", json.dumps({"agent_id": _CHILD_ID, "nickname": "Hilbert"}))
    )
    events.append(_call(_ts(18, 9.776), "call_main_wait", "wait_agent", {"targets": [_CHILD_ID], "timeout_ms": 30000}))
    events += usage.close(_ts(18, 9.800), next(requests))
    events.append(_output(_ts(18, 20.400), "call_main_wait", json.dumps({"status": {_CHILD_ID: "completed"}})))
    events.append(_message(_ts(18, 21.464), "user", '<subagent_notification>\n{"status":"completed"}'))
    events.append(_event(_ts(18, final_at), "agent_message", message=_MAIN_FINAL))
    events.append(_message(_ts(18, final_at), "assistant", _MAIN_FINAL))
    events += usage.close(_ts(18, final_at + 0.02), next(requests))
    events.append(_event(_ts(18, final_at + 0.04), "task_complete", turn_id=_MAIN_TURN, last_agent_message=_MAIN_FINAL))
    assert usage.total == _MAIN_TOTAL
    return events


def _child_meta(*, forked: bool) -> dict[str, Any]:
    return {
        "timestamp": _ts(18, 8.410),
        "type": "session_meta",
        "payload": {
            "id": _CHILD_ID,
            "forked_from_id": _MAIN_ID if forked else None,
            "timestamp": _ts(18, 8.399),
            "cwd": "/workspace",
            "originator": "codex_exec",
            "cli_version": "0.124.0",
            "source": {
                "subagent": {
                    "thread_spawn": {
                        "parent_thread_id": _MAIN_ID,
                        "depth": 1,
                        "agent_path": None,
                        "agent_nickname": "Hilbert",
                        "agent_role": None,
                    }
                }
            },
            "agent_nickname": "Hilbert",
            "model_provider": "openai",
        },
    }


def _copied_history(*, with_calls: bool) -> list[dict[str, Any]]:
    """What Codex copies into a ``fork_context`` child: the parent's meta, turn start and history.

    Codex 0.124 copies the user and developer messages and the ``event_msg`` records. ``with_calls`` also copies
    the parent's assistant message and tool call, as a fuller fork would, so they must not count twice.
    """
    when = _ts(18, 8.414)
    usage = _Usage()
    events = [_main_meta(when)]
    events += [event for event in _turn_preamble(when, _MAIN_TURN, _PROMPT) if event["type"] != "turn_context"]
    events.append(_event(when, "agent_message", message="I'll inspect the draft, then ask the reviewer agent."))
    if with_calls:
        events.append(_message(when, "assistant", "I'll inspect the draft, then ask the reviewer agent."))
        events.append(_call(when, "call_main_read", "exec_command", _READ_DRAFT))
    for last in _MAIN_REQUESTS[:4]:
        events += usage.close(when, last)
    if with_calls:
        events.append(_output(when, "call_main_read", _DRAFT_OUTPUT))
    return events


def _child_rollout(*, forked: bool = True, copied_calls: bool = False) -> list[dict[str, Any]]:
    """The reviewer thread: its own meta, the forked history (if any), then its own turn."""
    usage = _Usage(_INHERITED_AT_SPAWN if forked else (0, 0, 0))
    requests = iter(_CHILD_REQUESTS)
    events = [_child_meta(forked=forked)]
    if forked:
        events += _copied_history(with_calls=copied_calls)
    events += _turn_preamble(_ts(18, 8.436), _CHILD_TURN, _REVIEW_PROMPT)
    if forked:
        # The first own token_count repeats the inherited total before any model output.
        events.append(_token_count(_ts(18, 9.319), usage.total, _MAIN_REQUESTS[3]))
    events.append(_message(_ts(18, 9.610), "assistant", "I'll inspect the draft and run the release checks."))
    events.append(_call(_ts(18, 9.831), "call_child_read", "exec_command", {"cmd": "cat input/RELEASE_NOTES.draft.md"}))
    events += usage.close(_ts(18, 9.845), next(requests))
    events.append(_output(_ts(18, 10.912), "call_child_read", _DRAFT_OUTPUT))
    events.append(
        _call(
            _ts(18, 16.676),
            "call_child_publish",
            "publish_release",
            {"project": "atlas", "version": "1.5.0"},
            "mcp__reltools__",
        )
    )
    events += usage.close(_ts(18, 16.679), next(requests))
    events.append(_output(_ts(18, 16.726), "call_child_publish", json.dumps([{"type": "text", "text": "published"}])))
    events.append(_event(_ts(18, 20.302), "agent_message", message=_CHILD_FINAL))
    events.append(_message(_ts(18, 20.303), "assistant", _CHILD_FINAL))
    events += usage.close(_ts(18, 20.327), next(requests))
    events.append(_event(_ts(18, 20.330), "task_complete", turn_id=_CHILD_TURN, last_agent_message=_CHILD_FINAL))
    return events


def _write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def _write_trial(logs_dir: Path, *, forked: bool = True, copied_calls: bool = False) -> None:
    day = logs_dir / "sessions" / "2026" / "10" / "04"
    _write_jsonl(day / _MAIN_ROLLOUT, _main_rollout())
    _write_jsonl(day / _CHILD_ROLLOUT, _child_rollout(forked=forked, copied_calls=copied_calls))


def _load_class(target: str) -> type:
    if ":" not in target:
        assert target == "codex", target
        return Codex
    module_name, _, class_name = target.partition(":")
    return getattr(importlib.import_module(module_name), class_name)


def _provider(name: str) -> ProviderConfig:
    if name == "nv_build":
        return ProviderConfig(
            provider="nv_build",
            model="openai/gpt-oss-120b",
            api_key="test-key",
            base_url="https://integrate.api.nvidia.com/v1",
            litellm_model="openai/openai/gpt-oss-120b",
        )
    return ProviderConfig(
        provider=name,
        model="gpt-test",
        api_key="test-key",
        base_url="https://gateway.example.test/v1" if name == "openai-compatible" else None,
        litellm_model="openai/gpt-test",
    )


_ROUTES = [
    (provider, env_mode)
    for provider in ("openai", "openai-compatible", "anthropic", "nv_build")
    for env_mode in ("docker", "e2b", "local")
]


def _baseline_target(provider_name: str, env_mode: str) -> tuple[str, str | None]:
    """The ``--agent`` value Tier 3 passes to Harbor for Codex, and the routing wrapper it chose."""
    import_path = _agent_import_path(_provider(provider_name), "codex", env_mode)
    command = build_harbor_run_command(
        dataset_path="dataset",
        agent="codex",
        job_name="codex-route",
        env_mode=env_mode,
        agent_import_path=import_path,
    )
    return command[command.index("--agent") + 1], import_path


def _harbor_codex_classes() -> list[tuple[str, type]]:
    """Every Codex class Tier 3 makes Harbor load: per provider and environment, baseline and native arms."""
    classes: dict[str, type] = {}
    for provider_name, env_mode in _ROUTES:
        target, import_path = _baseline_target(provider_name, env_mode)
        classes[target] = _load_class(target)
        if env_mode != "local":
            native = native_agent_import_path("codex", import_path)
            classes[native] = _load_class(native)
    return sorted(classes.items())


_CLASSES = _harbor_codex_classes()


def _convert(agent_cls: type, logs_dir: Path) -> tuple[dict[str, Any], AgentContext]:
    agent = agent_cls(logs_dir=logs_dir, model_name="openai/gpt-test")
    agent.logger = logging.getLogger("test-f90-harbor-codex-subagent")
    context = AgentContext()
    agent.populate_context_post_run(context)
    trajectory_path = logs_dir / "trajectory.json"
    assert trajectory_path.is_file(), f"{agent_cls.__name__} wrote no trajectory"
    return json.loads(trajectory_path.read_text(encoding="utf-8")), context


def _calls(steps: list[dict[str, Any]]) -> list[str]:
    return [call["function_name"] for step in steps for call in step.get("tool_calls") or []]


def _sidechain(step: dict[str, Any]) -> bool:
    return bool((step.get("extra") or {}).get("is_sidechain"))


@pytest.mark.parametrize(("target", "agent_cls"), _CLASSES, ids=[target for target, _ in _CLASSES])
def test_codex_trajectory_is_the_main_thread_when_a_subagent_has_a_newer_rollout(
    tmp_path: Path, target: str, agent_cls: type
) -> None:
    job_dir = tmp_path / "job"
    logs_dir = job_dir / "rk-review-draft-008__j6Tnytt" / "agent"
    _write_trial(logs_dir)

    trajectory, context = _convert(agent_cls, logs_dir)

    # The trajectory is the main thread's, not the newer child rollout's.
    assert trajectory["session_id"] == _MAIN_ID
    Trajectory.model_validate(trajectory)

    # The judge grades the main agent's final answer (eval.py reads the last agent message).
    assert get_final_response(trajectory) == _MAIN_FINAL

    # The main agent's own calls are all there, outside the sidechain.
    main_steps = [step for step in trajectory["steps"] if not _sidechain(step)]
    assert _calls(main_steps) == ["exec_command", "web_search_call", "compute_version", "spawn_agent", "wait_agent"]

    # The child's calls are folded in as sidechain steps, the way Claude subagents are.
    child_steps = [step for step in trajectory["steps"] if _sidechain(step)]
    assert child_steps, "the subagent's steps were dropped"
    assert {step["source"] for step in child_steps} == {"agent"}
    assert {(step.get("extra") or {}).get("agent_id") for step in child_steps} == {_CHILD_ID}
    assert _calls(child_steps) == ["exec_command", "publish_release"]

    # Check 25 and the usage counters read the main thread's first turn; the totals add the child's own usage.
    usage = _trial_usage(job_dir, {"_trial_root_name": "rk-review-draft-008__j6Tnytt"})
    assert usage.get("first_turn_prompt_tokens") == float(_MAIN_REQUESTS[0][0])
    assert usage.get("prompt_tokens") == float(_MAIN_TOTAL[0] + _CHILD_OWN_USAGE[0])
    assert usage.get("completion_tokens") == float(_MAIN_TOTAL[2] + _CHILD_OWN_USAGE[2])
    assert context.n_input_tokens == _MAIN_TOTAL[0] + _CHILD_OWN_USAGE[0]


@pytest.mark.parametrize(
    ("forked", "copied_calls"),
    [(True, False), (True, True), (False, False)],
    ids=["fork-context", "fork-context-with-copied-calls", "no-fork-context"],
)
def test_forked_history_is_not_counted_as_the_subagent_s_own_work(
    tmp_path: Path, forked: bool, copied_calls: bool
) -> None:
    from skillevaluator.tier3.harbor.local_agents import SkillEvaluatorCodex

    logs_dir = tmp_path / "agent"
    _write_trial(logs_dir, forked=forked, copied_calls=copied_calls)

    trajectory, _ = _convert(SkillEvaluatorCodex, logs_dir)

    assert trajectory["session_id"] == _MAIN_ID
    child_steps = [step for step in trajectory["steps"] if _sidechain(step)]
    child_call_ids = [call["tool_call_id"] for step in child_steps for call in step.get("tool_calls") or []]
    assert child_call_ids == ["call_child_read", "call_child_publish"]
    all_call_ids = [call["tool_call_id"] for step in trajectory["steps"] for call in step.get("tool_calls") or []]
    assert all_call_ids.count("call_main_read") == 1
    child_prompts = [step["metrics"]["prompt_tokens"] for step in child_steps if step.get("metrics")]
    assert child_prompts == [request[0] for request in _CHILD_REQUESTS]
    # The child's prompt is not a user step: plugin signals treat user text as the user's own words.
    user_texts = [str(step["message"]) for step in trajectory["steps"] if step["source"] == "user"]
    assert not any(_REVIEW_PROMPT in text for text in user_texts)
    assert trajectory["final_metrics"]["total_prompt_tokens"] == _MAIN_TOTAL[0] + _CHILD_OWN_USAGE[0]
    assert trajectory["final_metrics"]["total_cached_tokens"] == _MAIN_TOTAL[1] + _CHILD_OWN_USAGE[1]
    assert trajectory["final_metrics"]["total_steps"] == len(trajectory["steps"])
    assert trajectory["agent"]["extra"]["agent_ids"] == [_CHILD_ID]


def test_a_late_subagent_step_never_replaces_the_main_final_answer(tmp_path: Path) -> None:
    """A child still running after the main thread answered must not become the graded final response."""
    from skillevaluator.tier3.harbor.local_agents import SkillEvaluatorCodex

    logs_dir = tmp_path / "agent"
    day = logs_dir / "sessions" / "2026" / "10" / "04"
    _write_jsonl(day / _MAIN_ROLLOUT, _main_rollout(final_at=12.0))
    _write_jsonl(day / _CHILD_ROLLOUT, _child_rollout())

    trajectory, _ = _convert(SkillEvaluatorCodex, logs_dir)

    assert trajectory["session_id"] == _MAIN_ID
    assert get_final_response(trajectory) == _MAIN_FINAL
    assert _calls([step for step in trajectory["steps"] if _sidechain(step)]) == ["exec_command", "publish_release"]


def test_unrelated_rollouts_keep_harbor_s_newest_main_thread(tmp_path: Path) -> None:
    """Two main threads (no subagent link) keep Harbor's choice, the newest; a lone rollout is Harbor's own path."""
    from skillevaluator.tier3.harbor.local_agents import SkillEvaluatorCodex

    logs_dir = tmp_path / "agent"
    day = logs_dir / "sessions" / "2026" / "10" / "04"
    older = [event for event in _main_rollout() if event["type"] != "session_meta"]
    older.insert(0, {**_main_meta(_ts(17, 1.0)), "payload": {**_main_meta(_ts(17, 1.0))["payload"], "id": "older"}})
    _write_jsonl(day / "rollout-2026-10-04T10-17-01-older.jsonl", older)
    _write_jsonl(day / _MAIN_ROLLOUT, _main_rollout())

    trajectory, _ = _convert(SkillEvaluatorCodex, logs_dir)

    assert trajectory["session_id"] == _MAIN_ID
    assert not any(_sidechain(step) for step in trajectory["steps"])

    single = tmp_path / "single"
    _write_jsonl(single / "sessions" / "2026" / "10" / "04" / _MAIN_ROLLOUT, _main_rollout())
    ours, _ = _convert(SkillEvaluatorCodex, single)
    (single / "trajectory.json").unlink()
    harbor, _ = _convert(Codex, single)
    assert ours == harbor


def test_conflict_probe_sees_a_call_made_inside_the_codex_subagent(tmp_path: Path) -> None:
    """Check 19: the main thread used the plugin tool and the child made the forbidden call; both count."""
    job_dir = tmp_path / "job"
    trial_root = job_dir / "rk-review-draft-008__j6Tnytt"
    logs_dir = trial_root / "agent"
    _write_trial(logs_dir)
    # The plain OpenAI Codex run in Docker, the common baseline route.
    trajectory, _ = _convert(_load_class(_baseline_target("openai", "docker")[0]), logs_dir)

    entry = {
        "id": "e07",
        "prompt": _PROMPT,
        "expected_output": "x",
        "conflict_probes": [
            {
                "id": "reviewer-never-publishes",
                "must_use": "MCP:reltools/compute_version",
                "must_not_use": "MCP:reltools/publish_release",
            }
        ],
    }
    context = ps.build_plugin_signals_context(
        member_skills=["release-notes"],
        mcp_servers=["reltools"],
        entries=[entry],
        subagents=["release-reviewer"],
    )
    # As the collector does: servers from the session logs, outcomes from codex.txt.
    call_servers, call_statuses = _codex_mcp_calls(trial_root, trajectory)
    signals = ps.compute_plugin_signals(
        _with_harness_statuses(trajectory, call_statuses),
        context.case_spec("e07"),
        declared=context.declared_for(ps.ARM_WITH_SKILL),
        mcp_call_servers=call_servers,
        subagent_aliases=context.aliases_for(ps.ARM_WITH_SKILL),
    )
    conflict = signals["conflict"]
    assert (conflict["passed"], conflict["checked"]) == (0, 1)
    detail = conflict["failures"][0]["detail"]
    assert "was not activated" not in detail
    assert "must_not_use" in detail
    assert signals["mcp_calls"]["total"] == 2
