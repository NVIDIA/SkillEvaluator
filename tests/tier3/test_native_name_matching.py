# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harness-native plugin names in plugin signals, MCP proof, and the Harbor verifier.

Each harness spells a plugin's MCP tools, skills, and commands its own way:
Claude Code ``mcp__plugin_<plugin>_<server>__<tool>`` and ``Skill(<plugin>:<name>)``,
OpenCode ``<server>_<tool>``, Hermes ``mcp_<server>_<tool>``, and Codex a bare
tool name whose server only its own log records. The same call must credit the
same declared component on every harness, and never a different one.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.tier3.eval_core import checks as host_checks
from skillevaluator.tier3.eval_core.plugin_signals import compute_plugin_signals, summarize_plugin_signals
from skillevaluator.tier3.mcp_proof import apply_in_agent_mcp_proof
from skillevaluator.tier3.plugin_runtime import apply_runtime_evidence

_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
)


def _load_template():
    spec = importlib.util.spec_from_file_location("harbor_eval_native_names", _TEMPLATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = _load_template()

DECLARED = {
    "skill": ["release-note", "codename-lookup"],
    "mcp": ["docs"],
    "command": ["deploy"],
}


def _trajectory(agent: str, *calls: tuple[str, dict[str, Any], Any]) -> dict[str, Any]:
    steps: list[dict[str, Any]] = [{"step_id": 1, "source": "user", "message": "Do the task."}]
    for index, (function, arguments, content) in enumerate(calls, start=1):
        result: dict[str, Any] = {"source_call_id": f"c{index}", "content": content}
        steps.append(
            {
                "step_id": index + 1,
                "source": "agent",
                "message": "",
                "tool_calls": [{"tool_call_id": f"c{index}", "function_name": function, "arguments": arguments}],
                "observation": {"results": [result]},
            }
        )
    return {"schema_version": "ATIF-v1.7", "agent": {"name": agent, "version": "1"}, "steps": steps}


def _provenance(*rows: dict[str, str]) -> dict[str, Any]:
    return {
        "component_coverage": {"components": [dict(row) for row in rows]},
        "mcp_proof": {"docs": {"status": "reachable-host", "tools": ["search"], "detail": "probe ok"}},
    }


def _engine(agent: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {"agents": {agent: {"plugin_signals_summary": {"with_skill": summary}}}}


# --------------------------------------------------------------------------- #
# MCP tool names                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("agent", "function", "call_servers"),
    [
        pytest.param("claude-code", "mcp__docs__search", None, id="claude-wrapper"),
        pytest.param("claude-code", "mcp__plugin_demo-plugin_docs__search", None, id="claude-native-plugin"),
        pytest.param("opencode", "docs_search", None, id="opencode"),
        pytest.param("hermes", "mcp_docs_search", None, id="hermes"),
        pytest.param("codex", "search", {"c1": "docs"}, id="codex-bare-name-with-log-server"),
    ],
)
def test_every_harness_spelling_of_a_plugin_mcp_call_credits_the_declared_server(
    agent: str, function: str, call_servers: dict[str, str] | None
) -> None:
    trajectory = _trajectory(agent, (function, {"query": "release"}, "3 results"))

    extra = {"mcp_call_servers": call_servers} if call_servers else {}
    signals = compute_plugin_signals(trajectory, {"expected_tools": ["MCP:docs/search"]}, declared=DECLARED, **extra)
    assert signals is not None
    summary = summarize_plugin_signals([signals])
    provenance = apply_runtime_evidence(
        _provenance({"type": "mcp", "name": "docs", "state": "loaded"}), _engine(agent, summary)
    )

    assert [(a["type"], a["name"], a["tool"]) for a in signals["activations"]] == [("mcp", "docs", "mcp__docs__search")]
    assert signals["tool_selection"]["recall"] == 1.0
    assert list(signals["mcp_calls"]["by_server"]) == ["docs"]
    assert "mcp:docs" in signals["activation_coverage"]["exercised"]
    assert provenance["component_coverage"]["components"][0]["state"] == "exercised"
    assert provenance["mcp_proof"]["docs"]["status"] == "used-successfully"


def test_claude_native_names_with_sanitized_server_names_map_back_to_the_declared_server() -> None:
    # Claude Code writes ``plugin:<plugin>:<server>`` as ``plugin_<plugin>_<server>`` and turns ``.`` into ``_``.
    declared = {**DECLARED, "mcp": ["my.docs", "tracker"]}
    trajectory = _trajectory("claude-code", ("mcp__plugin_demo-plugin_my_docs__search", {}, "ok"))

    signals = compute_plugin_signals(trajectory, {"expected_tools": ["MCP:my.docs"]}, declared=declared)

    assert signals is not None
    assert list(signals["mcp_calls"]["by_server"]) == ["my.docs"]
    assert signals["activation_coverage"]["exercised"] == ["mcp:my.docs"]


def test_harness_spellings_never_credit_a_different_server() -> None:
    # Hermes writes both ``my.docs`` and ``my-docs`` as ``my_docs``: the call cannot be attributed to either.
    declared = {**DECLARED, "mcp": ["my.docs", "my-docs"]}
    signals = compute_plugin_signals(_trajectory("hermes", ("mcp_my_docs_search", {}, "ok")), {}, declared=declared)
    assert signals is not None
    assert signals["activation_coverage"]["exercised"] == []
    assert signals["mcp_calls"]["total"] == 1

    # An exact name beats a normalized spelling of another server.
    declared = {**DECLARED, "mcp": ["my_docs", "my-docs"]}
    signals = compute_plugin_signals(_trajectory("hermes", ("mcp_my_docs_search", {}, "ok")), {}, declared=declared)
    assert signals is not None
    assert signals["activation_coverage"]["exercised"] == ["mcp:my_docs"]


def test_in_agent_mcp_proof_matches_exact_names_before_normalized_ones() -> None:
    proof = {
        "my.docs": {"status": "declared", "tools": [], "detail": ""},
        "my-docs": {"status": "declared", "tools": [], "detail": ""},
    }

    upgraded = apply_in_agent_mcp_proof(
        proof, _engine("codex", {"mcp_calls": {"by_server": {"my-docs": {"total": 1, "succeeded": 1}}}})
    )

    assert upgraded["my-docs"]["status"] == "used-successfully"
    assert upgraded["my.docs"]["status"] == "declared"


def test_in_agent_mcp_proof_reads_claude_plugin_server_keys() -> None:
    proof = {"docs": {"status": "reachable-host", "tools": [], "detail": "probe ok"}}

    upgraded = apply_in_agent_mcp_proof(
        proof,
        _engine("claude-code", {"mcp_calls": {"by_server": {"plugin_demo-plugin_docs": {"total": 2, "succeeded": 1}}}}),
    )

    assert upgraded["docs"]["status"] == "used-successfully"


def test_bare_server_prefix_is_only_read_for_harnesses_that_use_it() -> None:
    declared = {**DECLARED, "mcp": ["web", "docs"]}
    # A built-in tool is never a ``web`` server call, and Claude Code never names MCP tools ``<server>_<tool>``.
    for agent, function in (("opencode", "web_search"), ("hermes", "web_extract"), ("claude-code", "docs_search")):
        signals = compute_plugin_signals(_trajectory(agent, (function, {}, "ok")), {}, declared=declared)
        assert signals is not None
        assert signals["mcp_calls"]["total"] == 0, (agent, function)


# --------------------------------------------------------------------------- #
# Plugin commands and skill reads in plugin signals                            #
# --------------------------------------------------------------------------- #
def test_claude_skill_tool_call_of_a_plugin_command_is_a_command_activation() -> None:
    trajectory = _trajectory(
        "claude-code",
        ("Skill", {"skill": "demo-plugin:deploy"}, "Launching skill: demo-plugin:deploy"),
        ("Skill", {"skill": "demo-plugin:release-note"}, "Launching skill: demo-plugin:release-note"),
    )

    signals = compute_plugin_signals(
        trajectory, {"expected_tools": ["Command:deploy", "Skill:release-note"]}, declared=DECLARED
    )
    assert signals is not None
    summary = summarize_plugin_signals([signals])
    provenance = apply_runtime_evidence(
        _provenance({"type": "command", "name": "deploy", "state": "loaded"}), _engine("claude-code", summary)
    )

    # One component is one identity: the declared spelling, whatever the namespace prefix.
    assert [(a["type"], a["name"]) for a in signals["activations"]] == [
        ("command", "deploy"),
        ("skill", "release-note"),
    ]
    assert signals["routing"]["recall"] == 1.0
    assert signals["routing"]["precision"] == 1.0
    assert signals["activation_coverage"]["exercised"] == ["skill:release-note", "command:deploy"]
    assert provenance["component_coverage"]["components"][0]["state"] == "exercised"


SKILL_TEXT = "# Codename lookup\n\n1. Call the lookup tool.\n2. If the tool is not available, say so plainly.\n"


def test_a_chained_shell_read_credits_every_skill_manifest_and_skill_text_is_not_a_failure() -> None:
    command = (
        "cat /workspace/skills/release-note/SKILL.md && printf '\\n---\\n' && "
        "cat /workspace/skills/codename-lookup/SKILL.md"
    )
    trajectory = _trajectory(
        "codex",
        ("exec_command", {"cmd": command}, "Process exited with code 0\nOutput:\n# Release note\n---\n" + SKILL_TEXT),
    )

    signals = compute_plugin_signals(trajectory, {}, declared=DECLARED)

    assert signals is not None
    assert [(a["name"], a["succeeded"]) for a in signals["activations"]] == [
        ("release-note", True),
        ("codename-lookup", True),
    ]
    assert signals["activation_coverage"]["exercised"] == ["skill:release-note", "skill:codename-lookup"]
    assert signals["activation_coverage"]["unavailable"] == []


@pytest.mark.parametrize(
    ("agent", "function", "arguments", "content"),
    [
        pytest.param(
            "opencode",
            "skill",
            {"name": "codename-lookup"},
            '<skill_content name="codename-lookup">\n' + SKILL_TEXT + "</skill_content>",
            id="opencode-skill-tool",
        ),
        pytest.param(
            "claude-code", "Read", {"file_path": "/workspace/skills/codename-lookup/SKILL.md"}, SKILL_TEXT, id="read"
        ),
    ],
)
def test_skill_text_that_mentions_unavailability_does_not_fail_the_skill_load(
    agent: str, function: str, arguments: dict[str, Any], content: str
) -> None:
    signals = compute_plugin_signals(_trajectory(agent, (function, arguments, content)), {}, declared=DECLARED)

    assert signals is not None
    assert [(a["name"], a["succeeded"]) for a in signals["activations"]] == [("codename-lookup", True)]
    assert signals["activation_coverage"]["unavailable"] == []


def test_a_skill_load_that_really_failed_is_still_unavailable() -> None:
    trajectory = _trajectory(
        "claude-code",
        ("Skill", {"skill": "codename-lookup"}, "<tool_use_error>Unknown skill: codename-lookup</tool_use_error>"),
        ("Read", {"file_path": "/workspace/skills/release-note/SKILL.md"}, "File does not exist."),
    )

    signals = compute_plugin_signals(trajectory, {}, declared=DECLARED)

    assert signals is not None
    assert [(a["name"], a["succeeded"]) for a in signals["activations"]] == [
        ("codename-lookup", False),
        ("release-note", False),
    ]
    assert signals["activation_coverage"]["unavailable"] == ["skill:release-note", "skill:codename-lookup"]


def _codex_shell_output(exit_code: int, output: str) -> str:
    """Codex ``exec_command`` output: status lines first, then the command's own output."""
    return (
        f"Chunk ID: 1\nWall time: 0.0 seconds\nProcess exited with code {exit_code}\n"
        f"Original token count: 9\nOutput:\n{output}"
    )


def test_a_codex_shell_read_is_judged_past_its_status_lines() -> None:
    denied = "cat: /workspace/skills/release-note/SKILL.md: Permission denied\n"
    trajectory = _trajectory(
        "codex",
        ("exec_command", {"cmd": "cat /workspace/skills/codename-lookup/SKILL.md"}, _codex_shell_output(0, SKILL_TEXT)),
        ("exec_command", {"cmd": "cat /workspace/skills/release-note/SKILL.md"}, _codex_shell_output(1, denied)),
    )

    signals = compute_plugin_signals(trajectory, {}, declared=DECLARED)

    assert signals is not None
    assert [(a["name"], a["succeeded"]) for a in signals["activations"]] == [
        ("codename-lookup", True),
        ("release-note", False),
    ]
    assert signals["activation_coverage"]["unavailable"] == ["skill:release-note"]


def test_a_shell_error_about_a_skill_manifest_fails_the_read_after_other_output() -> None:
    command = "ls /workspace/skills && cat /workspace/skills/release-note/SKILL.md"
    output = "codename-lookup\nrelease-note\ncat: /workspace/skills/release-note/SKILL.md: Permission denied\n"
    trajectory = _trajectory("claude-code", ("Bash", {"command": command}, output))

    signals = compute_plugin_signals(trajectory, {}, declared=DECLARED)

    assert signals is not None
    assert [(a["name"], a["succeeded"]) for a in signals["activations"]] == [("release-note", False)]
    assert signals["activation_coverage"]["unavailable"] == ["skill:release-note"]


# --------------------------------------------------------------------------- #
# Collector: Codex MCP server names from the raw Codex logs                    #
# --------------------------------------------------------------------------- #
CODEX_REWARD = {
    "security": 1.0,
    "skill_execution": 1.0,
    "skill_efficiency": 1.0,
    "accuracy": 1.0,
    "goal_accuracy": 1.0,
    "behavior_check": 1.0,
}
CODEX_CASE = {"id": "case-1", "prompt": "Look up the codename.", "expected_tools": ["MCP:docs/lookup"]}


def _codex_trajectory() -> dict[str, Any]:
    # Harbor's Codex trajectory keeps only the bare MCP tool name.
    return _trajectory("codex", ("lookup", {"project": "Heron"}, "Codename: HERON-1"))


def _write_codex_job(jobs_dir: Path, variant: str, logs: dict[str, str]) -> None:
    job_dir = jobs_dir / f"demo-codex-{variant}"
    trial_name = "case-1__AbCd123"
    trial_dir = job_dir / trial_name
    (trial_dir / "verifier").mkdir(parents=True)
    (trial_dir / "verifier" / "reward.json").write_text(json.dumps(CODEX_REWARD), encoding="utf-8")
    (trial_dir / "agent").mkdir()
    (trial_dir / "agent" / "trajectory.json").write_text(json.dumps(_codex_trajectory()), encoding="utf-8")
    for relative, text in logs.items():
        path = trial_dir / "agent" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    stats = {"n_trials": 1, "n_errors": 0, "reward_stats": {"reward": {"1.0": [trial_name]}}}
    (job_dir / "result.json").write_text(
        json.dumps({"n_total_trials": 1, "stats": {"n_trials": 1, "n_errors": 0, "evals": {"codex__m__tasks": stats}}}),
        encoding="utf-8",
    )


SESSION_LOG = "\n".join(
    json.dumps(line)
    for line in (
        {"type": "session_meta", "payload": {"id": "s1"}},
        {
            "type": "response_item",
            "payload": {"type": "function_call", "name": "lookup", "namespace": "mcp__docs", "call_id": "c1"},
        },
        {
            "type": "event_msg",
            "payload": {"type": "item_completed", "item": {"type": "McpToolCall", "id": "c1", "server": "docs"}},
        },
    )
)
CODEX_STDOUT = "\n".join(
    [
        "Reading additional input from stdin...",
        json.dumps({"type": "item.started", "item": {"id": "item_1", "type": "mcp_tool_call", "server": "docs"}}),
        json.dumps(
            {
                "type": "item.completed",
                "item": {"id": "item_1", "type": "mcp_tool_call", "server": "docs", "tool": "lookup"},
            }
        ),
    ]
)


@pytest.mark.parametrize(
    "logs",
    [
        pytest.param({"sessions/2026/09/30/rollout-1.jsonl": SESSION_LOG}, id="session-log"),
        pytest.param({"codex.txt": CODEX_STDOUT}, id="codex-json-stream"),
    ],
)
def test_collector_credits_codex_mcp_calls_from_the_codex_logs(tmp_path: Path, logs: dict[str, str]) -> None:
    from skillevaluator.tier3.eval_core.plugin_signals import build_plugin_signals_context
    from skillevaluator.tier3.harbor.collector import collect_harbor_results

    jobs_dir = tmp_path / "jobs"
    for variant in ("with", "without"):
        _write_codex_job(jobs_dir, variant, logs if variant == "with" else {})
    context = build_plugin_signals_context(
        member_skills=["release-note"], mcp_servers=["docs"], wrapper_skills=["demo"], entries=[CODEX_CASE]
    )

    results = collect_harbor_results(
        skill_name="demo",
        agents=["codex"],
        output_dir=tmp_path / "out",
        jobs_dir=jobs_dir,
        expected_cases=1,
        expected_case_ids=["case-1"],
        expected_trials=1,
        plugin_signals=context,
    )

    summary = results["agents"]["codex"]["plugin_signals_summary"]["with_skill"]
    assert summary["mcp_calls"]["by_server"]["docs"]["succeeded"] == 1
    assert summary["activation_coverage"]["exercised"] == ["mcp:docs"]
    assert summary["tool_selection"]["recall"] == 1.0


def test_codex_session_walk_lists_session_logs_in_order_without_following_links(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor.collector import _codex_session_files

    sessions = tmp_path / "agent" / "sessions"
    for relative in ("2026/10/05/rollout-b.jsonl", "2026/10/04/rollout-a.jsonl", "top.jsonl", "notes.txt"):
        _write(sessions / relative, SESSION_LOG)
    outside = tmp_path / "outside"
    _write(outside / "rollout-outside.jsonl", SESSION_LOG)
    (sessions / "linked-dir").symlink_to(outside, target_is_directory=True)
    (sessions / "linked.jsonl").symlink_to(outside / "rollout-outside.jsonl")

    assert _codex_session_files(sessions) == [
        sessions / "top.jsonl",
        sessions / "2026" / "10" / "04" / "rollout-a.jsonl",
        sessions / "2026" / "10" / "05" / "rollout-b.jsonl",
    ]
    linked_sessions = tmp_path / "linked-agent" / "sessions"
    linked_sessions.parent.mkdir()
    linked_sessions.symlink_to(sessions, target_is_directory=True)
    assert _codex_session_files(linked_sessions) == []


def test_codex_session_walk_stops_at_its_entry_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3.harbor import collector

    sessions = tmp_path / "agent" / "sessions"
    for index in range(200):
        (sessions / f"empty-{index:03d}").mkdir(parents=True)
    _write(sessions / "zz" / "rollout.jsonl", SESSION_LOG)
    assert collector._codex_session_files(sessions) == [sessions / "zz" / "rollout.jsonl"]

    listed: list[str] = []
    real_scandir = os.scandir

    def counting_scandir(path: os.PathLike[str] | str):
        listed.append(os.fspath(path))
        return real_scandir(path)

    monkeypatch.setattr(collector, "_CODEX_SESSION_WALK_ENTRIES", 64)
    monkeypatch.setattr(collector.os, "scandir", counting_scandir)

    # The root alone holds more entries than the budget, so nothing below it is listed.
    assert collector._codex_session_files(sessions) == []
    assert listed == [os.fspath(sessions)]


@pytest.mark.parametrize("reverse", [False, True], ids=["sorted-listing", "reverse-listing"])
def test_codex_session_walk_does_not_depend_on_the_listing_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reverse: bool
) -> None:
    from skillevaluator.tier3.harbor import collector

    sessions = tmp_path / "agent" / "sessions"
    day = sessions / "2026" / "10" / "05"
    _write(day / "rollout-a.jsonl", SESSION_LOG)
    for index in range(99):
        _write(day / f"zz-junk-{index:02d}.txt", "")
    real_scandir = os.scandir

    class _ListedInOrder:
        """``os.scandir`` that returns a directory's entries in one name order, as a filesystem may."""

        def __init__(self, path: os.PathLike[str] | str) -> None:
            with real_scandir(path) as iterator:
                self._entries = iter(sorted(iterator, key=lambda entry: entry.name, reverse=reverse))

        def __iter__(self) -> _ListedInOrder:
            return self

        def __next__(self) -> os.DirEntry[str]:
            return next(self._entries)

        def __enter__(self) -> _ListedInOrder:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    monkeypatch.setattr(collector, "_CODEX_SESSION_WALK_ENTRIES", 64)
    monkeypatch.setattr(collector.os, "scandir", _ListedInOrder)

    # The day directory holds more entries than the budget has left, so the walk
    # ends there, whichever names the filesystem would have listed first.
    assert collector._codex_session_files(sessions) == []


# --------------------------------------------------------------------------- #
# Harbor verifier: namespaced skill and command names                         #
# --------------------------------------------------------------------------- #
def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _acme_plugin(tmp_path: Path) -> Path:
    """A plugin whose name is also one of its skills, so ``acme:billing`` contains ``acme``."""
    plugin = tmp_path / "acme"
    _write(plugin / ".claude-plugin" / "plugin.json", json.dumps({"name": "acme", "description": "Acme tools."}))
    for skill in ("acme", "billing"):
        _write(plugin / "skills" / skill / "SKILL.md", f"---\nname: {skill}\ndescription: {skill} skill.\n---\nBody\n")
    _write(plugin / "commands" / "deploy.md", "---\ndescription: Deploy\n---\nDeploy $ARGUMENTS\n")
    _write(
        plugin / "evals" / "evals.json",
        json.dumps(
            {
                "skill_name": "acme",
                "evals": [
                    {"id": "case-1", "prompt": "Use acme.", "expected_output": "Done.", "expected_skill": "acme"}
                ],
            }
        ),
    )
    return plugin


def _native_task_entry(tmp_path: Path, agent: str) -> dict[str, Any]:
    from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks
    from skillevaluator.tier3.harbor.native_staging import build_native_task_staging
    from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package
    from skillevaluator.tier3.plugin_native import HARNESS_ADAPTERS

    package = prepare_plugin_eval_package(_acme_plugin(tmp_path), stage_root=tmp_path / "stage", plugin_load="native")
    staging = build_native_task_staging(agent, HARNESS_ADAPTERS[agent], package.native_source)
    tasks = generate_harbor_tasks(
        package.package_path,
        tmp_path / "out" / agent,
        with_skill=True,
        workspace_skill_paths=list(package.include_skills),
        workspace_mode="group",
        native_plugin=staging,
    )
    return json.loads((Path(tasks[0]) / "tests" / "entry.json").read_text(encoding="utf-8"))


def test_native_claude_tasks_tell_the_verifier_the_plugin_prefix_and_commands(tmp_path: Path) -> None:
    entry = _native_task_entry(tmp_path, "claude-code")

    assert entry["native_plugin_prefix"] == "acme"
    assert entry["native_plugin_commands"] == ["deploy"]
    assert "acme:billing" in entry["workspace_skill_names"]


def test_harnesses_with_bare_names_get_no_native_prefix(tmp_path: Path) -> None:
    entry = _native_task_entry(tmp_path, "codex")

    assert "native_plugin_prefix" not in entry
    assert "native_plugin_commands" not in entry


def test_a_dataset_entry_cannot_set_the_trusted_native_name_fields(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor.adapter import _write_entry_json

    _write_entry_json(
        tmp_path,
        {"id": "case-1", "native_plugin_prefix": "acme", "native_plugin_commands": ["billing"]},
        True,
        evaluated_skill="demo",
    )

    entry = json.loads((tmp_path / "tests" / "entry.json").read_text(encoding="utf-8"))
    assert "native_plugin_prefix" not in entry
    assert "native_plugin_commands" not in entry


def _run_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: dict[str, Any], skill_calls: list[str]
) -> dict[str, Any]:
    """Run the verifier's main() on a trajectory of ``Skill`` calls; judges are stubbed (no model calls)."""
    module_name = f"harbor_eval_native_main_{tmp_path.name}"
    spec = importlib.util.spec_from_file_location(module_name, _TEMPLATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    logs_dir, tests_dir = tmp_path / "logs", tmp_path / "tests"
    agent_dir, verifier_dir = logs_dir / "agent", logs_dir / "verifier"
    for directory in (agent_dir, verifier_dir, tests_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for name, value in {
        "LOGS_DIR": logs_dir,
        "AGENT_LOGS_DIR": agent_dir,
        "VERIFIER_DIR": verifier_dir,
        "TESTS_DIR": tests_dir,
        "ATIF_PATH": agent_dir / "trajectory.json",
        "ENTRY_PATH": tests_dir / "entry.json",
        "REWARD_JSON": verifier_dir / "reward.json",
        "REWARD_TXT": verifier_dir / "reward.txt",
        "SKILL_EVALUATOR_REWARD_JSON": verifier_dir / "skill_evaluator_reward.json",
    }.items():
        monkeypatch.setattr(module, name, value)
    for judge in ("judge_accuracy", "judge_goal_accuracy", "judge_behavior_check"):
        monkeypatch.setattr(module, judge, lambda *_a, **_k: module._judge_not_applicable("stubbed in tests"))
    # Without a reference answer the judges are not applicable, so nothing is judged.
    entry = {**entry, "ground_truth": "", "expected_behavior": []}
    calls = [("Skill", {"skill": name}, f"Launching skill: {name}") for name in skill_calls]
    calls.append(("Bash", {"command": "echo done"}, "done"))
    (agent_dir / "trajectory.json").write_text(json.dumps(_trajectory("claude-code", *calls)), encoding="utf-8")
    (tests_dir / "entry.json").write_text(json.dumps(entry), encoding="utf-8")

    module.main()

    sidecar = json.loads((verifier_dir / "skill_evaluator_reward.json").read_text(encoding="utf-8"))
    return sidecar["details"]


def test_verifier_never_credits_a_different_plugin_skill_through_the_plugin_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = {**_native_task_entry(tmp_path, "claude-code"), "expected_skill": "acme", "should_trigger": True}

    wrong = _run_verifier(tmp_path / "wrong", monkeypatch, entry, ["acme:billing"])
    right = _run_verifier(tmp_path / "right", monkeypatch, entry, ["acme:acme"])

    assert wrong["skill_execution"]["activation"]["score"] == 0.0
    assert right["skill_execution"]["activation"]["score"] == 1.0
    assert right["skill_efficiency"]["routing"]["score"] == 1.0


def test_verifier_keeps_plugin_commands_out_of_skill_routing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry = {**_native_task_entry(tmp_path, "claude-code"), "expected_skill": "billing", "should_trigger": True}
    entry["skill_workspace_mode"] = "isolated"

    details = _run_verifier(tmp_path / "run", monkeypatch, entry, ["acme:billing", "acme:deploy"])

    assert details["skill_execution"]["activation"]["score"] == 1.0
    assert details["skill_efficiency"]["routing"]["score"] == 1.0, details["skill_efficiency"]["routing"]
    assert details["_native_plugin_commands"] == ["acme:deploy"]


def test_the_native_prefix_does_not_depend_on_the_skill_alias_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skillevaluator.tier3.plugin_native import ClaudeCodeAdapter

    build = ClaudeCodeAdapter.build

    def build_with_staged_skill_aliases(self: ClaudeCodeAdapter, source: Any) -> Any:
        bundle = build(self, source)
        # An alias rule that lists only the skills the staged plugin loads.
        bundle.skill_aliases = [f"{self.plugin_name(source)}:{name}" for name in ("acme", "billing")]
        return bundle

    monkeypatch.setattr(ClaudeCodeAdapter, "build", build_with_staged_skill_aliases)
    entry = {
        **_native_task_entry(tmp_path, "claude-code"),
        "expected_skill": "billing",
        "should_trigger": True,
        "skill_workspace_mode": "isolated",
    }

    assert entry["native_plugin_prefix"] == "acme"
    assert entry["native_plugin_commands"] == ["deploy"]
    details = _run_verifier(tmp_path / "run", monkeypatch, entry, ["acme:billing", "acme:deploy"])
    assert details["skill_efficiency"]["routing"]["score"] == 1.0, details["skill_efficiency"]["routing"]


@pytest.mark.parametrize("agent", ["claude-code", "codex", "opencode", "hermes"])
def test_only_claude_code_namespaces_plugin_skill_and_command_names(tmp_path: Path, agent: str) -> None:
    from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package
    from skillevaluator.tier3.plugin_native import HARNESS_ADAPTERS, ClaudeCodeAdapter

    package = prepare_plugin_eval_package(_acme_plugin(tmp_path), stage_root=tmp_path / "stage", plugin_load="native")
    source = package.native_source
    # Claude Code's namespace is the name in the plugin.json it stages.
    expected = ClaudeCodeAdapter().manifest(source)["name"] if agent == "claude-code" else ""

    assert HARNESS_ADAPTERS[agent].skill_namespace(source) == expected


def test_verifier_negative_case_sees_a_namespaced_skill_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry = {
        **_native_task_entry(tmp_path, "claude-code"),
        "expected_skill": "",
        "should_trigger": False,
        "evaluated_skill": "billing",
    }

    details = _run_verifier(tmp_path / "run", monkeypatch, entry, ["acme:billing"])

    assert details["skill_execution"]["negative_check"]["passed"] is False


@pytest.mark.parametrize("module", [verifier, host_checks], ids=["harbor-template", "host-checks"])
def test_both_verifier_copies_match_native_names_exactly(module: Any) -> None:
    def activation(expected: str, names: list[str]) -> float:
        return module.check_activation([], expected, skill_tool_names=names, native_prefix="acme")["score"]

    assert activation("acme", ["acme:billing"]) == 0.0
    assert activation("billing", ["acme:billing"]) == 1.0
    assert activation("billing", ["ACME:Billing"]) == 1.0
    # Names without the plugin prefix keep the legacy substring match.
    assert activation("billing", ["user:billing"]) == 1.0

    routing = module.check_routing([], "acme", skill_tool_names=["acme:billing"], native_prefix="acme")
    assert routing["score"] == 0.0 and routing["details"]["wrong_skills"] == ["Skill(acme:billing)"]

    kept, commands = module.split_native_command_calls(
        ["acme:billing", "acme:deploy", "deploy"], "acme", ["deploy"], ["billing", "acme:billing"]
    )
    assert (kept, commands) == (["acme:billing", "deploy"], ["acme:deploy"])

    negative = module.check_negative_case([], "billing", skill_tool_names=["acme:billing"], native_prefix="acme")
    assert negative["passed"] is False
