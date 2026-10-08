# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check 18 handoffs: value matching, windows, paths, shell forms, and validation (proof M28, L22).

The trajectories follow the check-18 replay examples, in the Harbor ATIF shapes
of Claude Code (``Skill`` loads, ``Bash``/``Write``/``Read``, sidechain steps
for subagents) and Codex (``exec_command`` with ``workdir``, bare MCP names).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from skillevaluator.tier3.eval_core.plugin_signals import (
    ARM_SUM_OF_PARTS,
    ARM_WITH_SKILL,
    build_plugin_signals_context,
    compute_plugin_signals,
    plugin_case_spec,
    validate_plugin_case_fields,
)

CONTEXT = build_plugin_signals_context(
    member_skills=["changelog-collect", "version-bump", "release-notes"],
    mcp_servers=["reltools", "fsx"],
    wrapper_skills=["release-kit-plugin-eval", "release-kit"],
    subagents=["release-reviewer"],
)
SKILLS = "/tmp/agent-home/.agents/skills"
CC, RN = "Skill:changelog-collect", "Skill:release-notes"


def _step(call_id: str, fn: str, args: dict[str, Any], content: str = "ok", **extra: Any) -> dict[str, Any]:
    return {
        "source": "agent",
        "tool_calls": [{"tool_call_id": call_id, "function_name": fn, "arguments": args}],
        "observation": {"results": [{"source_call_id": call_id, "content": content}]},
        "extra": {"cwd": "/workspace", "is_sidechain": False, **extra},
    }


def _claude(*steps: dict[str, Any]) -> dict[str, Any]:
    return {
        "agent": {"name": "claude-code", "extra": {"cwds": ["/workspace"]}},
        "steps": [{"source": "user", "message": "Collect the changes and write notes."}, *steps],
    }


def _codex(*steps: dict[str, Any]) -> dict[str, Any]:
    return {
        "agent": {"name": "codex", "extra": {"cwd": "/workspace"}},
        "steps": [{"source": "user", "message": "Collect the changes and write notes."}, *steps],
    }


def _launch(call_id: str, name: str, **extra: Any) -> dict[str, Any]:
    return _step(call_id, "Skill", {"skill": f"release-kit:{name}"}, f"Launching skill: release-kit:{name}", **extra)


def _bash(call_id: str, command: str, output: str = "", **extra: Any) -> dict[str, Any]:
    return _step(call_id, "Bash", {"command": command}, output or "(Bash completed with no output)", **extra)


def _exec(call_id: str, cmd: str, output: str = "", *, code: int = 0) -> dict[str, Any]:
    content = f"Chunk ID: c0\nWall time: 0.0000 seconds\nProcess exited with code {code}\nOutput:\n{output}"
    return _step(call_id, "exec_command", {"cmd": cmd, "workdir": "/workspace"}, content)


def _codex_read(call_id: str, *members: str) -> dict[str, Any]:
    cmd = " && ".join(f"sed -n '1,220p' {SKILLS}/{member}/SKILL.md" for member in members)
    return _exec(call_id, cmd, "".join(f"---\nname: {member}\n---\n" for member in members))


def _handoff(trajectory: dict[str, Any], *handoffs: dict[str, Any], arm: str = ARM_WITH_SKILL) -> dict[str, Any]:
    signals = compute_plugin_signals(
        trajectory,
        plugin_case_spec({"id": "c", "handoffs": list(handoffs)}),
        declared=CONTEXT.declared_for(arm),
        wrapper_skills=CONTEXT.wrapper_skills,
    )
    assert signals is not None
    return signals["handoff"]


def _passes(trajectory: dict[str, Any], **handoff: Any) -> bool:
    return _handoff(trajectory, {"producer": CC, "consumer": RN, **handoff})["passed"] == 1


# ---------------------------------------------------------------------------
# Values are compared as values (M28; check-18 e08)
# ---------------------------------------------------------------------------

CHANGES = 'AT-4821 say "hi"\npath C:\\rk\\out\nowner café-7\n'
MCP_CHANGES = json.dumps({"changes": [{"id": "AT-4835", "note": 'ok "go"'}]})
NOTES = '# notes\n- AT-4821: say "hi"\n- path C:\\rk\\out\n- owner café-7\n- AT-4835: ok "go"\n'


def _e08_claude() -> dict[str, Any]:
    return _claude(
        _launch("t1", "changelog-collect"),
        _bash("t2", "cat input/changes.txt", CHANGES),
        _step(
            "t3",
            "mcp__plugin_release-kit_reltools__list_changes",
            {"project": "atlas"},
            json.dumps({"type": "text", "text": MCP_CHANGES}),
        ),
        _launch("t4", "release-notes"),
        _step("t5", "Write", {"file_path": "/workspace/RELEASE_NOTES.md", "content": NOTES}),
    )


def _e08_codex() -> dict[str, Any]:
    mcp_output = "Wall time: 0.0010 seconds\nOutput:\n" + json.dumps([{"type": "text", "text": MCP_CHANGES}])
    return _codex(
        _codex_read("c1", "changelog-collect"),
        _exec("c2", "cat input/changes.txt", CHANGES),
        _step("c3", "mcp__reltools__list_changes", {"project": "atlas"}, mcp_output),
        _codex_read("c4", "release-notes"),
        _exec("c5", f"cat > RELEASE_NOTES.md <<'MD'\n{NOTES}MD"),
    )


@pytest.mark.parametrize("trajectory", [_e08_claude(), _e08_codex()], ids=["claude-code", "codex"])
@pytest.mark.parametrize(
    ("value", "passes"),
    [('say "hi"', True), ("C:\\rk\\out", True), ("café-7", True), ('ok "go"', True), ("AT-48", False)],
)
def test_values_with_quotes_and_backslashes_pass_and_a_prefix_of_a_longer_id_does_not(
    trajectory: dict[str, Any], value: str, passes: bool
) -> None:
    assert _passes(trajectory, value=value) is passes


def test_one_read_of_several_skill_files_does_not_open_a_shared_window() -> None:
    """check-18 e02: release-notes never does anything after a batched read."""
    tag = "1.5.0+rk.23b59e"
    compute = json.dumps({"build_tag": tag})
    claude = _claude(
        _bash("t1", f"cat {SKILLS}/version-bump/SKILL.md {SKILLS}/release-notes/SKILL.md", "# vb\n# rn\n"),
        _step("t2", "mcp__plugin_release-kit_reltools__compute_version", {"current": "1.4.2"}, compute),
        _bash("t3", f"echo '{tag}' > out/version.txt"),
    )
    codex = _codex(
        _codex_read("c1", "version-bump", "release-notes"),
        _step("c2", "mcp__reltools__compute_version", {"current": "1.4.2"}, compute),
        _exec("c3", f"echo '{tag}' > out/version.txt"),
    )
    for trajectory in (claude, codex):
        block = _handoff(trajectory, {"producer": "Skill:version-bump", "consumer": RN, "value": tag})
        assert block["passed"] == 0, trajectory["agent"]
        assert "read together with other skills" in block["failures"][0]["detail"]


# ---------------------------------------------------------------------------
# Windows (L22; check-18 e03, e04)
# ---------------------------------------------------------------------------


def test_another_skill_load_does_not_take_the_producers_write() -> None:
    """check-18 e03: a non-member skill load (Claude) or a peek at the consumer (Codex) before the save."""
    changes = json.dumps([{"id": "AT-4821"}])
    claude = _claude(
        _launch("t1", "changelog-collect"),
        _step("t2", "mcp__plugin_release-kit_reltools__list_changes", {"project": "atlas"}, changes),
        _launch("t3", "mcp_tool"),
        _step("t4", "Write", {"file_path": "/workspace/out/changes.json", "content": changes}),
        _launch("t5", "release-notes"),
        _step("t6", "Read", {"file_path": "/workspace/out/changes.json"}, f"1\t{changes}"),
    )
    codex = _codex(
        _codex_read("c1", "changelog-collect"),
        _step("c2", "mcp__reltools__list_changes", {"project": "atlas"}, changes),
        _codex_read("c3", "release-notes"),
        _exec("c4", f"cat > out/changes.json <<'EOF'\n{changes}\nEOF"),
        _exec("c5", "cat out/changes.json", changes),
    )
    for trajectory in (claude, codex):
        assert _passes(trajectory, artifact="out/changes.json"), trajectory["agent"]


def test_a_third_component_still_closes_the_producer_window() -> None:
    trajectory = _claude(
        _launch("t1", "changelog-collect"),
        _launch("t2", "version-bump"),
        _step("t3", "Write", {"file_path": "/workspace/out/changes.json", "content": "[]"}),
        _launch("t4", "release-notes"),
        _step("t5", "Read", {"file_path": "/workspace/out/changes.json"}, "1\t[]"),
    )
    assert not _passes(trajectory, artifact="out/changes.json")


def test_a_subagent_keeps_its_own_window_and_works_for_the_skill_that_started_it() -> None:
    """Harbor 0.20+ shape: sidechain steps are the subagent's own calls (check-18 e04)."""
    trajectory = _claude(
        _launch("t1", "changelog-collect"),
        _step("t2", "Agent", {"subagent_type": "general-purpose", "prompt": "Save the changes."}, "Saved."),
        {"source": "user", "message": "Save the changes.", "extra": {"is_sidechain": True}},
        _step("t3", "Write", {"file_path": "/workspace/out/changes.json", "content": "[]"}, is_sidechain=True),
        _launch("t4", "version-bump", is_sidechain=True),
        _step("t5", "Write", {"file_path": "/workspace/out/version.json", "content": "{}"}, is_sidechain=True),
        _step("t6", "Write", {"file_path": "/workspace/out/more.json", "content": "[]"}),
        _launch("t7", "release-notes"),
        _step("t8", "Read", {"file_path": "/workspace/out/changes.json"}, "1\t[]"),
        _step("t9", "Read", {"file_path": "/workspace/out/version.json"}, "1\t{}"),
        _step("t10", "Read", {"file_path": "/workspace/out/more.json"}, "1\t[]"),
    )
    # The subagent starts in the parent's changelog-collect window, so its first write is the producer's.
    assert _passes(trajectory, artifact="out/changes.json")
    # After the subagent switched to version-bump, its writes are version-bump's.
    assert not _passes(trajectory, artifact="out/version.json")
    # The subagent's switch does not move the parent out of changelog-collect's window.
    assert _passes(trajectory, artifact="out/more.json")


def test_concurrent_subagents_keep_separate_windows() -> None:
    """Harbor 0.24 interleaves concurrent subagents by time and tags each step with ``agent_id``."""
    trajectory = _claude(
        _launch("t1", "changelog-collect"),
        _step("t2", "Agent", {"subagent_type": "general-purpose", "prompt": "Bump."}, "Done."),
        _step("t3", "Agent", {"subagent_type": "general-purpose", "prompt": "Save."}, "Done."),
        _launch("a1", "version-bump", is_sidechain=True, agent_id="a-1"),
        _step(
            "b1",
            "Write",
            {"file_path": "/workspace/out/changes.json", "content": "[]"},
            is_sidechain=True,
            agent_id="b-2",
        ),
        _launch("t4", "release-notes"),
        _step("t5", "Read", {"file_path": "/workspace/out/changes.json"}, "1\t[]"),
    )
    # Subagent b-2 still works in the parent's changelog-collect window; a-1's switch is its own.
    assert _passes(trajectory, artifact="out/changes.json")


def test_a_subagent_producer_owns_the_writes_of_its_sidechain() -> None:
    trajectory = _claude(
        _step("t1", "Agent", {"subagent_type": "release-kit:release-reviewer", "prompt": "Review."}, "Done."),
        {"source": "user", "message": "Review.", "extra": {"is_sidechain": True}},
        _step("t2", "Write", {"file_path": "/workspace/out/review.json", "content": "{}"}, is_sidechain=True),
        _launch("t3", "release-notes"),
        _step("t4", "Read", {"file_path": "/workspace/out/review.json"}, "1\t{}"),
    )
    block = _handoff(trajectory, {"producer": "Agent:release-reviewer", "consumer": RN, "artifact": "out/review.json"})
    assert block["passed"] == 1, block["failures"]


def _agent_result(call_id: str, agent_type: str, agent_id: str, *, structured: bool) -> dict[str, Any]:
    """A parent ``Agent`` result as Harbor 0.24 records it for Claude Code (saved check-19 live run)."""
    use = {"status": "completed", "prompt": "Go.", "agentId": agent_id, "agentType": agent_type, "content": []}
    text = f"Done.\n\nagentId: {agent_id} (use SendMessage with to: '{agent_id}' to continue this agent)"
    result: dict[str, Any] = {"source_call_id": call_id, "content": f"{text}\n\n[metadata] {json.dumps(use)}"}
    if structured:
        result["extra"] = {"tool_result_metadata": {"tool_use_result": use}}
    return result


@pytest.mark.parametrize("structured", [True, False], ids=["result-metadata", "metadata-text"])
@pytest.mark.parametrize("together", [True, False], ids=["one-step", "back-to-back"])
def test_parallel_subagents_are_credited_by_their_agent_id(structured: bool, together: bool) -> None:
    """The plugin agent and a general-purpose agent start together; the plugin agent writes the artifact."""
    calls = [
        ("p1", {"subagent_type": "release-kit:release-reviewer", "prompt": "Review."}, "release-kit:release-reviewer"),
        ("p2", {"subagent_type": "general-purpose", "prompt": "Search."}, "general-purpose"),
    ]
    agent_ids = {"p1": "rev-1", "p2": "gp-2"}
    spawns: list[dict[str, Any]] = []
    for group in [calls] if together else [[call] for call in calls]:
        spawns.append(
            {
                "source": "agent",
                "tool_calls": [
                    {"tool_call_id": call_id, "function_name": "Agent", "arguments": args} for call_id, args, _ in group
                ],
                "observation": {
                    "results": [
                        _agent_result(call_id, agent_type, agent_ids[call_id], structured=structured)
                        for call_id, _, agent_type in group
                    ]
                },
                "extra": {"cwd": "/workspace", "is_sidechain": False},
            }
        )
    trajectory = _claude(
        *spawns,
        _step(
            "r1",
            "Write",
            {"file_path": "/workspace/out/review.json", "content": "{}"},
            is_sidechain=True,
            agent_id="rev-1",
        ),
        _step("g1", "Bash", {"command": "ls"}, "x", is_sidechain=True, agent_id="gp-2"),
        _launch("t3", "release-notes"),
        _step("t4", "Read", {"file_path": "/workspace/out/review.json"}, "1\t{}"),
    )

    block = _handoff(trajectory, {"producer": "Agent:release-reviewer", "consumer": RN, "artifact": "out/review.json"})
    assert block["passed"] == 1, block["failures"]
    # The general-purpose agent did not write it.
    other = _handoff(trajectory, {"producer": "Agent:general-purpose", "consumer": RN, "artifact": "out/review.json"})
    assert other["passed"] == 0


# ---------------------------------------------------------------------------
# Paths and shell forms (L22; check-18 e05, e06, e07, e10)
# ---------------------------------------------------------------------------


def test_paths_respect_the_working_directory() -> None:
    """check-18 e05: absolute vs relative, and the same name in another folder."""
    claude = _claude(
        _launch("t1", "changelog-collect"),
        _bash("t2", "cat > out/changes.json <<'EOF'\n[]\nEOF"),
        _step("t3", "Write", {"file_path": "/tmp/scratch/plan.json", "content": "{}"}),
        _launch("t4", "release-notes"),
        _bash("t5", "cat out/changes.json", "[]"),
        _step("t6", "Read", {"file_path": "/workspace/plan.json"}, '1\t{"old": true}'),
    )
    codex = _codex(
        _codex_read("c1", "changelog-collect"),
        _exec("c2", "cat > out/changes.json <<'EOF'\n[]\nEOF"),
        _exec("c3", "echo '{}' > /tmp/scratch/plan.json"),
        _codex_read("c4", "release-notes"),
        _exec("c5", "cat out/changes.json", "[]"),
        _exec("c6", "cat /workspace/plan.json", '{"old": true}'),
    )
    for trajectory in (claude, codex):
        assert _passes(trajectory, artifact="/workspace/out/changes.json"), trajectory["agent"]
        assert not _passes(trajectory, artifact="plan.json"), trajectory["agent"]


@pytest.mark.parametrize(
    ("write", "read"),
    [
        ("""python3 -c "open('out/a.json','w').write('[]')\"""", "cat out/a.json"),
        ("echo '[]' > out/b.json", """python3 -c "import json; print(json.load(open('out/b.json')))\""""),
        ("echo '[]' > out/c.json", "cd out && cat c.json"),
        ("sed -i 's/old/new/' out/d.json", "cat out/d.json"),
        ("echo '[]' > out/e.json", "cat out/*.json"),
        # Interpreter code fed as a here-document (the most common form in the saved live runs).
        ("python3 - <<'PY'\nimport json\nopen('out/f.json','w').write('[]')\nPY", "cat out/f.json"),
        (
            "python3 <<PY\nfrom pathlib import Path\nPath('/workspace/out/g.json').write_text('[]')\nPY",
            "cat out/g.json",
        ),
        ("echo '[]' > out/h.json", "python3 - <<'PY'\nimport json\nprint(json.load(open('out/h.json')))\nPY"),
        (
            "python3 - <<'PY'\nfrom pathlib import Path\nout = Path('/workspace/out/i.json')\nout.write_text('[]')\nPY",
            "python - <<'PY'\nfrom pathlib import Path\np=Path('/workspace/out/i.json')\nwith p.open(encoding='utf-8') as f:\n"
            "    print(f.read())\nPY",
        ),
        ("node - <<'JS'\nrequire('fs').writeFileSync('out/j.json', '[]')\nJS", "cat out/j.json"),
        ("cat <<'EOF' > out/k.json\n[]\nEOF", "wc -c out/k.json"),
    ],
    ids=[
        "python-write",
        "python-read",
        "cd",
        "sed-in-place",
        "glob",
        "heredoc-python-write",
        "heredoc-path-write",
        "heredoc-python-read",
        "heredoc-named-path",
        "heredoc-node-write",
        "heredoc-then-redirect",
    ],
)
def test_shell_writer_and_reader_forms_are_seen(write: str, read: str) -> None:
    """check-18 e06, on both harness shapes."""
    artifact = "out/" + next(name for name in "abcdefghijk" if f"out/{name}.json" in write + read) + ".json"
    claude = _claude(
        _launch("t1", "changelog-collect"), _bash("t2", write), _launch("t3", "release-notes"), _bash("t4", read, "[]")
    )
    codex = _codex(
        _codex_read("c1", "changelog-collect"),
        _exec("c2", write),
        _codex_read("c3", "release-notes"),
        _exec("c4", read, "[]"),
    )
    for trajectory in (claude, codex):
        assert _passes(trajectory, artifact=artifact), (trajectory["agent"], write, read)


def test_a_call_that_only_names_the_path_is_not_a_read() -> None:
    """check-18 e07: an MCP overwrite, ``rm``, and a subagent asked to delete the file."""
    trajectory = _claude(
        _launch("t1", "changelog-collect"),
        _step("t2", "Write", {"file_path": "/workspace/out/changes.json", "content": "[]"}),
        _step("t3", "Write", {"file_path": "/workspace/out/other.json", "content": "{}"}),
        _step("t4", "Write", {"file_path": "/workspace/out/third.json", "content": "{}"}),
        _launch("t5", "release-notes"),
        _step("t6", "mcp__plugin_release-kit_fsx__write_file", {"path": "out/changes.json", "content": "[]"}),
        _bash("t7", "rm out/other.json"),
        _step("t8", "Agent", {"subagent_type": "general-purpose", "prompt": "Delete /workspace/out/third.json now"}),
    )
    for artifact in ("out/changes.json", "out/other.json", "out/third.json"):
        assert not _passes(trajectory, artifact=artifact), artifact
    reader = _claude(
        _launch("t1", "changelog-collect"),
        _step("t2", "Write", {"file_path": "/workspace/out/changes.json", "content": "[]"}),
        _launch("t3", "release-notes"),
        _step("t4", "mcp__plugin_release-kit_fsx__read_file", {"path": "/workspace/out/changes.json"}, "[]"),
    )
    assert _passes(reader, artifact="out/changes.json")


def test_a_read_in_the_same_step_or_a_failed_read_does_not_count() -> None:
    """check-18 e10: a parallel write and read, and a read that says "No such file"."""
    same_step = _claude(
        _launch("t1", "changelog-collect"),
        {
            "source": "agent",
            "tool_calls": [
                {
                    "tool_call_id": "w",
                    "function_name": "Write",
                    "arguments": {"file_path": "/workspace/out/x.json", "content": "[]"},
                },
                {
                    "tool_call_id": "r",
                    "function_name": "mcp__plugin_release-kit_fsx__read_file",
                    "arguments": {"path": "out/x.json"},
                },
            ],
            "observation": {
                "results": [
                    {"source_call_id": "w", "content": "File created"},
                    {"source_call_id": "r", "content": "[]"},
                ]
            },
        },
        _launch("t2", "release-notes"),
    )
    assert not _handoff(same_step, {"producer": CC, "consumer": "MCP:fsx", "artifact": "out/x.json"})["passed"]

    failed_read = _claude(
        _launch("t1", "changelog-collect"),
        _step("t2", "Write", {"file_path": "/workspace/out/x.json", "content": "[]"}),
        _launch("t3", "release-notes"),
        _step("t4", "Read", {"file_path": "/workspace/out/y.json"}, "File does not exist."),
        _bash("t5", "cat out/x.json", "cat: out/x.json: No such file or directory"),
    )
    assert not _passes(failed_read, artifact="out/x.json")


# ---------------------------------------------------------------------------
# Components the arm cannot carry
# ---------------------------------------------------------------------------


def test_handoffs_whose_producer_or_consumer_the_arm_cannot_carry_are_skipped() -> None:
    """Codex loads no plugin subagents, and the member-skills arm stages no MCP servers."""
    skip = "this arm cannot carry that component type"
    typed = {"producer": "Agent:release-reviewer", "consumer": RN, "value": "AT-4821"}
    untyped = {"producer": RN, "consumer": "release-reviewer", "value": "AT-4821"}
    block = _handoff(_codex(_codex_read("c1", "release-notes")), typed, untyped)
    assert block["skipped"] == [
        {"producer": "Agent:release-reviewer", "consumer": RN, "reason": skip},
        {"producer": RN, "consumer": "release-reviewer", "reason": skip},
    ]
    assert (block["checked"], block["failures"], block["status"]) == (0, [], "not_applicable")

    mcp = {"producer": "MCP:reltools/list_changes", "consumer": RN, "value": "AT-4821"}
    members = {"producer": CC, "consumer": RN, "value": "AT-4821"}
    block = _handoff(_claude(_launch("t1", "release-notes")), mcp, members, arm=ARM_SUM_OF_PARTS)
    assert [item["producer"] for item in block["skipped"]] == ["MCP:reltools/list_changes"]
    assert (block["checked"], block["passed"], block["status"]) == (1, 0, "scored")


# ---------------------------------------------------------------------------
# Validation (L22; check-18 e09)
# ---------------------------------------------------------------------------


def test_self_handoff_is_rejected_and_control_characters_get_a_clear_message() -> None:
    problems = validate_plugin_case_fields({"id": "c", "handoffs": [{"producer": CC, "consumer": CC, "value": "v"}]})
    assert problems == [
        "handoffs[0]: producer and consumer must name different components (a self-handoff always passes)"
    ]
    problems = validate_plugin_case_fields(
        {"id": "c", "handoffs": [{"producer": CC, "consumer": RN, "artifact": "out/a\x07.json"}]}
    )
    assert problems == ["handoffs[0].artifact: must not contain control characters"]
