# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Activation evidence: plugin namespaces, failed reads, and all-failed components (proof M7, L19, L21).

Shapes follow the saved Harbor ATIF trajectories: Claude Code native skills are
``Skill(<plugin>:<skill>)``, Codex reads ``<member>/SKILL.md`` with
``exec_command`` and prints ``Process exited with code N`` before the output.
"""

from __future__ import annotations

from typing import Any

from skillevaluator.reporting.plugin_sections import signals_view
from skillevaluator.tier3.eval_core.plugin_signals import (
    ARM_WITH_SKILL,
    build_plugin_signals_context,
    compute_plugin_signals,
    plugin_case_spec,
    summarize_plugin_signals,
)

MEMBERS = ["changelog-collect", "version-bump", "release-notes"]
CONTEXT = build_plugin_signals_context(
    member_skills=MEMBERS,
    mcp_servers=["reltools"],
    wrapper_skills=["release-kit-plugin-eval", "release-kit"],
)
DECLARED = CONTEXT.declared_for(ARM_WITH_SKILL)


def _step(call_id: str, fn: str, args: dict[str, Any], content: str) -> dict[str, Any]:
    return {
        "source": "agent",
        "tool_calls": [{"tool_call_id": call_id, "function_name": fn, "arguments": args}],
        "observation": {"results": [{"source_call_id": call_id, "content": content}]},
    }


def _trajectory(agent: str, *steps: dict[str, Any]) -> dict[str, Any]:
    return {"agent": {"name": agent}, "steps": [{"source": "user", "message": "Do the release."}, *steps]}


def _codex_exec(call_id: str, cmd: str, output: str, *, code: int = 0) -> dict[str, Any]:
    content = f"Chunk ID: c0\nWall time: 0.0000 seconds\nProcess exited with code {code}\nOutput:\n{output}"
    return _step(call_id, "exec_command", {"cmd": cmd, "workdir": "/workspace"}, content)


def _signals(trajectory: dict[str, Any], case: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("declared", DECLARED)
    kwargs.setdefault("wrapper_skills", CONTEXT.wrapper_skills)
    signals = compute_plugin_signals(trajectory, plugin_case_spec({"id": "c", **(case or {})}), **kwargs)
    assert signals is not None
    return signals


def _launch(call_id: str, name: str) -> dict[str, Any]:
    return _step(call_id, "Skill", {"skill": name}, f"Launching skill: {name}")


# ---------------------------------------------------------------------------
# Namespaces (check-17 e01-namespace-collision-cc)
# ---------------------------------------------------------------------------


def test_context_names_the_plugin_from_the_wrapper_package() -> None:
    assert DECLARED["plugin"] == ["release-kit"]
    explicit = build_plugin_signals_context(member_skills=MEMBERS, plugin_name="release-kit")
    assert explicit.declared_for(ARM_WITH_SKILL)["plugin"] == ["release-kit"]


def test_another_plugins_same_named_skill_is_not_this_plugins_skill() -> None:
    trajectory = _trajectory(
        "claude-code",
        _launch("t1", "release-kit:changelog-collect"),
        _launch("t2", "acme-tools:release-notes"),
    )

    signals = _signals(trajectory, {"expected_tools": ["Skill:changelog-collect", "Skill:release-notes"]})

    coverage = signals["activation_coverage"]
    assert coverage["exercised"] == ["skill:changelog-collect"]
    assert "skill:release-notes" in coverage["unverified"]
    assert [a["name"] for a in signals["activations"]] == ["changelog-collect", "acme-tools:release-notes"]
    assert signals["routing"]["recall"] == 0.5


def test_this_plugins_namespace_still_matches_bare_and_qualified_refs() -> None:
    claude = _trajectory("claude-code", _launch("t1", "release-kit:release-notes"))
    codex = _trajectory(
        "codex",
        _codex_exec("c1", "cat /tmp/agent-home/.agents/skills/release-notes/SKILL.md", "---\nname: release-notes\n"),
    )
    for trajectory in (claude, codex):
        for ref in ("Skill:release-notes", "Skill:release-kit:release-notes", "Skill:Release-Kit:*"):
            signals = _signals(trajectory, {"expected_tools": [ref]})
            assert signals["routing"]["recall"] == 1.0, (trajectory["agent"], ref)
        assert signals["activation_coverage"]["exercised"] == ["skill:release-notes"]


def test_one_skill_is_one_identity_under_both_spellings() -> None:
    """check-15 e08 claude-native-duplicate: P 2/3, not 1/2."""
    trajectory = _trajectory(
        "claude-code",
        _launch("t1", "release-kit:changelog-collect"),
        _step("t2", "mcp__plugin_release-kit_reltools__list_changes", {"project": "atlas"}, "[]"),
        _launch("t3", "release-kit:release-notes"),
        _launch("t4", "release-notes"),
    )

    selection = _signals(trajectory, {"expected_tools": ["Skill:changelog-collect", "MCP:reltools/list_changes"]})[
        "routing"
    ]

    assert [label for label in selection["called"] if label.startswith("Skill:")] == [
        "Skill:changelog-collect",
        "Skill:release-notes",
    ]


# ---------------------------------------------------------------------------
# Failed and skipped SKILL.md reads (check-17 e02, check-15 e07)
# ---------------------------------------------------------------------------


def test_failed_codex_skill_read_is_not_an_activation_that_succeeded() -> None:
    trajectory = _trajectory(
        "codex",
        _codex_exec(
            "c1",
            "cat /tmp/codex-home/skills/.system/release-notes/SKILL.md && printf '\\n---FILE---\\n' "
            "&& sed -n '1,220p' /workspace/input/sensors.csv",
            "cat: /tmp/codex-home/skills/.system/release-notes/SKILL.md: No such file or directory\n",
            code=1,
        ),
    )

    signals = _signals(trajectory)

    assert [(a["name"], a["succeeded"]) for a in signals["activations"]] == [("release-notes", False)]
    coverage = signals["activation_coverage"]
    assert "skill:release-notes" not in coverage["exercised"]
    assert coverage["unavailable"] == ["skill:release-notes"]


def test_read_after_a_failed_and_chain_part_never_ran() -> None:
    """``cat missing && printf && cat member``: the second cat never ran."""
    trajectory = _trajectory(
        "codex",
        _codex_exec(
            "c1",
            "cat /tmp/codex-home/skills/.system/version-bump/SKILL.md && printf '\\n---\\n' "
            "&& cat /tmp/agent-home/.agents/skills/changelog-collect/SKILL.md",
            "cat: /tmp/codex-home/skills/.system/version-bump/SKILL.md: No such file or directory\n",
            code=1,
        ),
    )

    signals = _signals(
        trajectory, {"expected_tools": ["Skill:changelog-collect"], "decoy_tools": ["Skill:version-bump"]}
    )

    assert [(a["name"], a["succeeded"]) for a in signals["activations"]] == [("version-bump", False)]
    assert signals["activation_coverage"]["exercised"] == []
    assert signals["routing"]["decoy_calls"] == 0
    assert signals["routing"]["recall"] == 0.0


def test_a_good_read_in_a_list_that_also_fails_elsewhere_still_counts() -> None:
    trajectory = _trajectory(
        "codex",
        _codex_exec(
            "c1",
            "cat /tmp/agent-home/.agents/skills/changelog-collect/SKILL.md; cat /tmp/x/version-bump/SKILL.md",
            "---\nname: changelog-collect\n---\ncat: /tmp/x/version-bump/SKILL.md: No such file or directory\n",
            code=1,
        ),
    )

    activations = _signals(trajectory)["activations"]

    assert [(a["name"], a["succeeded"]) for a in activations] == [
        ("changelog-collect", True),
        ("version-bump", False),
    ]


def test_cp_and_stdin_to_a_non_reader_do_not_load_a_skill() -> None:
    """check-15 e07: ``cp`` sources and ``grep x < SKILL.md`` are not skill loads."""
    path = "/tmp/agent-home/.agents/skills/version-bump/SKILL.md"
    for cmd in (f"cp {path} /tmp/vb.md", f"grep -n bump < {path}", f"sed -i 's/a/b/' {path}"):
        trajectory = _trajectory("codex", _codex_exec("c1", cmd, ""))
        signals = _signals(trajectory, {"decoy_tools": ["Skill:version-bump"]})
        assert signals["activations"] == [], cmd
        assert signals["routing"]["decoy_calls"] == 0, cmd
    claude = _trajectory("claude-code", _step("t1", "Bash", {"command": f"cp {path} /tmp/vb.md"}, ""))
    assert _signals(claude)["activations"] == []
    reader = _trajectory("codex", _codex_exec("c1", f"cat < {path}", "---\nname: version-bump\n"))
    assert [a["name"] for a in _signals(reader)["activations"]] == ["version-bump"]


# ---------------------------------------------------------------------------
# Every activation failed (check-17 p07)
# ---------------------------------------------------------------------------


def _all_failed_trial() -> dict[str, Any]:
    return _trajectory(
        "claude-code",
        _step(
            "t1",
            "Skill",
            {"skill": "release-kit:version-bump"},
            "<tool_use_error>Unknown skill: release-kit:version-bump</tool_use_error>",
        ),
        _step(
            "t2",
            "mcp__plugin_release-kit_reltools__compute_version",
            {"current": "2.9.0", "bump": "major"},
            "MCP error -32000: Connection closed",
        ),
    )


def test_components_whose_every_activation_failed_are_unavailable_not_exercised() -> None:
    per_trial = [_signals(_all_failed_trial()) for _ in range(3)]

    coverage = per_trial[0]["activation_coverage"]
    assert coverage["exercised"] == []
    assert coverage["unavailable"] == ["skill:version-bump", "mcp:reltools"]

    summary = summarize_plugin_signals(per_trial)["activation_coverage"]
    assert summary["exercised"] == []
    assert summary["unavailable"] == ["skill:version-bump", "mcp:reltools"]
    assert "skill:version-bump" not in summary["unverified"]
    assert summary["exercise_rate"]["skill:version-bump"] == 0.0


def test_exercise_rate_counts_only_trials_that_really_exercised() -> None:
    good = _trajectory("claude-code", _launch("t1", "release-kit:version-bump"))
    summary = summarize_plugin_signals([_signals(good), _signals(_all_failed_trial())])["activation_coverage"]

    assert summary["exercised"] == ["skill:version-bump"]
    assert summary["unavailable"] == ["mcp:reltools"]
    assert summary["exercise_rate"]["skill:version-bump"] == 0.5


def test_report_does_not_count_all_failed_components_as_exercised() -> None:
    summary = summarize_plugin_signals([_signals(_all_failed_trial())])
    view = signals_view({"agents": {"claude-code": {"plugin_signals_summary": {"with_skill": summary}}}})

    assert view is not None
    activation = view["activation"]
    assert activation["exercised"] == []
    assert activation["unavailable"] == ["skill:version-bump", "mcp:reltools"]
    assert activation["summary"].startswith("0 of 4 declared components were exercised")
