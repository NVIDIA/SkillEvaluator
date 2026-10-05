# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 3 coverage reasons follow the resolved plugin-load plan.

A native arm stages rules as the harness's own rules and (for Claude Code)
stages hooks, subagents, and commands itself, so its coverage rows must not
describe the generated wrapper. Each test prepares a small plugin the way
``tier3 evaluate-plugin`` does; nothing starts Harbor, Docker, or a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package
from skillevaluator.tier3.plugin_native import apply_load_census

_HOOKS = {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "printf 'stop\\n' >> /tmp/stop.log"}]}]}}
_FILES: dict[str, Any] = {
    ".claude-plugin/plugin.json": {"name": "demo", "version": "1.0.0", "description": "Demo plugin"},
    "skills/alpha/SKILL.md": "---\nname: alpha\ndescription: Demo skill alpha\n---\n# alpha\nUse it.\n",
    "rules/style.md": "Write short sentences.\n",
    "hooks/hooks.json": _HOOKS,
    "agents/reviewer.md": "---\nname: reviewer\ndescription: Reviews code\n---\nReview the code.\n",
    "evals/evals.json": [{"id": "c1", "prompt": "Use the plugin.", "expected_output": "Done."}],
}


def _plugin(root: Path) -> Path:
    for rel, content in _FILES.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return root


def _rows(tmp_path: Path, plugin_load: str, agents: str | None) -> dict[str, dict[str, Any]]:
    stage = tmp_path / f"stage-{plugin_load}-{(agents or 'none').replace(',', '-')}"
    kwargs: dict[str, Any] = {"agents": agents, "env_mode": "docker"} if agents else {}
    package = prepare_plugin_eval_package(
        _plugin(tmp_path / "demo"), stage_root=stage, plugin_load=plugin_load, **kwargs
    )
    provenance = package.provenance()
    return {row["type"]: row for row in provenance["component_coverage"]["components"]}


def test_native_claude_code_rows_describe_native_staging_not_the_wrapper(tmp_path: Path) -> None:
    rows = _rows(tmp_path, "native", "claude-code")

    assert rows["rule"]["reason"] == "staged as a native rule for claude-code"
    assert (rows["hook"]["state"], rows["hook"]["reason"]) == ("staged", "staged natively for claude-code")
    assert (rows["agent"]["state"], rows["agent"]["reason"]) == ("staged", "staged natively for claude-code")
    for row in rows.values():
        assert "wrapper" not in row["reason"]
        assert "does not stage hook components yet" not in row["reason"]


@pytest.mark.parametrize(("agent", "adapter_id"), [("codex", "codex-home")])
def test_a_native_arm_that_cannot_load_hooks_names_its_adapter(tmp_path: Path, agent: str, adapter_id: str) -> None:
    rows = _rows(tmp_path, "native", agent)

    assert rows["rule"]["reason"] == f"staged as a native rule for {agent}"
    assert rows["hook"]["state"] == "unsupported"
    assert (
        rows["hook"]["reason"] == f"not staged for {agent} (unsupported by the {agent} native adapter ({adapter_id}))"
    )


def test_opencode_stages_subagents_but_not_hooks(tmp_path: Path) -> None:
    rows = _rows(tmp_path, "native", "opencode")

    assert (rows["agent"]["state"], rows["agent"]["reason"]) == ("staged", "staged natively for opencode")
    assert "unsupported by the opencode native adapter (opencode-config)" in rows["hook"]["reason"]


def test_mixed_native_arms_say_which_arm_stages_what(tmp_path: Path) -> None:
    rows = _rows(tmp_path, "native", "claude-code,codex")

    assert rows["rule"]["reason"] == "staged as a native rule for claude-code, codex"
    assert rows["hook"]["state"] == "staged"
    assert rows["hook"]["reason"] == (
        "staged natively for claude-code; not staged for codex (unsupported by the codex native adapter (codex-home))"
    )


def test_wrapper_rows_point_at_native_loading_for_hooks(tmp_path: Path) -> None:
    rows = _rows(tmp_path, "wrapper", None)

    assert rows["rule"]["reason"] == "rule embedded in the generated wrapper SKILL.md"
    assert rows["hook"]["state"] == "unsupported"
    assert rows["hook"]["reason"] == (
        "the generated wrapper does not stage hook components; --plugin-load native stages them for claude-code"
    )


def test_a_census_listing_does_not_repeat_the_planned_native_staging(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    package = prepare_plugin_eval_package(
        _plugin(tmp_path / "demo"), stage_root=stage, plugin_load="native", agents="claude-code", env_mode="docker"
    )
    coverage = package.provenance()["component_coverage"]
    hook = next(row for row in coverage["components"] if row["type"] == "hook")
    census = {"claude-code": {"mode": "native", "listed": [{"type": "hook", "name": hook["name"], "evidence": "x"}]}}
    plugin_load = {"by_agent": {"claude-code": {"mode": "native", "components": {"hook": "native"}}}}

    promoted = apply_load_census(coverage, census, plugin_load)

    reason = next(row["reason"] for row in promoted["components"] if row["type"] == "hook")
    assert reason.count("staged natively for claude-code") == 1
    assert "the load census listed it (x)" in reason


def test_a_census_listing_does_not_repeat_the_planned_native_rule(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    package = prepare_plugin_eval_package(
        _plugin(tmp_path / "demo"), stage_root=stage, plugin_load="native", agents="claude-code", env_mode="docker"
    )
    coverage = package.provenance()["component_coverage"]
    rule = next(row for row in coverage["components"] if row["type"] == "rule")
    census = {"claude-code": {"mode": "native", "listed": [{"type": "rule", "name": rule["name"], "evidence": "x"}]}}
    plugin_load = {"by_agent": {"claude-code": {"mode": "native", "components": {"rule": "native"}}}}

    promoted = apply_load_census(coverage, census, plugin_load)

    reason = next(row["reason"] for row in promoted["components"] if row["type"] == "rule")
    assert (
        reason
        == "staged as a native rule for claude-code; the load census listed it (x) but the harness did not confirm it was loaded"
    )


def test_rows_record_the_agents_that_stage_them_natively(tmp_path: Path) -> None:
    rows = _rows(tmp_path, "native", "claude-code,codex")

    assert rows["rule"]["native_agents"] == ["claude-code", "codex"]
    assert rows["hook"]["native_agents"] == ["claude-code"]
    assert "native_agents" not in _rows(tmp_path, "wrapper", None)["rule"]


def test_a_census_listing_by_some_planned_agents_does_not_repeat_the_plan(tmp_path: Path) -> None:
    rows = _rows(tmp_path, "native", "claude-code,codex")
    rule = rows["rule"]
    census = {"codex": {"mode": "native", "listed": [{"type": "rule", "name": rule["name"], "evidence": "x"}]}}
    plugin_load = {"by_agent": {"codex": {"mode": "native", "components": {"rule": "native"}}}}

    promoted = apply_load_census({"components": [rule]}, census, plugin_load)

    assert promoted["components"][0]["reason"] == (
        "staged as a native rule for claude-code, codex; the load census listed it (x) but the harness did not "
        "confirm it was loaded"
    )


def test_the_census_reads_the_recorded_agents_not_the_reason_text() -> None:
    census = {"claude-code": {"mode": "native", "listed": [{"type": "hook", "name": "h", "evidence": "x"}]}}
    plugin_load = {"by_agent": {"claude-code": {"mode": "native", "components": {"hook": "native"}}}}
    listed = "the load census listed it (x) but the harness did not confirm it was loaded"

    def reason(row: dict[str, Any]) -> str:
        return apply_load_census({"components": [row]}, census, plugin_load)["components"][0]["reason"]

    row = {"type": "hook", "name": "h", "path": "h", "state": "staged", "reason": "planned"}
    assert reason({**row, "native_agents": ["claude-code"]}) == f"planned; {listed}"
    assert reason(row) == f"planned; staged natively for claude-code; {listed}"
