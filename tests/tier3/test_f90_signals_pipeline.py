# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-signal checks through the real entry points: ``tier3 validate``, plugin staging, the runner context.

Covers the validation halves of proof L20, L22, L23, L26 and M29 (fields that
could never pass are refused before any agent runs) and the namespace half of
M7 on the path the runner and collector really take.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.tier3.eval_core.plugin_signals import ARM_WITH_SKILL, compute_plugin_signals
from skillevaluator.tier3.evals_spec import validate_tier3_source
from skillevaluator.tier3.harbor import runner
from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package

NEVER_PASSES = {
    "expected_order": [["Skill:alpha", "Skill:alpha"]],
    "handoffs": [{"producer": "Skill:alpha", "consumer": "skill:ALPHA", "value": "v"}],
    "conflict_probes": [
        {"id": "p", "must_use": "Skill:alpha", "must_not_use": "rule:policy"},
        {"id": "q", "must_use": "Skill:alpha", "must_not_use": "Skill:alpha"},
    ],
    "expected_tools": ["MCP:github/*"],
    "decoy_tools": ["MCP:github/delete_repo"],
    "acceptable_tools": ["Skil:alpha"],
}
EXPECTED_ERRORS = (
    "acceptable_tools[0]: unknown ref prefix 'Skil:'",
    "decoy_tools[0]: overlaps expected_tools ref 'MCP:github/*'",
    "expected_order[0]: can never pass",
    "handoffs[0]: producer and consumer must name different components",
    "conflict_probes[0].must_not_use: 'rule:' refs are not supported",
)


def _skill_with_dataset(tmp_path: Path, entry: dict[str, Any]) -> Path:
    skill = tmp_path / "source-skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: source-skill\ndescription: test\n---\n", encoding="utf-8")
    (skill / "evals" / "evals.json").write_text(
        json.dumps(
            {"skill_name": "source-skill", "evals": [{"id": "c1", "prompt": "p", "expected_output": "d", **entry}]}
        ),
        encoding="utf-8",
    )
    return skill


def _plugin_with_dataset(tmp_path: Path, entry: dict[str, Any]) -> Path:
    plugin = tmp_path / "plugin"
    manifest = plugin / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"name": "public-plugin", "skills": "./skills"}), encoding="utf-8")
    skill = plugin / "skills" / "alpha"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: Public test skill\n---\n# alpha\n", encoding="utf-8"
    )
    (plugin / "evals").mkdir()
    (plugin / "evals" / "evals.json").write_text(json.dumps([{"id": "c1", "prompt": "p", **entry}]), encoding="utf-8")
    return plugin


def test_tier3_validate_reports_fields_that_could_never_pass(tmp_path: Path) -> None:
    _source, checks = validate_tier3_source(_skill_with_dataset(tmp_path, NEVER_PASSES))

    errors = [check.message for check in checks if check.status == "error"]
    for expected in EXPECTED_ERRORS:
        assert any(expected in message for message in errors), (expected, errors)


@pytest.mark.parametrize("field", sorted(set(NEVER_PASSES) - {"expected_tools"}))
def test_plugin_staging_refuses_each_field_that_could_never_pass(tmp_path: Path, field: str) -> None:
    entry = {field: NEVER_PASSES[field]}
    if field == "decoy_tools":
        # A decoy is only wrong next to the expected ref it overlaps.
        entry["expected_tools"] = NEVER_PASSES["expected_tools"]
    with pytest.raises(ValueError, match="Invalid plugin signal fields"):
        prepare_plugin_eval_package(_plugin_with_dataset(tmp_path, entry), stage_root=tmp_path / "stage")


def test_runner_context_carries_the_plugin_namespace_to_the_graders(tmp_path: Path) -> None:
    package = tmp_path / "release-kit-plugin-eval"
    (package / "evals" / "environment").mkdir(parents=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    context = runner._plugin_signals_context(
        skill_path=package,
        evaluator_skill_path=package,
        workspace_skills=[tmp_path / "skills" / "release-notes"],
        run_dir=run_dir,
        baseline_has_members=False,
    )
    declared = context.declared_for(ARM_WITH_SKILL)
    assert declared["plugin"] == ["release-kit"]

    trajectory = {
        "agent": {"name": "claude-code"},
        "steps": [
            {"source": "user", "message": "Write the notes."},
            {
                "source": "agent",
                "tool_calls": [
                    {"tool_call_id": "t1", "function_name": "Skill", "arguments": {"skill": "acme-tools:release-notes"}}
                ],
                "observation": {"results": [{"source_call_id": "t1", "content": "Launching skill"}]},
            },
        ],
    }
    signals = compute_plugin_signals(trajectory, {}, declared=declared, wrapper_skills=context.wrapper_skills)
    assert signals is not None
    assert signals["activation_coverage"]["exercised"] == []
    assert signals["activation_coverage"]["unverified"] == ["skill:release-notes"]
