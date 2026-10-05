# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof H5 (verifier half): an arm without the skill gets no skill metrics.

The no-plugin (no-skill) baseline cannot discover, run or route to a skill that
is not installed. Before the fix the verifier still scored ``skill_execution``
and ``skill_efficiency`` for it (about 0.5 and 0.4), so a plugin earned lift
from skill activation alone (proof example check-13 edge-04: +0.16 with the
same judge verdicts in both arms). Now both metrics are N/A in that arm, while
the sum-of-parts arm, which stages the member skills, keeps them.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from skillevaluator.tier3.eval_core import checks as shared_checks
from skillevaluator.tier3.harbor.metrics import metric_is_not_applicable, overall_score

_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
)

# A Claude Code shape (Skill tool, Bash) and a Codex shape (exec_command reading SKILL.md).
_CLAUDE_STEPS = [
    {"source": "user", "message": "Draft the release notes."},
    {
        "source": "agent",
        "message": "",
        "tool_calls": [
            {"tool_call_id": "c1", "function_name": "Skill", "arguments": {"skill": "release-notes"}},
            {"tool_call_id": "c2", "function_name": "Bash", "arguments": {"command": "git log --oneline"}},
        ],
    },
    {"source": "agent", "message": "Release notes drafted."},
]
_CODEX_STEPS = [
    {"source": "user", "message": "Draft the release notes."},
    {
        "source": "agent",
        "message": "",
        "tool_calls": [
            {
                "tool_call_id": "x1",
                "function_name": "exec_command",
                "arguments": {"cmd": "cat skills/release-notes/SKILL.md"},
            },
            {"tool_call_id": "x2", "function_name": "exec_command", "arguments": {"cmd": "git log --oneline"}},
        ],
    },
    {"source": "agent", "message": "Release notes drafted."},
]
_BASELINE_STEPS = [
    {"source": "user", "message": "Draft the release notes."},
    {
        "source": "agent",
        "message": "",
        "tool_calls": [{"tool_call_id": "b1", "function_name": "Bash", "arguments": {"command": "git log"}}],
    },
    {"source": "agent", "message": "Release notes drafted."},
]


def _load_verifier(tmp_path: Path, *, steps: list[dict[str, Any]] | None, entry: dict[str, Any]) -> ModuleType:
    module_name = f"harbor_eval_f90_lift_{tmp_path.name}"
    spec = importlib.util.spec_from_file_location(module_name, _TEMPLATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    logs = tmp_path / "logs"
    (logs / "agent").mkdir(parents=True)
    (logs / "verifier").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    module.LOGS_DIR = logs
    module.AGENT_LOGS_DIR = logs / "agent"
    module.VERIFIER_DIR = logs / "verifier"
    module.TESTS_DIR = tmp_path / "tests"
    module.ATIF_PATH = logs / "agent" / "trajectory.json"
    module.ENTRY_PATH = tmp_path / "tests" / "entry.json"
    module.REWARD_JSON = logs / "verifier" / "reward.json"
    module.REWARD_TXT = logs / "verifier" / "reward.txt"
    module.SKILL_EVALUATOR_REWARD_JSON = logs / "verifier" / "skill_evaluator_reward.json"
    if steps is not None:
        module.ATIF_PATH.write_text(json.dumps({"schema_version": "ATIF-v1.6", "steps": steps}), encoding="utf-8")
    module.ENTRY_PATH.write_text(json.dumps(entry), encoding="utf-8")
    return module


def _entry(*, has_skill: bool, workspace_skill_names: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": "rk-cobalt-notes-002",
        "question": "Draft the release notes.",
        "ground_truth": "Release notes drafted.",
        "expected_behavior": ["Collect the changes", "Write the notes"],
        "expected_skill": "release-notes",
        "acceptable_skills": ["changelog-collect"],
        "evaluated_skill": "release-kit",
        "has_skill": has_skill,
        "skill_workspace_mode": "group",
        "workspace_skill_names": workspace_skill_names or [],
    }


def _run(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], dict[str, Any]]:
    def judge(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"score": 0.5, "reason": "replayed judge verdict"}

    monkeypatch.setattr(module, "judge_accuracy", judge)
    monkeypatch.setattr(module, "judge_goal_accuracy", judge)
    monkeypatch.setattr(module, "judge_behavior_check", judge)
    module.main()
    rich = json.loads(module.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    numeric = json.loads(module.REWARD_JSON.read_text(encoding="utf-8"))
    return rich, numeric


@pytest.mark.parametrize("steps", [_BASELINE_STEPS, _CLAUDE_STEPS, _CODEX_STEPS], ids=["plain", "claude", "codex"])
def test_no_plugin_arm_records_skill_metrics_as_not_applicable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: list[dict[str, Any]]
) -> None:
    module = _load_verifier(tmp_path, steps=steps, entry=_entry(has_skill=False))

    rich, numeric = _run(module, monkeypatch)

    for metric in ("skill_execution", "skill_efficiency"):
        assert rich[metric] is None
        assert rich["details"][metric]["status"] == "not_applicable"
        assert "no skill under test" in rich["details"][metric]["reason"]
        assert metric not in numeric  # reward.json stays numeric-only
        assert metric_is_not_applicable(rich, metric)
    # The overall is the mean of security and the three judged metrics only.
    assert rich["security"] == 1.0
    assert numeric["overall"] == pytest.approx((1.0 + 0.5 + 0.5 + 0.5) / 4, abs=1e-4)
    assert overall_score(rich) == pytest.approx(numeric["overall"], abs=1e-4)


def test_no_plugin_arm_without_a_trajectory_keeps_skill_metrics_not_applicable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_verifier(tmp_path, steps=None, entry=_entry(has_skill=False))

    rich, numeric = _run(module, monkeypatch)

    assert rich["skill_execution"] is None and rich["skill_efficiency"] is None
    assert metric_is_not_applicable(rich, "skill_execution")
    assert metric_is_not_applicable(rich, "skill_efficiency")
    assert rich["security"] == 0
    assert "skill_execution" not in numeric


@pytest.mark.parametrize("steps", [_CLAUDE_STEPS, _CODEX_STEPS], ids=["claude", "codex"])
def test_plugin_arm_and_member_skills_arm_keep_their_skill_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: list[dict[str, Any]]
) -> None:
    plugin_arm = _load_verifier(tmp_path / "with", steps=steps, entry=_entry(has_skill=True))
    members_arm = _load_verifier(
        tmp_path / "members",
        steps=steps,
        entry=_entry(has_skill=False, workspace_skill_names=["changelog-collect", "release-notes", "version-bump"]),
    )

    for module in (plugin_arm, members_arm):
        rich, numeric = _run(module, monkeypatch)
        assert isinstance(rich["skill_execution"], int | float)
        assert isinstance(rich["skill_efficiency"], int | float)
        assert "skill_execution" in numeric and "skill_efficiency" in numeric
        assert not metric_is_not_applicable(rich, "skill_execution")


def test_template_and_host_mirror_agree_on_when_skill_metrics_apply() -> None:
    spec = importlib.util.spec_from_file_location("harbor_eval_f90_lift_mirror", _TEMPLATE)
    assert spec and spec.loader
    template = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(template)

    cases = itertools.product(
        (True, False, None),
        (True, False, None),
        ("release-notes", ""),
        ([], ["changelog-collect"]),
        ("release-kit", ""),
        ([], ["release-notes"], ["Changelog-Collect"], ["release-kit"], ["unrelated"]),
    )
    seen = set()
    for has_skill, should_trigger, expected, acceptable, evaluated, staged in cases:
        args = (has_skill, should_trigger, expected, acceptable)
        kwargs = {"evaluated_skill": evaluated, "workspace_skill_names": staged}
        verdict = template.skill_metrics_applicable(*args, **kwargs)
        assert shared_checks.skill_metrics_applicable(*args, **kwargs) is verdict
        seen.add(verdict)
    assert seen == {True, False}
    assert template.SKILL_METRICS_NOT_APPLICABLE_REASON == shared_checks.SKILL_METRICS_NOT_APPLICABLE_REASON
    # The no-plugin baseline: nothing staged, so nothing to score.
    assert shared_checks.skill_metrics_applicable(False, True, "release-notes", ["changelog-collect"]) is False
    # A negative case in the member-skills arm: the plugin under test is not one of the members.
    assert (
        shared_checks.skill_metrics_applicable(
            False, False, "", [], evaluated_skill="release-kit", workspace_skill_names=["release-notes"]
        )
        is False
    )
