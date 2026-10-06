# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof L24 (PR #28 half): Integration verdict bands, the readiness gate and refusal wording.

- A plugin exactly as good as its parts (lift 0.00, interval [0.00, 0.00])
  was "inconclusive" because the interval includes zero (proof example
  check-20 edge-09); "cosmetic" could never be concluded for a real tie.
- "real" only needed the interval to exclude zero, not to clear +0.05.
- The readiness gate counted names that are not member skills (check-20
  edge-13), and a plugin with no member skills got no warning (neg-04).
- A refusal said "Plugin Integration is inconclusive", a verdict word for
  something that never ran.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from _f90_lift_fixtures import collect, metrics, payload
from click.testing import CliRunner

from skillevaluator import cli as cli_module
from skillevaluator.cli import _plugin_lift_mode_for_evidence
from skillevaluator.evaluation import EvaluationService
from skillevaluator.evaluation.tier3_report import _build_integration_report
from skillevaluator.tier3.plugin_eval import PluginEvalPackage, prepare_plugin_eval_package

CASES = [f"case-{index}" for index in range(1, 7)]
CONFIG = {
    "eval_target": {"kind": "plugin"},
    "skill_workspace": {"staged_skills": ["alpha", "beta"], "sum_of_parts_arm": True},
}


def _agent(low: float, high: float, *, n_cases: int = 12) -> dict:
    estimate = round((low + high) / 2, 4)
    return {
        "with_skill": 0.8,
        "sum_of_parts": round(0.8 - estimate, 4),
        "integration_completeness": {"complete": True},
        "lift_uncertainty": {
            "integration": {
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "confidence": 0.95,
                "n_cases": n_cases,
                "precision": "adequate",
                "ci_includes_zero": low <= 0 <= high,
            }
        },
    }


def test_a_real_tie_is_cosmetic_bundling(tmp_path: Path) -> None:
    # Proof check-20 edge-09: the plugin scores exactly like its parts.
    arms = {
        "with": {case: [metrics(0.65)] for case in CASES},
        "without": {case: [metrics(0.3)] for case in CASES},
        "sumofparts": {case: [metrics(0.65)] for case in CASES},
    }
    collect(tmp_path, {"claude-code": arms}, n_attempts=1)

    integration = payload(tmp_path, "both")["integration"]

    assert (integration["integration_lift"], integration["lift_uncertainty"]["ci_low"]) == (0.0, 0.0)
    assert integration["verdict"] == "cosmetic_bundling"
    assert integration["reason"] is None


@pytest.mark.parametrize(
    ("low", "high", "verdict", "reason_part"),
    [
        (0.06, 0.30, "real_integration", None),
        (0.01, 0.30, "inconclusive", "+0.05 band edge, so it cannot tell a real effect"),
        (-0.30, -0.06, "negative_integration", None),
        (-0.30, -0.01, "inconclusive", "-0.05 band edge, so it cannot tell a negative effect"),
        (-0.03, 0.04, "cosmetic_bundling", None),
        (-0.10, 0.30, "inconclusive", "includes zero"),
    ],
)
def test_the_verdict_needs_the_whole_interval_inside_its_band(
    low: float, high: float, verdict: str, reason_part: str | None
) -> None:
    report = _build_integration_report(_agent(low, high), CONFIG)

    assert report is not None
    assert report["verdict"] == verdict
    if reason_part is None:
        assert report["reason"] is None
    else:
        assert reason_part in report["reason"]


def _plugin(tmp_path: Path, expected_skills: list[str], *, members: tuple[str, ...] = ("alpha", "beta")) -> Path:
    plugin = tmp_path / "plugin"
    manifest = plugin / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"name": "public-plugin", "skills": "./skills"}), encoding="utf-8")
    for name in members:
        skill = plugin / "skills" / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Public test skill\n---\n# {name}\n\nUse this skill.\n", encoding="utf-8"
        )
    (plugin / "evals").mkdir()
    (plugin / "evals" / "evals.json").write_text(
        json.dumps(
            [
                {
                    "id": "composition",
                    "prompt": "Use both skills.",
                    "expected_skills": expected_skills,
                    "cross_component": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    return plugin


def test_readiness_gate_counts_only_member_skills(tmp_path: Path) -> None:
    # Proof check-20 edge-13: names that are not member skills unlocked Integration.
    members_case = prepare_plugin_eval_package(_plugin(tmp_path / "ok", ["alpha", "Beta"]), stage_root=tmp_path / "s1")
    strangers = prepare_plugin_eval_package(
        _plugin(tmp_path / "bad", ["not-a-member-a", "not-a-member-b"]), stage_root=tmp_path / "s2"
    )

    assert members_case.cross_component_case_count == 1
    assert strangers.cross_component_case_count == 0
    assert "member skills of the plugin" in str(strangers.integration_evidence_error())


def test_a_plugin_without_member_skills_is_warned_up_front() -> None:
    # Proof check-20 neg-04: an MCP-only plugin whose dataset still has a cross-component case.
    package = PluginEvalPackage(
        plugin_name="mcp-only",
        package_path=Path("/unused"),
        include_skills=(),
        unresolved_mcp_servers=(),
        runnable_mcp_servers=("docs",),
        dataset_case_count=2,
        cross_component_case_count=1,
    )

    effective, reason = _plugin_lift_mode_for_evidence(package, "both")

    assert effective == "effectiveness"
    assert reason is not None and "no member skills" in reason


def test_a_refused_integration_request_is_not_worded_as_a_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (tmp_path / "package").mkdir()
    prepared = SimpleNamespace(
        skipped=False,
        skip_reason=None,
        package_path=tmp_path / "package",
        include_skills=(),
        unresolved_skill_refs=(),
        unresolved_rule_refs=(),
        unresolved_mcp_servers=(),
        integration_evidence_error=lambda: "the plugin has no member skills",
        provenance=lambda: {"plugin_name": "plugin", "partial": False},
    )
    monkeypatch.setattr("skillevaluator.tier3.plugin_eval.prepare_plugin_eval_package", lambda *_a, **_k: prepared)
    monkeypatch.setattr(EvaluationService, "evaluate", lambda *_a, **_k: pytest.fail("evaluation must not start"))

    result = CliRunner().invoke(
        cli_module.cli, ["tier3", "evaluate-plugin", str(plugin), "--lift-mode", "integration", "--progress", "off"]
    )

    assert result.exit_code != 0
    assert "Plugin Integration was not run: the plugin has no member skills" in result.output
    assert "inconclusive" not in result.output
