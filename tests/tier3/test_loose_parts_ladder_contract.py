# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The seam between the loose-parts staging and the lift ladder statistics and report.

The staging (:mod:`skillevaluator.tier3.loose_parts`) writes the ``loose_parts``
info block and the per-case load failures; the statistics
(``stats.lift_ladder``), the report block (``tier3_report``) and the display
model (``plugin_sections.lift_ladder_view``) read them. Each test feeds the
staging's real output, built from the fake ``shipit`` plugin, to the real
reader, so a shape change on either side fails here. Nothing here starts
Harbor, Docker, or an agent.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _loose_parts_fixtures import make_plugin, native_source

from skillevaluator.evaluation.tier3_report import _build_ladder_report
from skillevaluator.reporting.plugin_sections import lift_ladder_view
from skillevaluator.tier3.harbor import stats
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS
from skillevaluator.tier3.harbor.stats import ArmObservations, TrialObservation
from skillevaluator.tier3.loose_parts import (
    LOOSE_PARTS_ARM,
    build_loose_parts,
    compare_content_hashes,
    load_failures_by_case,
    loose_parts_load_evidence,
)

NONE, SKILLS, LOOSE, FULL = "without_skill", "sum_of_parts", "loose_parts", "with_skill"
CROSS = [f"x-{index}" for index in range(1, 7)]
OTHER = ["o-1", "o-2", "o-3"]
CASES = [*CROSS, *OTHER]
COMPLETE = {"complete": True, "missing_cases": [], "failed_arms": [], "attempt_shortfall": []}
RUN_CONFIG = {"eval_target": {"kind": "plugin"}, "lift_mode": {"requested": "ladder", "effective": "ladder"}}
NATIVE = {"requested": "native", "by_agent": {"claude-code": {"mode": "native"}}}


@pytest.fixture(scope="module")
def loose(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("loose-contract")
    parts = build_loose_parts(native_source(make_plugin(root / "plugins"), root / "stage").native_source)
    assert parts.error is None
    return parts


@pytest.fixture(scope="module")
def info(loose) -> dict[str, Any]:
    return loose.info()


def _init(**overrides: Any) -> dict[str, Any]:
    """A Claude Code ``system/init`` event of the loose arm that lists every staged part."""
    event: dict[str, Any] = {
        "type": "system",
        "subtype": "init",
        "skills": ["changelog-collect", "version-bump"],
        "slash_commands": ["changelog-collect", "version-bump", "release-check"],
        "agents": ["Explore", "release-reviewer"],
        "mcp_servers": [{"name": "reltools", "status": "connected"}],
        "tools": ["Bash", "mcp__reltools__list_changes"],
        "plugins": [],
    }
    event.update(overrides)
    return event


def _reward(case_id: str, value: float, *, skill: bool = True) -> dict[str, Any]:
    reward = {"entry_id": case_id, "metric_set": DEFAULT_METRIC_SET, **dict.fromkeys(DEFAULT_METRICS, value)}
    if not skill:
        reward.update(skill_execution=None, skill_efficiency=None)
    return reward


def _arms() -> dict[str, ArmObservations]:
    """no plugin 0.4, skills only 0.6, loose parts 0.7, full plugin 0.8 on every case."""
    scores = {NONE: 0.4, SKILLS: 0.6, LOOSE: 0.7, FULL: 0.8}
    return {
        arm: ArmObservations(
            "succeeded", tuple(TrialObservation(case, _reward(case, value, skill=arm != NONE)) for case in CASES)
        )
        for arm, value in scores.items()
    }


def _failures(loose) -> dict[str, list[dict[str, Any]]]:
    """x-1: the loose arm lost its subagent and MCP server. x-2: the full arm's plugin was not listed."""
    ok = loose_parts_load_evidence([_init()], None, loose)
    lost = loose_parts_load_evidence(
        [_init(agents=["Explore"], mcp_servers=[{"name": "reltools", "status": "failed"}])], None, loose
    )
    full_missing = loose_parts_load_evidence([_init(plugins=[])], None, loose, arm=FULL)
    return load_failures_by_case(
        {
            LOOSE_PARTS_ARM: {"x-1": [lost], "x-2": [ok]},
            FULL: {"x-2": [full_missing]},
        }
    )


def _report(
    ladder: dict[str, Any] | None, loose_parts: dict[str, Any], agent_name: str = "claude-code"
) -> dict[str, Any]:
    agent = {"lift_ladder": ladder, "integration_completeness": {**COMPLETE, "loose_parts": COMPLETE}}
    provenance = {"plugin_load": NATIVE, "loose_parts": loose_parts}
    block = _build_ladder_report(agent, RUN_CONFIG, provenance, agent_name=agent_name)
    assert block is not None
    return block


# ---------------------------------------------------------------------------
# Load failures: staging -> statistics
# ---------------------------------------------------------------------------


def test_staging_load_failures_drop_the_case_from_every_ladder_arm(loose) -> None:
    failures = _failures(loose)
    assert sorted(failures) == ["x-1", "x-2"]

    ladder = stats.lift_ladder(_arms(), CROSS, expected_case_ids=CASES, load_failures_by_case=failures)

    excluded = {entry["case"]: entry for entry in ladder["excluded_cases"]}
    assert sorted(excluded) == ["x-1", "x-2"]
    assert ladder["n_cases"] == len(CASES) - 2
    assert ladder["partial"] is True
    for entry in excluded.values():
        assert entry["reason"] == stats.EXCLUDED_LOAD_FAILED
        assert entry["missing_arms"] == []
    # Each failure is kept under the arm it happened in, naming the piece and the attempt.
    assert set(excluded["x-1"]["load_failures"]) == {LOOSE}
    lost = excluded["x-1"]["load_failures"][LOOSE]
    assert "agent:release-reviewer (attempt 1)" in lost
    assert "mcp:reltools (attempt 1)" in lost
    assert "failed" in lost
    assert set(excluded["x-2"]["load_failures"]) == {FULL}
    assert "not listed in its init event" in excluded["x-2"]["load_failures"][FULL]
    # The glue loses the same cross-component cases, so its comparison is partial too.
    assert ladder["glue"]["n"] == len(CROSS) - 2
    assert sorted(ladder["glue"]["excluded_cases"]) == ["x-1", "x-2"]
    assert ladder["glue"]["partial"] is True


def test_build_agent_statistics_passes_staging_load_failures_through(loose) -> None:
    failures = _failures(loose)

    result = stats.build_agent_statistics(
        _arms(),
        expected_case_ids=CASES,
        n_attempts=1,
        stop_on_pass=False,
        pass_threshold=0.5,
        sum_of_parts_requested=True,
        cross_component_case_ids=frozenset(CROSS),
        load_failures_by_case=failures,
    )

    assert [entry["case"] for entry in result["lift_ladder"]["excluded_cases"]] == ["x-1", "x-2"]


def test_no_staging_load_failure_keeps_every_case(loose) -> None:
    ok = loose_parts_load_evidence([_init()], None, loose)
    failures = load_failures_by_case({LOOSE_PARTS_ARM: {case: [ok] for case in CASES}})
    assert failures == {}

    ladder = stats.lift_ladder(_arms(), CROSS, expected_case_ids=CASES, load_failures_by_case=failures)

    assert (ladder["n_cases"], ladder["excluded_cases"]) == (len(CASES), [])


def test_a_malformed_failure_record_never_crashes_and_still_drops_its_case() -> None:
    failures = {"x-1": [{"piece": "skill:alpha", "reason": "boom"}, "junk", {"arm": LOOSE, "attempt": "two"}]}

    ladder = stats.lift_ladder(_arms(), CROSS, expected_case_ids=CASES, load_failures_by_case=failures)

    (entry,) = ladder["excluded_cases"]
    assert entry["case"] == "x-1" and entry["reason"] == stats.EXCLUDED_LOAD_FAILED
    assert set(entry["load_failures"]) == {"unknown", LOOSE}


def test_many_failed_pieces_are_bounded_per_arm(loose) -> None:
    rows = [{"arm": LOOSE, "attempt": 1, "piece": f"skill:s{index}", "reason": "x" * 400} for index in range(40)]

    ladder = stats.lift_ladder(_arms(), CROSS, expected_case_ids=CASES, load_failures_by_case={"x-1": rows})

    reason = ladder["excluded_cases"][0]["load_failures"][LOOSE]
    assert len(reason) <= 1000
    assert "(+" in reason and "more)" in reason


# ---------------------------------------------------------------------------
# The info block: staging -> report block and display model
# ---------------------------------------------------------------------------


def test_the_report_reads_the_staging_info_block(info) -> None:
    ladder = stats.lift_ladder(_arms(), CROSS, expected_case_ids=CASES)

    block = _report(ladder, info)

    loose_parts = block["loose_parts"]
    assert info["equal_across_arms"] is True
    assert (loose_parts["equal_across_arms"], loose_parts["mismatches"]) == (True, [])
    glue_ids = [piece["id"] for piece in info["dropped_as_glue"]]
    assert glue_ids and "manifest:.claude-plugin/plugin.json" in glue_ids
    assert loose_parts["dropped_as_glue"] == glue_ids
    assert (loose_parts["rules"], loose_parts["mcp_servers"]) == (info["rules"], info["mcp_servers"])
    assert loose_parts["subagents"] == ["release-reviewer"]
    assert loose_parts["commands"] == ["release-check"]
    assert loose_parts["hooks"] == ["hooks/hooks.json"]
    assert loose_parts["content_hashes"] == info["content_hashes"]
    # Same capability, complete, 6 cross-component pairs at +0.10 everywhere: a real integration lift.
    assert block["glue"]["verdict"] == "real_integration", block["glue"]["reason"]


def test_report_parity_agrees_with_the_staging_on_a_changed_piece(loose, info) -> None:
    hashes = {arm: dict(pieces) for arm, pieces in info["content_hashes"].items()}
    hashes[LOOSE]["rule:release-policy.md"] = "0" * 64
    hashes[LOOSE].pop("command:release-check")
    staged = compare_content_hashes(hashes)
    changed = loose.info(content=staged)
    ladder = stats.lift_ladder(_arms(), CROSS, expected_case_ids=CASES)

    block = _report(ladder, changed)

    assert staged["equal_across_arms"] is False
    assert block["loose_parts"]["equal_across_arms"] is False
    assert {item["piece"] for item in block["loose_parts"]["mismatches"]} == set(staged["mismatches"])
    assert block["glue"]["verdict"] == "inconclusive"
    assert "capability not identical across arms" in block["glue"]["reason"]


def test_the_rendered_ladder_names_glue_by_id_and_every_staged_kind(info) -> None:
    ladder = stats.lift_ladder(_arms(), CROSS, expected_case_ids=CASES)
    block = _report(ladder, info)

    view = lift_ladder_view({"lift_ladder": block})

    assert view is not None
    entry = view["entries"][0]
    line = entry["loose_parts_line"]
    assert "{" not in line, line
    assert "rules: release-files.md, release-policy.md" in line
    assert "MCP servers: reltools" in line
    assert "subagents: release-reviewer" in line
    assert "commands: release-check" in line
    assert "hooks: hooks/hooks.json" in line
    assert "Plugin-only, counted as glue: " in line and "manifest:.claude-plugin/plugin.json" in line
    # v1 stages hooks as loose parts, so nothing may say they are glue.
    assert "hooks as glue" not in " ".join(str(value) for value in entry.values())
    for words in ("subagents", "commands", "hooks"):
        assert words in view["definition"], view["definition"]
    assert "Capability identical across arms: yes" in entry["capability_line"]


def test_an_agent_the_staging_refused_shows_the_staging_reason(loose, info) -> None:
    refused = build_loose_parts(loose.source, "codex")
    assert refused.error is not None and refused.error.startswith("not supported:")
    record = {**info, "agents": {"codex": refused.info(content=refused.content_report())}}

    block = _report(None, record, agent_name="codex")

    assert block["measured"] is False
    assert block["reason"] == refused.error[:300]
    # The refused agent inherits nothing from the measured agent's hashes.
    assert block["loose_parts"]["content_hashes"] is None
    assert block["loose_parts"]["equal_across_arms"] is None


def test_a_refused_agent_inherits_no_staged_parts_from_the_measured_agent(info) -> None:
    record = {**info, "agents": {"codex": {"not_measured_reason": "not supported: codex is not measured"}}}

    block = _report(None, record, agent_name="codex")

    assert (block["measured"], block["reason"]) == (False, "not supported: codex is not measured")
    loose_parts = block["loose_parts"]
    for key in ("rules", "mcp_servers", "subagents", "commands", "hooks", "dropped_as_glue"):
        assert loose_parts[key] == [], key
    assert (loose_parts["content_hashes"], loose_parts["equal_across_arms"]) == (None, None)


def test_the_report_never_calls_equal_what_the_staging_called_unequal(info) -> None:
    """The staging's hashing is the source of truth; the report's recomputation may only add mismatches."""
    record = {**info, "equal_across_arms": False, "mismatches": ["rule:release-policy.md"]}
    ladder = stats.lift_ladder(_arms(), CROSS, expected_case_ids=CASES)

    block = _report(ladder, record)

    assert block["loose_parts"]["equal_across_arms"] is False
    assert [item["piece"] for item in block["loose_parts"]["mismatches"]] == ["rule:release-policy.md"]
    assert block["glue"]["verdict"] == "inconclusive"
    assert "capability not identical across arms" in block["glue"]["reason"]


def test_one_piece_failing_in_every_attempt_is_listed_once() -> None:
    rows = [
        *({"arm": LOOSE, "attempt": attempt, "piece": "agent:a", "reason": "not reported"} for attempt in (1, 2, 3)),
        {"arm": LOOSE, "attempt": 2, "piece": "mcp:x", "reason": "failed"},
    ]

    ladder = stats.lift_ladder(_arms(), CROSS, expected_case_ids=CASES, load_failures_by_case={"x-1": rows})

    reason = ladder["excluded_cases"][0]["load_failures"][LOOSE]
    assert reason == "agent:a (attempts 1, 2, 3): not reported; mcp:x (attempt 2): failed"
