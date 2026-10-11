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

from skillevaluator.tier3.harbor import stats
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS
from skillevaluator.tier3.harbor.stats import ArmObservations, TrialObservation
from skillevaluator.tier3.loose_parts import (
    LOOSE_PARTS_ARM,
    build_loose_parts,
    load_failures_by_case,
    loose_parts_load_evidence,
)

NONE, SKILLS, LOOSE, FULL = "without_skill", "sum_of_parts", "loose_parts", "with_skill"
CROSS = [f"x-{index}" for index in range(1, 7)]
OTHER = ["o-1", "o-2", "o-3"]
CASES = [*CROSS, *OTHER]


@pytest.fixture(scope="module")
def loose(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("loose-contract")
    parts = build_loose_parts(native_source(make_plugin(root / "plugins"), root / "stage").native_source)
    assert parts.error is None
    return parts


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
