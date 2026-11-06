# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from skillevaluator.tier3.harbor.resource_usage import (
    agent_execution_seconds,
    normalized_trial_timing,
    resource_comparison,
)


def trial(entry="case", tokens=100, seconds=2):
    return {"entry_id": entry, "tokens": {"prompt": tokens, "completion": 10}, "agent_seconds": seconds}


def test_actual_resources_are_independent_and_signed():
    resources = resource_comparison(
        [trial()], [trial(tokens=200, seconds=1)], with_complete=True, baseline_complete=True
    )
    assert resources["tokens"]["delta"] == -100
    assert resources["agent_seconds"]["delta"] == 1


@pytest.mark.parametrize(
    "baseline,complete", [([], False), ([trial(entry="other")], True), ([trial(), trial()], True), ([trial()], False)]
)
def test_incomplete_or_unmatched_baseline_does_not_claim_delta(baseline, complete):
    resources = resource_comparison([trial()], baseline, with_complete=True, baseline_complete=complete)
    assert resources["tokens"]["delta"] is None
    assert resources["agent_seconds"]["delta"] is None


def test_missing_counters_are_not_zero_and_do_not_hide_other_measurements():
    item = trial()
    item.pop("tokens")
    resources = resource_comparison([item], [trial()], with_complete=True, baseline_complete=True)
    assert resources["tokens"]["with_skill"]["total"] is None
    assert resources["tokens"]["with_skill"]["measured_trials"] == 0
    assert resources["tokens"]["delta"] is None
    assert resources["agent_seconds"]["delta"] == 0


def test_measured_zero_is_preserved():
    resources = resource_comparison([trial(tokens=0, seconds=0)], [trial()], with_complete=True, baseline_complete=True)
    assert resources["agent_seconds"]["with_skill"]["total"] == 0


@pytest.mark.parametrize("value", [True, -1, float("inf"), float("nan"), 10**400, None])
def test_invalid_time_counter_cannot_become_a_measurement(value):
    resources = resource_comparison([trial(seconds=value)], [trial()], with_complete=True, baseline_complete=True)
    assert resources["agent_seconds"]["with_skill"]["total"] is None
    assert resources["agent_seconds"]["delta"] is None


@pytest.mark.parametrize(
    "timing",
    [
        {},
        {"started_at": None, "finished_at": None},
        {"started_at": "2026-10-11T00:00:02Z", "finished_at": "2026-10-11T00:00:01Z"},
        {"started_at": "2026-10-11T00:00:00", "finished_at": "2026-10-11T00:00:01"},
    ],
)
def test_unknown_or_invalid_time_is_not_zero(timing):
    assert agent_execution_seconds({"agent_execution": timing}) is None


def test_agent_phase_excludes_build_and_verifier():
    assert (
        agent_execution_seconds(
            {
                "started_at": "2026-10-11T00:00:00Z",
                "finished_at": "2026-10-11T01:00:00Z",
                "agent_execution": {"started_at": "2026-10-11T00:01:00Z", "finished_at": "2026-10-11T00:01:03Z"},
            }
        )
        == 3
    )


def timed(entry="case", llm=10.0, tool=2.0, source="step_timestamps", prompt=100, completion=20):
    item = trial(entry=entry, tokens=prompt)
    item["tokens"]["completion"] = completion
    item["timing"] = {"llm_seconds": llm, "tool_seconds": tool, "source": source}
    return item


def test_normalized_trial_timing_rejects_empty_and_untagged_values():
    assert normalized_trial_timing({}) is None
    assert normalized_trial_timing({"llm_seconds": 1.0}) is None
    assert normalized_trial_timing({"llm_seconds": 1.0, "source": "reward"}) is None
    assert normalized_trial_timing({"llm_seconds": 1.0, "source": "atif"}) == {
        "llm_seconds": 1.0,
        "tool_seconds": None,
        "source": "atif",
    }


def test_timing_pairs_arms_and_derives_ms_per_token():
    resources = resource_comparison(
        [timed(llm=10.0, tool=2.0, prompt=100, completion=20)],
        [timed(llm=20.0, tool=4.0, prompt=150, completion=30)],
        with_complete=True,
        baseline_complete=True,
    )
    assert resources["llm_seconds"]["with_skill"]["total"] == 10.0
    assert resources["llm_seconds"]["baseline"]["total"] == 20.0
    assert resources["llm_seconds"]["delta"] == -10.0
    assert resources["tool_seconds"]["delta"] == -2.0
    per_token = resources["llm_ms_per_token"]
    assert per_token["with_skill"]["value"] == round(10.0 * 1000 / 120, 3)
    assert per_token["baseline"]["value"] == round(20.0 * 1000 / 180, 3)
    assert per_token["delta"] == round(per_token["with_skill"]["value"] - per_token["baseline"]["value"], 3)
    assert per_token["with_skill"]["measured_trials"] == 1
    assert resources["timing_sources"] == {"with_skill": ["step_timestamps"], "baseline": ["step_timestamps"]}
    assert resources["caveats"] == []


def test_timing_totals_are_rounded_to_milliseconds():
    resources = resource_comparison(
        [timed(entry="a", llm=0.1), timed(entry="b", llm=0.2)],
        [timed(entry="a", llm=0.1), timed(entry="b", llm=0.1)],
        with_complete=True,
        baseline_complete=True,
    )
    assert resources["llm_seconds"]["with_skill"]["total"] == 0.3
    assert resources["llm_seconds"]["delta"] == 0.1


def test_tool_total_requires_every_trial():
    resources = resource_comparison(
        [timed(entry="a", llm=10.0, tool=2.0), timed(entry="b", llm=8.0, tool=None)],
        [timed(entry="a", llm=5.0, tool=1.0), timed(entry="b", llm=4.0, tool=1.0)],
        with_complete=True,
        baseline_complete=True,
    )
    assert resources["llm_seconds"]["with_skill"]["total"] == 18.0
    assert resources["llm_seconds"]["delta"] == 9.0
    assert resources["tool_seconds"]["with_skill"] == {
        "status": "unavailable",
        "measured_trials": 1,
        "trials": 2,
        "total": None,
    }
    assert resources["tool_seconds"]["delta"] is None


def test_timing_comparison_does_not_require_token_counters():
    with_trial, baseline_trial = timed(llm=10.0), timed(llm=20.0)
    del with_trial["tokens"], baseline_trial["tokens"]
    resources = resource_comparison([with_trial], [baseline_trial], with_complete=True, baseline_complete=True)
    assert resources["llm_seconds"]["delta"] == -10.0
    assert resources["tokens"]["delta"] is None
    assert resources["llm_ms_per_token"]["with_skill"]["value"] is None
    assert resources["llm_ms_per_token"]["delta"] is None


@pytest.mark.parametrize(
    "baseline,complete",
    [([], False), ([timed(entry="other")], True), ([timed(), timed()], True), ([timed()], False)],
)
def test_partial_or_mismatched_cases_do_not_get_timing_deltas(baseline, complete):
    resources = resource_comparison([timed()], baseline, with_complete=True, baseline_complete=complete)
    for field in ("llm_seconds", "tool_seconds", "llm_ms_per_token"):
        assert resources[field]["delta"] is None
    assert resources["llm_seconds"]["with_skill"]["total"] == 10.0


def test_missing_timing_is_not_zero():
    untimed = trial()
    resources = resource_comparison([untimed], [timed()], with_complete=True, baseline_complete=True)
    assert resources["llm_seconds"]["with_skill"] == {
        "status": "unavailable",
        "measured_trials": 0,
        "trials": 1,
        "total": None,
    }
    assert resources["llm_seconds"]["delta"] is None
    assert resources["timing_sources"] == {"with_skill": [], "baseline": ["step_timestamps"]}
    assert resources["caveats"] == []


def test_measured_zero_timing_is_preserved():
    resources = resource_comparison([timed(llm=0.0, tool=0.0)], [timed()], with_complete=True, baseline_complete=True)
    assert resources["llm_seconds"]["with_skill"]["total"] == 0.0
    assert resources["tool_seconds"]["delta"] == -2.0
    assert resources["llm_ms_per_token"]["with_skill"]["value"] == 0.0


def test_mixed_timing_source_comparison_is_flagged():
    resources = resource_comparison(
        [timed(llm=12.0, source="step_timestamps")],
        [timed(llm=9.0, source="atif")],
        with_complete=True,
        baseline_complete=True,
    )
    assert resources["llm_seconds"]["delta"] == 3.0
    assert resources["timing_sources"] == {"with_skill": ["step_timestamps"], "baseline": ["atif"]}
    assert resources["caveats"] == ["mixed_timing_sources"]


def test_mixed_sources_within_one_arm_are_flagged():
    resources = resource_comparison(
        [timed(entry="a", source="atif"), timed(entry="b", source="step_timestamps")],
        [timed(entry="a", source="atif"), timed(entry="b", source="atif")],
        with_complete=True,
        baseline_complete=True,
    )
    assert resources["timing_sources"]["with_skill"] == ["atif", "step_timestamps"]
    assert resources["caveats"] == ["mixed_timing_sources"]


@pytest.mark.parametrize("source", ["atif", "step_timestamps"])
def test_consistent_timing_sources_are_not_flagged(source):
    resources = resource_comparison(
        [timed(llm=10.0, source=source)], [timed(llm=12.0, source=source)], with_complete=True, baseline_complete=True
    )
    assert resources["llm_seconds"]["delta"] == -2.0
    assert resources["timing_sources"] == {"with_skill": [source], "baseline": [source]}
    assert resources["caveats"] == []


@pytest.mark.parametrize("value", [True, -1, "1.0", float("nan"), float("inf"), 10**1000, None])
def test_invalid_timing_stays_unavailable(value):
    assert normalized_trial_timing({"llm_seconds": value, "source": "atif"}) is None
    assert normalized_trial_timing({"llm_seconds": value, "tool_seconds": 1.0, "source": "atif"}) == {
        "llm_seconds": None,
        "tool_seconds": 1.0,
        "source": "atif",
    }
    item = timed()
    item["timing"]["llm_seconds"] = value
    resources = resource_comparison([item], [timed()], with_complete=True, baseline_complete=True)
    assert resources["llm_seconds"]["with_skill"]["total"] is None
    assert resources["llm_seconds"]["delta"] is None
    assert resources["tool_seconds"]["delta"] == 0.0
