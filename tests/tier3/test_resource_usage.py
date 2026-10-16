# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from skillevaluator.tier3.harbor.resource_usage import agent_execution_seconds, resource_comparison


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
