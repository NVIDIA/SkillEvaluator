# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How the measured context delta reads in HTML and the CLI (check 25, label slips and baselines).

The payloads copy the shapes of the check-25 examples: a 9-pair Claude Code run,
a one-pair run (tier3-12), an unavailable run whose reason already ends in a
period (tier3-04), a partial run, and a ``--lift-mode integration`` run whose
baseline holds the member skills (tier3-11). The real view builder, Jinja macro
and Rich printer render each one.
"""

from __future__ import annotations

import re
from io import StringIO
from typing import Any

import pytest
from rich.console import Console

from skillevaluator.reporting import HTMLReporter
from skillevaluator.reporting.cli import print_plugin_tier3
from skillevaluator.reporting.plugin_sections import statistics_view, tier3_plugin_view
from skillevaluator.tier3.harbor import stats
from skillevaluator.tier3.harbor.stats import ArmObservations, TrialObservation


def _payload(measured: dict[str, Any], *, lift_mode: str = "effectiveness") -> dict[str, Any]:
    return {
        "eval_target": {"kind": "plugin"},
        "lift_mode_requested": lift_mode,
        "lift_mode_effective": lift_mode,
        "best_agent": "claude-code",
        "agents": {"claude-code": {"context_cost_measured": measured}},
    }


def _measured(**fields: Any) -> dict[str, Any]:
    return {"method": "paired_first_turn_prompt_tokens", "reason": None, **fields}


def _html(payload: dict[str, Any]) -> str:
    environment = HTMLReporter(include_timestamp=False)._create_environment()
    module = environment.get_template("plugin_sections.html.j2").module
    html = str(module.statistics_section(statistics_view(payload)))
    match = re.search(r'id="tier3-plugin-context-measured">(.*?)</p>', html, re.DOTALL)
    assert match is not None
    return " ".join(re.sub(r"<[^>]+>", " ", match.group(1)).split())


def _cli(payload: dict[str, Any]) -> str:
    console = Console(file=StringIO(), width=220, color_system=None)
    view = tier3_plugin_view(payload)
    assert view is not None
    print_plugin_tier3(view, console)
    line = next(line for line in console.file.getvalue().splitlines() if "Measured context delta" in line)
    return " ".join(line.split())


def test_l29_measured_delta_counts_paired_cases_not_trials() -> None:
    payload = _payload(_measured(delta_tokens_mean=635.0, n_pairs=9, status="measured", min_pairs=3))

    html, cli = _html(payload), _cli(payload)

    for rendered in (html, cli):
        assert "+635 tokens" in rendered
        assert "9 paired cases" in rendered
        assert "trial(s)" not in rendered


def test_l29_one_pair_reads_as_too_few_not_as_a_measurement() -> None:
    payload = _payload(
        _measured(
            delta_tokens_mean=1351.0,
            n_pairs=1,
            status="insufficient",
            min_pairs=3,
            reason="Only 1 case has first-turn prompt tokens in both arms; at least 3 are needed.",
        )
    )

    for rendered in (_html(payload), _cli(payload)):
        assert "+1,351 tokens" in rendered
        assert "1 paired case" in rendered
        assert "1 pairs" not in rendered
        assert "too few" in rendered


def test_l29_unavailable_reason_does_not_end_in_two_periods() -> None:
    payload = _payload(
        _measured(delta_tokens_mean=None, n_pairs=0, status="unavailable", reason="No baseline (without) arm was run.")
    )

    html = _html(payload)

    assert "No baseline (without) arm was run." in html
    assert ".." not in html


@pytest.mark.parametrize("render", [_html, _cli])
def test_l29_integration_mode_names_the_member_skills_baseline(render: Any) -> None:
    """tier3-11: +477 was plugin minus member skills, printed with no label."""
    payload = _payload(
        _measured(delta_tokens_mean=477.0, n_pairs=9, status="measured", min_pairs=3), lift_mode="integration"
    )

    assert "member skills" in render(payload)


@pytest.mark.parametrize("render", [_html, _cli])
def test_m4_partial_measurement_says_so(render: Any) -> None:
    """tier3-04: one with-plugin trial failed; the other nine pairs still measure +884."""
    cases = [f"case-{index}" for index in range(1, 11)]
    # The failed trial's usage was read, but its trajectory has no per-step counts.
    with_trials = [_trial(case, first_turn_prompt_tokens=25_884) for case in cases[:9]]
    with_trials.append(_trial(cases[9], prompt_tokens=40_000.0))
    without_trials = [_trial(case, first_turn_prompt_tokens=25_000) for case in cases]
    measured = stats.context_cost_measured(
        ArmObservations(execution_status="failed", trials=tuple(with_trials)),
        ArmObservations(execution_status="succeeded", trials=tuple(without_trials)),
    )

    rendered = render(_payload(measured))

    assert "+884 tokens" in rendered
    assert "9 paired cases" in rendered
    assert "Partial: 1 trial(s) of the with-plugin arm have no first-turn prompt token count" in rendered


def _trial(case_id: str, **usage: float) -> TrialObservation:
    return TrialObservation(case_id=case_id, reward={"entry_id": case_id}, usage=usage)
