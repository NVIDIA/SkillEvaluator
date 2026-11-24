# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report-only statistics for an ordinary Tier 3 skill run.

The collector hands this module the arms it has already scored. The module
asks :mod:`skillevaluator.tier3.harbor.stats` for the Skill Lift interval and
pass^k beside pass@k, and writes them to ``<agent>/statistics.json``.

Nothing here is scored. A failure never reaches the caller: :func:`attach`
leaves every score, lift, pass@k, status and exit code as it was and adds one
bounded warning.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillevaluator.tier3.harbor import collector, stats
from skillevaluator.tier3.runtime_arms import ARM_WITH, ARM_WITHOUT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArmInput:
    """One scored arm as the collector holds it.

    ``rewards`` are the arm's logical attempt rewards. ``pass_summary`` is the
    collector's raw pass@k summary, with its ``_pairing_cases``.
    """

    rewards: Sequence[Mapping[str, Any]]
    execution_status: str
    pass_summary: Mapping[str, Any]


def arm_observations(arm: ArmInput, *, expected_case_ids: Sequence[str] | None) -> stats.ArmObservations:
    """Build the observations of a succeeded arm from the collector's own case and score rules.

    An attempt whose overall score is unavailable is left out, as the headline
    Skill Lift leaves it out.
    """
    expected = set(expected_case_ids) if expected_case_ids else None
    scored: list[stats.TrialObservation] = []
    for reward in sorted(arm.rewards, key=collector._attempt_sort_key):
        case = collector._entry_id(reward, expected)
        if expected and case not in expected:
            continue
        score = collector._overall_score(reward)
        if score is not None:
            logical = score if reward.get("_logical_mixed_contracts") else None
            scored.append(stats.TrialObservation(case, reward, logical_overall=logical))
    return stats.ArmObservations(arm.execution_status, tuple(scored))


def _same_attempts_per_case(arms: Iterable[stats.ArmObservations]) -> bool:
    """Whether every case of each arm has the same number of scored attempts."""
    return all(len(set(Counter(trial.case_id for trial in arm.trials).values())) <= 1 for arm in arms)


def agent_statistics(
    arms: Mapping[str, ArmInput],
    *,
    expected_case_ids: Sequence[str] | None,
    stop_on_pass: bool,
    lift_eligible: bool,
) -> dict[str, Any]:
    """Return the :data:`stats.RUN_STATISTICS_FIELDS` blocks of one agent, or ``{}`` when no arm ran.

    ``arms`` holds the arms that ran, keyed ``with_skill`` and ``without_skill``.
    An arm that did not succeed has no reliability. The lift interval is built
    only when the collector reports the headline Skill Lift (``lift_eligible``:
    both arms succeeded and the skill activated), so a withheld lift never
    costs a bootstrap. It is also withheld when an arm's cases made different
    numbers of attempts, as after ``stop_on_pass``: the headline weights every
    attempt, the interval weights every case, and the two would disagree.
    """
    if not arms:
        return {}
    interval = None
    if lift_eligible:
        observed = {name: arm_observations(arm, expected_case_ids=expected_case_ids) for name, arm in arms.items()}
        if _same_attempts_per_case(observed.values()):
            interval = stats.paired_lift(
                observed.get(ARM_WITH),
                observed.get(ARM_WITHOUT),
                treatment_arm=ARM_WITH,
                control_arm=ARM_WITHOUT,
                expected_case_ids=expected_case_ids,
            )
    reliability = {}
    for name, arm in arms.items():
        if arm.execution_status == "succeeded" and (
            summary := stats.reliability_summary(arm.pass_summary, stop_on_pass=stop_on_pass)
        ):
            reliability[name] = summary
    return {"lift_uncertainty": {"effectiveness": interval}, "reliability": reliability}


def attach(
    all_results: dict[str, Any],
    agent: str,
    *,
    agent_dir: Path,
    output_dir: Path,
    arms: Mapping[str, ArmInput],
    expected_case_ids: Sequence[str] | None,
    stop_on_pass: bool,
    lift_eligible: bool,
) -> None:
    """Compute, write and attach one agent's statistics. Never raises.

    On success ``statistics.json`` is written through the collector's bounded,
    link-refusing writer and the agent's results gain the
    :data:`stats.RUN_STATISTICS_FIELDS` blocks. On any failure the results are
    left as they were, apart from one warning that names only the agent and the
    exception type.
    """
    try:
        statistics = agent_statistics(
            arms, expected_case_ids=expected_case_ids, stop_on_pass=stop_on_pass, lift_eligible=lift_eligible
        )
        if statistics:
            collector._write_generated_root_json(agent_dir / "statistics.json", output_dir, statistics)
            all_results["agents"][agent].update(statistics)
    except Exception as exc:
        logger.debug("Run statistics were not computed for %s", agent, exc_info=True)
        all_results.setdefault("warnings", []).append(
            f"Report-only run statistics were not computed for {agent} ({type(exc).__name__})."
        )
