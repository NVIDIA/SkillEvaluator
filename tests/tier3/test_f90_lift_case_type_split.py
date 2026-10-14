# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Bundle lift split by case type, end to end.

The split (``by_case_type``) reports the plugin-minus-member-skills lift on
cross-component cases and on all other cases. It must reuse the headline's
pairs and bootstrap, partition the paired cases, use the same notion of a
cross-component case as the dataset count in the same report, and stay
report-only. The reports call the lift the Bundle lift, and no human label
may still promise coordination.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import pytest
from _f90_lift_fixtures import MEMBERS, benchmark, cli, collect, html, markdown, metrics, payload, result_for

from skillevaluator.constants import LIFT_BOOTSTRAP_RESAMPLES, LIFT_BOOTSTRAP_SEED, LIFT_CI_MIN_PAIRED_CASES
from skillevaluator.tier3.eval_core.plugin_signals import build_plugin_signals_context
from skillevaluator.tier3.harbor import stats
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS
from skillevaluator.tier3.harbor.stats import ArmObservations, TrialObservation

CROSS = [f"x-{index}" for index in range(1, 6)]
OTHER = [f"o-{index}" for index in range(1, 5)]
ALL_CASES = [*CROSS, *OTHER]
_OLD_LABELS = ("Real integration", "Negative integration", "REAL INTEGRATION", "NEGATIVE INTEGRATION")


def _entries(cross: list[str] = CROSS, other: list[str] = OTHER) -> list[dict[str, Any]]:
    """Dataset entries: cross-component cases name both member skills, as the readiness gate requires."""
    return [
        *({"id": case, "cross_component": True, "expected_skills": list(MEMBERS)} for case in cross),
        *({"id": case, "expected_skills": [MEMBERS[0]]} for case in other),
    ]


def _context(entries: list[dict[str, Any]] | None = None) -> Any:
    return build_plugin_signals_context(member_skills=MEMBERS, entries=_entries() if entries is None else entries)


def _three_arms(cross_plugin: float, other_plugin: float) -> dict[str, dict[str, list[Any]]]:
    return {
        "with": {
            **{case: [metrics(cross_plugin)] for case in CROSS},
            **{case: [metrics(other_plugin)] for case in OTHER},
        },
        "without": {case: [metrics(0.4, skill=False)] for case in ALL_CASES},
        "sumofparts": {case: [metrics(0.65)] for case in ALL_CASES},
    }


# ---------------------------------------------------------------------------
# Statistics: same pairs, same bootstrap, a partition of the paired cases
# ---------------------------------------------------------------------------


def _random_reward(case_id: str, rng: random.Random, *, skill: bool) -> dict[str, Any]:
    reward: dict[str, Any] = {"entry_id": case_id, "metric_set": DEFAULT_METRIC_SET}
    for metric in DEFAULT_METRICS:
        roll = rng.random()
        if roll < 0.05:
            reward[metric] = None  # an N/A metric
        elif not skill and metric in ("skill_execution", "skill_efficiency"):
            reward[metric] = None  # no skill in this arm to score
        else:
            reward[metric] = round(rng.random(), 3)
    return reward


def _random_arms(rng: random.Random) -> tuple[ArmObservations, ArmObservations, set[str]]:
    """Random plugin and member-skills arms: 0-20 cases, 1-3 attempts, some cases in one arm only."""
    cases = [f"case-{index}" for index in range(rng.randint(0, 20))]
    plugin: list[TrialObservation] = []
    members: list[TrialObservation] = []
    for case in cases:
        presence = rng.random()
        attempts = rng.randint(1, 3)
        if presence > 0.1:
            plugin.extend(TrialObservation(case, _random_reward(case, rng, skill=True)) for _ in range(attempts))
        if presence < 0.9:
            skill = rng.random() < 0.5
            members.extend(TrialObservation(case, _random_reward(case, rng, skill=skill)) for _ in range(attempts))
    share = rng.choice((0.0, 0.3, 0.5, 1.0))
    cross = {case for case in cases if rng.random() < share} | {"not-a-case"}
    succeeded = "succeeded"
    return ArmObservations(succeeded, tuple(plugin)), ArmObservations(succeeded, tuple(members)), cross


def test_split_partitions_the_headline_pairs_with_the_same_bootstrap_on_random_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Property: on random data the two groups are the headline's paired cases, split once.

    The tail-resample floor is lowered so 60 random runs stay fast; the headline
    and both groups read the same module setting, so they still share one
    bootstrap configuration.
    """
    monkeypatch.setattr(stats, "LIFT_BOOTSTRAP_TAIL_RESAMPLES", 5)
    rng = random.Random(20261010)
    checked = 0
    for _ in range(60):
        plugin, members, cross = _random_arms(rng)
        whole = stats.paired_lift(plugin, members, treatment_arm="with_skill", control_arm="sum_of_parts")
        split = stats.lift_by_case_type(plugin, members, cross)
        if whole is None:
            assert split is None
            continue
        assert split is not None
        treatment, control, _ = stats.shared_case_scores(plugin.trials, members.trials)
        groups = {
            "cross_component": sorted(set(treatment) & cross),
            "other": sorted(set(treatment) - cross),
        }
        # A partition of the headline's pairs: nothing dropped, nothing counted twice.
        assert set(groups["cross_component"]).isdisjoint(groups["other"])
        assert split["cross_component"]["n"] + split["other"]["n"] == whole["n_cases"] == len(treatment)
        weighted = 0.0
        for key, ids in groups.items():
            side = split[key]
            assert side["n"] == len(ids)
            expected = stats.paired_case_bootstrap(
                {case: treatment[case] for case in ids},
                {case: control[case] for case in ids},
                seed=LIFT_BOOTSTRAP_SEED,
                resamples=LIFT_BOOTSTRAP_RESAMPLES,
            )
            assert side["lift"] == expected["estimate"]
            if len(ids) >= LIFT_CI_MIN_PAIRED_CASES:
                assert (side["ci_low"], side["ci_high"]) == (expected["ci_low"], expected["ci_high"])
                assert side["reason"] is None
            else:
                assert side["ci_low"] is None and side["ci_high"] is None
                assert side["reason"]
            if ids:
                weighted += len(ids) * side["lift"]
        # The case-weighted mean of the groups is the headline lift (each value is rounded to 4 places).
        assert weighted / whole["n_cases"] == pytest.approx(whole["estimate"], abs=1e-4)
        checked += 1
    assert checked >= 40


def test_every_case_cross_component_makes_the_cross_group_the_headline() -> None:
    plugin = ArmObservations(
        "succeeded", tuple(TrialObservation(c, _flat(c, 0.8 + 0.01 * i)) for i, c in enumerate(CROSS))
    )
    members = ArmObservations("succeeded", tuple(TrialObservation(c, _flat(c, 0.6)) for c in CROSS))

    whole = stats.paired_lift(plugin, members, treatment_arm="with_skill", control_arm="sum_of_parts")
    split = stats.lift_by_case_type(plugin, members, set(CROSS))

    assert whole is not None and split is not None
    cross = split["cross_component"]
    assert (cross["n"], cross["lift"], cross["ci_low"], cross["ci_high"]) == (
        whole["n_cases"],
        whole["estimate"],
        whole["ci_low"],
        whole["ci_high"],
    )
    assert split["other"] == {
        "n": 0,
        "lift": None,
        "ci_low": None,
        "ci_high": None,
        "reason": "No other case was scored in both arms.",
    }


def _flat(case_id: str, value: float | None) -> dict[str, Any]:
    return {"entry_id": case_id, "metric_set": DEFAULT_METRIC_SET, **dict.fromkeys(DEFAULT_METRICS, value)}


def test_five_pairs_get_an_interval_and_four_pairs_only_a_point_estimate() -> None:
    plugin = [TrialObservation(c, _flat(c, 0.7 + 0.02 * i)) for i, c in enumerate(ALL_CASES)]
    members = [TrialObservation(c, _flat(c, 0.6)) for c in ALL_CASES]

    split = stats.lift_by_case_type(ArmObservations("succeeded", plugin), ArmObservations("succeeded", members), CROSS)

    assert split is not None
    assert split["cross_component"]["n"] == 5 == LIFT_CI_MIN_PAIRED_CASES
    assert split["cross_component"]["ci_low"] is not None and split["cross_component"]["reason"] is None
    assert split["other"]["n"] == 4
    # o-1..o-4 sit at positions 5-8: plugin 0.80-0.86 against 0.60.
    assert split["other"]["lift"] == pytest.approx(0.23)
    assert split["cross_component"]["lift"] == pytest.approx(0.14)
    assert split["other"]["ci_low"] is None and split["other"]["ci_high"] is None
    assert split["other"]["reason"] == "Only 4 paired other case(s); at least 5 are needed for an interval."


def test_a_case_in_one_arm_only_is_in_neither_group() -> None:
    plugin = [TrialObservation(c, _flat(c, 0.9)) for c in ALL_CASES]
    # x-1 never scored in the member-skills arm; o-1 has only an unscored attempt there.
    members = [TrialObservation(c, _flat(c, 0.6)) for c in ALL_CASES if c not in {"x-1", "o-1"}]
    members.append(TrialObservation("o-1", _flat("o-1", None)))
    plugin_arm, members_arm = ArmObservations("succeeded", plugin), ArmObservations("succeeded", members)

    whole = stats.paired_lift(
        plugin_arm, members_arm, treatment_arm="with_skill", control_arm="sum_of_parts", expected_case_ids=ALL_CASES
    )
    split = stats.lift_by_case_type(plugin_arm, members_arm, CROSS)

    assert whole is not None and split is not None
    assert whole["n_cases"] == 7 and whole["partial"] is True
    assert (split["cross_component"]["n"], split["other"]["n"]) == (4, 3)


# ---------------------------------------------------------------------------
# One rule for "cross-component case" across the report
# ---------------------------------------------------------------------------


def test_split_uses_the_same_cross_component_rule_as_the_dataset_count() -> None:
    """A case counts only when it is marked cross_component and names two or more member skills.

    The dataset section of the same report prints ``cross_component_case_count``
    (plugin_eval), so the split must not count a case that names one member, or
    a member plus an outsider, as cross-component.
    """
    from skillevaluator.tier3.plugin_eval import _cross_component_case_count

    entries = [
        {"id": "both-members", "cross_component": True, "expected_skills": ["alpha", "beta"]},
        {"id": "case-folded", "cross_component": True, "expected_skills": [" Alpha ", "BETA"]},
        {"id": "one-member", "cross_component": True, "expected_skills": ["alpha"]},
        {"id": "outsider", "cross_component": True, "expected_skills": ["alpha", "gamma"]},
        {"id": "no-list", "cross_component": True, "expected_skills": "alpha,beta"},
        {"id": "string-flag", "cross_component": "true", "expected_skills": ["alpha", "beta"]},
        {"id": "unflagged", "expected_skills": ["alpha", "beta"]},
    ]

    context = build_plugin_signals_context(member_skills=MEMBERS, entries=entries)

    assert context.cross_component_cases == frozenset({"both-members", "case-folded"})
    assert len(context.cross_component_cases) == _cross_component_case_count(
        entries, tuple(Path("/stage") / name for name in MEMBERS)
    )


def test_an_unread_dataset_records_no_split_instead_of_zero_cross_component_cases(tmp_path: Path) -> None:
    """When the runner could not read the staged cases, the split is "not recorded", not "0 cross-component"."""
    assert _context(entries=[]).cross_component_cases is None
    assert build_plugin_signals_context(member_skills=MEMBERS).cross_component_cases is None

    collect(tmp_path, {"codex": _three_arms(0.9, 0.65)}, n_attempts=1, plugin_signals=_context(entries=[]))
    statistics = json.loads((tmp_path / "results" / "codex" / "statistics.json").read_text(encoding="utf-8"))
    assert "by_case_type" not in statistics["lift_uncertainty"]["integration"]
    built = payload(tmp_path, "both")
    assert built["integration"]["by_case_type"] is None
    assert "On cross-component cases" not in markdown(built)


# ---------------------------------------------------------------------------
# Reports: real collector -> payload -> every format, one and two agents
# ---------------------------------------------------------------------------


def test_one_agent_run_carries_the_split_into_every_report(tmp_path: Path) -> None:
    collect(tmp_path, {"codex": _three_arms(0.9, 0.65)}, n_attempts=1, plugin_signals=_context())
    built = payload(tmp_path, "both")

    block = built["integration"]
    assert block["by_case_type"] == block["lift_uncertainty"]["by_case_type"]
    assert block["by_case_type"]["cross_component"]["n"] == 5
    assert block["by_case_type"]["other"]["n"] == 4
    line = "On cross-component cases: +0.25 (n=5). On other cases: +0.00 (n=4)."
    assert line in markdown(built)
    assert line in " ".join(cli(built).split())
    section = _integration_section(html(built))
    assert line in section
    assert f"Bundle lift by case type: {line}" in benchmark(built)

    # Report-only: the same run without the split has the same block, verdict and summary.
    plain_dir = tmp_path / "plain"
    collect(plain_dir, {"codex": _three_arms(0.9, 0.65)}, n_attempts=1)
    plain = payload(plain_dir, "both")
    assert plain["integration"]["by_case_type"] is None
    without_split = {key: value for key, value in block.items() if key != "by_case_type"}
    without_split["lift_uncertainty"] = {
        key: value for key, value in block["lift_uncertainty"].items() if key != "by_case_type"
    }
    assert without_split == {key: value for key, value in plain["integration"].items() if key != "by_case_type"}
    assert block["integration_lift"] == pytest.approx(0.1389, abs=1e-4)
    assert built["summary"] == plain["summary"]


def test_each_agent_shows_its_own_split_in_every_report(tmp_path: Path) -> None:
    collect(
        tmp_path,
        {"claude-code": _three_arms(0.9, 0.65), "codex": _three_arms(0.65, 0.9)},
        n_attempts=1,
        plugin_signals=_context(),
    )
    built = payload(tmp_path, "both")

    claude = "On cross-component cases: +0.25 (n=5). On other cases: +0.00 (n=4)."
    codex = "On cross-component cases: +0.00 (n=5). On other cases: +0.25 (n=4)."
    agents = built["agents"]
    assert agents["claude-code"]["integration"]["by_case_type"]["cross_component"]["lift"] == 0.25
    assert agents["codex"]["integration"]["by_case_type"]["other"]["lift"] == 0.25

    report = markdown(built)
    assert report.index("**claude-code — ") < report.index(claude) < report.index("**codex — ") < report.index(codex)
    output = " ".join(cli(built).split())
    assert output.index("Integration (claude-code)") < output.index(claude) < output.index("Integration (codex)")
    assert output.index("Integration (codex)") < output.index(codex)
    section = _integration_section(html(built))
    assert section.index("claude-code") < section.index(claude) < section.index(codex)
    card = benchmark(built)
    assert f"Bundle lift by case type — claude-code: {claude}" in card
    assert f"Bundle lift by case type — codex: {codex}" in card


def test_a_run_saved_without_the_split_renders_every_format(tmp_path: Path) -> None:
    """Runs saved before the split (no plugin signals context) render with no split line and no crash."""
    collect(tmp_path, {"codex": _three_arms(0.9, 0.65)}, n_attempts=1)
    built = payload(tmp_path, "both")

    assert "by_case_type" not in built["agents"]["codex"]["lift_uncertainty"]["integration"]
    assert built["integration"]["by_case_type"] is None
    for rendered in (markdown(built), cli(built), html(built), benchmark(built)):
        assert "On cross-component cases" not in rendered
        assert "Bundle lift" in rendered


def test_hostile_split_values_are_shown_as_n_a_and_escaped(tmp_path: Path) -> None:
    collect(tmp_path, {"codex": _three_arms(0.9, 0.65)}, n_attempts=1, plugin_signals=_context())
    built = payload(tmp_path, "both")
    built["integration"]["by_case_type"] = {
        "cross_component": {"n": "<b>6</b>", "lift": "<script>alert(1)</script>"},
        "other": {"n": True, "lift": "NaN"},
    }

    page = html(built)
    section = _integration_section(page)
    assert "<script>alert(1)" not in page and "<b>6</b>" not in page
    assert "On cross-component cases: n/a (n=n/a). On other cases: n/a (n=n/a)." in section
    built["integration"]["by_case_type"] = "<img src=x onerror=alert(1)>"
    for rendered in (html(built), markdown(built), cli(built), benchmark(built)):
        assert "<img src=x onerror" not in rendered
        assert "On cross-component cases" not in rendered


# ---------------------------------------------------------------------------
# Human labels: the verdicts no longer promise coordination
# ---------------------------------------------------------------------------


def test_verdict_labels_say_bundle_lift_and_json_values_stay(tmp_path: Path) -> None:
    arms = {
        "claude-code": {
            "with": {case: [metrics(0.45)] for case in ALL_CASES},
            "without": {case: [metrics(0.4, skill=False)] for case in ALL_CASES},
            "sumofparts": {case: [metrics(0.65)] for case in ALL_CASES},
        },
        "codex": _three_arms(0.9, 0.9),
    }
    collect(tmp_path, arms, n_attempts=1, plugin_signals=_context())
    built = payload(tmp_path, "both")

    agents = built["agents"]
    assert agents["codex"]["integration"]["verdict"] == "real_integration"
    assert agents["claude-code"]["integration"]["verdict"] == "negative_integration"
    report, output, page, card = markdown(built), cli(built), html(built), benchmark(built)
    assert "**codex — Real bundle lift:**" in report and "**claude-code — Negative bundle lift:**" in report
    assert re.search(r"Integration \(codex\):\s+REAL BUNDLE LIFT", output)
    assert re.search(r"Integration \(claude-code\):\s+NEGATIVE BUNDLE LIFT", output)
    assert "Real bundle lift" in _integration_section(page) and "Negative bundle lift" in _integration_section(page)
    assert "| Bundle lift (plugin vs. its skills alone) — codex | Real bundle lift, +25 points |" in card
    for rendered in (report, output, page, card):
        for old in _OLD_LABELS:
            assert old not in rendered
        # The interpretation sentence must not credit coordination either.
        assert "coordinated plugin" not in rendered
        assert "coordination overhead" not in rendered


def test_json_report_keeps_the_verdict_values_and_carries_the_split(tmp_path: Path) -> None:
    from skillevaluator.reporting.json_reporter import JSONReporter

    collect(tmp_path, {"codex": _three_arms(0.9, 0.65)}, n_attempts=1, plugin_signals=_context())
    built = payload(tmp_path, "both")

    rendered = JSONReporter().render_all([result_for(built)])
    parsed = json.loads(rendered)
    blocks = list(_find_integration_blocks(parsed))
    assert blocks
    for block in blocks:
        assert block["verdict"] in {"real_integration", "cosmetic_bundling", "negative_integration", "inconclusive"}
        assert set(block["by_case_type"]) == {"cross_component", "other"}
    assert "NaN" not in rendered and "Infinity" not in rendered


def _find_integration_blocks(value: Any) -> Any:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "integration" and isinstance(child, dict) and "verdict" in child:
                yield child
            yield from _find_integration_blocks(child)
    elif isinstance(value, list):
        for child in value:
            yield from _find_integration_blocks(child)


def _integration_section(page: str) -> str:
    section = page[page.index('id="tier3-integration"') :]
    section = section[: section.index("</details>")]
    return " ".join(re.sub(r"<[^>]+>", " ", section).split())
