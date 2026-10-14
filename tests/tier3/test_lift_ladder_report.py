# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The lift ladder in the report payload and in every report format.

The collector does not run a loose-parts arm yet (the plumbing comes later),
so each test runs the real collector on the three existing arms, writes the
``lift_ladder`` statistics block where the collector will write it
(``statistics.json``), and then runs the real report data loader, payload
builder and reporters. The block itself comes from ``stats.lift_ladder`` on
synthetic observations.
"""

from __future__ import annotations

import json
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest
from _f90_lift_fixtures import SKILL, benchmark, cli, collect, html, markdown, metrics, result_for

from skillevaluator.evaluation.tier3_report import (
    _capability_parity,
    _plugin_lift_modes,
    build_agent_eval_payload,
)
from skillevaluator.tier3.harbor import stats
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS
from skillevaluator.tier3.harbor.report_data import load_agent_data
from skillevaluator.tier3.harbor.stats import ArmObservations, TrialObservation

NONE, SKILLS, LOOSE, FULL = "without_skill", "sum_of_parts", "loose_parts", "with_skill"
CROSS = [f"x-{index}" for index in range(1, 6)]
OTHER = [f"o-{index}" for index in range(1, 5)]
CASES = [*CROSS, *OTHER]
WRAPPER_LABEL = "Wrapper effect (full wrapper arm minus loose parts)"
GLUE_PIECES = ["hook:hooks/hooks.json", "agent:release-reviewer", "command:release-check"]
HASHES = {
    FULL: {
        "skill:alpha": "a1",
        "skill:beta": "b1",
        "rule:release-policy.md": "r1",
        "mcp:reltools": "m1",
        "hook:hooks/hooks.json": "h1",
        "agent:release-reviewer": "g1",
        "command:release-check": "c1",
    },
    LOOSE: {"skill:alpha": "a1", "skill:beta": "b1", "rule:release-policy.md": "r1", "mcp:reltools": "m1"},
    SKILLS: {"skill:alpha": "a1", "skill:beta": "b1"},
}
LOOSE_PARTS = {
    "loader": "plain",
    "rules": ["release-policy.md"],
    "mcp_servers": ["reltools"],
    "dropped_as_glue": GLUE_PIECES,
    "content_hashes": HASHES,
}
COMPLETE = {"complete": True, "missing_cases": [], "failed_arms": [], "attempt_shortfall": []}
SIGNALS = {
    LOOSE: {
        "n_trials": 9,
        "routing": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "n_scored": 9},
        "order": {"satisfied": 1, "edges": 2, "n_scored": 9},
        "handoff": {"passed": 1, "checked": 2, "n_scored": 9},
        "conflict": {"passed": 2, "checked": 2, "n_scored": 9},
    },
    FULL: {
        "n_trials": 9,
        "routing": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_scored": 9},
        "order": {"satisfied": 2, "edges": 2, "n_scored": 9},
        "handoff": {"passed": 2, "checked": 2, "n_scored": 9},
        "conflict": {"passed": 2, "checked": 2, "n_scored": 9},
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reward(case_id: str, value: float) -> dict[str, Any]:
    return {"entry_id": case_id, "metric_set": DEFAULT_METRIC_SET, **dict.fromkeys(DEFAULT_METRICS, value)}


def _observations(glue: list[float], *, cross: list[str] = CROSS, drop_loose: tuple[str, ...] = ()) -> dict:
    """no plugin 0.4, skills only 0.6, loose parts 0.7; full = loose + glue[i] on cross-component case i."""
    deltas = {case: glue[index % len(glue)] for index, case in enumerate(cross)}
    arms: dict[str, list[TrialObservation]] = {NONE: [], SKILLS: [], LOOSE: [], FULL: []}
    for case in CASES:
        none = _reward(case, 0.4)
        none.update(skill_execution=None, skill_efficiency=None)
        arms[NONE].append(TrialObservation(case, none))
        arms[SKILLS].append(TrialObservation(case, _reward(case, 0.6)))
        if case not in drop_loose:
            arms[LOOSE].append(TrialObservation(case, _reward(case, 0.7)))
        arms[FULL].append(TrialObservation(case, _reward(case, round(0.7 + deltas.get(case, 0.0), 6))))
    return {arm: ArmObservations("succeeded", tuple(trials)) for arm, trials in arms.items()}


def _ladder(glue: list[float] | None = None, *, cross: list[str] = CROSS, **kwargs: Any) -> dict[str, Any]:
    return stats.lift_ladder(_observations(glue or [0.2], cross=cross, **kwargs), cross, expected_case_ids=CASES)


def _three_arm_jobs() -> dict[str, dict[str, list[Any]]]:
    return {
        "with": {case: [metrics(0.8)] for case in CASES},
        "without": {case: [metrics(0.4, skill=False)] for case in CASES},
        "sumofparts": {case: [metrics(0.6)] for case in CASES},
    }


def _build(
    tmp_path: Path,
    ladders: dict[str, dict[str, Any] | None],
    *,
    modes: dict[str, str] | None = None,
    loose_parts: dict[str, Any] | None = LOOSE_PARTS,
    completeness: dict[str, Any] | None = None,
    signals: dict[str, Any] | None = None,
    requested: str = "ladder",
    effective: str = "ladder",
    provenance_extra: dict[str, Any] | None = None,
    collected: bool = False,
    record_load: bool = True,
) -> dict[str, Any]:
    """Collect three real arms per agent, add each agent's ladder block, and build the report payload."""
    if not collected:
        collect(tmp_path, {agent: _three_arm_jobs() for agent in ladders}, n_attempts=1)
    for agent, block in ladders.items():
        path = tmp_path / "results" / agent / "statistics.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["lift_ladder"] = block
        if block is not None:
            data["integration_completeness"]["loose_parts"] = completeness or COMPLETE
        path.write_text(json.dumps(data), encoding="utf-8")
    agents = load_agent_data(tmp_path / "results")
    if signals is not None:
        for info in agents.values():
            info["plugin_signals_summary"] = {**(info.get("plugin_signals_summary") or {}), **signals}
    plugin_load = {
        "requested": "native",
        "by_agent": {agent: {"mode": (modes or {}).get(agent, "native")} for agent in ladders},
    }
    run_config = {
        "eval_target": {"kind": "plugin"},
        "skill_workspace": {
            "mode": "group",
            "staged_skills": ["alpha", "beta"],
            "include": ["/stage/alpha", "/stage/beta"],
            "sum_of_parts_arm": True,
            "loose_parts_arm": effective == "ladder",
        },
        "lift_mode": {"requested": requested, "effective": effective},
    }
    provenance: dict[str, Any] = {
        "plugin_name": SKILL,
        "requested_lift_mode": requested,
        "effective_lift_mode": effective,
        "partial": False,
        "cross_component_case_count": len(CROSS),
        "integration_evidence_ready": True,
        **({"plugin_load": plugin_load} if record_load else {}),
        **(provenance_extra or {}),
    }
    if loose_parts is not None:
        provenance["loose_parts"] = loose_parts
    built = build_agent_eval_payload(
        SKILL, agents, run_config=run_config, plugin_provenance=provenance, use_llm_judge=False
    )
    assert built is not None
    return built


_VOID_ELEMENTS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "wbr"}
)


class _Section(HTMLParser):
    """Collect the text and the table rows inside the element with one id."""

    def __init__(self, element_id: str) -> None:
        super().__init__(convert_charrefs=True)
        self.element_id = element_id
        self.depth = 0
        self.parts: list[str] = []
        self.rows: list[list[str]] = []
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _VOID_ELEMENTS:
            return
        if self.depth:
            self.depth += 1
            if tag == "tr":
                self.rows.append([])
            elif tag in {"td", "th"}:
                self._cell = []
        elif dict(attrs).get("id") == self.element_id:
            self.depth = 1

    def handle_endtag(self, tag: str) -> None:
        if not self.depth:
            return
        if tag in {"td", "th"} and self._cell is not None and self.rows:
            self.rows[-1].append(" ".join("".join(self._cell).split()))
            self._cell = None
        self.depth -= 1

    def handle_data(self, data: str) -> None:
        if self.depth:
            self.parts.append(data)
            if self._cell is not None:
                self._cell.append(data)

    @property
    def text(self) -> str:
        return " ".join(" ".join(self.parts).split())


def _html_section(page: str, element_id: str = "tier3-lift-ladder") -> _Section:
    parser = _Section(element_id)
    parser.feed(page)
    return parser


def _flat(rendered: str) -> str:
    """Whitespace-collapsed text as a reader sees it: HTML entities and Markdown backslash escapes undone."""
    readable = re.sub(r"\\([!-/:-@\[-`{-~])", r"\1", unescape(rendered))
    return " ".join(readable.split())


def _glue_line(block: dict[str, Any], verdict_words: str) -> str:
    steps, glue = block["steps"], block["glue"]
    return (
        f"Bundle lift {block['bundle']['estimate']:+.2f} = capability {steps['capability']['estimate']:+.2f} "
        f"+ integration (glue) {steps['integration']['estimate']:+.2f} (common basis, {block['n_cases']} cases). "
        f"Integration lift on cross-component cases: {glue['lift']:+.2f} [{glue['ci_low']:+.2f}, "
        f"{glue['ci_high']:+.2f}] (n={glue['n']}), {verdict_words}."
    )


# ---------------------------------------------------------------------------
# Lift modes
# ---------------------------------------------------------------------------


def test_lift_modes_know_ladder() -> None:
    config = {"eval_target": {"kind": "plugin"}, "lift_mode": {"requested": "ladder", "effective": "ladder"}}
    assert _plugin_lift_modes(config, None)["effective"] == "ladder"
    assert _plugin_lift_modes(config, None)["requested"] == "ladder"

    inferred = {
        "eval_target": {"kind": "plugin"},
        "skill_workspace": {"sum_of_parts_arm": True, "loose_parts_arm": True},
    }
    # The return shape is unchanged; the ladder's own skip reason is read elsewhere.
    assert _plugin_lift_modes(inferred, None) == {
        "requested": "ladder",
        "effective": "ladder",
        "integration_skip_reason": None,
    }

    fell_back = _plugin_lift_modes(
        {"eval_target": {"kind": "plugin"}},
        {"requested_lift_mode": "ladder", "effective_lift_mode": "both"},
    )
    assert (fell_back["requested"], fell_back["effective"]) == ("ladder", "both")


# ---------------------------------------------------------------------------
# The payload block
# ---------------------------------------------------------------------------


def test_a_native_run_carries_the_ladder_per_agent_and_for_the_run(tmp_path: Path) -> None:
    ladder = _ladder([0.2])
    built = _build(tmp_path, {"claude-code": ladder}, signals=SIGNALS)

    block = built["agents"]["claude-code"]["lift_ladder"]
    assert built["lift_ladder"] == block
    assert block["agent"] == "claude-code"
    assert block["measured"] is True
    # The statistics are carried as computed.
    for key in ("steps", "bundle", "by_case_type", "n_cases", "excluded_cases", "dimensions", "basis", "arms"):
        assert block[key] == ladder[key]
    assert block["full_arm_load"] == "native"
    assert block["integration_label"] == "Integration lift"
    glue = block["glue"]
    assert (glue["n"], glue["lift"]) == (5, pytest.approx(0.2))
    assert (glue["point_verdict"], glue["verdict"], glue["reason"]) == ("real_integration", "real_integration", None)
    assert glue["complete"] is True
    assert glue["signals"]["routing"] == {LOOSE: SIGNALS[LOOSE]["routing"], FULL: SIGNALS[FULL]["routing"]}
    assert set(glue["signals"]) == {"routing", "order", "handoff", "conflict"}
    loose = block["loose_parts"]
    assert loose["loader"] == "plain"
    assert loose["rules"] == ["release-policy.md"] and loose["mcp_servers"] == ["reltools"]
    assert loose["dropped_as_glue"] == GLUE_PIECES
    assert loose["content_hashes"] == HASHES
    assert (loose["equal_across_arms"], loose["mismatches"]) == (True, [])
    assert block["completeness"] == COMPLETE
    assert (block["lift_mode_requested"], block["lift_mode_effective"]) == ("ladder", "ladder")
    # A ladder run measures the Bundle lift too, unchanged.
    assert built["integration"]["measured"] is True
    assert built["integration"]["lift_mode_requested"] == "ladder"


@pytest.mark.parametrize(
    ("glue", "cross", "verdict", "point", "reason_part"),
    [
        ([0.2], CROSS, "real_integration", "real_integration", None),
        ([0.02], CROSS, "cosmetic_bundling", "cosmetic_bundling", None),
        ([-0.2], CROSS, "negative_integration", "negative_integration", None),
        ([0.0, 0.05, 0.1, 0.15, 0.2], CROSS, "inconclusive", "real_integration", "for the Integration lift"),
        ([0.2], CROSS[:4], "inconclusive", "real_integration", "Only 4 paired cross-component case(s)"),
    ],
)
def test_glue_verdict_uses_the_bundle_lift_band_rules(
    tmp_path: Path, glue: list[float], cross: list[str], verdict: str, point: str, reason_part: str | None
) -> None:
    built = _build(tmp_path, {"claude-code": _ladder(glue, cross=cross)})

    result = built["lift_ladder"]["glue"]
    assert (result["verdict"], result["point_verdict"]) == (verdict, point)
    if reason_part is None:
        assert result["reason"] is None
    else:
        assert reason_part in result["reason"]


def test_incomplete_loose_arm_makes_the_glue_verdict_inconclusive(tmp_path: Path) -> None:
    shortfall = {
        "complete": False,
        "missing_cases": [],
        "failed_arms": [],
        "attempt_shortfall": [{"case": "o-1", "arm": LOOSE, "expected": 2, "observed": 1}],
    }
    built = _build(tmp_path, {"claude-code": _ladder([0.2])}, completeness=shortfall)

    glue = built["lift_ladder"]["glue"]
    assert (glue["verdict"], glue["point_verdict"], glue["complete"]) == ("inconclusive", "real_integration", False)
    assert "loose-parts arm" in glue["reason"] and "o-1" in glue["reason"]


def test_a_cross_component_case_lost_in_any_arm_makes_the_glue_verdict_inconclusive(tmp_path: Path) -> None:
    built = _build(tmp_path, {"claude-code": _ladder([0.2], drop_loose=("x-1",))})

    block = built["lift_ladder"]
    assert block["excluded_cases"] == [
        {"case": "x-1", "reason": "not_scored", "missing_arms": [LOOSE], "load_failures": {}}
    ]
    assert block["glue"]["verdict"] == "inconclusive"
    assert "x-1" in block["glue"]["reason"]
    # The glue row says how many of the cross-component cases it kept (4 of 5: too few for an interval).
    row = next(line for line in benchmark(built).splitlines() if line.startswith("| Integration lift (glue"))
    assert "No interval: 4 of 5 paired case(s), too few" in row


def test_capability_not_identical_shows_the_number_but_no_verdict(tmp_path: Path) -> None:
    hashes = {**HASHES, LOOSE: {**HASHES[LOOSE], "rule:release-policy.md": "r2"}}
    built = _build(tmp_path, {"claude-code": _ladder([0.2])}, loose_parts={**LOOSE_PARTS, "content_hashes": hashes})

    block = built["lift_ladder"]
    assert block["loose_parts"]["equal_across_arms"] is False
    assert block["loose_parts"]["mismatches"] == [
        {"piece": "rule:release-policy.md", "issue": "differs", "arms": [FULL, LOOSE]}
    ]
    glue = block["glue"]
    assert glue["lift"] == pytest.approx(0.2)
    assert (glue["verdict"], glue["point_verdict"]) == ("inconclusive", "real_integration")
    assert glue["reason"].startswith("capability not identical across arms")


@pytest.mark.parametrize(
    ("hashes", "dropped", "equal", "mismatches"),
    [
        (HASHES, GLUE_PIECES, True, []),
        (None, GLUE_PIECES, None, []),
        ({FULL: HASHES[FULL]}, GLUE_PIECES, None, []),
        (
            {**HASHES, LOOSE: {k: v for k, v in HASHES[LOOSE].items() if k != "mcp:reltools"}},
            GLUE_PIECES,
            False,
            [{"piece": "mcp:reltools", "issue": "missing", "arms": [LOOSE]}],
        ),
        (
            HASHES,
            GLUE_PIECES[:2],
            False,
            [{"piece": "command:release-check", "issue": "missing", "arms": [LOOSE]}],
        ),
        (
            {**HASHES, LOOSE: {**HASHES[LOOSE], "hook:hooks/hooks.json": "h1"}},
            GLUE_PIECES,
            False,
            [{"piece": "hook:hooks/hooks.json", "issue": "glue_in_loose_parts", "arms": [LOOSE]}],
        ),
        (
            {**HASHES, LOOSE: {**HASHES[LOOSE], "rule:extra.md": "e1"}},
            GLUE_PIECES,
            False,
            [{"piece": "rule:extra.md", "issue": "missing", "arms": [FULL]}],
        ),
        (
            {**HASHES, SKILLS: {**HASHES[SKILLS], "rule:release-policy.md": "r1"}},
            GLUE_PIECES,
            False,
            [{"piece": "rule:release-policy.md", "issue": "not_a_skill", "arms": [SKILLS]}],
        ),
        (
            {"with_plugin": HASHES[FULL], LOOSE: {**HASHES[LOOSE], "mcp:reltools": "m2"}},
            GLUE_PIECES,
            False,
            [{"piece": "mcp:reltools", "issue": "differs", "arms": [FULL, LOOSE]}],
        ),
        (
            {**HASHES, SKILLS: {"skill:alpha": "a2"}},
            GLUE_PIECES,
            False,
            [
                {"piece": "skill:alpha", "issue": "differs", "arms": [SKILLS, FULL]},
                {"piece": "skill:beta", "issue": "missing", "arms": [SKILLS]},
            ],
        ),
    ],
)
def test_capability_parity_rules(
    hashes: dict[str, Any] | None, dropped: list[str], equal: bool | None, mismatches: list[dict[str, Any]]
) -> None:
    assert _capability_parity(hashes, dropped) == (equal, mismatches)


def test_capability_parity_compares_every_recorded_piece_not_only_the_stored_copy() -> None:
    """A plugin with many pieces: parity reads them all (the payload keeps a bounded copy)."""
    many_glue = [f"hook:hooks/h{index}.json" for index in range(100)]
    skills = {f"skill:s{index}": f"h{index}" for index in range(150)}
    hashes = {
        FULL: {**skills, **dict.fromkeys(many_glue, "g")},
        LOOSE: dict(skills),
        SKILLS: dict(skills),
    }
    assert _capability_parity(hashes, many_glue) == (True, [])


def test_unrecorded_content_hashes_give_no_glue_verdict(tmp_path: Path) -> None:
    """Nothing shows the loose arm staged the same bytes, so the glue is shown but not judged."""
    no_hashes = {key: value for key, value in LOOSE_PARTS.items() if key != "content_hashes"}
    built = _build(tmp_path, {"claude-code": _ladder([0.2])}, loose_parts=no_hashes)

    block = built["lift_ladder"]
    assert block["loose_parts"]["equal_across_arms"] is None
    glue = block["glue"]
    assert glue["lift"] == pytest.approx(0.2)
    assert (glue["point_verdict"], glue["verdict"]) == ("real_integration", "inconclusive")
    assert glue["reason"].startswith("capability not verified across arms")
    assert "Capability identical across arms: not recorded (no content hashes)." in _flat(markdown(built))


def test_an_unrecorded_load_mode_is_never_called_the_integration_lift(tmp_path: Path) -> None:
    built = _build(tmp_path, {"claude-code": _ladder([0.2])}, record_load=False)

    block = built["lift_ladder"]
    assert (block["full_arm_load"], block["integration_label"]) == ("unknown", WRAPPER_LABEL)
    assert (block["glue"]["verdict"], block["glue"]["point_verdict"]) == (None, None)
    assert "did not record that the full-plugin arm loaded the plugin natively" in block["glue"]["reason"]
    assert "Integration lift on cross-component cases" not in _flat(markdown(built))


def test_the_line_says_when_rounded_terms_do_not_add_up() -> None:
    from skillevaluator.reporting.plugin_sections import lift_ladder_view

    step = {"ci_low": None, "ci_high": None, "n_cases": 9, "precision": "low"}
    estimates = {"skill": 0.2, "capability": 0.1149, "integration": 0.0649, "plugin": 0.3798}
    block = {
        "agent": "claude-code",
        "measured": True,
        "full_arm_load": "native",
        "n_cases": 9,
        "steps": {name: {**step, "estimate": value} for name, value in estimates.items()},
        "bundle": {**step, "estimate": 0.1798},
        "glue": {"lift": 0.07, "n": 9, "verdict": "inconclusive"},
    }
    line = lift_ladder_view({"lift_ladder": block})["entries"][0]["line"]
    assert line.startswith(
        "Bundle lift +0.18 = capability +0.11 + integration (glue) +0.06 (common basis, 9 cases; terms rounded)."
    )
    block["steps"]["integration"]["estimate"] = 0.0651
    line = lift_ladder_view({"lift_ladder": block})["entries"][0]["line"]
    assert "(common basis, 9 cases)." in line


def test_the_ladder_stays_out_of_the_statistics_scopes() -> None:
    from skillevaluator.reporting.plugin_sections import _statistics_block

    source = {"lift_ladder": {"steps": {}}, "cost": {"with_skill": {}}}
    assert set(_statistics_block(source)) == {"cost"}


def test_wrapper_mode_reports_a_wrapper_effect_and_no_integration_verdict(tmp_path: Path) -> None:
    built = _build(tmp_path, {"claude-code": _ladder([0.2])}, modes={"claude-code": "wrapper"})

    block = built["lift_ladder"]
    assert block["full_arm_load"] == "wrapper"
    assert block["integration_label"] == WRAPPER_LABEL
    glue = block["glue"]
    assert glue["lift"] == pytest.approx(0.2)
    assert (glue["verdict"], glue["point_verdict"]) == (None, None)
    assert "wrapper" in glue["reason"].lower()

    for rendered in (markdown(built), cli(built), html(built), benchmark(built)):
        flat = _flat(rendered)
        assert WRAPPER_LABEL in flat
        assert "no verdict (wrapper mode)" in flat
        assert "Integration lift on cross-component cases" not in flat
        assert "integration (glue)" not in flat
    section = _html_section(html(built))
    assert "Integration lift" not in section.text


def test_an_agent_whose_loose_arm_did_not_run_says_why_in_every_format(tmp_path: Path) -> None:
    reason = "v1 builds the loose-parts arm for native Claude Code only."
    loose_parts = {**LOOSE_PARTS, "agents": {"codex": {"not_measured_reason": reason}}}
    built = _build(tmp_path, {"claude-code": _ladder([0.2]), "codex": None}, loose_parts=loose_parts)

    codex = built["agents"]["codex"]["lift_ladder"]
    assert (codex["measured"], codex["reason"], codex["agent"]) == (False, reason, "codex")
    assert codex["glue"]["verdict"] is None
    assert built["agents"]["claude-code"]["lift_ladder"]["measured"] is True
    for rendered in (markdown(built), cli(built), html(built), benchmark(built)):
        flat = _flat(rendered)
        assert reason in flat
        assert "codex" in flat


def test_a_ladder_that_fell_back_explains_itself(tmp_path: Path) -> None:
    reason = "No rule or runnable MCP server: loose parts equal the skills alone."
    built = _build(
        tmp_path,
        {"claude-code": None},
        effective="both",
        loose_parts=None,
        provenance_extra={"ladder_skip_reason": reason},
    )

    block = built["lift_ladder"]
    assert (block["measured"], block["reason"]) == (False, reason)
    assert (block["lift_mode_requested"], block["lift_mode_effective"]) == ("ladder", "both")
    for rendered in (markdown(built), cli(built), html(built), benchmark(built)):
        assert reason in _flat(rendered)
        # Nothing was measured, so there is no ladder total to compare with the headline.
        assert "labeled 'Plugin lift on the common basis'" not in _flat(rendered)


def test_the_ladder_is_report_only(tmp_path: Path) -> None:
    """Scores, verdicts, bands, the Bundle lift block and SARIF are the same with and without the ladder."""
    from skillevaluator.reporting.sarif_reporter import SARIFReporter

    with_ladder = _build(tmp_path / "a", {"claude-code": _ladder([-0.2])}, signals=SIGNALS)
    without = _build(tmp_path / "b", {"claude-code": None}, signals=SIGNALS)

    for key in ("integration", "summary", "lift_band", "verdict", "overall_score", "overall_lift", "dimensions"):
        assert with_ladder.get(key) == without.get(key), key
    agent_a, agent_b = with_ladder["agents"]["claude-code"], without["agents"]["claude-code"]
    for key in ("dimensions", "lift", "with_skill", "baseline", "integration", "lift_uncertainty"):
        assert agent_a.get(key) == agent_b.get(key), key
    sarif = SARIFReporter()

    def _sarif(built: dict[str, Any]) -> dict[str, Any]:
        rendered = json.loads(sarif.render_all([result_for(built)]))
        for run in rendered.get("runs", []):
            run.get("invocations", [{}])[0].pop("endTimeUtc", None)
        return rendered

    assert _sarif(with_ladder) == _sarif(without)


# ---------------------------------------------------------------------------
# Every report format
# ---------------------------------------------------------------------------


def test_every_format_shows_the_ladder_line_rows_basis_note_parts_and_signals(tmp_path: Path) -> None:
    ladder = _ladder([0.2])
    built = _build(tmp_path, {"claude-code": ladder}, signals=SIGNALS)
    line = _glue_line(ladder, "real integration lift")
    basis_note = "labeled 'Plugin lift on the common basis' and can differ from the headline Plugin lift"
    parts_line = (
        "Loose parts (plain loader): rules: release-policy.md; MCP servers: reltools. Plugin-only, counted as glue: "
        "hook:hooks/hooks.json, agent:release-reviewer, command:release-check."
    )
    capability = "Capability identical across arms: yes (7 hashed pieces)."
    labels = ("Skill lift", "Capability lift", "Integration lift (glue)", "Plugin lift on the common basis")

    report, output, page, card = markdown(built), cli(built), html(built), benchmark(built)
    for rendered in (report, output, page, card):
        flat = _flat(rendered)
        assert line in flat
        assert basis_note in flat
        assert parts_line in flat
        assert capability in flat
    for rendered in (report, output, page):
        flat = _flat(rendered)
        for label in labels:
            assert label in flat
        assert "Bundle lift on the common basis" in flat
        assert "Routing (P/R/F1)" in flat and "50% / 50% / 50%" in flat and "100% / 100% / 100%" in flat
        assert "1/2 edges in order (50%)" in flat and "2/2 edges in order (100%)" in flat

    section = _html_section(page)
    assert section.text.startswith("Lift Ladder")
    rows = {row[0]: row for row in section.rows if row}
    skill = ladder["steps"]["skill"]
    assert rows["Skill lift"][1:4] == [
        "skills only minus no plugin",
        f"{skill['estimate']:+.2f}",
        f"[{skill['ci_low']:+.2f}, {skill['ci_high']:+.2f}]",
    ]
    assert rows["Integration lift (glue)"][1] == "full plugin minus loose parts"
    assert rows["Plugin lift on the common basis"][1] == "full plugin minus no plugin"
    assert rows["Bundle lift on the common basis"][1] == "full plugin minus skills only"

    assert "| Skill lift | skills only minus no plugin |" in report
    assert "| Integration lift (glue: full plugin vs. loose parts) | Real integration lift, +20 points |" in card


def test_each_agent_has_its_own_named_ladder_in_every_format(tmp_path: Path) -> None:
    claude, codex = _ladder([0.2]), _ladder([-0.2])
    built = _build(tmp_path, {"claude-code": claude, "codex": codex})

    assert built["agents"]["claude-code"]["lift_ladder"]["glue"]["verdict"] == "real_integration"
    assert built["agents"]["codex"]["lift_ladder"]["glue"]["verdict"] == "negative_integration"
    assert built["lift_ladder"]["agent"] == built["best_agent"]
    claude_line = _glue_line(claude, "real integration lift")
    codex_line = _glue_line(codex, "negative integration lift")
    for rendered in (markdown(built), cli(built), html(built), benchmark(built)):
        flat = _flat(rendered)
        assert flat.index("claude-code") < flat.index(claude_line)
        assert codex_line in flat
        assert flat.index(claude_line) < flat.index(codex_line)


def test_a_result_saved_before_the_ladder_renders_every_format(tmp_path: Path) -> None:
    """A ``both`` run saved before this change (no lift_ladder key at all) and a new run with ``null``."""
    collect(tmp_path, {"codex": _three_arm_jobs()}, n_attempts=1)
    statistics_path = tmp_path / "results" / "codex" / "statistics.json"
    data = json.loads(statistics_path.read_text(encoding="utf-8"))
    for saved in ("missing", "null"):
        if saved == "missing":
            data.pop("lift_ladder", None)
        else:
            data["lift_ladder"] = None
        statistics_path.write_text(json.dumps(data), encoding="utf-8")
        built = build_agent_eval_payload(
            SKILL,
            load_agent_data(tmp_path / "results"),
            run_config={
                "eval_target": {"kind": "plugin"},
                "skill_workspace": {"staged_skills": ["alpha", "beta"], "sum_of_parts_arm": True},
                "lift_mode": {"requested": "both", "effective": "both"},
            },
            plugin_provenance={"plugin_name": SKILL, "requested_lift_mode": "both", "effective_lift_mode": "both"},
            use_llm_judge=False,
        )
        assert built is not None
        assert "lift_ladder" not in built and "lift_ladder" not in built["agents"]["codex"]
        for rendered in (markdown(built), cli(built), html(built), benchmark(built)):
            assert "Lift ladder" not in rendered and "Lift Ladder" not in rendered
            assert "Bundle lift" in rendered


def test_hostile_ladder_strings_are_escaped(tmp_path: Path) -> None:
    ladder = _ladder([0.2])
    ladder["excluded_cases"] = [
        {
            "case": "<script>alert(1)</script>|x",
            "reason": "load_failed",
            "missing_arms": [],
            "load_failures": {LOOSE: "<b>boom</b>"},
        }
    ]
    loose_parts = {**LOOSE_PARTS, "rules": ["<img src=x onerror=alert(1)>"], "loader": "<i>plain</i>"}
    built = _build(tmp_path, {"claude-code": ladder}, loose_parts=loose_parts)

    page = html(built)
    assert "<script>alert(1)" not in page and "<img src=x onerror" not in page and "<b>boom</b>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    report = markdown(built)
    assert "<script>alert(1)" not in report and "<b>boom</b>" not in report
    # A pipe in a case id is an entity, so it can never split a Markdown table row.
    assert "&#124;x" in report
    card = benchmark(built)
    assert "<script>alert(1)" not in card and "<img src=x onerror" not in card


def test_json_report_carries_the_ladder_field_names(tmp_path: Path) -> None:
    from skillevaluator.reporting.json_reporter import JSONReporter

    built = _build(tmp_path, {"claude-code": _ladder([0.2])}, signals=SIGNALS)
    rendered = JSONReporter().render_all([result_for(built)])
    block = _find(json.loads(rendered), "lift_ladder")

    assert block is not None
    assert set(block["steps"]) == {"skill", "capability", "integration", "plugin"}
    assert {"bundle", "glue", "loose_parts", "excluded_cases", "by_case_type", "integration_label"} <= set(block)
    assert {"content_hashes", "equal_across_arms", "mismatches", "dropped_as_glue"} <= set(block["loose_parts"])
    assert "NaN" not in rendered and "Infinity" not in rendered


def _find(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if isinstance(value.get(key), dict):
            return value[key]
        for child in value.values():
            found = _find(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find(child, key)
            if found is not None:
                return found
    return None


def test_old_bundle_labels_stay_out_of_the_bundle_section(tmp_path: Path) -> None:
    """The glue verdict says "integration"; the Bundle lift section keeps its own labels."""
    built = _build(tmp_path, {"claude-code": _ladder([0.2])})
    page = html(built)
    bundle = _html_section(page, "tier3-integration").text
    assert "Real integration lift" not in bundle
    assert "Real integration lift" in _html_section(page).text
    assert re.search(r"Integration \(Bundle Lift\)", bundle)


def test_a_partial_glue_interval_says_so(tmp_path: Path) -> None:
    """Six cross-component cases, one lost in the loose arm: five pairs keep an interval marked partial."""
    cross = [*CROSS, "o-1"]
    built = _build(tmp_path, {"claude-code": _ladder([0.2], cross=cross, drop_loose=("x-1",))})

    glue = built["lift_ladder"]["glue"]
    assert (glue["n"], glue["partial"], glue["verdict"]) == (5, True, "inconclusive")
    row = next(line for line in benchmark(built).splitlines() if line.startswith("| Integration lift (glue"))
    assert "partial: 5 of 6 cases" in row


# ---------------------------------------------------------------------------
# The run status (reporting/run_status.py) with a ladder in the payload
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blocking", [False, True], ids=["advisory", "gated"])
@pytest.mark.parametrize("shape", ["negative_glue", "wrapper", "fell_back"])
def test_the_ladder_never_changes_the_run_status_on_any_surface(tmp_path: Path, shape: str, blocking: bool) -> None:
    """A ladder, whatever its glue says, leaves every status to run_status, and every surface still shows it."""
    from tests.reporting._saved_run import render_surfaces

    from skillevaluator.reporting.run_status import resolve_run_status

    if shape == "negative_glue":
        built = _build(tmp_path / "ladder", {"claude-code": _ladder([-0.2])})
        shown = "Negative integration lift"
    elif shape == "wrapper":
        built = _build(tmp_path / "ladder", {"claude-code": _ladder([-0.2])}, modes={"claude-code": "wrapper"})
        shown = WRAPPER_LABEL
    else:
        reason = "No rule or runnable MCP server: loose parts equal the skills alone."
        built = _build(
            tmp_path / "ladder",
            {"claude-code": None},
            effective="both",
            loose_parts=None,
            provenance_extra={"ladder_skip_reason": reason},
        )
        shown = reason
    plain = _build(tmp_path / "plain", {"claude-code": None}, requested="both", effective="both", loose_parts=None)
    assert plain.get("lift_ladder") is None

    def _result(payload: dict[str, Any]) -> Any:
        result = result_for(payload)
        result.metadata["gating"]["blocking"] = blocking
        return result

    status, without = resolve_run_status([_result(built)]), resolve_run_status([_result(plain)])
    assert (status.headline, status.publication, status.plugin, status.reasons) == (
        without.headline,
        without.publication,
        without.plugin,
        without.reasons,
    )
    assert (status.tiers[3].status, status.tiers[3].publication) == (
        without.tiers[3].status,
        without.tiers[3].publication,
    )

    surfaces = render_surfaces([_result(built)], name=SKILL)
    assert surfaces["headlines"] == dict.fromkeys(surfaces["headlines"], status.headline.value)
    assert surfaces["benchmark"] == status.publication.verdict
    assert surfaces["html_tier_cards"]["tier3"] == status.tiers[3].status.verdict
    outputs = surfaces["outputs"]
    for rendered in (outputs["markdown"], outputs["html"], outputs["terminal"]):
        assert shown in _flat(rendered)
    assert shown in _flat(benchmark(built))
