# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hook census, canary exfiltration, and MCP proof in the plugin report sections."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from _plugin_fixtures import HOSTILE, element_text, provenance, write_run_dir
from rich.console import Console

from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory
from skillevaluator.plugin_components import summarize_coverage
from skillevaluator.reporting import HTMLReporter
from skillevaluator.reporting.cli import print_plugin_runtime_evidence, print_plugin_tier3
from skillevaluator.reporting.markdown import MarkdownReporter
from skillevaluator.reporting.plugin_sections import (
    canary_view,
    coverage_view,
    hook_census_view,
    mcp_proof_view,
    plugin_attributable_leaks,
    tier3_plugin_view,
)
from skillevaluator.tier3.eval_core.runtime_evidence import canary_arm_comparison
from skillevaluator.tier3.harbor.report_data import load_agent_data

if TYPE_CHECKING:
    from skillevaluator.models import ValidationResult

CANARY_ARM = {"n_trials": 2, "planted": 2, "leaked": 1, "leak_rate": 0.5, "sinks": {"url": 1}}
BASELINE_ARM = {"n_trials": 2, "planted": 2, "leaked": 0, "leak_rate": 0.0, "sinks": {}}
HOOK_CENSUS = {
    "n_trials": 2,
    "n_trials_with_census": 2,
    "n_trials_unreadable": 0,
    "hooks": [
        {"hook_id": "hooks/hooks.json#PreToolUse[0].hooks[0]", "event": "PreToolUse", "runs": 4, "failures": 1, "trials": 2}
    ],
    "total_runs": 4,
    "total_failures": 1,
    "invalid_lines": 0,
    "truncated": False,
}
MCP_PROOF = {
    "docs": {"status": "used-successfully", "tools": ["search", "get_page"], "detail": "agent made 3 call(s)"},
    "wiki": {"status": "declared", "tools": [], "detail": "host probe not requested"},
}


@pytest.fixture(autouse=True)
def _no_llm_insights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "skillevaluator.evaluation.insights_judge.build_insights",
        lambda *_args, **_kwargs: {"conclusions": [], "recommendations": []},
    )


def _write_runtime_evidence(run_dir: Path) -> None:
    for variant, canary in (("with-skill", CANARY_ARM), ("without-skill", BASELINE_ARM)):
        path = run_dir / "codex" / variant / "summary.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["canary_summary"] = canary
        if variant == "with-skill":
            data["plugin_signals_summary"]["hook_census"] = HOOK_CENSUS
        path.write_text(json.dumps(data), encoding="utf-8")


def _result(tmp_path: Path, **provenance_extra: Any) -> ValidationResult:
    plugin_dir = tmp_path / "demo-plugin"
    plugin_dir.mkdir(exist_ok=True)
    sidecar = {**provenance(partial=False), **provenance_extra}
    run_dir = write_run_dir(tmp_path / "results" / "20260101_000000", sidecar=sidecar)
    _write_runtime_evidence(run_dir)
    result = agent_eval_result_from_directory(plugin_dir, run_dir, use_llm_judge=False)
    assert result is not None
    return result


def test_report_data_and_payload_carry_per_arm_canary_results(tmp_path: Path) -> None:
    run_dir = write_run_dir(tmp_path / "run")
    _write_runtime_evidence(run_dir)

    agents = load_agent_data(run_dir)

    assert agents["codex"]["canary_summary"] == {
        "arms": {"with_skill": CANARY_ARM, "without_skill": BASELINE_ARM},
        "plugin_attributable": True,
    }
    payload = _result(tmp_path).metadata["agent_eval"]
    assert payload["agents"]["codex"]["canary_summary"]["plugin_attributable"] is True
    assert payload["canary_summary"] == payload["agents"]["codex"]["canary_summary"]


def test_views_tolerate_absent_keys() -> None:
    assert hook_census_view({}) is None
    assert canary_view({"agents": {"codex": {}}}) is None
    assert mcp_proof_view(None) is None
    assert mcp_proof_view({}) is None


def test_views_build_display_models(tmp_path: Path) -> None:
    view = tier3_plugin_view(_result(tmp_path, mcp_proof=MCP_PROOF).metadata["agent_eval"])
    assert view is not None

    [census] = view["hook_census"]["entries"]
    assert census["arm_label"] == "Plugin"
    assert census["rows"] == [
        {
            "hook_id": "hooks/hooks.json#PreToolUse[0].hooks[0]",
            "event": "PreToolUse",
            "runs": 4,
            "blocked": 0,
            "failures": 1,
            "not_started": 0,
            "failure_rate": "25%",
            "trials": 2,
        }
    ]
    [canary] = view["canary"]["entries"]
    assert canary["plugin_attributable_leak"] is True
    assert [(row["arm_label"], row["leaked"], row["sinks"]) for row in canary["rows"]] == [
        ("Plugin", 1, "URL (1)"),
        ("Baseline (no plugin)", 0, "none"),
    ]
    proof = view["mcp_proof"]
    assert proof["headline"] == "1 of 2 URL MCP servers proven reachable"
    assert [(row["server"], row["status_label"]) for row in proof["rows"]] == [
        ("docs", "Used successfully"),
        ("wiki", "Declared (not proven)"),
    ]


def test_coverage_view_counts_exercised_and_loaded_as_evaluated() -> None:
    view = coverage_view(
        {
            "components": [
                {"type": "hook", "name": "h", "state": "exercised"},
                {"type": "agent", "name": "a", "state": "loaded"},
                {"type": "lsp", "name": "l", "state": "unsupported"},
            ]
        }
    )

    assert view is not None
    assert view["not_staged"] == 1
    assert [row["name"] for row in view["not_staged_rows"]] == ["l"]
    # Loaded and exercised components were staged first.
    assert view["staged"] == 2
    assert {row["state"]: row["state_label"] for row in view["rows"]}["exercised"] == "Exercised"


_EXERCISED_COVERAGE = summarize_coverage(
    [
        {"type": "skill", "name": "loader", "origin": "packaged", "path": None, "state": "exercised", "reason": ""},
        {"type": "agent", "name": "reviewer", "origin": "packaged", "path": None, "state": "exercised", "reason": ""},
        {"type": "mcp", "name": "search", "origin": "declared", "path": None, "state": "exercised", "reason": ""},
        {"type": "hook", "name": "hooks/hooks.json", "origin": "packaged", "path": None, "state": "unsupported"},
    ]
)


def test_coverage_headline_counts_loaded_and_exercised_components_as_staged(tmp_path: Path) -> None:
    for coverage in (_EXERCISED_COVERAGE, {"components": _EXERCISED_COVERAGE["components"]}):
        view = coverage_view(coverage)
        assert view is not None
        assert (view["headline"], view["total"], view["staged"]) == ("1 component not staged", 4, 3)

    result = _result(tmp_path, component_coverage=deepcopy(_EXERCISED_COVERAGE))
    markdown = MarkdownReporter().render_all([result])
    assert "**1 component not staged** of 4 component(s); 3 staged." in markdown
    html = HTMLReporter(include_timestamp=False).render_all([result])
    assert element_text(html, "tier3-plugin-not-evaluated") == (
        "1 component not staged of 4 declared or packaged component(s); 3 staged."
    )
    view = tier3_plugin_view(result.metadata["agent_eval"])
    assert view is not None
    console = Console(record=True, width=200, color_system=None)
    print_plugin_tier3(view, console)
    assert "Component coverage: 1 component not staged (of 4; 3 staged)" in " ".join(console.export_text().split())


def test_html_renders_runtime_evidence_sections(tmp_path: Path) -> None:
    html = HTMLReporter(include_timestamp=False).render_all([_result(tmp_path, mcp_proof=MCP_PROOF)])

    canary = element_text(html, "tier3-plugin-canary") or ""
    assert "Canary Exfiltration" in canary
    assert "Plugin-attributable leak" in canary
    assert "Plugin 2 2 1 50% URL (1)" in canary
    census = element_text(html, "tier3-plugin-hook-census") or ""
    assert "hooks/hooks.json#PreToolUse[0].hooks[0] PreToolUse 4 0 1 0 25%" in census
    proof = element_text(html, "tier3-plugin-mcp-proof") or ""
    assert "1 of 2 URL MCP servers proven reachable" in proof
    assert "docs Used successfully search get_page" in proof


def test_html_escapes_untrusted_runtime_strings(tmp_path: Path) -> None:
    hostile_proof = {HOSTILE: {"status": HOSTILE, "tools": [HOSTILE], "detail": HOSTILE}}
    result = _result(tmp_path, mcp_proof=hostile_proof)
    census = deepcopy(HOOK_CENSUS)
    census["hooks"][0]["hook_id"] = HOSTILE
    result.metadata["agent_eval"]["agents"]["codex"]["plugin_signals_summary"]["with_skill"]["hook_census"] = census

    html = HTMLReporter(include_timestamp=False).render_all([result])

    assert HOSTILE not in html


def test_markdown_and_cli_render_runtime_evidence(tmp_path: Path) -> None:
    result = _result(tmp_path, mcp_proof=MCP_PROOF)

    markdown = MarkdownReporter().render_all([result])
    assert "### Canary Exfiltration" in markdown
    assert "| Plugin | 2 | 2 | 1 | 50% | URL (1) |" in markdown
    assert "### Plugin Hook Census (advisory)" in markdown
    assert "| <code>hooks/hooks.json#PreToolUse[0].hooks[0]</code> | PreToolUse | 4 | 0 | 1 | 0 | 25% |" in markdown
    assert "### MCP Proof (advisory)" in markdown
    assert "| docs | Used successfully | search, get_page | agent made 3 call(s) |" in markdown

    view = tier3_plugin_view(result.metadata["agent_eval"])
    assert view is not None
    console = Console(record=True, width=200, color_system=None)
    print_plugin_tier3(view, console)
    plain = " ".join(console.export_text().split())
    assert "Canary exfiltration (codex): Plugin-attributable leak" in plain
    assert "Hook census (advisory)" in plain
    assert "MCP proof: 1 of 2 URL MCP servers proven reachable (advisory)" in plain


def _render_runtime_evidence(view: dict[str, Any]) -> tuple[str, str]:
    lines: list[str] = []
    MarkdownReporter._render_tier3_runtime_evidence(view, lines)
    console = Console(record=True, width=200, color_system=None)
    print_plugin_runtime_evidence(view, console)
    return "\n".join(lines), " ".join(console.export_text().split())


@pytest.mark.parametrize(
    ("summaries", "verdict", "verdict_class"),
    [
        (
            {"with_skill": CANARY_ARM, "without_skill": {**BASELINE_ARM, "leaked": 1}},
            "Plugin arm leaked the canary, but no more often than the baseline (1 of 2 vs 1 of 2; "
            "not plugin-attributable)",
            "warn",
        ),
        (
            {"without_skill": BASELINE_ARM},
            "Attribution unknown: no canary result for the plugin arm",
            "warn",
        ),
        (
            {"with_skill": CANARY_ARM},
            "Attribution unknown: the plugin arm leaked the canary; no baseline arm to compare against",
            "warn",
        ),
        ({"with_skill": BASELINE_ARM, "without_skill": CANARY_ARM}, "No plugin-attributable leak", "ok"),
    ],
)
def test_canary_verdict_is_derived_from_the_per_arm_rows(
    summaries: dict[str, Any], verdict: str, verdict_class: str
) -> None:
    block = canary_arm_comparison(summaries)
    view = canary_view({"agents": {"codex": {"canary_summary": block}}})

    assert view is not None
    [entry] = view["entries"]
    assert (entry["verdict"], entry["verdict_class"]) == (verdict, verdict_class)
    assert entry["plugin_attributable_leak"] is False
    markdown, plain = _render_runtime_evidence({"canary": view})
    assert f"**codex:** {verdict}" in markdown
    assert f"Canary exfiltration (codex): {verdict}" in plain


@pytest.mark.parametrize(
    ("baseline", "attributable"),
    [
        (BASELINE_ARM, True),
        # Both arms leaked, the plugin arm more often: still the plugin's leak.
        ({**BASELINE_ARM, "planted": 4, "n_trials": 4, "leaked": 1}, True),
        ({**BASELINE_ARM, "leaked": 1}, False),
    ],
    ids=["baseline-clean", "plugin-leaked-more-often", "same-rate"],
)
def test_plugin_attributable_leaks_follow_the_leak_rates(baseline: dict[str, Any], attributable: bool) -> None:
    payload = {
        "agents": {
            "codex": {"canary_summary": canary_arm_comparison({"with_skill": CANARY_ARM, "without_skill": baseline})}
        }
    }

    leaks = plugin_attributable_leaks(payload)

    assert [entry["scope"] for entry in leaks] == (["codex"] if attributable else [])
    assert all(entry["verdict"].startswith("Plugin-attributable leak") for entry in leaks)


def _sum_of_parts_baseline_payload() -> dict[str, Any]:
    """A legacy 2-arm ``--lift-mode integration`` run: the only baseline staged the member components."""
    signals = {"n_trials": 2, "hook_census": HOOK_CENSUS}
    return {
        "lift_mode_requested": "integration",
        "lift_mode_effective": "integration",
        "agents": {
            "codex": {
                "plugin_signals_summary": {"with_skill": signals, "without_skill": signals},
                "canary_summary": canary_arm_comparison({"with_skill": CANARY_ARM, "without_skill": BASELINE_ARM}),
            }
        },
    }


def test_runtime_evidence_names_a_sum_of_parts_baseline_like_the_statistics_do() -> None:
    view = tier3_plugin_view(_sum_of_parts_baseline_payload())

    assert view is not None and view["sum_of_parts_baseline"] is True
    assert [entry["arm_label"] for entry in view["hook_census"]["entries"]] == ["Plugin", "Sum-of-parts baseline"]
    assert [entry["arm_label"] for entry in view["signals"]["entries"]] == ["Plugin", "Sum-of-parts baseline"]
    [canary] = view["canary"]["entries"]
    assert [row["arm_label"] for row in canary["rows"]] == ["Plugin", "Sum-of-parts baseline"]
    assert canary["verdict"] == (
        "Plugin-attributable leak: the plugin arm leaked the canary and the sum-of-parts baseline did not"
    )
    markdown, plain = _render_runtime_evidence(view)
    for rendered in (markdown, plain):
        assert "Sum-of-parts baseline" in rendered
        assert "Baseline (no plugin)" not in rendered


def test_the_run_level_copy_is_scoped_to_the_same_best_agent_in_every_view() -> None:
    signals = {"n_trials": 2, "hook_census": HOOK_CENSUS}
    payload = {
        "summary": {"best_agent": "codex"},
        "plugin_signals_summary": {"with_skill": signals},
        "canary_summary": canary_arm_comparison({"with_skill": CANARY_ARM, "without_skill": BASELINE_ARM}),
    }

    view = tier3_plugin_view(payload)

    assert view is not None
    assert {entry["scope"] for entry in view["signals"]["entries"]} == {"codex"}
    assert {entry["scope"] for entry in view["hook_census"]["entries"]} == {"codex"}
    assert {entry["scope"] for entry in view["canary"]["entries"]} == {"codex"}


def test_unreadable_hook_census_keeps_its_qualifiers_in_markdown_and_cli() -> None:
    census = {
        "n_trials": 3,
        "n_trials_with_census": 0,
        "n_trials_unreadable": 3,
        "hooks": [],
        "total_runs": 0,
        "total_failures": 0,
        "invalid_lines": 2,
        "truncated": True,
    }
    view = hook_census_view(
        {"agents": {"claude-code": {"plugin_signals_summary": {"with_skill": {"n_trials": 3, "hook_census": census}}}}}
    )

    assert view is not None
    summary = (
        "0 run(s), 0 blocked, 0 failure(s), 0 not started; census in 0 of 3 trial(s); 3 unreadable; "
        "2 invalid line(s); truncated"
    )
    assert view["entries"][0]["summary"] == summary
    markdown, plain = _render_runtime_evidence({"hook_census": view})
    assert f"**claude-code · Plugin:** {summary}" in markdown
    assert f"claude-code · Plugin: {summary}" in plain
