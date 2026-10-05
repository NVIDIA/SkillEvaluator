# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin sections in the JSON, Markdown, SARIF, HTML, and terminal reports."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

import pytest
from _plugin_fixtures import (
    HOSTILE,
    SIGNALS_SUMMARY,
    STATISTICS,
    element_text,
    provenance,
    tier1_plugin_result,
    tier2_plugin_result,
    write_run_dir,
)

from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory
from skillevaluator.models import ValidationResult
from skillevaluator.reporting import HTMLReporter, JSONReporter
from skillevaluator.reporting.cli import CLIReporter
from skillevaluator.reporting.markdown import MarkdownReporter
from skillevaluator.reporting.plugin_sections import (
    component_for_path,
    coverage_view,
    is_plugin_payload,
    split_display_prefix,
    statistics_view,
    tier3_plugin_view,
)
from skillevaluator.reporting.sarif_reporter import SARIFReporter
from skillevaluator.tier3.harbor.report_data import load_agent_data
from skillevaluator.tier3.result_display import render_result

PLUGIN_SECTION_IDS = (
    "plugin-overview",
    "tier3-plugin-incomplete",
    "tier3-plugin-coverage",
    "tier3-plugin-completeness",
    "tier3-integration",
    "tier3-plugin-statistics",
    "tier3-plugin-signals",
)


@pytest.fixture(autouse=True)
def _no_llm_insights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "skillevaluator.evaluation.insights_judge.build_insights",
        lambda *_args, **_kwargs: {"conclusions": [], "recommendations": []},
    )


def _tier3_result(
    tmp_path: Path,
    *,
    partial: bool = True,
    statistics: bool = True,
    integration: dict | None = None,
    sidecar: bool = True,
) -> ValidationResult:
    plugin_dir = tmp_path / "demo-plugin"
    plugin_dir.mkdir(exist_ok=True)
    run_dir = write_run_dir(
        tmp_path / "results" / "20260101_000000",
        sidecar=provenance(partial=partial) if sidecar else None,
    )
    result = agent_eval_result_from_directory(plugin_dir, run_dir, use_llm_judge=False)
    assert result is not None
    payload = result.metadata["agent_eval"]
    if statistics:
        # Stats places the best agent's statistics at the top level and per agent.
        payload.update(deepcopy(STATISTICS))
        payload["agents"]["codex"].update(deepcopy(STATISTICS))
        payload["agents"]["codex"]["integration_completeness"] = {
            "with_plugin": {"execution_status": "succeeded"},
            "sum_of_parts": {"execution_status": "failed"},
            "complete": False,
            "missing_cases": ["case-2"],
            "failed_arms": ["sum_of_parts"],
            "attempt_shortfall": [{"case": "case-3", "arm": "sum_of_parts", "expected": 3, "observed": 1}],
        }
        payload["lift_mode_requested"] = "both"
        payload["lift_mode_effective"] = "effectiveness"
    if integration is not None:
        payload["integration"] = integration
    return result


def _skill_results() -> list[ValidationResult]:
    tier1 = ValidationResult(validator_name="Schema Check", validator_description="Tier 1")
    tier1.add_success("schema", "valid")
    return [tier1]


# ---------------------------------------------------------------------------
# Tier 1 / Tier 2 plugin block
# ---------------------------------------------------------------------------


def test_json_plugin_block_merges_tier1_and_tier2_keys() -> None:
    data = json.loads(JSONReporter(include_timestamp=False).render_all([tier1_plugin_result(), tier2_plugin_result()]))

    plugin = data["plugin"]
    assert plugin["name"] == "demo-plugin"
    assert plugin["manifest_type"] == "agent_plugin"
    assert plugin["dependency_status_counts"]["missing"] == 1
    assert plugin["component_inventory"]["unsupported_types_present"] == ["hook"]
    assert plugin["mcp"]["pinning"]["ratio"] == 0.5
    assert plugin["context_cost"]["method"] == "static_estimate"
    assert plugin["catalog_skill_similarity"]["matches"][0]["match"] == "catalog-loader"
    assert plugin["inter_plugin_similarity"]["status"] == "compared"


def test_json_plugin_block_is_json_safe() -> None:
    result = tier1_plugin_result()
    result.metadata["plugin"]["mcp"]["pinning"]["ratio"] = float("nan")

    data = json.loads(JSONReporter(include_timestamp=False).render_all([result]))

    assert data["plugin"]["mcp"]["pinning"]["ratio"] is None


def test_legacy_plugin_results_derive_bundled_skills_from_prefixed_findings() -> None:
    result = tier1_plugin_result(finding_path="[loader] SKILL.md")
    del result.metadata["plugin"]["bundled_skills"]

    html = HTMLReporter(include_timestamp=False).render_all([result])

    assert "Bundled skills: loader" in (element_text(html, "plugin-overview") or "")


def test_json_without_plugin_has_no_plugin_block() -> None:
    data = json.loads(JSONReporter(include_timestamp=False).render_all(_skill_results()))

    assert "plugin" not in data


def test_markdown_plugin_section_renders_contract_blocks() -> None:
    markdown = MarkdownReporter(include_timestamp=False).render_all([tier1_plugin_result(), tier2_plugin_result()])
    section = markdown.split("## Plugin", 1)[1].split("## Results", 1)[0]

    assert "| demo-plugin | ✅ PASSED | agent_plugin | bundle |" in section
    assert "**Declared dependencies:** mcp=2, rules=1, skills=2" in section
    assert "**Bundled skills (1):** skills/loader" in section
    assert "**Status counts:** provided=1, referenced=0, missing=1, external=1, unresolved=0" in section
    assert "| rule | rules/missing.md | missing | — |" in section
    assert "&lt;script&gt;" in section and HOSTILE not in section
    assert "**Unsupported component types present:** hook." in section
    assert "| hook | pre-commit | packaged | Unsupported | 0 |" in section
    assert "**Pinned:** 1/2 pinned (50%)" in section
    assert "| search | command | npx search-server@latest |" in section
    assert "### Context cost (Static estimate (characters ÷ 4))" in section
    assert "**Always-on:** 1,200 tokens · **On-demand:** 3,400 tokens" in section
    assert "not a measured token count" in section
    assert "| loader | catalog-loader | 0.91 |" in section
    assert "| other-plugin | 0.84 | 50% | overlapping |" in section


def test_markdown_tier3_plugin_blocks_state_what_was_not_evaluated(tmp_path: Path) -> None:
    integration = {"verdict": "inconclusive", "measured": False, "reason": "No cross-component case completed."}
    markdown = MarkdownReporter(include_timestamp=False).render_all([_tier3_result(tmp_path, integration=integration)])

    assert "INCOMPLETE: 1 unresolved skill ref, 1 provider-only MCP server" in markdown
    assert "**2 components not staged** of 4 declared or packaged component(s); 2 staged." in markdown
    assert "| mcp | docs | Unavailable | provider-only MCP server |" in markdown
    excluded = markdown.split("**Not evaluated by this run:**", 1)[1].split("###", 1)[0]
    assert "- 2 components not staged: mcp docs, hook pre-commit" in excluded
    assert "- Unresolved skill refs were not evaluated: github::org/repo::skills::remote" in excluded
    assert "- Provider-only MCP servers were not exercised: docs" in excluded
    assert "- Integration (the plugin versus its own parts) was not measured: No cross-component case completed." in (
        excluded
    )
    assert "**Lift mode:** requested <code>both</code>, effective <code>effectiveness</code>" in markdown
    assert "**INCONCLUSIVE:** No cross-component case completed." in markdown
    assert "Effectiveness lift: +0.30 [-0.02, +0.55] (95% CI), precision low — ⚠️ CI includes zero" in markdown


def test_markdown_without_plugin_has_no_plugin_section() -> None:
    markdown = MarkdownReporter(include_timestamp=False).render_all(_skill_results())

    assert "## Plugin" not in markdown


def test_markdown_with_an_empty_plugin_block_has_no_plugin_section() -> None:
    result = ValidationResult(validator_name="Schema Check", validator_description="Tier 1", metadata={"plugin": {}})
    result.add_success("schema", "valid")

    markdown = MarkdownReporter(include_timestamp=False).render_all([result])

    assert "## Plugin" not in markdown
    assert "## Results" in markdown


def test_sarif_run_and_results_carry_plugin_context(tmp_path: Path) -> None:
    results = [tier1_plugin_result(), tier2_plugin_result(), _tier3_result(tmp_path)]
    document = json.loads(SARIFReporter(include_timestamp=False).render_all(results))
    run = document["runs"][0]

    plugin = run["properties"]["plugin"]
    assert plugin["name"] == "demo-plugin"
    assert plugin["manifestType"] == "agent_plugin"
    assert plugin["dependencyStatusCounts"]["missing"] == 1
    assert plugin["componentCounts"] == {"hook": 1, "mcp": 1, "skill": 1}
    assert plugin["unsupportedTypesPresent"] == ["hook"]
    assert plugin["mcpPinning"]["ratio"] == 0.5
    assert plugin["contextCost"]["alwaysOnTokens"] == 1200
    assert plugin["catalogSkillSimilarity"] == {
        "status": "compared",
        "catalogEntries": 12,
        "matches": 1,
        "advisory": True,
    }
    assert plugin["evaluationIncomplete"] is True
    assert plugin["componentsNotStaged"] == 2
    assert plugin["componentsStagedNotObserved"] == 0
    finding = next(item for item in run["results"] if item["properties"]["checkName"] == "description_short")
    assert finding["properties"]["pluginComponent"] == {
        "type": "skill",
        "name": "loader",
        "path": "skills/loader",
        "support": "evaluated",
    }


def test_sarif_without_plugin_has_no_run_properties() -> None:
    document = json.loads(SARIFReporter(include_timestamp=False).render_all(_skill_results()))

    assert "properties" not in document["runs"][0]


@pytest.mark.parametrize(
    ("file_path", "expected"),
    [
        ("skills/loader/SKILL.md", "loader"),
        ("./skills/loader/scripts/run.py", "loader"),
        ("/work/demo-plugin/hooks/pre.sh", "pre-commit"),
        ("[loader] skills/loader/SKILL.md", "loader"),
        # A bundled skill's path relative to the skill, as Tier 2 reports it.
        ("[loader] SKILL.md", "loader"),
        # A file inside the skill, not the plugin's own hooks/pre.sh.
        ("[loader] hooks/pre.sh", "loader"),
        ("[loader] /work/demo-plugin/skills/loader/SKILL.md", "loader"),
        ("README.md", None),
        ("/elsewhere/skills/loader/SKILL.md", None),
        ("skills/loader-extra/SKILL.md", None),
    ],
)
def test_component_for_path_maps_findings_to_components(file_path: str, expected: str | None) -> None:
    block = tier1_plugin_result().metadata["plugin"]

    component = component_for_path(file_path, block)

    assert (component or {}).get("name") == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("[loader] skills/loader/SKILL.md", ("loader", "skills/loader/SKILL.md")),
        ("[a] [b] SKILL.md", ("a", "[b] SKILL.md")),
        ("skills/loader/SKILL.md", (None, "skills/loader/SKILL.md")),
        ("[draft]notes.md", (None, "[draft]notes.md")),
    ],
)
def test_split_display_prefix_splits_only_the_merge_label(path: str, expected: tuple[str | None, str]) -> None:
    assert split_display_prefix(path) == expected


def test_html_tier1_plugin_section_renders_and_escapes() -> None:
    html = HTMLReporter(include_timestamp=False, content_label="Plugin").render_all(
        [tier1_plugin_result(), tier2_plugin_result()]
    )

    overview = element_text(html, "plugin-overview") or ""
    assert "Plugin: demo-plugin" in overview
    assert "Manifest: agent_plugin ( agent_plugin.yaml )" in overview
    dependencies = element_text(html, "plugin-dependency-resolution") or ""
    assert "1 missing" in dependencies
    assert "github::org/repo::skills::remote external" in dependencies
    assert HOSTILE not in html
    assert "&lt;script&gt;alert(&#39;x&#39;)&lt;/script&gt;" in html
    inventory = element_text(html, "plugin-component-inventory") or ""
    assert "hook pre-commit hooks/pre.sh packaged Unsupported 0" in inventory
    assert "Unsupported component types present: hook" in (element_text(html, "plugin-unsupported-types") or "")
    pinning = element_text(html, "plugin-mcp-pinning") or ""
    assert "50% pinned ratio" in pinning and "search command inline npx search-server@latest" in pinning
    cost = element_text(html, "plugin-context-cost") or ""
    assert "Static estimate (characters ÷ 4)" in cost
    assert "1,200 always-on tokens" in cost and "3,400 on-demand tokens" in cost
    skills = element_text(html, "plugin-catalog-skill-similarity") or ""
    assert "Advisory" in skills and "loader catalog-loader 0.91" in skills
    plugins = element_text(html, "plugin-inter-plugin-similarity") or ""
    assert "other-plugin 0.84 50% overlapping" in plugins


def test_html_skill_report_has_no_plugin_sections() -> None:
    html = HTMLReporter(include_timestamp=False).render_all(_skill_results())

    for section_id in PLUGIN_SECTION_IDS:
        assert element_text(html, section_id) is None
    assert ".plugin-section" not in html


# ---------------------------------------------------------------------------
# Tier 3 plugin report
# ---------------------------------------------------------------------------


def test_html_tier3_coverage_states_what_was_not_demonstrated(tmp_path: Path) -> None:
    html = HTMLReporter(include_timestamp=False).render_all([_tier3_result(tmp_path)])

    coverage = element_text(html, "tier3-plugin-coverage") or ""
    assert "2 components not staged of 4 declared or packaged component(s); 2 staged." in coverage
    assert "Files staged ≠ components loaded ≠ behavior verified." in coverage
    assert "does not verify the component's behavior" in coverage
    assert "mcp docs declared Unavailable not observed provider-only MCP server" in coverage
    assert "hook pre-commit hooks/pre.sh packaged Unsupported unavailable hooks are not evaluated" in coverage
    assert "skill loader skills/loader declared+packaged Staged exercised" in coverage
    excluded = element_text(html, "tier3-plugin-excluded") or ""
    assert "Provider-only MCP servers were not exercised: docs" in excluded
    assert "Unresolved skill refs were not evaluated: github::org/repo::skills::remote" in excluded
    assert element_text(html, "tier3-plugin-not-evaluated").startswith("2 components not staged")


def test_staged_but_unexercised_components_are_not_reported_as_evaluated(tmp_path: Path) -> None:
    """Every component staged, none exercised: the headline counts staging and the rest is listed as excluded."""
    from io import StringIO

    from rich.console import Console

    from skillevaluator.reporting import BenchmarkReporter
    from skillevaluator.reporting.cli import print_plugin_tier3

    result = _tier3_result(tmp_path, partial=False, statistics=False)
    declared = ["skill:loader", "mcp:search", "mcp:docs", "hook:pre-commit"]
    result.metadata["agent_eval"]["agents"]["codex"]["plugin_signals_summary"]["with_skill"]["activation_coverage"] = {
        "declared": declared,
        "exercised": [],
        "unverified": declared,
        "unavailable": [],
    }

    view = tier3_plugin_view(result.metadata["agent_eval"])
    assert view is not None
    coverage = view["coverage"]
    assert coverage["headline"] == "0 components not staged"
    assert coverage["staged_not_observed"] == 4
    assert coverage["observed_headline"] == "4 staged components not observed in any plugin trial"
    assert coverage["all_exercised"] is False
    unobserved = "Staged but not observed in any plugin trial: skill loader, mcp search, mcp docs, hook pre-commit"
    assert unobserved in view["excluded"]

    html = HTMLReporter(include_timestamp=False).render_all([result])
    section = element_text(html, "tier3-plugin-coverage") or ""
    assert "not evaluated" not in (element_text(html, "tier3-plugin-not-evaluated") or "")
    assert (
        "0 components not staged of 4 declared or packaged component(s); 4 staged; "
        "4 staged components not observed in any plugin trial."
    ) in section
    assert '<span class="t3-pill warn">0 components not staged</span>' in html
    assert "4 staged, not observed" in section
    assert unobserved in (element_text(html, "tier3-plugin-excluded") or "")
    markdown = MarkdownReporter(include_timestamp=False).render_all([result])
    assert (
        "**0 components not staged** of 4 declared or packaged component(s); 4 staged; 4 staged components not observed in any plugin "
        "trial."
    ) in markdown
    console = Console(file=StringIO(), width=200, color_system=None)
    print_plugin_tier3(view, console)
    assert (
        "Component coverage: 0 components not staged (of 4 declared or packaged component(s); 4 staged; "
        "4 staged components not observed in any plugin trial)"
    ) in " ".join(console.file.getvalue().split())
    sarif = json.loads(SARIFReporter(include_timestamp=False).render_all([tier1_plugin_result(), result]))
    assert sarif["runs"][0]["properties"]["plugin"]["componentsStagedNotObserved"] == 4
    card = BenchmarkReporter(include_timestamp=False, content_type="plugin", skill_name="demo-plugin").render_all(
        [result]
    )
    assert "- Component coverage: 0 components not staged; 4 staged components not observed" in card
    assert f"- {unobserved}" in card
    assert "No declared component or measurement was recorded as excluded" not in card


def test_coverage_reads_complete_only_when_every_component_was_exercised() -> None:
    coverage = {
        "components": [
            {"type": "skill", "name": "loader", "state": "staged"},
            {"type": "mcp", "name": "search", "state": "staged"},
        ]
    }
    exercised = {
        "activation": {
            "declared": ["skill:loader", "mcp:search"],
            "exercised": ["skill:loader", "mcp:search"],
            "unverified": [],
            "unavailable": [],
        }
    }

    complete = coverage_view(coverage, exercised)
    assert complete is not None
    assert complete["all_exercised"] is True
    assert complete["observed_headline"] == ""
    # Without activation data only staging is known, which never reads as complete.
    staged_only = coverage_view(coverage)
    assert staged_only is not None
    assert staged_only["all_exercised"] is False
    assert staged_only["staged_not_observed"] is None


def test_html_tier3_integration_renders_inconclusive_with_reason(tmp_path: Path) -> None:
    integration = {
        "verdict": "inconclusive",
        "measured": False,
        "reason": "No cross-component case completed in both arms.",
        "lift_mode_requested": "both",
        "lift_mode_effective": "effectiveness",
    }
    html = HTMLReporter(include_timestamp=False).render_all([_tier3_result(tmp_path, integration=integration)])

    block = element_text(html, "tier3-integration") or ""
    assert "INCONCLUSIVE" in block
    assert "Lift mode: requested both · effective effectiveness fell back" in block
    inconclusive = element_text(html, "tier3-integration-inconclusive") or ""
    assert "Integration was not measured." in inconclusive
    assert "No cross-component case completed in both arms." in inconclusive


def test_html_tier3_integration_is_synthesized_when_requested_but_missing(tmp_path: Path) -> None:
    html = HTMLReporter(include_timestamp=False).render_all([_tier3_result(tmp_path)])

    inconclusive = element_text(html, "tier3-integration-inconclusive") or ""
    assert "Integration was requested, but this run recorded no sum-of-parts comparison." in inconclusive


def test_html_tier3_measured_integration_shows_ci_and_point_verdict(tmp_path: Path) -> None:
    integration = {
        "verdict": "inconclusive",
        "point_verdict": "real_integration",
        "measured": True,
        "with_plugin": 0.8,
        "sum_of_parts": 0.65,
        "integration_lift": 0.15,
        "components": ["loader", "summarizer"],
        "interpretation": "The sum-of-parts comparison did not produce complete comparable evidence.",
        "lift_uncertainty": {
            "estimate": 0.15,
            "ci_low": -0.05,
            "ci_high": 0.3,
            "confidence": 0.95,
            "precision": "insufficient",
            "ci_includes_zero": True,
        },
        "lift_mode_requested": "both",
        "lift_mode_effective": "both",
    }
    result = _tier3_result(tmp_path, integration=integration)
    html = HTMLReporter(include_timestamp=False).render_all([result])

    block = element_text(html, "tier3-integration") or ""
    assert "Inconclusive point estimate: Real integration 0.80 0.65 +0.15" in block
    assert "[-0.05, +0.30] precision: insufficient CI includes zero" in block
    assert "loader summarizer" in block
    assert element_text(html, "tier3-integration-inconclusive") is None


def test_html_tier3_statistics_render_ci_reliability_cost_and_context(tmp_path: Path) -> None:
    html = HTMLReporter(include_timestamp=False).render_all([_tier3_result(tmp_path)])

    statistics = element_text(html, "tier3-plugin-statistics") or ""
    assert "Advisory · Report-only" in statistics
    assert "Effectiveness lift +0.30 [-0.02, +0.55] 95% CI precision: low CI includes zero 4" in statistics
    assert "includes zero: this run cannot distinguish that lift from no effect" in statistics
    assert "Baseline (no plugin) 50% 25% 3 4 n/a n/a 0.41" in statistics
    # Tokens without USD read "not priced", not a blank column or $0.
    assert "USD / success" in statistics
    assert "Plugin 75% 50% 3 4 12,000 not priced 0.62" in statistics
    assert "not priced: the run recorded tokens but no USD cost" in statistics
    assert "pass^k: a case passes only when all k attempts pass." in statistics
    context = element_text(html, "tier3-plugin-context-measured") or ""
    assert "+1,450 tokens per first turn across 4 paired trial(s)" in context
    assert "Missing cases: case-2" in statistics
    assert "Failed arms: Sum of parts" in statistics
    assert "Attempt shortfall in case-3 (Sum of parts): 1 of 3 attempts observed" in statistics


def test_integration_lift_mode_labels_the_baseline_as_sum_of_parts(tmp_path: Path) -> None:
    """With --lift-mode integration the baseline arm and its interval describe the sum of parts."""
    from io import StringIO

    from rich.console import Console

    from skillevaluator.reporting.cli import print_plugin_tier3

    result = _tier3_result(tmp_path, partial=False)
    result.metadata["agent_eval"]["lift_mode_requested"] = "integration"
    result.metadata["agent_eval"]["lift_mode_effective"] = "integration"

    view = tier3_plugin_view(result.metadata["agent_eval"])
    assert view is not None and view["sum_of_parts_baseline"] is True
    assert [row["label"] for row in view["statistics"]["primary"]["lift_ci"]] == [
        "Integration lift (sum-of-parts baseline)"
    ]
    assert [arm["label"] for arm in view["statistics"]["primary"]["arms"]] == ["Plugin", "Sum-of-parts baseline"]
    assert view["statistics"]["primary"]["completeness"]["failed_arms"] == ["Sum of parts"]

    markdown = MarkdownReporter(include_timestamp=False).render_all([result])
    assert "Integration lift (sum-of-parts baseline): +0.30 [-0.02, +0.55] (95% CI)" in markdown
    html = HTMLReporter(include_timestamp=False).render_all([result])
    statistics = element_text(html, "tier3-plugin-statistics") or ""
    assert "Integration lift (sum-of-parts baseline) +0.30 [-0.02, +0.55]" in statistics
    assert "Sum-of-parts baseline 50% 25% 3 4" in statistics
    console = Console(file=StringIO(), width=200, color_system=None)
    print_plugin_tier3(view, console)
    plain = " ".join(console.file.getvalue().split())
    assert "Integration lift (sum-of-parts baseline)" in plain
    assert "Sum-of-parts baseline" in plain
    for rendered in (markdown, statistics, plain):
        assert "Effectiveness lift" not in rendered
        assert "Baseline (no plugin)" not in rendered


def test_html_tier3_statistics_show_usd_only_when_present(tmp_path: Path) -> None:
    result = _tier3_result(tmp_path)
    for target in (result.metadata["agent_eval"], result.metadata["agent_eval"]["agents"]["codex"]):
        target["cost"]["with_skill"]["usd_per_success"] = 0.0425

    html = HTMLReporter(include_timestamp=False).render_all([result])

    statistics = element_text(html, "tier3-plugin-statistics") or ""
    assert "USD / success" in statistics
    assert "$0.0425" in statistics


def test_html_tier3_signals_are_advisory_and_render_every_block(tmp_path: Path) -> None:
    html = HTMLReporter(include_timestamp=False).render_all([_tier3_result(tmp_path)])

    signals = element_text(html, "tier3-plugin-signals") or ""
    assert signals.startswith("Plugin Signals — Advisory")
    assert "codex · Plugin Advisory 3 trial(s); 1 without a trajectory" in signals
    assert "Tool selection (precision / recall / F1) 80% / 67% / 73% 1 decoy call(s) (10% of calls)" in signals
    assert "Tool arguments 75% pass rate 3/4 checks passed" in signals
    assert "MCP call success 80% 4/5 succeeded; 1 failed; 0 unknown" in signals
    assert "Order checks 1/2 passed (50%)" in signals
    assert "Handoff checks not configured for this dataset" in signals
    assert "Conflict checks 1/1 passed (100%)" in signals
    assert "Activation coverage 67% exercised declared 3, exercised 2, unverified 0, unavailable 1" in signals
    assert "mcp__search__query q required missing q 1" in signals
    assert "search 5 4 1 0 80% mcp__search__query" in signals


def test_html_tier3_signals_render_null_ratios_as_not_available(tmp_path: Path) -> None:
    result = _tier3_result(tmp_path, statistics=False)
    summary = result.metadata["agent_eval"]["agents"]["codex"]["plugin_signals_summary"]["with_skill"]
    summary["mcp_calls"].update({"total": 0, "succeeded": 0, "success_rate": None, "by_server": {}})

    html = HTMLReporter(include_timestamp=False).render_all([result])

    assert "MCP call success n/a 0/0 succeeded" in (element_text(html, "tier3-plugin-signals") or "")


def test_html_tier3_escapes_untrusted_plugin_strings(tmp_path: Path) -> None:
    result = _tier3_result(tmp_path)
    payload = result.metadata["agent_eval"]
    payload["plugin_provenance"]["component_coverage"]["components"][2]["reason"] = HOSTILE
    payload["agents"]["codex"]["plugin_signals_summary"]["with_skill"]["mcp_calls"]["by_server"] = {
        HOSTILE: {"total": 1, "succeeded": 1, "tools": [HOSTILE]}
    }

    html = HTMLReporter(include_timestamp=False).render_all([result])

    assert HOSTILE not in html
    assert "&lt;script&gt;alert(&#39;x&#39;)&lt;/script&gt;" in (html or "")


def test_html_tier3_verdict_card_of_a_partial_plugin_run_is_warn_toned(tmp_path: Path) -> None:
    result = _tier3_result(tmp_path)
    assert result.metadata["agent_eval"]["verdict"] == "pass"

    html = HTMLReporter(include_timestamp=False).render_all([result])

    card = re.search(
        r'<div class="dashboard-card ([^"]*)">\s*<h3 class="dashboard-card-title">Verdict</h3>\s*'
        r'<p class="dashboard-card-value">([^<]*)</p>',
        html,
    )
    assert card is not None
    # The raw verdict passes, but a partial run reads INCOMPLETE and must not be colored as a success.
    assert card.groups() == ("warning", "INCOMPLETE")


def test_complete_plugin_run_reports_complete_and_no_incomplete_callout(tmp_path: Path) -> None:
    result = _tier3_result(tmp_path, partial=False)
    html = HTMLReporter(include_timestamp=False).render_all([result])

    assert result.passed is True
    assert element_text(html, "tier3-plugin-incomplete") is None
    assert "0 components not staged" in (element_text(html, "tier3-plugin-coverage") or "")
    completeness = element_text(html, "tier3-plugin-completeness") or ""
    assert completeness.startswith("Plugin Dependency Completeness — Complete")
    assert element_text(html, "tier3-plugin-dependency-counts") == (
        "Declared dependency resolution: provided: 1 referenced: 0 missing: 0 external: 1 unresolved: 0"
    )


def test_tier3_payload_carries_plugin_signals_per_agent_and_for_the_best_agent(tmp_path: Path) -> None:
    result = _tier3_result(tmp_path, statistics=False)
    payload = result.metadata["agent_eval"]

    agent_signals = payload["agents"]["codex"]["plugin_signals_summary"]
    assert agent_signals["with_skill"]["n_trials"] == SIGNALS_SUMMARY["n_trials"]
    assert agent_signals["with_skill"]["arguments"]["top_failures"] == [
        {"tool": "mcp__search__query", "arg": "q", "rule": "required", "detail": "missing q", "count": 1}
    ]
    assert payload["plugin_signals_summary"] == agent_signals


def test_report_data_loads_per_arm_plugin_signals(tmp_path: Path) -> None:
    run_dir = write_run_dir(tmp_path / "run")

    agents = load_agent_data(run_dir)

    assert agents["codex"]["plugin_signals_summary"] == {"with_skill": SIGNALS_SUMMARY}


def test_statistics_view_prefers_per_agent_blocks_with_best_agent_primary() -> None:
    other = deepcopy(STATISTICS)
    other["lift_uncertainty"]["effectiveness"]["estimate"] = -0.1
    payload = {
        "best_agent": "codex",
        **deepcopy(STATISTICS),
        "agents": {"aider": deepcopy(other), "codex": deepcopy(STATISTICS)},
    }

    view = statistics_view(payload)

    assert view is not None
    assert [scope["label"] for scope in view["scopes"]] == ["codex", "aider"]
    assert view["primary"]["lift_ci"][0]["estimate"] == "+0.30"


def test_skill_payloads_are_not_plugin_payloads(tmp_path: Path) -> None:
    payload = {"agents": {"codex": {"with_skill": 0.9}}, **deepcopy(STATISTICS)}

    assert is_plugin_payload(payload) is False
    assert tier3_plugin_view(payload) is None
    assert tier3_plugin_view({"eval_target": {"kind": "plugin"}}) is None


def test_cli_reporter_prints_tier3_plugin_blocks(tmp_path: Path) -> None:
    integration = {"verdict": "inconclusive", "measured": False, "reason": "No cross-component case completed."}
    output = CLIReporter().render_all([_tier3_result(tmp_path, integration=integration)])
    plain = re.sub(r"\x1b\[[0-9;]*m", "", output)
    plain = " ".join(plain.split())

    assert "INCOMPLETE: 1 unresolved skill ref" in plain
    assert "Component coverage: 2 components not staged (of 4 declared or packaged component(s); 2 staged)" in plain
    assert "Files staged ≠ components loaded ≠ behavior verified." in plain
    assert "Lift mode: requested both · effective effectiveness (fell back)" in plain
    assert "Integration: INCONCLUSIVE — No cross-component case completed. (advisory)" in plain
    assert "CI includes zero" in plain
    assert "Measured context delta: +1,450 tokens per first turn (4 pairs)" in plain
    assert "Plugin signals (advisory, report-only; never changes a score or verdict)" in plain
    assert "handoff not configured for this dataset" in plain


def test_engine_result_display_renders_plugin_signals_and_statistics() -> None:
    engine = {
        "execution_status": "succeeded",
        "run_config": {"eval_target": {"kind": "plugin"}},
        "agents": {
            "codex": {
                "with_skill": {"accuracy": 0.9},
                "without_skill": {"accuracy": 0.5},
                "execution_status": "succeeded",
                "conditions": {
                    "with_skill": {"execution_status": "succeeded"},
                    "without_skill": {"execution_status": "succeeded"},
                },
                "plugin_signals_summary": {"with_skill": deepcopy(SIGNALS_SUMMARY)},
                **deepcopy(STATISTICS),
            }
        },
    }

    output = " ".join(render_result(engine).split())

    assert "Plugin signals (advisory" in output
    assert "tool selection P/R/F1 80% / 67% / 73%" in output
    assert "Lift Uncertainty (advisory)" in output
    assert "Reliability and Cost (advisory)" in output


def test_engine_result_display_is_unchanged_for_skill_runs() -> None:
    engine = {
        "execution_status": "succeeded",
        "run_config": {"eval_target": {"kind": "skill"}},
        "agents": {"codex": {"with_skill": {"accuracy": 0.9}, "execution_status": "succeeded"}},
    }

    output = render_result(engine)

    assert "Plugin signals" not in output
    assert "Lift Uncertainty" not in output


# Saved by a run without a sum-of-parts arm (default effectiveness lift or --skip-baseline) before
# ``complete`` became None for such runs; newer runs save ``complete: None``.
_NO_SUM_OF_PARTS_COMPLETENESS = {
    "with_plugin": {
        "execution_status": "succeeded",
        "execution_errors": [],
        "expected_attempts": 2,
        "scored_attempts": 2,
    },
    "sum_of_parts": {"execution_status": "skipped", "execution_errors": []},
    "complete": False,
    "missing_cases": [],
    "failed_arms": [],
    "attempt_shortfall": [],
}


@pytest.mark.parametrize("complete", [False, None], ids=["saved-before", "saved-now"])
def test_a_run_without_a_sum_of_parts_arm_reports_no_integration_completeness_issue(
    tmp_path: Path, complete: bool | None
) -> None:
    result = _tier3_result(tmp_path)
    payload = result.metadata["agent_eval"]
    completeness = {**deepcopy(_NO_SUM_OF_PARTS_COMPLETENESS), "complete": complete}
    payload["integration_completeness"] = deepcopy(completeness)
    payload["agents"]["codex"]["integration_completeness"] = deepcopy(completeness)
    payload["lift_mode_requested"] = payload["lift_mode_effective"] = "effectiveness"

    cli = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", CLIReporter().render_all([result])).split())
    html = HTMLReporter(include_timestamp=False).render_all([result])
    markdown = MarkdownReporter(include_timestamp=False).render_all([result])

    for rendered in (cli, html, markdown):
        assert "completeness issues" not in rendered.lower()
    assert "Effectiveness lift" in (element_text(html, "tier3-plugin-statistics") or "")


def test_a_sum_of_parts_arm_that_ran_without_comparable_cases_is_still_an_issue(tmp_path: Path) -> None:
    result = _tier3_result(tmp_path)
    payload = result.metadata["agent_eval"]
    completeness = {**deepcopy(_NO_SUM_OF_PARTS_COMPLETENESS), "sum_of_parts": {"execution_status": "succeeded"}}
    payload["agents"]["codex"]["integration_completeness"] = completeness

    cli = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", CLIReporter().render_all([result])).split())

    assert "Integration completeness issues: the compared arms did not score the same cases" in cli


def test_html_activation_coverage_reads_the_per_component_rate_the_arm_summary_saves(tmp_path: Path) -> None:
    result = _tier3_result(tmp_path, statistics=False)
    summary = result.metadata["agent_eval"]["agents"]["codex"]["plugin_signals_summary"]["with_skill"]
    # summarize_plugin_signals saves one rate per declared component, not a single number.
    summary["activation_coverage"] = {
        "declared": ["skill:codename-lookup", "skill:release-note", "mcp:acme"],
        "exercised": ["skill:release-note", "mcp:acme"],
        "unverified": ["skill:codename-lookup"],
        "unavailable": [],
        "exercise_rate": {"skill:codename-lookup": 0.0, "skill:release-note": 0.5, "mcp:acme": 1.0},
    }

    html = HTMLReporter(include_timestamp=False).render_all([result])

    signals = element_text(html, "tier3-plugin-signals") or ""
    assert "Activation coverage 67% exercised declared 3, exercised 2, unverified 1, unavailable 0" in signals
    assert "n/a exercised" not in signals


def _many_components_view(count: int) -> dict:
    coverage = {"components": [{"type": "hook", "name": f"h{index}", "state": "unsupported"} for index in range(count)]}
    view = tier3_plugin_view({"plugin_provenance": {"plugin_name": "p", "component_coverage": coverage}})
    assert view is not None
    return view


def test_cli_counts_the_not_staged_components_past_the_row_limit() -> None:
    from io import StringIO

    from rich.console import Console

    from skillevaluator.reporting.cli import print_plugin_tier3

    view = _many_components_view(205)
    console = Console(file=StringIO(), width=200, color_system=None)

    print_plugin_tier3(view, console)

    plain = " ".join(console.file.getvalue().split())
    assert "Component coverage: 205 components not staged" in plain
    # Ten are listed; the other 195 are counted, not just the 190 left of the 200 kept rows.
    assert "... and 195 more" in plain


def test_the_not_staged_statement_counts_the_components_it_does_not_name() -> None:
    view = _many_components_view(15)

    [statement] = [line for line in view["excluded"] if line.startswith("15 components not staged")]

    assert statement.endswith("h11 (+3 more)")


def test_html_and_cli_say_how_many_integration_components_were_left_out(tmp_path: Path) -> None:
    from io import StringIO

    from rich.console import Console

    from skillevaluator.reporting.cli import print_plugin_tier3

    integration = {
        "verdict": "real_integration",
        "measured": True,
        "with_plugin": 0.8,
        "sum_of_parts": 0.6,
        "integration_lift": 0.2,
        "components": [f"skill-{index}" for index in range(70)],
    }
    result = _tier3_result(tmp_path, integration=integration)

    html = HTMLReporter(include_timestamp=False).render_all([result])
    view = tier3_plugin_view(result.metadata["agent_eval"])
    assert view is not None
    console = Console(file=StringIO(), width=400, color_system=None)
    print_plugin_tier3(view, console)

    assert "skill-63 (+6 more)" in (element_text(html, "tier3-integration") or "")
    assert "skill-63 (+6 more)" in " ".join(console.file.getvalue().split())


def test_a_partial_plugin_run_recorded_only_under_the_summary_is_incomplete() -> None:
    from skillevaluator.reporting.base import ReporterBase, is_partial_plugin_agent_eval

    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Tier 3")
    result.metadata["agent_eval"] = {"summary": {"plugin_provenance": provenance(partial=True)}}

    assert is_partial_plugin_agent_eval(result) is True
    assert ReporterBase._plugin_status([tier1_plugin_result(), result]) == "incomplete"
    assert tier3_plugin_view(result.metadata["agent_eval"])["partial"] is True


def test_similarity_views_carry_their_title_columns_and_summary() -> None:
    from skillevaluator.reporting.plugin_sections import similarity_view

    skills = similarity_view({"status": "compared", "catalog_entries": 1, "matches": []}, kind="skills")
    plugins = similarity_view(tier2_plugin_result().metadata["plugin"]["inter_plugin_similarity"], kind="plugins")

    assert skills is not None and plugins is not None
    assert skills["title"] == "Bundled skills vs. local skills catalog"
    assert skills["summary"] == "1 catalog entry compared. No similar entries found."
    assert [column["label"] for column in plugins["columns"]] == [
        "Catalog plugin",
        "Similarity",
        "Member overlap",
        "Verdict",
    ]
    markdown = MarkdownReporter(include_timestamp=False).render_all([tier1_plugin_result(), tier2_plugin_result()])
    assert "### Plugin vs. other plugins in the local catalog (advisory)" in markdown
    assert "**Status:** Compared · 3 catalog entries compared." in markdown
    assert "| Catalog plugin | Similarity | Member overlap | Verdict |" in markdown
