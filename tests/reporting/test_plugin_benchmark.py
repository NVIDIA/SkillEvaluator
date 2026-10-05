# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin BENCHMARK.md cards: complete, partial, and legacy runs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from _plugin_fixtures import STATISTICS, provenance, tier1_plugin_result, tier2_plugin_result
from click.testing import CliRunner
from scripts.ci import check_public_benchmarks as benchmark_gate

from skillevaluator import cli as cli_module
from skillevaluator.models import ValidationResult
from skillevaluator.reporting import BenchmarkReporter

if TYPE_CHECKING:
    import pytest

_DIMENSIONS = ("security", "correctness", "discoverability", "effectiveness", "efficiency")


def _tier3(*, partial: bool, integration: dict[str, Any] | None = None, coverage: bool = True) -> ValidationResult:
    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Run live agent evaluation")
    payload: dict[str, Any] = {
        "skill_name": "demo-plugin",
        "verdict": "pass",
        "execution_status": "succeeded",
        "evaluated_at": "2026-09-01T12:00:00+00:00",
        "evaluator_version": "0.3.0",
        "overall_lift": 0.3,
        "summary": {"environment": "docker", "verdict": "pass", "execution_status": "succeeded"},
        "dataset_summary": {
            "total_tasks": 4,
            "positive_tasks": 3,
            "negative_tasks": 1,
            "unclassified_tasks": 0,
            "source": "dataset",
        },
        "dataset_digest": "sha256:" + "0123456789abcdef" * 4,
        "dataset_digest_algorithm": "skill-evaluator-dataset-snapshot/1",
        "attempt_policy": {"max_attempts": 3, "pass_threshold": 0.5},
        "best_agent": "codex",
        "agents": {
            "codex": {
                "model": "gpt-codex",
                "execution_status": "succeeded",
                "baseline": 0.5,
                "with_skill": 0.8,
                "dimensions": [
                    {"id": dimension, "baseline": 0.5, "with_skill": 0.8, "lift": 0.3} for dimension in _DIMENSIONS
                ],
                **deepcopy(STATISTICS),
            }
        },
        "plugin_provenance": provenance(partial=partial, coverage=coverage),
        "lift_mode_requested": "both",
        "lift_mode_effective": "both",
        **deepcopy(STATISTICS),
    }
    if integration is not None:
        payload["integration"] = integration
    result.metadata["agent_eval"] = payload
    result.add_success("agent_eval", "Live evaluation completed")
    if partial:
        result.passed = False
        result.metadata["execution_status"] = "skipped"
        result.metadata["skip_reason"] = (
            "INCOMPLETE: 1 unresolved skill ref, 1 provider-only MCP server could not be resolved or evaluated at Tier 3"
        )
    result.metadata["gating"] = {"tier": 3, "blocking": False}
    return result


_MEASURED_INTEGRATION = {
    "verdict": "real_integration",
    "measured": True,
    "with_plugin": 0.8,
    "sum_of_parts": 0.62,
    "integration_lift": 0.18,
    "components": ["loader", "search"],
    "lift_uncertainty": {
        "estimate": 0.18,
        "ci_low": 0.06,
        "ci_high": 0.31,
        "confidence": 0.95,
        "precision": "adequate",
        "ci_includes_zero": False,
    },
}


def _gate(tmp_path: Path, rendered: str) -> list[str]:
    card = tmp_path / "BENCHMARK.md"
    card.write_text(rendered, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([card])
    return [str(offender) for offender in offenders]


def _render(results: list[ValidationResult]) -> str:
    return BenchmarkReporter(include_timestamp=False, content_type="plugin", skill_name="demo-plugin").render_all(
        results
    )


def test_complete_plugin_card(tmp_path: Path) -> None:
    rendered = _render(
        [tier1_plugin_result(), tier2_plugin_result(), _tier3(partial=False, integration=_MEASURED_INTEGRATION)]
    )

    assert rendered.startswith("# Plugin Benchmark: demo-plugin\n")
    assert "Overall verdict: PASS" in rendered
    assert "- Plugin: `demo-plugin`" in rendered
    assert "- Agents: Codex (`gpt-codex`)" in rendered
    assert "- Tasks: 4 evaluation tasks (3 positive, 1 negative)" in rendered
    assert "- Cross-component tasks: 2" in rendered
    assert "- Component coverage: 0 components not staged" in rendered
    assert "- Plugin run: complete" in rendered
    assert "| Codex (Baseline → Plugin Uplift) |" in rendered
    assert (
        "| Effectiveness (plugin vs. no plugin) | +30 points "
        "| \\[-2, +55\\] points (95% CI); precision low; CI includes zero |"
    ) in rendered
    assert (
        "| Integration (plugin vs. its own parts) | Real integration, +18 points "
        "| \\[+6, +31\\] points (95% CI); precision adequate |"
    ) in rendered
    assert "## Component Coverage" in rendered
    assert "| loader | skill | Staged | — |" in rendered
    assert "Files staged ≠ components loaded ≠ behavior verified." in rendered
    assert "- Status: complete — every declared dependency was resolved for Tier 3." in rendered
    assert "- Dataset: 4 case(s), 2 cross-component case(s)" in rendered
    # The hook is staged in this run, so it is not listed as excluded.
    assert "no check evaluates them" not in rendered
    assert "Tier 1 checks them statically" not in rendered
    assert "| Tier 3 | Live agent evaluation | **PASS** |" in rendered
    assert "Regenerate this benchmark when the plugin or any of its components" in rendered
    assert _gate(tmp_path, rendered) == []


def test_integration_lift_mode_card_does_not_claim_an_effectiveness_result(tmp_path: Path) -> None:
    """Legacy 2-arm --lift-mode integration: the only baseline is the sum of parts, never "no plugin"."""
    effectiveness_interval = deepcopy(STATISTICS["lift_uncertainty"]["effectiveness"])
    integration = {
        "verdict": "inconclusive",
        "point_verdict": "real_integration",
        "measured": True,
        "with_plugin": 0.8,
        "sum_of_parts": 0.5,
        "integration_lift": 0.3,
        "components": ["loader", "search"],
        # The Integration report reuses the effectiveness interval in this mode.
        "lift_uncertainty": effectiveness_interval,
        "reason": "The paired case bootstrap interval for the Integration lift includes zero.",
    }
    tier3 = _tier3(partial=False, integration=integration)
    tier3.metadata["agent_eval"]["lift_mode_requested"] = "integration"
    tier3.metadata["agent_eval"]["lift_mode_effective"] = "integration"

    rendered = _render([tier1_plugin_result(), tier2_plugin_result(), tier3])

    assert (
        "| Effectiveness (plugin vs. no plugin) | Not measured — lift mode integration compares against "
        "sum-of-parts | Not measured |"
    ) in rendered
    assert (
        "| Integration (plugin vs. its own parts) | Inconclusive, +30 points "
        "| \\[-2, +55\\] points (95% CI); precision low; CI includes zero |"
    ) in rendered
    assert "so the plugin was never compared with a run without it" in rendered
    assert (
        "baseline is the same task attempted with the plugin's member components staged individually "
        "(sum of parts), not without the plugin"
    ) in rendered
    assert "without the target plugin" not in rendered
    assert "47% sum-of-parts baseline" in rendered
    assert _gate(tmp_path, rendered) == []


def test_partial_plugin_card_is_incomplete_and_lists_excluded_behavior(tmp_path: Path) -> None:
    integration = {"verdict": "inconclusive", "measured": False, "reason": "No cross-component case completed."}
    rendered = _render([tier1_plugin_result(), _tier3(partial=True, integration=integration)])

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "Tier 3 evaluated only part of this plugin" in rendered
    assert "## Publication Recommendation" not in rendered
    assert "- Plugin run: INCOMPLETE (partial)" in rendered
    assert "| Integration (plugin vs. its own parts) | INCONCLUSIVE — No cross-component case completed. |" in rendered
    assert "**2 components not staged** of 4 declared or packaged component(s); 2 staged." in rendered
    assert "- mcp docs (Unavailable) — provider-only MCP server" in rendered
    assert "- Status: **INCOMPLETE** — 1 unresolved skill ref, 1 provider-only MCP server" in rendered
    assert "- Provider-only MCP servers were not exercised: docs" in rendered
    assert "- Unresolved skill refs were not evaluated: github::org/repo::skills::remote" in rendered
    assert "- Integration (the plugin versus its own parts) was not measured: No cross-component case completed." in (
        rendered
    )
    assert "| Tier 3 | Live agent evaluation | **INCOMPLETE** | Partial plugin run:" in rendered
    assert _gate(tmp_path, rendered) == []


def test_partial_plugin_card_still_fails_on_blocking_findings(tmp_path: Path) -> None:
    tier1 = tier1_plugin_result()
    tier1.passed = False

    rendered = _render([tier1, _tier3(partial=True)])

    assert "Overall verdict: FAIL" in rendered
    assert _gate(tmp_path, rendered) == []


def test_legacy_plugin_card_says_coverage_was_not_recorded(tmp_path: Path) -> None:
    tier3 = _tier3(partial=False, coverage=False)
    del tier3.metadata["agent_eval"]["plugin_provenance"]

    rendered = _render([tier1_plugin_result(), tier3])

    assert "Per-component coverage was not recorded for this run." in rendered
    assert "Plugin provenance was not recorded for this run" in rendered
    assert "- No declared component or measurement was recorded as excluded" not in rendered
    assert _gate(tmp_path, rendered) == []


def test_complete_card_without_exclusions_says_so(tmp_path: Path) -> None:
    tier1 = tier1_plugin_result()
    tier1.metadata["plugin"]["component_inventory"]["unsupported_types_present"] = []

    rendered = _render([tier1, _tier3(partial=False, integration=_MEASURED_INTEGRATION)])

    assert "- No declared component or measurement was recorded as excluded from this run." in rendered
    assert _gate(tmp_path, rendered) == []


def test_tier1_only_plugin_card_passes_gate(tmp_path: Path) -> None:
    rendered = _render([tier1_plugin_result()])

    assert rendered.startswith("# Plugin Benchmark: demo-plugin\n")
    assert "Overall verdict: INCOMPLETE" in rendered
    assert _gate(tmp_path, rendered) == []


def test_plugin_card_rejects_non_canonical_names() -> None:
    tier1 = tier1_plugin_result()
    tier1.metadata["plugin"]["name"] = "Demo Plugin <b>"

    rendered = BenchmarkReporter(include_timestamp=False, content_type="plugin", skill_name="Not Kebab").render_all(
        [tier1]
    )

    assert rendered.startswith("# Plugin Benchmark: plugin\n")


def test_gate_still_requires_a_benchmark_title(tmp_path: Path) -> None:
    rendered = _render([tier1_plugin_result()]).replace("# Plugin Benchmark:", "# Something Else:")

    assert "missing required section: # Skill Benchmark:" in " ".join(_gate(tmp_path, rendered))


def test_validate_writes_plugin_benchmark_card(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plugin = tmp_path / "demo-plugin"
    plugin.mkdir()
    (plugin / "agent_plugin.yaml").write_text("name: demo-plugin\n", encoding="utf-8")
    output = tmp_path / "reports"

    monkeypatch.setattr(cli_module, "run_validation", lambda *_args, **_kwargs: [tier1_plugin_result()], raising=False)

    outcome = CliRunner().invoke(
        cli_module.cli,
        ["validate", str(plugin), "--type", "plugin", "--no-dedup", "-r", "json", "-o", str(output)],
    )

    card = output / "BENCHMARK.md"
    assert card.is_file(), outcome.output
    assert card.read_text(encoding="utf-8").startswith("# Plugin Benchmark: demo-plugin\n")
