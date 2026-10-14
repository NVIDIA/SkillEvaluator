# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-render a saved ``validate`` run on every surface (test helper).

No command re-renders reports from a saved JSON report, so this rebuilds the validation results from one: the
findings, legacy messages, summaries, scanner evidence, gates and the Tier 3 payload the JSON keeps, plus the two
stamps it does not serialize but ``validate`` set on the same results (an advisory catalog check that did not run,
and a plugin run that did not complete). Every reporter then renders the rebuilt results, and the headline each
surface shows is read back the way a reader sees it.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

from rich.console import Console

from skillevaluator.evaluation.tier3_report import incomplete_reason
from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.models.result import SuccessDetail, ValidationSummary
from skillevaluator.reporting import (
    BenchmarkReporter,
    CLIReporter,
    HTMLReporter,
    JSONReporter,
    MarkdownReporter,
    SARIFReporter,
)
from skillevaluator.reporting.plugin_sections import completeness_view, plugin_provenance

_CATALOG_KEYS = ("catalog_skill_similarity", "inter_plugin_similarity")


def saved_report(run_dir: Path) -> Path:
    """The JSON report a saved run wrote (``reports/skillevaluator-output-<stamp>.json``)."""
    reports = [path for path in (run_dir / "reports").glob("skillevaluator-output-*.json") if ".sarif" not in path.name]
    if len(reports) != 1:
        raise FileNotFoundError(f"expected one JSON report under {run_dir / 'reports'}, found {len(reports)}")
    return reports[0]


def load_saved_results(report_path: Path) -> list[ValidationResult]:
    """Rebuild the validation results a saved JSON report was rendered from."""
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    results = [_result(item) for item in payload["results"]]
    plugin = payload.get("plugin")
    if isinstance(plugin, dict):
        # The JSON report folds every result's plugin block into one; the Tier 1 plugin result carried it.
        first_tier1 = next((r for r in results if (r.metadata.get("gating") or {}).get("tier") == 1), results[0])
        first_tier1.metadata["plugin"] = {**plugin, **first_tier1.metadata.get("plugin", {})}
    if isinstance(payload.get("policy"), dict):
        for result in results:
            result.metadata["policy"] = payload["policy"]
    return results


def _result(item: dict[str, Any]) -> ValidationResult:
    result = ValidationResult(validator_name=item["validator"], validator_description=item.get("description") or "")
    for finding in item.get("findings") or []:
        result.findings.append(
            Finding(
                category=finding["category"],
                severity=Severity(finding["severity"]),
                check_name=finding["check_name"],
                message=finding["message"],
                file_path=finding.get("file_path") or "",
                line_number=finding.get("line_number"),
                line_content=finding.get("line_content"),
                suggestion=finding.get("suggestion"),
                metadata=finding.get("metadata") or {},
            )
        )
    result.success_details = [
        SuccessDetail(check_name=detail["check"], message=detail["message"], metadata=detail.get("metadata") or {})
        for detail in item.get("success_details") or []
    ]
    legacy = item.get("legacy") or {}
    result.errors = list(legacy.get("errors") or [])
    result.warnings = list(legacy.get("warnings") or [])
    result.messages = list(legacy.get("messages") or [])
    result.summary = ValidationSummary(**item["summary"])
    result.passed = bool(item["passed"])
    metadata = result.metadata
    if item.get("incomplete_scans"):
        metadata["incomplete_scans"] = list(item["incomplete_scans"])
    for source, target in (
        ("gating", "gating"),
        ("quality", "quality_scores"),
        ("tier3", "agent_eval"),
        ("rubric_eval", "rubric_eval"),
        ("workflow", "workflow"),
        ("llm_analysis", "llm_analysis"),
        ("plugin", "plugin"),
    ):
        if item.get(source) is not None:
            metadata[target] = item[source]
    _restore_unserialized_stamps(result)
    return result


def _restore_unserialized_stamps(result: ValidationResult) -> None:
    """Re-apply the stamps ``validate`` set that the JSON report does not serialize."""
    plugin = result.metadata.get("plugin") if isinstance(result.metadata.get("plugin"), dict) else {}
    for key in _CATALOG_KEYS:
        block = plugin.get(key)
        if isinstance(block, dict) and block.get("status") == "skipped":
            # deduplication.result_status.mark_advisory_skip: an advisory catalog check that did not run.
            reason = str(block.get("reason") or "The local catalog check did not run.")
            result.metadata.update(
                advisory_tier2=True, skipped=True, execution_status="skipped", skip_reason=reason, optional=True
            )
    payload = result.metadata.get("agent_eval")
    if result.validator_name == "AGENT_EVAL" and isinstance(payload, dict) and not result.passed:
        provenance = plugin_provenance(payload)
        completeness = completeness_view(provenance)
        if completeness and completeness["partial"] and payload.get("execution_status") != "succeeded":
            # cli._incomplete_plugin_agent_eval_result: a plugin run that did not complete is INCOMPLETE.
            result.metadata["execution_status"] = "skipped"
            result.metadata["skip_reason"] = incomplete_reason(dict(provenance))


_MARKDOWN = {"✅ PASSED": "passed", "❌ FAILED": "failed", "⚠️ INCOMPLETE": "incomplete"}
_HTML_HEADER = {"All Passed": "passed", "Issues Found": "failed", "Incomplete": "incomplete"}
# The terminal summary's last line; a per-result block above it can also say "[FAIL] Validation failed".
_SUMMARY = re.compile(
    r"\[PASS\] (?:All|Required) validations passed|\[FAIL\] Validation failed|"
    r"\[INCOMPLETE\] Validation evidence is incomplete"
)


def _summary_status(terminal: str) -> str | None:
    lines = _SUMMARY.findall(terminal)
    if not lines:
        return None
    return {"[PASS]": "passed", "[FAIL]": "failed", "[INCOMPLETE]": "incomplete"}[lines[-1].split(" ", 1)[0]]


class _FooterView:
    """Stands in for the quiet pipeline view: keeps the verdict ``_finish_pipeline_view`` hands it."""

    blocks: tuple[()] = ()
    verdict: Any = None

    def finish(self, verdict: Any, _links: list[tuple[str, str]]) -> None:
        self.verdict = verdict


def render_surfaces(results: list[ValidationResult], *, name: str) -> dict[str, Any]:
    """Render *results* on every surface; return each surface's headline, the HTML tier cards and the outputs."""
    import skillevaluator.cli as cli_module

    json_report = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    markdown = MarkdownReporter(include_timestamp=False).render_all(results)
    html = HTMLReporter(include_timestamp=False, content_label="Plugin").render_all(results)
    sarif = json.loads(SARIFReporter(include_timestamp=False).render_all(results))
    benchmark = BenchmarkReporter(include_timestamp=False, content_type="plugin", skill_name=name).render_all(results)
    stream = io.StringIO()
    CLIReporter(Console(file=stream, width=200, emoji=False, color_system=None)).print_all(results)
    terminal = " ".join(stream.getvalue().split())

    gate = [result for result in results if (result.metadata.get("gating") or {}).get("blocking", True)]
    view = _FooterView()
    cli_module._finish_pipeline_view(
        view,
        tier_gate_results=gate,
        tier3_result=next((result for result in results if result.validator_name == "AGENT_EVAL"), None),
        gate_failed=not all(result.passed for result in gate),
        output_dir=Path("reports"),
        basename="saved",
        report_formats=("json",),
        target_path=Path(name),
        agent_eval=True,
    )
    footer = "passed" if view.verdict.passed else "incomplete" if view.verdict.incomplete else "failed"

    markdown_status = re.search(r"\*\*Status:\*\* (✅ PASSED|❌ FAILED|⚠️ INCOMPLETE)", markdown)
    html_status = re.search(r'class="header-status-badge [a-z]+"[^>]*>([^<]+)<', html)
    verdict = re.search(r"Overall verdict: (PASS|FAIL|INCOMPLETE|NEUTRAL)", benchmark)
    cards = dict(
        re.findall(
            r'data-target-tab="(tier\d)".*?<span class="tier-card-verdict">([^<]+)</span>', html, flags=re.DOTALL
        )
    )
    return {
        "headlines": {
            "json": json_report.get("overall_status"),
            "markdown": _MARKDOWN.get(markdown_status.group(1)) if markdown_status else None,
            "html": _HTML_HEADER.get(html_status.group(1)) if html_status else None,
            "sarif": sarif["runs"][0].get("properties", {}).get("overallStatus"),
            "cli-summary": _summary_status(terminal),
            "cli-footer": footer,
        },
        "benchmark": verdict.group(1) if verdict else None,
        "html_tier_cards": cards,
        "outputs": {"json": json_report, "markdown": markdown, "html": html, "terminal": terminal},
    }
