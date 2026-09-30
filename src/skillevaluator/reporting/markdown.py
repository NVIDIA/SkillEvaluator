# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Markdown reporter for PR comments and documentation.

This reporter produces Markdown output suitable for:
- GitHub pull request comments
- Documentation wikis
- Slack/Teams messages (with markdown support)
- README files

The output is optimized for readability in code review contexts.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from skillevaluator.evidence import evidence_ref_identity
from skillevaluator.reporting.base import (
    ReporterBase,
    additional_errors,
    is_advisory_agent_eval_skip,
    passes_required_gate,
)
from skillevaluator.reporting.harbor_viewer import (
    harbor_evidence_link_text,
    normalize_harbor_viewer_for_display,
    safe_url,
)
from skillevaluator.reporting.plugin_sections import tier1_plugin_view, tier3_plugin_view

if TYPE_CHECKING:
    from skillevaluator.models import Finding, ValidationResult


def _markdown_table_cell(value: object) -> str:
    """Return one safe physical Markdown table cell."""
    normalized = str(value).replace("\r\n", "\n").replace("\r", "\n")
    escaped = html.escape(normalized, quote=False)
    return escaped.replace("|", "&#124;").replace("`", "&#96;").replace("\n", "<br>")


def _related_paths(finding: Finding) -> list[str]:
    """Return distinct path-like string values carried in finding metadata."""
    metadata = finding.metadata if isinstance(finding.metadata, dict) else {}
    paths: list[str] = []
    for key, value in metadata.items():
        normalized_key = str(key).casefold()
        if not (normalized_key == "path" or normalized_key.startswith("path_") or normalized_key.endswith("_path")):
            continue
        if isinstance(value, str) and value and value not in paths:
            paths.append(value)
    return paths


class MarkdownReporter(ReporterBase):
    """Markdown report generator for PR comments.

    Produces clean, readable Markdown suitable for GitHub
    pull request comments and documentation.
    """

    def __init__(
        self,
        *,
        include_timestamp: bool = True,
        include_details: bool = True,
        max_findings_shown: int = 10,
    ) -> None:
        """Initialize Markdown reporter.

        Args:
            include_timestamp: Whether to include generation timestamp
            include_details: Whether to include expandable details sections
            max_findings_shown: Maximum findings to show per validator
        """
        self.include_timestamp = include_timestamp
        self.include_details = include_details
        self.max_findings_shown = max_findings_shown

    @property
    def name(self) -> str:
        return "markdown"

    @property
    def description(self) -> str:
        return "Markdown for PR comments and documentation"

    def render(self, result: ValidationResult) -> str:
        """Render single result to Markdown."""
        lines = []
        self._render_result(result, lines)
        return "\n".join(lines)

    def render_all(self, results: list[ValidationResult]) -> str:
        """Render all results to Markdown with summary."""
        lines = []

        # Header
        lines.append("# SkillEvaluator Validation Report")
        lines.append("")

        # Overall status
        all_passed = all(passes_required_gate(r) for r in results)
        has_incomplete = any(r.is_incomplete for r in results)
        advisory_skip_count = sum(1 for r in results if is_advisory_agent_eval_skip(r))
        status = "⚠️ INCOMPLETE" if has_incomplete else "✅ PASSED" if all_passed else "❌ FAILED"
        lines.append(f"**Status:** {status}")

        policy = next(
            (
                result.metadata.get("policy")
                for result in results
                if isinstance(result.metadata.get("policy"), dict)
            ),
            None,
        )
        if policy is not None:
            lines.append(f"**Profile:** {policy.get('profile', 'external')}")
            if policy.get("digest"):
                lines.append(f"**Policy digest:** `{policy['digest']}`")

        if self.include_timestamp:
            timestamp = datetime.now(tz=UTC).strftime("%B %d, %Y at %I:%M %p UTC")
            lines.append(f"**Generated:** {timestamp}")
        lines.append("")

        # Summary table
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Validator Results | {len(results)} |")
        lines.append(f"| ✅ Passed | {sum(1 for r in results if r.status == 'passed')} |")
        lines.append(
            f"| ❌ Failed | {sum(1 for r in results if r.status == 'failed' and not is_advisory_agent_eval_skip(r))} |"
        )
        lines.append(f"| ⚠️ Incomplete | {sum(1 for r in results if r.is_incomplete)} |")
        if advisory_skip_count:
            lines.append(f"| ⏭️ Advisory skips | {advisory_skip_count} |")

        total_errors = sum(r.summary.errors for r in results)
        total_warnings = sum(r.summary.warnings for r in results)
        critical = sum(r.summary.critical_count for r in results)
        high = sum(r.summary.high_count for r in results)
        medium = sum(r.summary.medium_count for r in results)

        severity_breakdown = []
        if critical > 0:
            severity_breakdown.append(f"{critical} critical")
        if high > 0:
            severity_breakdown.append(f"{high} high")
        if medium > 0:
            severity_breakdown.append(f"{medium} medium")

        issue_str = f"{total_errors + total_warnings}"
        if severity_breakdown:
            issue_str += f" ({', '.join(severity_breakdown)})"
        lines.append(f"| Total Issues | {issue_str} |")
        lines.append("")

        plugin = self._plugin_block_from_results(results)
        if plugin is not None:
            self._render_plugin_section(results, plugin, lines)

        # Quality Score summary (if any QUALITY results present)
        quality_results = [r for r in results if r.metadata.get("quality_scores")]
        if quality_results:
            lines.append("## Quality Score")
            lines.append("")
            lines.append("| Skill | Score | Grade | Type | Correctness | Discoverability | Reliability | Efficiency |")
            lines.append("|-------|-------|-------|------|-------------|-----------------|-------------|------------|")
            for qr in quality_results:
                qs = qr.metadata["quality_scores"]
                dims = qs.get("dimensions", {})
                lines.append(
                    f"| {qs.get('skill_name', '—')} "
                    f"| {qs.get('overall_score', 0):.1f} "
                    f"| {qs.get('grade', '?')} "
                    f"| {qs.get('skill_type', '—')} "
                    f"| {dims.get('correctness', {}).get('score', 0):.1f} "
                    f"| {dims.get('discoverability', {}).get('score', 0):.1f} "
                    f"| {dims.get('reliability', {}).get('score', 0):.1f} "
                    f"| {dims.get('efficiency', {}).get('score', 0):.1f} |"
                )
            lines.append("")

        # Tier 3: Agent Evaluation summary (if present)
        tier3_results = [r for r in results if r.metadata.get("agent_eval")]
        if tier3_results:
            ae = tier3_results[0].metadata["agent_eval"]
            verdict = ae.get("verdict", "unknown").upper()
            composite = ae.get("composite_lift")
            runtime = ae.get("runtime_seconds", 0.0)

            lines.append("## Tier 3: Agent Evaluation")
            lines.append("")
            composite_text = f"{composite:+.2f}" if isinstance(composite, int | float) else "N/A"
            lines.append(f"**Verdict:** {verdict} (composite lift = {composite_text})")
            lines.append(f"**Runtime:** {runtime:.1f}s")
            harbor_viewer = normalize_harbor_viewer_for_display(ae)
            if harbor_viewer.get("job_url"):
                lines.append(f"**Harbor logs:** [Open Harbor logs]({harbor_viewer['job_url']})")
            if harbor_viewer.get("analysis_url"):
                lines.append(f"**Harbor analysis:** [Open Harbor analysis]({harbor_viewer['analysis_url']})")
            lines.append("")

            tier3_plugin = tier3_plugin_view(ae)
            if tier3_plugin is not None:
                self._render_tier3_plugin(tier3_plugin, lines)

            evaluators = ae.get("evaluators", {})
            if evaluators:
                lines.append("### Evaluator Scores")
                lines.append("")
                lines.append("| Evaluator | With Skill | Baseline | Lift |")
                lines.append("|-----------|-----------|----------|------|")
                for name, scores in evaluators.items():
                    ws = scores.get("with_skill", 0.0)
                    bl = scores.get("baseline", 0.0)
                    lift = scores.get("lift", 0.0)
                    lines.append(f"| {name.replace('_', ' ').title()} | {ws:.2f} | {bl:.2f} | {lift:+.2f} |")
                lines.append("")

            insights = ae.get("insights", {})
            if any(v.get("score") is not None for v in insights.values()):
                lines.append("### LLM-as-Judge Insights")
                lines.append("")
                lines.append("| Dimension | Score | Explanation |")
                lines.append("|-----------|-------|-------------|")
                for dim, info in insights.items():
                    score = info.get("score")
                    if score is None:
                        continue
                    score_str = str(score).upper() if isinstance(score, str) else f"{score:.2f}"
                    explanation = info.get("explanation", "")[:60]
                    lines.append(f"| {dim.title()} | {score_str} | {explanation} |")
                lines.append("")

            suggestions_v2 = ae.get("suggestions_v2") or []
            if suggestions_v2:
                lines.append("### Evidence-Backed Suggestions")
                lines.append("")
                for idx, suggestion in enumerate(suggestions_v2, start=1):
                    recommendation = str(suggestion.get("recommendation") or "").strip()
                    if not recommendation:
                        continue
                    metric = suggestion.get("metric", "unknown")
                    lines.append(f"{idx}. **{metric}**: {recommendation}")
                    harbor_evidence = suggestion.get("harbor_evidence") or suggestion.get("evidence")
                    if isinstance(harbor_evidence, dict):
                        url = safe_url(harbor_evidence.get("url"))
                        if url:
                            label = harbor_evidence_link_text(harbor_evidence)
                            lines.append(f"   - Evidence: [{label}]({url})")
                    for ref in (suggestion.get("evidence_refs") or [])[:3]:
                        pointer = evidence_ref_identity(ref)
                        excerpt = str(ref.get("excerpt") or ref.get("label") or "")[:120]
                        lines.append(f"   - Evidence: `{ref.get('kind', 'evidence')}` `{pointer}` {excerpt}")
                lines.append("")
            elif ae.get("recommendations"):
                lines.append("### Recommendations")
                lines.append("")
                for idx, recommendation in enumerate(ae.get("recommendations") or [], start=1):
                    if not isinstance(recommendation, dict):
                        continue
                    message = str(recommendation.get("message") or recommendation.get("title") or "").strip()
                    if not message:
                        continue
                    lines.append(f"{idx}. {message}")
                    evidence = recommendation.get("evidence")
                    if isinstance(evidence, dict):
                        url = safe_url(evidence.get("url"))
                        if url:
                            label = harbor_evidence_link_text(evidence)
                            lines.append(f"   - Evidence: [{label}]({url})")
                lines.append("")

        # Results per validator
        lines.append("## Results")
        lines.append("")

        for result in results:
            self._render_result(result, lines)

        # Footer
        lines.append("---")
        lines.append("*Generated by SkillEvaluator*")

        return "\n".join(lines)

    def _render_plugin_section(self, results: list[ValidationResult], plugin: dict, lines: list[str]) -> None:
        """Render the Tier 1 plugin block: manifest, dependencies, components, MCP, context."""
        view = (
            tier1_plugin_view(
                plugin,
                status=self._plugin_status(results),
                bundled_skills=self._plugin_child_names(results),
            )
            or {}
        )
        cell = _markdown_table_cell
        status = {"failed": "❌ FAILED", "incomplete": "⚠️ INCOMPLETE", "passed": "✅ PASSED"}.get(view["status"], "")
        lines.append("## Plugin")
        lines.append("")
        lines.append("| Plugin | Status | Manifest | Mode |")
        lines.append("|--------|--------|----------|------|")
        lines.append(
            f"| {cell(view['name'] or 'plugin')} | {status} "
            f"| {cell(view['manifest_type'] or 'unknown')} | {cell(view['plugin_mode'] or 'unknown')} |"
        )
        lines.append("")
        if view["declared_dependencies"]:
            declared = ", ".join(f"{cell(row['kind'])}={row['count']}" for row in view["declared_dependencies"])
            lines.append(f"**Declared dependencies:** {declared}")
            lines.append("")
        if view["bundled_skills"]:
            bundled = ", ".join(cell(name) for name in view["bundled_skills"])
            more = f" (+{view['bundled_skills_omitted']} more)" if view["bundled_skills_omitted"] else ""
            lines.append(
                f"**Bundled skills ({view['in_plugin_skills'] or len(view['bundled_skills'])}):** {bundled}{more}"
            )
            lines.append("")

        dependencies = view["dependencies"]
        if dependencies:
            lines.append("### Dependency resolution")
            lines.append("")
            if dependencies["counts"]:
                counts = ", ".join(f"{cell(row['state'])}={row['count']}" for row in dependencies["counts"])
                lines.append(f"**Status counts:** {counts}")
                lines.append("")
            if dependencies["rows"]:
                lines.append("| Kind | Ref | State | Path | Reason |")
                lines.append("|------|-----|-------|------|--------|")
                for row in dependencies["rows"]:
                    lines.append(
                        f"| {cell(row['kind'])} | {cell(row['ref'])} | {cell(row['state'])} "
                        f"| {cell(row['path'] or '—')} | {cell(row['reason'])} |"
                    )
                if dependencies["omitted"]:
                    lines.append(f"| … | *{dependencies['omitted']} more refs* | | | |")
                lines.append("")

        inventory = view["inventory"]
        if inventory:
            lines.append(f"### Component inventory ({inventory['total']})")
            lines.append("")
            if inventory["unsupported_types"]:
                unsupported = ", ".join(cell(name) for name in inventory["unsupported_types"])
                lines.append(
                    f"> ⚠️ **Unsupported component types present:** {unsupported}. "
                    "SkillEvaluator lists these components but cannot evaluate them."
                )
                lines.append("")
            if inventory["rows"]:
                lines.append("| Type | Name | Origin | Support | Findings |")
                lines.append("|------|------|--------|---------|----------|")
                for row in inventory["rows"]:
                    lines.append(
                        f"| {cell(row['type'])} | {cell(row['name'])} | {cell(row['origin'])} "
                        f"| {cell(row['support_label'])} | {row['findings']} |"
                    )
                if inventory["omitted"]:
                    lines.append(f"| … | *{inventory['omitted']} more components* | | | |")
                lines.append("")

        mcp = view["mcp"]
        if mcp and mcp["pinning"]:
            pinning = mcp["pinning"]
            lines.append("### MCP pinning")
            lines.append("")
            lines.append(f"**Pinned:** {cell(pinning['summary'])} ({cell(pinning['ratio_label'])})")
            lines.append("")
            if mcp["unpinned"]:
                lines.append("| Unpinned server | Kind | Detail |")
                lines.append("|-----------------|------|--------|")
                for server in mcp["unpinned"]:
                    lines.append(f"| {cell(server['name'])} | {cell(server['kind'])} | {cell(server['pin_detail'])} |")
                lines.append("")

        cost = view["context_cost"]
        if cost:
            lines.append(f"### Context cost ({cell(cost['label'])})")
            lines.append("")
            lines.append(f"**Always-on:** {cost['always_on']} tokens · **On-demand:** {cost['on_demand']} tokens")
            lines.append("")
            lines.append(f"*{cell(cost['note'])}*")
            lines.append("")

        for key, title, headers in (
            (
                "catalog_skill_similarity",
                "Bundled skills vs. local skills catalog",
                ("Bundled skill", "Catalog match", "Similarity"),
            ),
            (
                "inter_plugin_similarity",
                "Plugin vs. other plugins in the local catalog",
                ("Catalog plugin", "Similarity", "Member overlap", "Verdict"),
            ),
        ):
            similarity = view.get(key)
            if not similarity:
                continue
            lines.append(f"### {title} (advisory)")
            lines.append("")
            entries = similarity["catalog_entries"]
            summary = f"**Status:** {cell(similarity['status_label'])}"
            if entries is not None:
                summary += f" · {entries} catalog entries"
            if similarity["reason"]:
                summary += f" · {cell(similarity['reason'])}"
            lines.append(summary)
            lines.append("")
            if similarity["matches"]:
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("|" + "|".join("---" for _ in headers) + "|")
                for match in similarity["matches"]:
                    values = [
                        match[field]
                        for field in ("subject", "match", "similarity", "member_overlap", "verdict")
                        if field in match
                    ]
                    lines.append("| " + " | ".join(cell(value) for value in values) + " |")
                lines.append("")

    @staticmethod
    def _render_tier3_plugin(view: dict, lines: list[str]) -> None:
        """Render what the Tier 3 plugin run did and did not demonstrate."""
        cell = _markdown_table_cell
        if view["partial"]:
            lines.append(
                f"> ⚠️ **INCOMPLETE: {cell(view['incomplete_reason'])}.** This is a partial result, not a pass."
            )
            lines.append("")
        coverage = view["coverage"]
        if coverage:
            lines.append("### Plugin Component Coverage")
            lines.append("")
            observed = f"; {cell(coverage['observed_headline'])}" if coverage["observed_headline"] else ""
            lines.append(
                f"**{cell(coverage['headline'])}** of {coverage['total']} component(s); "
                f"{coverage['staged']} staged{observed}."
            )
            lines.append("")
            lines.append(f"*Files staged ≠ components loaded ≠ behavior verified. {cell(coverage['note'])}*")
            lines.append("")
            if coverage["staged_not_observed_rows"]:
                names = ", ".join(f"{row['type']} {row['name']}" for row in coverage["staged_not_observed_rows"])
                lines.append(f"Staged but not observed in any plugin trial: {cell(names)}")
                lines.append("")
            if coverage["not_staged_rows"]:
                lines.append("| Type | Component | State | Reason |")
                lines.append("|------|-----------|-------|--------|")
                for row in coverage["not_staged_rows"]:
                    lines.append(
                        f"| {cell(row['type'])} | {cell(row['name'])} | {cell(row['state_label'])} "
                        f"| {cell(row['reason'])} |"
                    )
                lines.append("")
        integration = view["integration"]
        modes = (integration or {}).get("modes") or view["lift_modes"]
        if integration or modes:
            lines.append("### Integration (advisory)")
            lines.append("")
            if modes:
                lines.append(
                    f"**Lift mode:** requested `{cell(modes['requested'])}`, effective `{cell(modes['effective'])}`"
                )
                lines.append("")
        if integration:
            if integration["measured"]:
                ci = f" {cell(integration['ci']['summary'])}" if integration["ci"] else ""
                point = (
                    f"; point estimate: {cell(integration['point_verdict_label'])}"
                    if integration["point_verdict_label"]
                    else ""
                )
                lines.append(
                    f"**{cell(integration['verdict_label'])}:** plugin {integration['with_plugin']} vs "
                    f"sum-of-parts {integration['sum_of_parts']} (lift {integration['integration_lift']}{ci}{point})"
                )
            else:
                lines.append(f"**INCONCLUSIVE:** {cell(integration['reason'])}")
            lines.append("")
        statistics = view["statistics"]
        if statistics:
            for row in statistics["primary"]["lift_ci"]:
                warning = " — ⚠️ CI includes zero" if row["ci_includes_zero"] else ""
                lines.append(
                    f"- {cell(row['label'])}: {cell(row['summary'])}, precision {cell(row['precision'])}{warning}"
                )
            if statistics["primary"]["lift_ci"]:
                lines.append("")

    def _render_result(self, result: ValidationResult, lines: list[str]) -> None:
        """Render a single validation result."""
        qs = result.metadata.get("quality_scores")

        advisory_skip = is_advisory_agent_eval_skip(result)
        if result.is_incomplete:
            status_emoji = "⚠️ INCOMPLETE"
            lines.append(f"### {status_emoji} {result.validator_name}")
        elif advisory_skip:
            lines.append(f"### ⏭️ SKIPPED {result.validator_name}")
        elif qs and qs.get("grade"):
            grade = qs["grade"]
            status_emoji = "✅" if result.passed else "❌"
            lines.append(f"### {status_emoji} {grade} {result.validator_name}")
        else:
            status_emoji = "✅" if result.passed else "❌"
            lines.append(f"### {status_emoji} {result.validator_name}")

        if result.validator_description:
            lines.append(f"*{result.validator_description}*")
        lines.append("")

        # Quality dimension breakdown
        if qs and qs.get("dimensions"):
            score = qs.get("overall_score", 0)
            grade = qs.get("grade", "?")
            stype = qs.get("skill_type", "unknown")
            lines.append(f"**Overall: {score:.1f}/100 (Grade: {grade})** | Skill Type: {stype}")
            lines.append("")
            lines.append("| Dimension | Score | Weight |")
            lines.append("|-----------|-------|--------|")
            for dname, ddata in qs["dimensions"].items():
                lines.append(f"| {dname.title()} | {ddata.get('score', 0):.1f} | {ddata.get('weight', 0) * 100:.0f}% |")
            lines.append("")

        if result.is_incomplete:
            self._render_incomplete(result, lines)
        elif advisory_skip:
            payload = result.metadata.get("agent_eval", {}) if result.metadata else {}
            provenance = payload.get("provenance", {}) if isinstance(payload, dict) else {}
            message = provenance.get("message") if isinstance(provenance, dict) else None
            lines.append(f"- {message or 'Live evaluation did not run.'}")
        elif result.passed:
            self._render_success(result, lines)
            if result.findings:
                lines.append("")
                lines.append(f"**Non-blocking findings: {len(result.findings)}**")
                lines.append("")
                self._render_findings(result.findings, lines)
        else:
            self._render_failure(result, lines)

        lines.append("")

    def _render_success(self, result: ValidationResult, lines: list[str]) -> None:
        """Render success details."""
        if result.success_details:
            for detail in result.success_details:
                lines.append(f"- [OK] **{detail.check_name}**: {detail.message}")
        elif result.messages:
            for msg in result.messages:
                lines.append(f"- {msg}")
        else:
            lines.append("- All checks passed")

    def _render_failure(self, result: ValidationResult, lines: list[str]) -> None:
        """Render failure details with findings table and expandable details."""
        # Summary counts
        s = result.summary
        lines.append(f"**{s.errors} errors, {s.warnings} warnings**")
        lines.append("")

        errors = additional_errors(result)
        if errors:
            label = "Execution errors" if result.is_incomplete else "Errors"
            lines.extend([f"**{label}:**", ""])
            for error in errors[: self.max_findings_shown]:
                lines.append(f"- ❌ {html.escape(error, quote=False)}")
            remaining = len(errors) - self.max_findings_shown
            if remaining > 0:
                lines.append(f"- *... and {remaining} more errors*")
            lines.append("")

        if result.findings:
            self._render_findings(result.findings, lines)

    def _render_findings(self, findings: list[Finding], lines: list[str]) -> None:
        """Render a shared findings table for blocking and non-blocking results."""
        lines.append("| Severity | Issue | Location |")
        lines.append("|----------|-------|----------|")

        shown_findings = findings[: self.max_findings_shown]
        for finding in shown_findings:
            emoji = finding.severity.emoji
            severity_upper = finding.severity.value.upper()
            message = _markdown_table_cell(finding.message)
            location = _markdown_table_cell(finding.location)
            lines.append(f"| {emoji} {severity_upper} | {message} | <code>{location}</code> |")

        remaining = len(findings) - len(shown_findings)
        if remaining > 0:
            lines.append(f"| ... | *{remaining} more issues* | |")

        lines.append("")
        if self.include_details:
            lines.append("<details>")
            lines.append("<summary>View Details</summary>")
            lines.append("")

            for index, finding in enumerate(shown_findings, 1):
                self._render_finding_detail(index, finding, lines)

            if remaining > 0:
                lines.append(f"*... and {remaining} more issues*")
                lines.append("")

            lines.append("</details>")
            lines.append("")

    def _render_incomplete(self, result: ValidationResult, lines: list[str]) -> None:
        """Render missing scanner evidence without presenting a failure as a pass."""
        lines.append(f"**Incomplete scanners:** {', '.join(result.incomplete_scans)}")
        lines.append("")
        self._render_failure(result, lines)
        if not result.findings and not result.errors:
            for warning in result.warnings[: self.max_findings_shown]:
                lines.append(f"- ⚠️ {warning}")

    def _render_finding_detail(self, index: int, finding: Finding, lines: list[str]) -> None:
        """Render detailed information for a single finding."""
        lines.append(f"**{index}. {finding.message}**")
        lines.append(f"- File: `{finding.location}`")
        lines.append(f"- Check: `{finding.check_name}`")
        related_paths = _related_paths(finding)
        if related_paths:
            lines.append(f"- Related paths: {' <-> '.join(f'`{path}`' for path in related_paths)}")

        if finding.line_content:
            content = finding.line_content.strip()
            if len(content) > 60:
                content = content[:57] + "..."
            lines.append(f"- Content: `{content}`")

        if finding.suggestion:
            lines.append(f"- Fix: {finding.suggestion}")

        lines.append("")

    def get_file_extension(self) -> str:
        return ".md"
