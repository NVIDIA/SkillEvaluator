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
import math
import unicodedata
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from skillevaluator.evidence import evidence_ref_identity
from skillevaluator.reporting.base import (
    ReporterBase,
    additional_errors,
    additional_warnings,
    assess_publication,
    get_skip_reason,
    is_advisory_agent_eval_skip,
    is_cleanly_skipped,
    select_agent_eval_payload,
)
from skillevaluator.reporting.finding_summary import (
    describe_issue_counts,
    groups_by_display_priority,
    summarize_findings,
    summarize_messages,
)
from skillevaluator.reporting.finding_verification import finding_verification
from skillevaluator.reporting.harbor_viewer import (
    harbor_evidence_link_text,
    harbor_job_links,
    safe_url,
)
from skillevaluator.reporting.run_status import Status, resolve_run_status, result_status
from skillevaluator.reporting.summary_copy import SKILL_LIFT_DEFINITION, pass_at_k_summary
from skillevaluator.tier3.baseline_store import baseline_description
from skillevaluator.tier3.dataset_provenance import dataset_provenance_display
from skillevaluator.tier3.version_comparison import version_comparison_lines

if TYPE_CHECKING:
    from skillevaluator.models import Finding, ValidationResult


_SKIP_REASON_MAX_CHARS = 1024
# Case rows of the Tier 3 token usage table; the JSON payload keeps every case.
_MAX_TOKEN_USAGE_CASE_ROWS = 50


def _markdown_safe_characters(value: str, *, preserve_newlines: bool = False) -> str:
    """Remove control and separator characters from untrusted Markdown text."""
    safe: list[str] = []
    for character in value.replace("\r\n", "\n").replace("\r", "\n"):
        category = unicodedata.category(character)
        if category.startswith("C") or category in {"Zl", "Zp"}:
            safe.append("\n" if preserve_newlines and character == "\n" else " ")
        else:
            safe.append(character)
    return "".join(safe)


def _markdown_table_cell(value: object) -> str:
    """Return one safe physical Markdown table cell."""
    normalized = _markdown_safe_characters(str(value), preserve_newlines=True)
    escaped = html.escape(normalized, quote=False)
    return escaped.replace("|", "&#124;").replace("`", "&#96;").replace("\n", "<br>")


def _finite_report_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_report_number(value: object, *, signed: bool = False) -> str:
    number = _finite_report_number(value)
    if number is None:
        return "N/A"
    return f"{number:+.2f}" if signed else f"{number:.2f}"


def _markdown_inline_text(value: object, *, limit: int | None = None) -> str:
    """Flatten untrusted metadata into one inert Markdown text fragment."""
    if isinstance(value, str):
        text = value
    elif (
        isinstance(value, bool)
        or (isinstance(value, int) and value.bit_length() <= 256)
        or (isinstance(value, float) and math.isfinite(value))
    ):
        text = str(value)
    else:
        return ""
    flattened = " ".join(_markdown_safe_characters(text).split())
    if limit is not None:
        flattened = flattened[:limit]
    escaped = html.escape(flattened, quote=False)
    for character, entity in (
        ("#", "&#35;"),
        ("\\", "&#92;"),
        ("`", "&#96;"),
        ("*", "&#42;"),
        ("_", "&#95;"),
        ("[", "&#91;"),
        ("]", "&#93;"),
        ("@", "&#64;"),
    ):
        escaped = escaped.replace(character, entity)
    return escaped


def _markdown_code_text(value: object, *, limit: int | None = None) -> str:
    """Flatten untrusted metadata for a code span without obscuring paths."""
    if not isinstance(value, str):
        return ""
    flattened = " ".join(_markdown_safe_characters(value).split())
    if limit is not None:
        flattened = flattened[:limit]
    return html.escape(flattened, quote=False).replace("`", "&#96;")


def _mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _dict_items(value: object) -> list[dict]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _markdown_untrusted_table_cell(value: object, *, limit: int | None = None) -> str:
    """Return one inert table cell without double-escaping HTML entities."""
    return _markdown_inline_text(value, limit=limit).replace("|", "&#124;")


def _markdown_safe_url(value: object) -> str | None:
    """Return an HTTP(S) URL that cannot terminate a Markdown destination."""
    url = safe_url(value)
    if url is None:
        return None
    if any(character.isspace() or unicodedata.category(character).startswith("C") for character in url):
        return None
    if any(character in "()<>" for character in url):
        return None
    return url


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
        expected_skill_name: str | None = None,
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
        self.expected_skill_name = expected_skill_name

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
        finding_summary = summarize_findings(results)
        message_summary = summarize_messages(results)

        # Header
        lines.append("# SkillEvaluator Validation Report")
        lines.append("")

        # Overall status
        run_status = resolve_run_status(results)
        all_passed = run_status.headline == Status.PASSED
        has_incomplete = run_status.headline == Status.INCOMPLETE
        skip_count = sum(1 for r in results if is_cleanly_skipped(r))
        advisory_skip_count = sum(1 for r in results if is_advisory_agent_eval_skip(r))
        executed_count = len(results) - skip_count
        passed_count = sum(1 for r in results if r.passed and not r.is_incomplete and not is_cleanly_skipped(r))
        status = "⚠️ INCOMPLETE" if has_incomplete else "✅ PASSED" if all_passed else "❌ FAILED"
        lines.append(f"**Status:** {status}")
        publication = assess_publication(results, expected_skill_name=self.expected_skill_name)
        publication_label = {
            "pass": "✅ PASS",
            "fail": "❌ FAIL",
            "neutral": "⚠️ NEUTRAL",
            "incomplete": "⚠️ INCOMPLETE",
        }.get(publication.status, publication.status.upper())
        lines.append(f"**Publication status:** {publication_label}")

        policy = next(
            (result.metadata.get("policy") for result in results if isinstance(result.metadata.get("policy"), dict)),
            None,
        )
        if policy is not None:
            lines.append(f"**Profile:** {policy.get('profile', 'default')}")
            if policy.get("digest"):
                lines.append(f"**Policy digest:** `{policy['digest']}`")

        quality_config = next(
            (
                result.metadata.get("quality_config")
                for result in results
                if isinstance(result.metadata.get("quality_config"), dict)
            ),
            None,
        )
        if quality_config is not None:
            source = quality_config.get("source", "packaged")
            file_label = f" (`{quality_config['file']}`)" if quality_config.get("file") else ""
            min_score_label = (
                f", min score {quality_config['min_score']}" if quality_config.get("min_score") is not None else ""
            )
            lines.append(
                f"**Quality config:** {source}{file_label}, `{quality_config.get('digest', '')}`{min_score_label}"
            )

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
        lines.append(f"| Validators Run | {executed_count} |")
        lines.append(f"| ✅ Passed | {passed_count} |")
        lines.append(
            f"| ❌ Failed | {sum(1 for r in results if result_status(r) == Status.FAILED and not is_advisory_agent_eval_skip(r))} |"
        )
        lines.append(f"| ⚠️ Incomplete | {sum(1 for r in results if r.is_incomplete)} |")
        if skip_count:
            lines.append(f"| ⏭️ Skipped | {skip_count} |")
        if advisory_skip_count:
            lines.append(f"| ⏭️ Advisory skips | {advisory_skip_count} |")

        # The breakdown qualifies the unique count, so it counts unique findings
        # at their highest severity rather than raw validator records.
        unique_by_severity = finding_summary["unique_by_severity"]
        severity_breakdown = [
            f"{unique_by_severity[severity]} {severity}"
            for severity in ("critical", "high", "medium")
            if unique_by_severity[severity] > 0
        ]
        issue_str = describe_issue_counts(
            finding_summary,
            message_summary,
            unique_suffix=f" ({', '.join(severity_breakdown)})" if severity_breakdown else "",
            raw_records=True,
        )
        lines.append(f"| Total Issues | {issue_str} |")
        lines.append(f"| Duplicate Occurrences Reconciled | {finding_summary['duplicates_removed']} |")
        lines.append("")

        if finding_summary["groups"]:
            lines.append("## Finding Reconciliation")
            lines.append("")
            lines.append("| Finding ID | Severity | Finding | Occurrences | Validators |")
            lines.append("|------------|----------|---------|-------------|------------|")
            # The table is capped, so list the most severe findings first.
            shown_groups = groups_by_display_priority(finding_summary)[: self.max_findings_shown]
            for group in shown_groups:
                # Finding text and validator names are untrusted report input.
                lines.append(
                    f"| `{group['finding_id']}` "
                    f"| {_markdown_untrusted_table_cell(str(group['severity']).upper())} "
                    f"| {_markdown_untrusted_table_cell(group['message'])} "
                    f"| {group['occurrence_count']} "
                    f"| {_markdown_untrusted_table_cell(', '.join(group['validators']))} |"
                )
            remaining_groups = len(finding_summary["groups"]) - len(shown_groups)
            if remaining_groups > 0:
                lines.append(f"| ... | | *{remaining_groups} more unique findings* | | |")
            lines.append("")

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
        ae = publication.tier3.payload or select_agent_eval_payload(results)
        if ae:
            verdict = publication.tier3.status.upper()
            composite = ae.get("composite_lift")
            runtime = ae.get("runtime_seconds", 0.0)

            lines.append("## Tier 3: Agent Evaluation")
            lines.append("")
            if ae.get("baseline"):
                lines.append(baseline_description(ae["baseline"]))
                lines.append("")
            if comparison := ae.get("prior_version_comparison"):
                lines.extend(html.escape(line) for line in version_comparison_lines(comparison))
                lines.append("")
            composite_text = _format_report_number(composite, signed=True)
            lines.append(f"**Verdict:** {verdict} (Skill Lift = {composite_text})")
            if _finite_report_number(composite) is not None:
                lines.append(SKILL_LIFT_DEFINITION)
            if pass_summary := pass_at_k_summary(ae.get("pass_at_k"), ae.get("attempt_policy")):
                lines.append(pass_summary)
            provenance_line, _ = dataset_provenance_display(ae.get("dataset_provenance"))
            if provenance_line:
                lines.append(_markdown_inline_text(provenance_line))
            if verdict == "INCONCLUSIVE":
                lines.append("Skill never activated. Scores are diagnostic only; no lift verdict was computed.")
            runtime_number = _finite_report_number(runtime)
            runtime_text = f"{runtime_number:.1f}s" if runtime_number is not None else "N/A"
            lines.append(f"**Runtime:** {runtime_text}")
            # One list item per job, so CommonMark keeps each job's links separate.
            job_lines = []
            for job in harbor_job_links(ae):
                links = [
                    f"[{label}]({url})"
                    for label, url in (
                        ("Harbor logs", _markdown_safe_url(job.get("url"))),
                        ("Harbor analysis", _markdown_safe_url(job.get("analysis_url"))),
                    )
                    if url
                ]
                if links:
                    job_lines.append(f"- **{_markdown_inline_text(job['name'])}:** " + " · ".join(links))
            if job_lines:
                lines.append("")
                lines.extend(job_lines)
            lines.append("")

            evaluators = ae.get("evaluators", {})
            if isinstance(evaluators, dict) and evaluators:
                lines.append("### Evaluator Scores")
                lines.append("")
                lines.append("| Evaluator | With Skill | Baseline | Lift |")
                lines.append("|-----------|-----------|----------|------|")
                for name, scores in evaluators.items():
                    if not isinstance(name, str) or not isinstance(scores, dict):
                        continue
                    ws = scores.get("with_skill")
                    bl = scores.get("baseline")
                    lift = scores.get("lift")
                    # A baseline-only run (--skip-with-skill) never ran the with-skill arm.
                    with_skipped = scores.get("with_skill_status") == "skipped"
                    ws_text = (
                        "N/A"
                        if scores.get("with_skill_status") == "not_applicable" and ws is None
                        else "skipped"
                        if with_skipped and ws is None
                        else "NO SCORE"
                        if ws is None
                        else _format_report_number(ws)
                    )
                    bl_text = (
                        "N/A"
                        if scores.get("baseline_status") == "not_applicable" and bl is None
                        else "NO SCORE"
                        if bl is None
                        else _format_report_number(bl)
                    )
                    lift_text = (
                        _format_report_number(lift, signed=True)
                        if lift is not None
                        else "N/A"
                        if "not_applicable" in (scores.get("with_skill_status"), scores.get("baseline_status"))
                        else "skipped"
                        if with_skipped
                        else "NO SCORE"
                    )
                    lines.append(
                        f"| {_markdown_untrusted_table_cell(name.replace('_', ' ').title())} "
                        f"| {ws_text} | {bl_text} | {lift_text} |"
                    )
                lines.append("")

            from skillevaluator.reporting.tier3_token_usage import token_usage_markdown_lines

            lines.extend(
                token_usage_markdown_lines(
                    ae.get("token_usage"),
                    heading="### Token Usage",
                    agent_label=lambda agent: _markdown_untrusted_table_cell("All agents" if agent is None else agent),
                    cell=_markdown_untrusted_table_cell,
                    max_case_rows=_MAX_TOKEN_USAGE_CASE_ROWS,
                )
            )

            insights = _mapping(ae.get("insights"))
            insight_rows = [
                (dimension, info)
                for dimension, info in insights.items()
                if isinstance(dimension, str) and isinstance(info, dict) and info.get("score") is not None
            ]
            if insight_rows:
                lines.append("### LLM-as-Judge Insights")
                lines.append("")
                lines.append("| Dimension | Score | Explanation |")
                lines.append("|-----------|-------|-------------|")
                for dim, info in insight_rows:
                    score = info.get("score")
                    score_number = _finite_report_number(score)
                    score_str = (
                        f"{score_number:.2f}"
                        if score_number is not None
                        else _markdown_inline_text(score, limit=32).upper() or "N/A"
                    )
                    explanation = _markdown_untrusted_table_cell(info.get("explanation"), limit=60)
                    lines.append(f"| {_markdown_untrusted_table_cell(dim.title())} | {score_str} | {explanation} |")
                lines.append("")

            suggestions_v2 = _dict_items(ae.get("suggestions_v2"))
            if suggestions_v2:
                lines.append("### Evidence-Backed Suggestions")
                lines.append("")
                for idx, suggestion in enumerate(suggestions_v2, start=1):
                    recommendation = _markdown_inline_text(suggestion.get("recommendation"))
                    if not recommendation:
                        continue
                    metric = _markdown_inline_text(suggestion.get("metric")) or "unknown"
                    lines.append(f"{idx}. **{metric}**: {recommendation}")
                    harbor_evidence = suggestion.get("harbor_evidence") or suggestion.get("evidence")
                    if isinstance(harbor_evidence, dict):
                        url = _markdown_safe_url(harbor_evidence.get("url"))
                        if url:
                            label = _markdown_inline_text(harbor_evidence_link_text(harbor_evidence)) or "Evidence"
                            lines.append(f"   - Evidence: [{label}]({url})")
                    for ref in _dict_items(suggestion.get("evidence_refs"))[:3]:
                        pointer = _markdown_code_text(evidence_ref_identity(ref))
                        excerpt = _markdown_inline_text(ref.get("excerpt") or ref.get("label"), limit=120)
                        kind = _markdown_code_text(ref.get("kind")) or "evidence"
                        lines.append(f"   - Evidence: `{kind}` `{pointer}` {excerpt}")
                lines.append("")
            elif recommendations := _dict_items(ae.get("recommendations")):
                lines.append("### Recommendations")
                lines.append("")
                for idx, recommendation in enumerate(recommendations, start=1):
                    message = _markdown_inline_text(recommendation.get("message") or recommendation.get("title"))
                    if not message:
                        continue
                    lines.append(f"{idx}. {message}")
                    evidence = recommendation.get("evidence")
                    if isinstance(evidence, dict):
                        url = _markdown_safe_url(evidence.get("url"))
                        if url:
                            label = _markdown_inline_text(harbor_evidence_link_text(evidence)) or "Evidence"
                            lines.append(f"   - Evidence: [{label}]({url})")
                lines.append("")

            if next_steps := _dict_items(ae.get("next_steps")):
                lines.append("### Next Steps")
                lines.append("")
                for idx, step in enumerate(next_steps, start=1):
                    problem = _markdown_inline_text(step.get("problem"))
                    action = _markdown_inline_text(step.get("action"))
                    if not problem or not action:
                        continue
                    target = _markdown_inline_text(step.get("target")) or "evaluation"
                    strength = _markdown_inline_text(step.get("strength"))
                    label = target if strength in ("", "supported") else f"{target} · {strength}"
                    lines.append(f"{idx}. **{problem}** [{label}]: {action}")
                    if check := _markdown_inline_text(step.get("check")):
                        lines.append(f"   - Check: {check}")
                    if evidence := _markdown_inline_text(step.get("evidence")):
                        lines.append(f"   - Evidence: {evidence}")
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

    def _render_result(self, result: ValidationResult, lines: list[str]) -> None:
        """Render a single validation result."""
        qs = result.metadata.get("quality_scores")

        clean_skip = is_cleanly_skipped(result)
        if result_status(result) == Status.INCOMPLETE:
            status_emoji = "⚠️ INCOMPLETE"
            lines.append(f"### {status_emoji} {result.validator_name}")
        elif clean_skip:
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

        self._render_reference_summary(result, lines)

        if result.is_incomplete:
            self._render_incomplete(result, lines)
        elif clean_skip:
            lines.append(
                f"- Skip reason: {_markdown_inline_text(get_skip_reason(result), limit=_SKIP_REASON_MAX_CHARS)}"
            )
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
            verification = finding_verification(finding)
            if verification:
                label = f"{verification.verdict} ({verification.confidence} confidence)"
                message += f"<br><strong>LLM verification:</strong> {_markdown_untrusted_table_cell(label)}"
                if verification.reasoning:
                    message += f"<br>Reasoning: {_markdown_untrusted_table_cell(verification.reasoning)}"
                message += f"<br>{_markdown_untrusted_table_cell(verification.severity_effect)}"
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

    @staticmethod
    def _render_reference_summary(result: ValidationResult, lines: list[str]) -> None:
        """List how many referenced artifacts SkillSpector did not completely inspect in a failed result."""
        from skillevaluator.skillspector_references import reference_summary_message

        summary = reference_summary_message(result.metadata.get("skillspector_reference_summary"))
        if summary and not result.passed:
            lines.append(f"- {summary}")
            lines.append("")

    def _render_incomplete(self, result: ValidationResult, lines: list[str]) -> None:
        """Render missing scanner evidence without presenting a failure as a pass."""
        lines.append(f"**Incomplete scanners:** {', '.join(result.incomplete_scans)}")
        lines.append("")
        self._render_failure(result, lines)
        # Warnings say why a scan did not complete. Show them unless an error
        # already does; the findings table never repeats them. They can name
        # target files, so escape them as the error list does.
        if not additional_errors(result):
            for warning in additional_warnings(result)[: self.max_findings_shown]:
                lines.append(f"- ⚠️ {html.escape(warning, quote=False)}")

    def _render_finding_detail(self, index: int, finding: Finding, lines: list[str]) -> None:
        """Render detailed information for a single finding."""
        lines.append(f"**{index}. {finding.message}**")
        lines.append(f"- File: `{finding.location}`")
        lines.append(f"- Check: `{finding.check_name}`")
        lines.append(f"- Finding ID: `{finding.finding_id}`")
        lines.append(f"- Occurrence ID: `{finding.occurrence_id}`")
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
