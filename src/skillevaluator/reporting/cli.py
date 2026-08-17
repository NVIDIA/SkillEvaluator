# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI reporter using Rich for terminal output.

This reporter provides colorful, formatted terminal output using the Rich
library. It's the default reporter for interactive use.

Features:
- Colored output with severity-based highlighting
- Summary table for multiple validators
- Tree-structured findings display
- Progress indicators and spinners
"""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markup import escape as rich_escape
from rich.panel import Panel
from rich.table import Table

from skillevaluator.reporting.base import ReporterBase, passes_required_gate
from skillevaluator.reporting.harbor_viewer import (
    harbor_evidence_link_text,
    normalize_harbor_viewer_for_display,
    safe_url,
)

if TYPE_CHECKING:
    from skillevaluator.models import Finding, ValidationResult


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


class CLIReporter(ReporterBase):
    """Terminal-based reporter with Rich formatting.

    Provides colorful, structured output for terminal/console display.
    Supports both direct printing and string capture for testing.
    """

    def __init__(self, console: Console | None = None) -> None:
        """Initialize CLI reporter.

        Args:
            console: Rich Console instance (creates new one if not provided)
        """
        self._console = console

    @property
    def console(self) -> Console:
        """Get or create the Rich console instance."""
        if self._console is None:
            self._console = Console()
        return self._console

    @property
    def name(self) -> str:
        return "cli"

    @property
    def description(self) -> str:
        return "Terminal output with Rich formatting"

    def render(self, result: ValidationResult) -> str:
        """Render single result to string (captures console output)."""
        string_io = StringIO()
        temp_console = Console(file=string_io, force_terminal=True)
        self.render_result(result, temp_console)
        return string_io.getvalue()

    def render_all(self, results: list[ValidationResult]) -> str:
        """Render all results to string with summary table."""
        string_io = StringIO()
        temp_console = Console(file=string_io, force_terminal=True)
        self._render_all_results(results, temp_console)
        return string_io.getvalue()

    def print(self, result: ValidationResult) -> None:
        """Print single result directly to console."""
        self.render_result(result, self.console)

    def print_all(self, results: list[ValidationResult]) -> None:
        """Print all results directly to console."""
        self._render_all_results(results, self.console)

    def print_summary(self, results: list[ValidationResult]) -> None:
        """Print only the summary table (no failure details, no overall verdict).

        Used for progressive/interim CLI output -- e.g. flushing Tier 1 and
        Tier 2 results to the terminal before the long-running Tier 3 agent
        evaluation, so they stay visible in CI logs even when Tier 3 is slow,
        errors, or is interrupted before the final combined report is emitted.
        """
        self._print_summary_table(results, self.console)
        self.console.print()

    def _render_all_results(self, results: list[ValidationResult], console: Console) -> None:
        """Render all results with summary table and failure details."""
        # Print summary table
        self._print_summary_table(results, console)
        console.print()

        # Print detailed results for failures
        failed = [r for r in results if not passes_required_gate(r)]
        if failed:
            console.print(
                Panel.fit(
                    "[bold]Failure Details[/bold]",
                    style="yellow",
                    border_style="yellow",
                )
            )
            for result in failed:
                self.render_result(result, console)

        non_blocking = [result for result in results if passes_required_gate(result) and result.findings]
        if non_blocking:
            console.print(
                Panel.fit(
                    "[bold]Non-blocking Findings[/bold]",
                    style="yellow",
                    border_style="yellow",
                )
            )
            for result in non_blocking:
                self.render_result(result, console)

        # Print overall status
        advisory_skips = [result for result in results if self._is_advisory_agent_eval_skip(result)]
        required_passed = all(passes_required_gate(result) for result in results)
        if any(result.is_incomplete for result in results):
            console.print("\n[bold yellow][INCOMPLETE] Validation evidence is incomplete[/bold yellow]\n")
        elif required_passed:
            if advisory_skips:
                console.print(
                    "\n[bold green][PASS] Required validations passed[/bold green] "
                    f"[yellow]({len(advisory_skips)} live evaluation skipped)[/yellow]\n"
                )
            else:
                console.print("\n[bold green][PASS] All validations passed[/bold green]\n")
        else:
            console.print("\n[bold red][FAIL] Validation failed[/bold red]\n")

    def render_result(self, result: ValidationResult, console: Console) -> None:
        """Render a single validation result."""
        # Header
        console.print(f"\n[bold][{result.validator_name}][/bold]")
        if result.validator_description:
            console.print(f"[dim]{result.validator_description}[/dim]")

        # Quality score display (unified table for single and multi-skill)
        qs = result.metadata.get("quality_scores") if result.metadata else None
        qs_all = result.metadata.get("quality_scores_all") if result.metadata else None
        if qs and qs.get("dimensions"):
            grade_colors = {"A": "green", "B": "green", "C": "yellow", "D": "red", "F": "red"}
            skills_list = qs_all or [qs]
            self._print_quality_table(qs, skills_list, grade_colors, console)

        # LLM rubric evaluation display
        rubric = result.metadata.get("rubric_eval") if result.metadata else None
        if rubric and rubric.get("checks"):
            self._print_rubric_table(rubric, console)

        # Tier 3: Agent evaluation display
        agent_eval = result.metadata.get("agent_eval") if result.metadata else None
        if agent_eval:
            self._print_agent_eval_tables(agent_eval, console)

        if self._is_advisory_agent_eval_skip(result):
            console.print("[yellow][SKIP] Live evaluation did not run[/yellow]\n")
            self._print_summary_stats(result, console)
        elif result.is_incomplete:
            tools = ", ".join(result.incomplete_scans)
            console.print(f"[yellow][INCOMPLETE] {tools} did not complete[/yellow]\n")
            self._print_summary_stats(result, console)
            self._print_findings(result, console)
        elif result.passed:
            console.print("[green][PASS] Validation passed[/green]\n")
            self._print_summary_stats(result, console)
            self._print_success_details(result, console)
            if result.findings:
                self._print_findings(result, console)
        else:
            console.print("[red][FAIL] Validation failed[/red]\n")
            self._print_summary_stats(result, console)
            self._print_findings(result, console)

    @staticmethod
    def _score_cell(dims: dict, dname: str) -> str:
        """Format a dimension score cell with color coding."""
        d = dims.get(dname, {})
        s = d.get("score", 0)
        c = "green" if s >= 80 else ("yellow" if s >= 60 else "red")
        return f"[{c}]{s:.0f}[/{c}]"

    @staticmethod
    def _print_quality_table(
        qs: dict,
        skills_list: list[dict],
        grade_colors: dict[str, str],
        console: Console,
    ) -> None:
        """Print a unified quality scores table for single or multi-skill runs."""
        multi = len(skills_list) > 1
        avg_score = qs.get("overall_score", 0)
        avg_grade = qs.get("grade", "?")
        gc = grade_colors.get(avg_grade, "white")

        if multi:
            console.print(f"\n  [{gc}]Average: {avg_score:.1f}/100 (Grade: {avg_grade})[/{gc}]")
            console.print(f"  [dim]Skills analyzed:[/dim] {qs.get('skill_count', len(skills_list))}\n")
        else:
            stype = skills_list[0].get("skill_type", "unknown")
            console.print(f"\n  [{gc}]Overall: {avg_score:.1f}/100 (Grade: {avg_grade})[/{gc}]")
            console.print(f"  [dim]Skill Type:[/dim] {stype}\n")

        table = Table(title="Quality Scores by Skill", border_style="cyan", show_header=True)
        table.add_column("Skill", style="bold")
        table.add_column("Grade", justify="center")
        table.add_column("Score", justify="right")
        table.add_column("Correctness", justify="right")
        table.add_column("Discoverability", justify="right")
        table.add_column("Reliability", justify="right")
        table.add_column("Efficiency", justify="right")

        for skill_qs in skills_list:
            sname = skill_qs.get("skill_name", "?")
            sgrade = skill_qs.get("grade", "?")
            sscore = skill_qs.get("overall_score", 0)
            sgc = grade_colors.get(sgrade, "white")
            dims = skill_qs.get("dimensions", {})
            table.add_row(
                sname,
                f"[{sgc}]{sgrade}[/{sgc}]",
                f"[{sgc}]{sscore:.1f}[/{sgc}]",
                CLIReporter._score_cell(dims, "correctness"),
                CLIReporter._score_cell(dims, "discoverability"),
                CLIReporter._score_cell(dims, "reliability"),
                CLIReporter._score_cell(dims, "efficiency"),
            )

        if multi:
            table.add_section()
            agc = grade_colors.get(avg_grade, "white")
            avg_dims = qs.get("dimensions", {})
            table.add_row(
                "[bold]Average[/bold]",
                f"[{agc}]{avg_grade}[/{agc}]",
                f"[{agc}]{avg_score:.1f}[/{agc}]",
                CLIReporter._score_cell(avg_dims, "correctness"),
                CLIReporter._score_cell(avg_dims, "discoverability"),
                CLIReporter._score_cell(avg_dims, "reliability"),
                CLIReporter._score_cell(avg_dims, "efficiency"),
            )

        console.print(table)
        console.print()

    @staticmethod
    def _print_rubric_table(rubric: dict, console: Console) -> None:
        """Print LLM rubric evaluation results as a table."""
        score = rubric.get("overall_score", 0)
        color = "green" if score >= 80 else ("yellow" if score >= 60 else "red")
        console.print(f"\n  [{color}]LLM Rubric Score: {score}/100[/{color}]")
        summary = rubric.get("summary", "")
        if summary:
            console.print(f"  [dim]{summary}[/dim]")
        console.print()

        table = Table(title="Rubric Evaluation", border_style="cyan", show_header=True)
        table.add_column("Criterion", style="bold")
        table.add_column("Score", justify="center")
        table.add_column("Pass", justify="center")
        table.add_column("Notes")

        for check in rubric.get("checks", []):
            cs = check.get("score", 0)
            cc = "green" if cs >= 7 else ("yellow" if cs >= 5 else "red")
            passed = "[green]Yes[/green]" if check.get("pass") else "[red]No[/red]"
            table.add_row(
                check.get("id", "?").replace("_", " ").title(),
                f"[{cc}]{cs}/10[/{cc}]",
                passed,
                check.get("notes", ""),
            )

        console.print(table)
        console.print()

    @staticmethod
    def _print_agent_eval_tables(agent_eval: dict, console: Console) -> None:
        """Print Tier 3 agent evaluation results."""
        verdict = agent_eval.get("verdict", "unknown")
        composite = agent_eval.get("composite_lift")
        runtime = agent_eval.get("runtime_seconds", 0.0)

        vc = "green" if verdict == "pass" else ("red" if verdict == "fail" else "yellow")
        composite_text = f"{composite:+.2f}" if isinstance(composite, int | float) else "N/A"
        console.print(f"\n  [{vc}]Verdict: {verdict.upper()} (composite lift = {composite_text})[/{vc}]")
        if runtime:
            console.print(f"  [dim]Runtime: {runtime:.1f}s[/dim]")
        harbor_viewer = normalize_harbor_viewer_for_display(agent_eval)
        if harbor_viewer.get("job_url") or harbor_viewer.get("analysis_url"):
            console.print("  [dim]Harbor artifacts:[/dim]")
            if harbor_viewer.get("job_url"):
                console.print(f"    [dim]Harbor logs:[/dim] [cyan]{harbor_viewer['job_url']}[/cyan]", soft_wrap=True)
            if harbor_viewer.get("analysis_url"):
                console.print(
                    f"    [dim]Harbor analysis:[/dim] [cyan]{harbor_viewer['analysis_url']}[/cyan]",
                    soft_wrap=True,
                )
        console.print()

        evaluators = agent_eval.get("evaluators", {})
        if evaluators:
            table = Table(
                title="Evaluator Scores (Skill Lift)",
                border_style="cyan",
                show_header=True,
            )
            table.add_column("Evaluator", style="bold")
            table.add_column("With Skill", justify="right")
            table.add_column("Baseline", justify="right")
            table.add_column("Lift", justify="right")

            for name, scores in evaluators.items():
                ws = scores.get("with_skill", 0.0)
                bl = scores.get("baseline", 0.0)
                lift = scores.get("lift", 0.0)
                lc = "green" if lift > 0.01 else ("red" if lift < -0.01 else "dim")
                table.add_row(
                    name.replace("_", " ").title(),
                    f"{ws:.2f}",
                    f"{bl:.2f}",
                    f"[{lc}]{lift:+.2f}[/{lc}]",
                )

            console.print(table)
            console.print()

        recommendations = agent_eval.get("recommendations") or []
        if recommendations:
            printed = False
            for recommendation in recommendations[:5]:
                if not isinstance(recommendation, dict):
                    continue
                message = str(recommendation.get("message") or recommendation.get("title") or "").strip()
                if not message:
                    continue
                if not printed:
                    console.print("[bold]Recommendations[/bold]")
                    printed = True
                console.print(f"  • {message}", soft_wrap=True)
                evidence = recommendation.get("evidence")
                if isinstance(evidence, dict):
                    url = safe_url(evidence.get("url"))
                    if url:
                        console.print(
                            f"    [dim]{harbor_evidence_link_text(evidence)}:[/dim] [cyan]{url}[/cyan]", soft_wrap=True
                        )
            if printed:
                console.print()

        insights = agent_eval.get("insights", {})
        if any(v.get("score") is not None for v in insights.values()):
            table = Table(
                title="LLM-as-Judge Insights",
                border_style="cyan",
                show_header=True,
            )
            table.add_column("Dimension", style="bold")
            table.add_column("Score", justify="center")
            table.add_column("Explanation")

            for dim, info in insights.items():
                score = info.get("score")
                if score is None:
                    continue
                explanation = info.get("explanation", "")
                if isinstance(score, str):
                    sc = "green" if score.upper() == "PASS" else "red"
                    score_str = f"[{sc}]{score}[/{sc}]"
                else:
                    sc = "green" if score >= 0.7 else ("yellow" if score >= 0.4 else "red")
                    score_str = f"[{sc}]{score:.2f}[/{sc}]"
                table.add_row(dim.title(), score_str, explanation[:80])

            console.print(table)
            console.print()

    def _print_summary_stats(self, result: ValidationResult, console: Console) -> None:
        """Print summary statistics."""
        s = result.summary
        console.print("[dim]Summary:[/dim]")
        if s.files_scanned > 0:
            console.print(f"  • Files scanned: {s.files_scanned}")
        if s.checks_performed > 0:
            console.print(f"  • Checks performed: {s.checks_performed}")
        if not result.passed:
            error_detail = f"{s.errors}"
            if s.critical_count > 0 or s.high_count > 0:
                parts = []
                if s.critical_count > 0:
                    parts.append(f"{s.critical_count} critical")
                if s.high_count > 0:
                    parts.append(f"{s.high_count} high")
                error_detail += f" ({', '.join(parts)})"
            console.print(f"  • Errors: {error_detail}")
            if s.warnings > 0:
                console.print(f"  • Warnings: {s.warnings}")
        console.print()

    def _print_success_details(self, result: ValidationResult, console: Console) -> None:
        """Print success details."""
        if not result.success_details:
            # Fall back to legacy messages
            if result.messages:
                console.print("[dim]Details:[/dim]")
                for msg in result.messages:
                    console.print(f"  {rich_escape(str(msg))}")
            return

        console.print("[dim]Details:[/dim]")
        for detail in result.success_details:
            meta_str = ""
            if detail.metadata:
                meta_parts = [f"{k}={v}" for k, v in detail.metadata.items()]
                if meta_parts:
                    meta_str = f" ({', '.join(meta_parts)})"
            console.print(
                "  [green][OK][/green] "
                f"{rich_escape(str(detail.check_name))}: "
                f"{rich_escape(str(detail.message))}{rich_escape(meta_str)}"
            )

    def _print_findings(self, result: ValidationResult, console: Console) -> None:
        """Print findings with tree structure."""
        if not result.findings:
            # Fall back to legacy errors/warnings
            if result.errors:
                console.print("[dim]Errors:[/dim]")
                for error in result.errors:
                    console.print(f"  [red]•[/red] {rich_escape(str(error))}")
            if result.warnings:
                console.print("[dim]Warnings:[/dim]")
                for warning in result.warnings:
                    console.print(f"  [yellow][WARN][/yellow] {rich_escape(str(warning))}")
            return

        console.print("[dim]Issues:[/dim]")
        for i, finding in enumerate(result.findings, 1):
            self._print_finding(i, finding, console)

    def _print_finding(self, index: int, finding: Finding, console: Console) -> None:
        """Print a single finding with structured details.

        Dynamic finding fields (message, location, content, suggestion) are
        escaped so values containing ``[...]`` are not parsed as Rich markup
        (which would raise ``rich.errors.MarkupError`` and abort the report).
        """
        severity_color = finding.severity.color
        console.print(
            f"  {index}. [{severity_color}]{rich_escape(finding.tag)}[/{severity_color}] {rich_escape(finding.message)}"
        )
        console.print(f"     [dim]File:[/dim]    {rich_escape(str(finding.location))}")
        console.print(f"     [dim]Check:[/dim]   {rich_escape(str(finding.check_name))}")
        related_paths = _related_paths(finding)
        if related_paths:
            console.print(f"     [dim]Related paths:[/dim] {rich_escape(' <-> '.join(related_paths))}")

        if finding.line_content:
            content = finding.line_content.strip()
            if len(content) > 70:
                content = content[:67] + "..."
            console.print(f"     [dim]Content:[/dim] [italic]{rich_escape(content)}[/italic]")

        if finding.suggestion:
            label = (
                "Recommendation" if finding.category in ("WHOLE_DUPLICATE", "PARTIAL_OVERLAP", "DUPLICATE") else "Fix"
            )
            console.print(f"     [dim]{label}:[/dim]     {rich_escape(finding.suggestion)}")
        console.print()

    def _print_summary_table(self, results: list[ValidationResult], console: Console) -> None:
        """Print summary table for all validators."""
        table = Table(
            title="SkillEvaluator Validation Results",
            show_header=True,
            header_style="bold white on dark_green",
            border_style="green",
            title_style="bold green",
        )
        table.add_column("Validator", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Details")

        for result in results:
            advisory_skip = self._is_advisory_agent_eval_skip(result)
            status = (
                "[yellow]SKIP[/yellow]"
                if advisory_skip
                else "[bold yellow]INCOMPLETE[/bold yellow]"
                if result.is_incomplete
                else "[green]PASS[/green]"
                if result.passed
                else "[red]FAIL[/red]"
            )
            s = result.summary
            static_test_evidence = self._static_test_evidence_message(result)

            if advisory_skip:
                agent_eval = result.metadata.get("agent_eval", {})
                provenance = agent_eval.get("provenance", {}) if isinstance(agent_eval, dict) else {}
                details = str(provenance.get("message") or "Live evaluation did not run")
            elif result.is_incomplete:
                details = f"[bold yellow]{', '.join(result.incomplete_scans)} did not complete[/bold yellow]"
                counts = []
                if s.errors:
                    counts.append(f"{s.errors} errors")
                if s.warnings:
                    counts.append(f"{s.warnings} warnings")
                if counts:
                    details += f" ({', '.join(counts)})"
            elif result.passed:
                if result.metadata.get("skipped"):
                    details = "Skipped (see warnings)"
                elif static_test_evidence:
                    details = rich_escape(static_test_evidence)
                elif s.checks_performed > 0:
                    details = f"{s.checks_performed} checks passed"
                else:
                    details = "OK"
            else:
                parts = []
                if s.errors > 0:
                    parts.append(f"{s.errors} errors")
                if s.warnings > 0:
                    parts.append(f"{s.warnings} warnings")
                details = ", ".join(parts) if parts else "Failed"

            table.add_row(result.validator_name, status, details)

        console.print(table)

    @staticmethod
    def _static_test_evidence_message(result: ValidationResult) -> str | None:
        """Return the static test limitation from direct or folder-aggregated results."""
        for detail in result.success_details:
            if detail.check_name == "test_discovery":
                return detail.message
        for detail in result.success_details:
            checks = detail.metadata.get("checks") if isinstance(detail.metadata, dict) else None
            if not isinstance(checks, list):
                continue
            for check in checks:
                if isinstance(check, dict) and check.get("name") == "test_discovery":
                    return "Target tests were not executed and coverage was not measured for any discovered skill"
        return None

    @staticmethod
    def _is_advisory_agent_eval_skip(result: ValidationResult) -> bool:
        """Return whether an AGENT_EVAL result records a skipped live run."""
        if result.validator_name != "AGENT_EVAL":
            return False
        payload = result.metadata.get("agent_eval", {}) if result.metadata else {}
        provenance = payload.get("provenance", {}) if isinstance(payload, dict) else {}
        return bool(
            isinstance(provenance, dict) and provenance.get("advisory") and provenance.get("reason") == "skipped"
        )
