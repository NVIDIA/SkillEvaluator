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

Text that SkillEvaluator does not author (skill and plugin content, paths,
LLM judge output, external tool messages) goes through ``escape_markup`` before
it is interpolated into Rich markup: ``[/x]`` would otherwise raise
``rich.errors.MarkupError`` and ``[link=...]`` could restyle or spoof output.
Consoles are created with ``emoji=False`` so ``:name:`` codes in that text
(``root:x:0:0:``) print literally.
"""

from __future__ import annotations

import math
from io import StringIO
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from skillevaluator.constants import (
    DIMENSION_VERDICT_NEUTRAL_THRESHOLD,
    DIMENSION_VERDICT_PASS_THRESHOLD,
)
from skillevaluator.reporting.base import (
    ReporterBase,
    _agent_eval_finite_number,
    additional_errors,
    additional_warnings,
    get_skip_reason,
    is_advisory_agent_eval_skip,
    is_cleanly_skipped,
    passes_required_gate,
    source_validation_errors,
    source_validation_preview,
)
from skillevaluator.reporting.finding_verification import finding_verification
from skillevaluator.reporting.harbor_viewer import (
    harbor_evidence_link_text,
    harbor_job_links,
    safe_url,
)
from skillevaluator.reporting.presentation import FinalRenderContext, build_final_render_context, trial_coverage
from skillevaluator.reporting.run_status import Status, result_status
from skillevaluator.reporting.summary_copy import SKILL_LIFT_DEFINITION, pass_at_k_summary
from skillevaluator.reporting.terminal_styles import AMBER, FAINT, GREEN, RED, TEXT, grade_style, lift_style
from skillevaluator.tier3.baseline_store import baseline_description
from skillevaluator.tier3.dataset_provenance import dataset_provenance_display
from skillevaluator.tier3.evidence_volume import format_evidence_volume
from skillevaluator.tier3.version_comparison import version_comparison_lines
from skillevaluator.utils.rich_markup import escape_markup

if TYPE_CHECKING:
    from collections.abc import Iterable

    from skillevaluator.models import Finding, ValidationResult


_SKIP_REASON_MAX_CHARS = 1024
_SOURCE_ERROR_MAX_CHARS = 1024


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


def _rich_source_error(error: dict[str, str]) -> str:
    """Return one failed Tier 3 source check as a bounded line inert to Rich markup."""
    text = f"{error['path']}: {error['message']}" if error["path"] else error["message"]
    text = " ".join(text.split())
    if len(text) > _SOURCE_ERROR_MAX_CHARS:
        text = text[: _SOURCE_ERROR_MAX_CHARS - 1] + "…"
    return escape_markup(text)


def _rich_line(text: str) -> str:
    """Return one bounded, single-line message inert to Rich markup."""
    line = " ".join(text.replace("\r\n", "\n").replace("\r", "\n").split())
    if len(line) > _SKIP_REASON_MAX_CHARS:
        line = line[: _SKIP_REASON_MAX_CHARS - 1] + "…"
    return escape_markup(line)


def _rich_skip_reason(result: ValidationResult) -> str:
    """Return one bounded, single-line skip reason inert to Rich markup."""
    return _rich_line(get_skip_reason(result))


def _coverage_notice_messages(agent_eval: object) -> list[str]:
    """Return the Tier 3 coverage notice messages the Tier 3 block prints."""
    notices = agent_eval.get("coverage_notices") if isinstance(agent_eval, dict) else None
    return [
        notice["message"]
        for notice in (notices if isinstance(notices, list) else [])
        if isinstance(notice, dict) and isinstance(notice.get("message"), str) and notice["message"]
    ]


class CLIReporter(ReporterBase):
    """Terminal-based reporter with Rich formatting.

    Provides colorful, structured output for terminal/console display.
    Supports both direct printing and string capture for testing.
    """

    def __init__(
        self,
        console: Console | None = None,
        *,
        report_formats: Iterable[str] = (),
        presentation_context: FinalRenderContext | None = None,
    ) -> None:
        """Initialize CLI reporter.

        Args:
            console: Rich Console instance (creates new one if not provided)
            report_formats: Every format the run emits alongside this report.
                Failed Tier 3 source checks are cut to a preview only when one
                of these files lists all of them.
        """
        self._console = console
        self._report_formats = tuple(report_formats)
        self._presentation_context = presentation_context

    @property
    def console(self) -> Console:
        """Get or create the Rich console instance."""
        if self._console is None:
            self._console = Console(emoji=False)
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
        temp_console = Console(file=string_io, force_terminal=True, emoji=False)
        self.render_result(result, temp_console)
        return string_io.getvalue()

    def render_all(self, results: list[ValidationResult]) -> str:
        """Render all results to string with summary table."""
        string_io = StringIO()
        temp_console = Console(file=string_io, force_terminal=True, emoji=False)
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
        context = self._presentation_context or build_final_render_context(results)
        # Print summary table
        self._print_summary_table(results, console, presentation=context)
        console.print()

        failed = [result for result in results if not passes_required_gate(result)]
        # A non-blocking Tier 3 result (the ``validate`` default) can fail with
        # no findings; its coverage notes still say which agent arm failed.
        non_blocking = [
            result
            for result in results
            if passes_required_gate(result)
            and (result.findings or _coverage_notice_messages((result.metadata or {}).get("agent_eval")))
        ]
        rendered_result_ids = {id(result) for result in (*failed, *non_blocking)}
        for result in results:
            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            agent_eval = metadata.get("agent_eval")
            if (
                result.validator_name == "AGENT_EVAL"
                and id(result) not in rendered_result_ids
                and not self._is_advisory_agent_eval_skip(result)
                and isinstance(agent_eval, dict)
                and result_status(result) not in {Status.SKIPPED, Status.INCOMPLETE}
            ):
                self._print_agent_eval_tables(agent_eval, console)
                rendered_result_ids.add(id(result))

        # Print detailed results for failures
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

        # Successful results do not enter the detailed failure/finding sections.
        rendered = rendered_result_ids
        for result in results:
            if id(result) not in rendered:
                agent_eval = result.metadata.get("agent_eval")
                if isinstance(agent_eval, dict):
                    provenance_line, provenance_is_template = dataset_provenance_display(
                        agent_eval.get("dataset_provenance")
                    )
                    if provenance_line:
                        console.print(Text(f"  {provenance_line}", style="yellow" if provenance_is_template else "dim"))
                    self._print_harbor_artifacts(agent_eval, console)

        color = {Status.PASSED: "green", Status.FAILED: "red"}.get(context.command_status, "yellow")
        console.print(f"\n[bold {color}][{context.command_status.verdict}] {context.command_label}[/bold {color}]")
        for number, tier in context.tiers.items():
            if not tier.displayed:
                continue
            console.print(f"Tier {number}: {tier.summary}", highlight=False, soft_wrap=True)
        if context.tiers[3].status not in (Status.NOT_RUN, Status.SKIPPED):
            console.print(context.coverage.text, highlight=False, soft_wrap=True)
            for line in context.coverage.details:
                console.print(line, highlight=False, soft_wrap=True)
        console.print(f"Publication eligibility: {context.publication_label}\n")

    def render_result(self, result: ValidationResult, console: Console) -> None:
        """Render a single validation result."""
        # Header
        console.print(f"\n[bold]{escape_markup(f'[{result.validator_name}]')}[/bold]")
        if result.validator_description:
            console.print(f"[dim]{escape_markup(str(result.validator_description))}[/dim]")

        # Quality score display (unified table for single and multi-skill)
        qs = result.metadata.get("quality_scores") if result.metadata else None
        qs_all = result.metadata.get("quality_scores_all") if result.metadata else None
        if qs and qs.get("dimensions"):
            grade_colors = {grade: grade_style(grade, status=result_status(result).value) for grade in "ABCDF"}
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

        if result_status(result) == Status.SKIPPED:
            label = (
                "Live evaluation did not run"
                if self._is_advisory_agent_eval_skip(result)
                else "Validation check skipped"
            )
            console.print(f"[yellow][SKIP] {label}[/yellow]\n")
            self._print_summary_stats(result, console)
        elif result_status(result) == Status.INCOMPLETE:
            tools = ", ".join(result.incomplete_scans)
            console.print(f"[yellow][INCOMPLETE] {escape_markup(tools)} did not complete[/yellow]\n")
            self._print_reference_summary(result, console)
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
            self._print_reference_summary(result, console)
            self._print_summary_stats(result, console)
            self._print_findings(result, console)

    @staticmethod
    def _print_reference_summary(result: ValidationResult, console: Console) -> None:
        """Print how many referenced artifacts SkillSpector did not completely inspect in a failed result."""
        from skillevaluator.skillspector_references import reference_summary_message

        summary = reference_summary_message(result.metadata.get("skillspector_reference_summary"))
        if summary and not result.passed:
            console.print(f"  {escape_markup(summary)}", soft_wrap=True, highlight=False)
            console.print()

    @staticmethod
    def _score_cell(dims: dict, dname: str) -> str:
        """Format a dimension score cell with color coding."""
        d = dims.get(dname, {})
        s = d.get("score", 0)
        c = GREEN if s >= 80 else (AMBER if s >= 60 else RED)
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
        avg_grade_text = escape_markup(str(avg_grade))

        if multi:
            console.print(f"\n  [{gc}]Average: {avg_score:.1f}/100 (Grade: {avg_grade_text})[/{gc}]")
            skill_count = escape_markup(str(qs.get("skill_count", len(skills_list))))
            console.print(f"  [dim]Skills analyzed:[/dim] {skill_count}\n")
        else:
            stype = escape_markup(str(skills_list[0].get("skill_type", "unknown")))
            console.print(f"\n  [{gc}]Overall: {avg_score:.1f}/100 (Grade: {avg_grade_text})[/{gc}]")
            console.print(f"  [dim]Skill Type:[/dim] {stype}\n")

        table = Table(title="Quality Scores by Skill", border_style=FAINT, show_header=True)
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
                escape_markup(str(sname)),
                f"[{sgc}]{escape_markup(str(sgrade))}[/{sgc}]",
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
                f"[{agc}]{avg_grade_text}[/{agc}]",
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
            console.print(f"  [dim]{escape_markup(str(summary))}[/dim]")
        console.print()

        table = Table(title="Rubric Evaluation", border_style=FAINT, show_header=True)
        table.add_column("Criterion", style="bold")
        table.add_column("Score", justify="center")
        table.add_column("Pass", justify="center")
        table.add_column("Notes")

        for check in rubric.get("checks", []):
            cs = check.get("score", 0)
            cc = "green" if cs >= 7 else ("yellow" if cs >= 5 else "red")
            passed = "[green]Yes[/green]" if check.get("pass") else "[red]No[/red]"
            table.add_row(
                escape_markup(str(check.get("id", "?")).replace("_", " ").title()),
                f"[{cc}]{cs}/10[/{cc}]",
                passed,
                escape_markup(str(check.get("notes") or "")),
            )

        console.print(table)
        console.print()

    @staticmethod
    def _print_harbor_artifacts(agent_eval: dict, console: Console) -> None:
        jobs = harbor_job_links(agent_eval)
        if jobs:
            console.print("  [bold]Harbor artifacts:[/bold]")
        for job in jobs:
            console.print(f"    [bold]{escape_markup(job['name'])}[/bold]", highlight=False)
            if job.get("url"):
                console.print(f"    Harbor logs: Results {escape_markup(job['url'])}", soft_wrap=True, highlight=False)
            if job.get("analysis_url"):
                console.print(
                    f"    Harbor analysis: {escape_markup(job['analysis_url'])}", soft_wrap=True, highlight=False
                )

    @staticmethod
    def _print_agent_eval_tables(agent_eval: dict, console: Console) -> None:
        """Print Tier 3 agent evaluation results."""
        if agent_eval.get("baseline"):
            console.print(escape_markup(baseline_description(agent_eval["baseline"])))
        verdict = agent_eval.get("verdict", "unknown")
        best_agent = agent_eval.get("best_agent")
        multiple_agents = isinstance(agent_eval.get("agents_run"), list) and len(agent_eval["agents_run"]) > 1
        composite = agent_eval.get("composite_lift")
        runtime = agent_eval.get("runtime_seconds", 0.0)

        vc = GREEN if verdict == "pass" else (RED if verdict == "fail" else AMBER)
        scored_lift = (
            isinstance(composite, int | float) and not isinstance(composite, bool) and math.isfinite(composite)
        )
        composite_text = f"{composite:+.2f}" if scored_lift else "N/A"
        composite_style = lift_style(composite) if scored_lift else "dim"
        verdict_text = escape_markup(str(verdict).upper())
        verdict_reason = agent_eval.get("verdict_reason")
        safe_reason = f" — {escape_markup(str(verdict_reason))}" if verdict == "fail" and verdict_reason else ""
        lift_label = (
            "Skill Lift (for reference)" if verdict == "fail" and scored_lift and composite > 0 else "Skill Lift"
        )
        console.print(
            f"\n  [{vc}]Verdict: {verdict_text}[/{vc}]{safe_reason} "
            f"({lift_label} = [{composite_style}]{composite_text}[/{composite_style}])"
        )
        if scored_lift:
            console.print(Text(SKILL_LIFT_DEFINITION, style="dim"), soft_wrap=True)
        if pass_summary := pass_at_k_summary(agent_eval.get("pass_at_k"), agent_eval.get("attempt_policy")):
            console.print(f"  [dim]{pass_summary}[/dim]")
        for label, count_data, numerator in (
            ("Required checks passed", agent_eval.get("required_checks"), "passed"),
            ("Goals met", agent_eval.get("correct_answers"), "correct"),
            ("Security clean", agent_eval.get("security_clean"), "clean"),
        ):
            with_skill = count_data.get("with_skill") if isinstance(count_data, dict) else None
            without_skill = count_data.get("without_skill") if isinstance(count_data, dict) else None
            difference = count_data.get("difference") if isinstance(count_data, dict) else None
            with_counts = (with_skill.get(numerator), with_skill.get("scored")) if isinstance(with_skill, dict) else ()
            without_counts = (
                (without_skill.get(numerator), without_skill.get("scored")) if isinstance(without_skill, dict) else ()
            )
            valid_with = (
                len(with_counts) == 2
                and all(isinstance(count, int) and not isinstance(count, bool) and count >= 0 for count in with_counts)
                and with_counts[1] > 0
            )
            valid_without = (
                len(without_counts) == 2
                and all(
                    isinstance(count, int) and not isinstance(count, bool) and count >= 0 for count in without_counts
                )
                and without_counts[1] > 0
            )
            if valid_with:
                suffix = f" ({best_agent})" if multiple_agents and best_agent else ""
                line = f"  {label}{suffix}: {with_counts[0]}/{with_counts[1]}"
                if valid_without:
                    if isinstance(difference, int) and not isinstance(difference, bool):
                        delta = f"{difference:+d}"
                    elif isinstance(difference, int | float) and not isinstance(difference, bool):
                        delta = f"{difference:+.1f} pp"
                    else:
                        delta = "N/A"
                    line += f" | {without_counts[0]}/{without_counts[1]} | {delta}"
                else:
                    line += " | N/A | N/A"
                console.print(line)
        provenance_line, provenance_is_template = dataset_provenance_display(agent_eval.get("dataset_provenance"))
        if provenance_line:
            console.print(Text(f"  {provenance_line}", style="yellow" if provenance_is_template else "dim"))
        if verdict == "inconclusive":
            console.print(
                "  [yellow]Skill never activated. Scores are diagnostic only; no lift verdict was computed.[/yellow]"
            )
        if runtime:
            console.print(f"  [dim]Runtime: {runtime:.1f}s[/dim]")
        evidence_line = format_evidence_volume(agent_eval.get("evidence_volume"))
        if evidence_line:
            style = "yellow" if evidence_line.startswith("Limited evidence:") else "dim"
            console.print(f"  [{style}]{evidence_line}[/{style}]")
        sufficiency = agent_eval.get("dataset_sufficiency")
        if isinstance(sufficiency, dict):
            status = str(sufficiency.get("status") or "provisional")
            style = "yellow" if status == "provisional" else "green"
            line = (
                sufficiency.get("coverage_summary")
                or f"Dataset sufficiency: {status} — {sufficiency.get('reason', '')}"
            )
            console.print(f"  [{style}]{line}[/{style}]")
        CLIReporter._print_harbor_artifacts(agent_eval, console)
        pass_at_k = agent_eval.get("pass_at_k")
        with_skill_pass = pass_at_k.get("with_skill") if isinstance(pass_at_k, dict) else None
        case_arm_label = "with skill"
        case_summaries = with_skill_pass.get("cases") if isinstance(with_skill_pass, dict) else None
        if not isinstance(case_summaries, dict) or not case_summaries:
            without_skill_pass = pass_at_k.get("without_skill") if isinstance(pass_at_k, dict) else None
            case_summaries = without_skill_pass.get("cases") if isinstance(without_skill_pass, dict) else None
            case_arm_label = "without skill"
        if isinstance(case_summaries, dict) and case_summaries:
            title = "Tier 3 Cases" if case_arm_label == "with skill" else "Tier 3 Cases (without skill)"
            table = Table(title=title, border_style=FAINT, show_header=True)
            table.add_column("Case", style="bold")
            table.add_column("Score", justify="right")
            table.add_column("Result")
            table.add_column("Warnings")
            for case_id, case in case_summaries.items():
                if not isinstance(case, dict):
                    continue
                score = case.get("best_score")
                score_text = f"{score:.2f}" if isinstance(score, int | float) and not isinstance(score, bool) else "N/A"
                if case.get("passed") is True:
                    result_text = "[green]PASS[/green]"
                elif (
                    isinstance(case.get("attempts_used"), int)
                    and not isinstance(case.get("attempts_used"), bool)
                    and case.get("attempts_used") > 0
                ):
                    result_text = "[red]FAIL[/red]"
                    reason = case.get("reason")
                    if isinstance(reason, str) and reason:
                        result_text += f" — {escape_markup(reason)}"
                else:
                    result_text = "[yellow]not scored[/yellow]"
                warnings = case.get("warnings")
                warning_text = (
                    "; ".join(str(item) for item in warnings if isinstance(item, str))
                    if isinstance(warnings, list)
                    else ""
                )
                table.add_row(
                    escape_markup(str(case_id)),
                    score_text,
                    result_text,
                    f"[yellow]WARNING[/yellow] {escape_markup(warning_text)}" if warning_text else "",
                )
            if table.rows:
                console.print(table)
                console.print()
        console.print(f"  {trial_coverage(agent_eval).text}")
        console.print()

        evaluators = agent_eval.get("evaluators", {})
        if evaluators:
            table = Table(
                title="Evaluator Scores (Skill Lift)",
                border_style=FAINT,
                show_header=True,
            )
            table.add_column("Evaluator", style="bold")
            table.add_column("With Skill", justify="right")
            table.add_column("Baseline", justify="right")
            table.add_column("Lift", justify="right")

            for name, scores in evaluators.items():
                ws = scores.get("with_skill")
                bl = scores.get("baseline")
                lift = scores.get("lift")
                # A baseline-only run (--skip-with-skill) never ran the with-skill arm.
                with_skipped = scores.get("with_skill_status") == "skipped"
                if lift is None:
                    lift_text = (
                        "N/A"
                        if "not_applicable" in (scores.get("with_skill_status"), scores.get("baseline_status"))
                        else "skipped"
                        if with_skipped
                        else "NO SCORE"
                    )
                else:
                    lc = lift_style(lift)
                    lift_text = f"[{lc}]{lift:+.2f}[/{lc}]"
                table.add_row(
                    escape_markup(str(name).replace("_", " ").title()),
                    f"{ws:.2f}"
                    if ws is not None
                    else "N/A"
                    if scores.get("with_skill_status") == "not_applicable"
                    else "skipped"
                    if with_skipped
                    else "NO SCORE",
                    f"{bl:.2f}"
                    if bl is not None
                    else "N/A"
                    if scores.get("baseline_status") == "not_applicable"
                    else "NO SCORE",
                    lift_text,
                )

            console.print(table)
            console.print()

        # Name arms that ran but failed, so a missing baseline or score is not read as an arm that never ran.
        coverage_messages = _coverage_notice_messages(agent_eval)
        if coverage_messages:
            console.print("[bold]Agent coverage notes[/bold]")
            for message in coverage_messages:
                console.print(f"  • {escape_markup(message)}", soft_wrap=True)
            console.print()

        from skillevaluator.reporting.tier3_token_usage import VALIDATE_CASE_POINTER, render_token_usage_table

        if render_token_usage_table(console, agent_eval.get("token_usage"), case_pointer=VALIDATE_CASE_POINTER):
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
                console.print(f"  • {escape_markup(message)}", soft_wrap=True)
                evidence = recommendation.get("evidence")
                if isinstance(evidence, dict):
                    url = safe_url(evidence.get("url"))
                    if url:
                        link_text = escape_markup(harbor_evidence_link_text(evidence))
                        console.print(f"    [dim]{link_text}:[/dim] [dim]{escape_markup(url)}[/dim]", soft_wrap=True)
            if printed:
                console.print()

        insights = agent_eval.get("insights", {})
        if any(v.get("score") is not None for v in insights.values()):
            table = Table(
                title="LLM-as-Judge Insights",
                border_style=FAINT,
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
                    score_str = f"[{sc}]{escape_markup(score)}[/{sc}]"
                else:
                    sc = (
                        "green"
                        if score >= DIMENSION_VERDICT_PASS_THRESHOLD
                        else ("yellow" if score >= DIMENSION_VERDICT_NEUTRAL_THRESHOLD else "red")
                    )
                    score_str = f"[{sc}]{score:.2f}[/{sc}]"
                # Truncate before escaping so the cut cannot split an escape sequence.
                table.add_row(
                    escape_markup(str(dim).title()),
                    score_str,
                    escape_markup(str(explanation or "")[:80]),
                )

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
                    console.print(f"  {escape_markup(str(msg))}")
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
                f"{escape_markup(str(detail.check_name))}: "
                f"{escape_markup(str(detail.message))}{escape_markup(meta_str)}"
            )

    def _print_findings(self, result: ValidationResult, console: Console) -> None:
        """Print findings with tree structure."""
        errors = additional_errors(result)
        if errors:
            label = "Execution errors" if result.is_incomplete else "Errors"
            console.print(f"[dim]{label}:[/dim]")
            for error in errors:
                console.print(f"  [red]•[/red] {escape_markup(str(error))}")
        if not result.findings:
            # Preserve legacy warnings when there are no structured findings.
            if result.warnings:
                console.print("[dim]Warnings:[/dim]")
                for warning in result.warnings:
                    console.print(f"  [yellow][WARN][/yellow] {escape_markup(str(warning))}")
            return

        # Warnings no finding repeats, such as why a scan is incomplete.
        warnings = additional_warnings(result)
        if warnings:
            if errors:
                console.print()
            console.print("[dim]Warnings:[/dim]")
            for warning in warnings:
                console.print(f"  [yellow][WARN][/yellow] {escape_markup(str(warning))}")
        if errors or warnings:
            console.print()
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
            f"  {index}. [{severity_color}]{escape_markup(finding.tag)}[/{severity_color}] {escape_markup(finding.message)}"
        )
        console.print(f"     [dim]File:[/dim]    {escape_markup(str(finding.location))}")
        console.print(f"     [dim]Check:[/dim]   {escape_markup(str(finding.check_name))}")
        verification = finding_verification(finding)
        if verification:
            label = f"{verification.verdict} ({verification.confidence} confidence)"
            console.print(f"     [dim]LLM verification:[/dim] {escape_markup(label)}")
            if verification.reasoning:
                console.print(f"     [dim]Reasoning:[/dim] {escape_markup(verification.reasoning)}")
            console.print(f"     [dim]{escape_markup(verification.severity_effect)}[/dim]")
        related_paths = _related_paths(finding)
        if related_paths:
            console.print(f"     [dim]Related paths:[/dim] {escape_markup(' <-> '.join(related_paths))}")

        if finding.line_content:
            content = finding.line_content.strip()
            if len(content) > 70:
                content = content[:67] + "..."
            console.print(f"     [dim]Content:[/dim] [italic]{escape_markup(content)}[/italic]")

        if finding.suggestion:
            label = (
                "Recommendation" if finding.category in ("WHOLE_DUPLICATE", "PARTIAL_OVERLAP", "DUPLICATE") else "Fix"
            )
            console.print(f"     [dim]{label}:[/dim]     {escape_markup(finding.suggestion)}")
        source_errors = source_validation_errors(finding.metadata)
        if source_errors:
            shown, overflow_note = source_validation_preview(source_errors, self._report_formats)
            console.print("     [dim]Source validation:[/dim]")
            for error in shown:
                console.print(f"       {_rich_source_error(error)}")
            if overflow_note:
                console.print(f"       [dim]{escape_markup(overflow_note)}[/dim]")
        console.print()

    def _print_summary_table(
        self, results: list[ValidationResult], console: Console, *, presentation: FinalRenderContext | None = None
    ) -> None:
        """Print summary table for all validators."""
        table = Table(
            title="SkillEvaluator Validation Results",
            show_header=True,
            header_style="bold white on dark_green",
            border_style="green",
            title_style="bold green",
        )
        table.add_column("Validator", style=TEXT)
        table.add_column("Status", justify="center")
        table.add_column("Details")

        for result in results:
            outcome = presentation.result_statuses[id(result)] if presentation else result_status(result).value
            advisory_skip = outcome == "skipped"
            status = {
                "skipped": "[yellow]SKIP[/yellow]",
                "not_applicable": "[dim]N/A[/dim]",
                "incomplete": "[bold yellow]INCOMPLETE[/bold yellow]",
                "passed": "[green]PASS[/green]",
                "pass": "[green]PASS[/green]",
            }.get(outcome, "[red]FAIL[/red]")
            s = result.summary
            static_test_evidence = self._static_test_evidence_message(result)

            if advisory_skip or (outcome == "not_applicable" and is_cleanly_skipped(result)):
                details = _rich_skip_reason(result)
            elif outcome == "not_applicable":
                # Keep the recorded reason; "N/A │ 1 checks passed" contradicts itself.
                details = _rich_line(result.success_details[0].message) if result.success_details else "Not applicable"
            elif outcome == "incomplete":
                scanners = escape_markup(", ".join(result.incomplete_scans))
                details = f"[bold yellow]{scanners} did not complete[/bold yellow]"
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
                    details = escape_markup(static_test_evidence)
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

            agent_eval = result.metadata.get("agent_eval")
            if (
                result.validator_name == "AGENT_EVAL"
                and isinstance(agent_eval, dict)
                and agent_eval.get("verdict") == "fail"
            ):
                raw_summary = agent_eval.get("summary")
                summary = raw_summary if isinstance(raw_summary, dict) else agent_eval
                reason = agent_eval.get("verdict_reason") or summary.get("verdict_reason")
                if isinstance(reason, str) and reason:
                    details = escape_markup(reason)
                    lift = _agent_eval_finite_number(summary.get("overall_lift"))
                    if lift is None:
                        lift = _agent_eval_finite_number(agent_eval.get("composite_lift"))
                    if lift is not None:
                        details += f"; lift [{lift_style(lift)}]{lift:+.2f}[/]"

            if outcome in {"pass", "passed"}:
                quality = result.metadata.get("quality_scores")
                if isinstance(quality, dict):
                    score = _agent_eval_finite_number(quality.get("overall_score"))
                    if score is not None:
                        grade = str(quality.get("grade", "?"))
                        details += f"; quality {escape_markup(grade)} {score:.1f}/100"
                agent = result.metadata.get("agent_eval")
                if isinstance(agent, dict):
                    raw_summary = agent.get("summary")
                    summary = raw_summary if isinstance(raw_summary, dict) else agent
                    lift = _agent_eval_finite_number(summary.get("overall_lift", agent.get("composite_lift")))
                    if (
                        lift is not None
                        and agent.get("verdict") != "inconclusive"
                        and summary.get("verdict") != "inconclusive"
                    ):
                        details += f"; lift [{lift_style(lift)}]{lift:+.2f}[/]"
                    elif agent.get("verdict") == "inconclusive":
                        details += "; inconclusive (skill never activated)"
            table.add_row(escape_markup(str(result.validator_name)), status, details)

        console.print(table)

        for result in results:
            agent_eval = result.metadata.get("agent_eval")
            if isinstance(agent_eval, dict) and (comparison := agent_eval.get("prior_version_comparison")):
                for line in version_comparison_lines(comparison):
                    console.print(escape_markup(line))

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
        return is_advisory_agent_eval_skip(result)
