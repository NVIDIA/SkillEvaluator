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

from io import StringIO
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from skillevaluator.constants import (
    DIMENSION_VERDICT_NEUTRAL_THRESHOLD,
    DIMENSION_VERDICT_PASS_THRESHOLD,
)
from skillevaluator.reporting.base import (
    ReporterBase,
    additional_errors,
    passes_required_gate,
    plugin_catalog_similarity_summary,
)
from skillevaluator.reporting.harbor_viewer import (
    harbor_evidence_link_text,
    normalize_harbor_viewer_for_display,
    safe_url,
)
from skillevaluator.reporting.plugin_sections import (
    NOT_CONFIGURED,
    format_score,
    number,
    static_risk_view,
    tier3_plugin_view,
)
from skillevaluator.utils.rich_markup import escape_markup

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


# Rich styles for the ok / warn / fail status classes of the plugin views.
_STATUS_STYLES = {"ok": "green", "warn": "yellow", "fail": "bold red"}
# How many rows of a plugin list the terminal shows before "... and N more".
_CLI_LIST_ITEMS = 10


def _print_remaining(console: Console, remaining: int) -> None:
    if remaining > 0:
        console.print(f"    [dim]... and {remaining} more[/dim]")


def print_plugin_tier1_static(risk: dict, console: Console) -> None:
    """Print compact Tier 1 plugin static-risk lines: privileges, hooks, CVE audit, parity, endpoints."""
    privileges = risk.get("privileges")
    if privileges:
        console.print(
            f"[bold]Plugin privileges:[/bold] {privileges['agents']} subagent(s), {privileges['commands']} command(s); "
            f"{privileges['flagged']} flagged"
        )
        for row in [row for row in privileges["rows"] if row["risky"]][:10]:
            console.print(
                f"  - {escape_markup(row['type'])} {escape_markup(row['name'])}: {escape_markup(', '.join(row['flags']))}"
            )
    hooks = risk.get("hooks")
    if hooks:
        console.print(f"[bold]Plugin hooks:[/bold] {hooks['total']} handler(s); {hooks['flagged']} flagged")
        for row in [row for row in hooks["rows"] if row["flagged"]][:10]:
            # Escape the brackets with the plugin-controlled matcher: "[/x]" is a
            # closing tag and "[mcp__memory__.*]" a style tag to Rich markup.
            matcher = escape_markup(f"[{row['matcher']}]")
            console.print(
                f"  - {escape_markup(row['event'])} {matcher} {escape_markup(row['handler_type'])}: "
                f"{escape_markup(', '.join(row['flags']))}"
            )
    cve = risk.get("cve")
    if cve:
        parts = [
            f"{row['ecosystem']} {row['status_label']} ({row['audited']} audited, {row['unverified']} unverified, "
            f"{row['severity_label']})"
            for row in cve["rows"]
        ]
        console.print(f"[bold]Dependency CVE audit:[/bold] {escape_markup('; '.join(parts))}")
    parity = risk.get("parity")
    if parity:
        if parity["status"] == "compared":
            console.print(
                f"[bold]claude plugin validate --strict:[/bold] {escape_markup(parity['claude_verdict'])} "
                f"({parity['error_count']} errors, {parity['warning_count']} warnings); SkillEvaluator "
                f"{escape_markup(parity['skillevaluator_verdict'])} ({escape_markup(parity['agreement'])})"
            )
        else:
            console.print(
                f"[bold]claude plugin validate parity:[/bold] {escape_markup(parity['status_label'])}: "
                f"{escape_markup(parity['reason'])}"
            )
    endpoints = risk.get("endpoints")
    if endpoints:
        counts = ", ".join(f"{row['status']}={row['count']}" for row in endpoints["counts"]) or "none"
        redirects = endpoints.get("redirects_flagged")
        flagged = f"; {redirects} flagged redirect(s)" if redirects else ""
        console.print(
            f"[bold]Endpoint DNS/redirect checks:[/bold] {endpoints['total']} endpoint(s) ({escape_markup(counts)})"
            f"{flagged}"
        )
    console.print()


def print_plugin_tier3(view: dict, console: Console) -> None:
    """Print the Tier 3 plugin blocks: completeness, coverage, Integration, statistics, signals."""
    esc = escape_markup
    if view.get("partial"):
        console.print(
            f"  [bold red]INCOMPLETE: {esc(view['incomplete_reason'])}.[/bold red] [red]Partial result, not a pass.[/red]"
        )
    coverage = view.get("coverage")
    if coverage:
        style = _STATUS_STYLES[coverage["status_class"]]
        console.print(
            f"  [{style}]Component coverage: {esc(coverage['headline'])}[/{style}] ({esc(coverage['detail'])})",
            soft_wrap=True,
        )
        console.print(f"    [dim]{esc(coverage['caveat'])}[/dim]")
        shown = coverage["not_staged_rows"][:_CLI_LIST_ITEMS]
        for row in shown:
            reason = f": {esc(row['reason'])}" if row["reason"] else ""
            console.print(f"    [dim]- {esc(row['type'])} {esc(row['name'])} ({esc(row['state_label'])}){reason}[/dim]")
        _print_remaining(console, coverage["not_staged"] - len(shown))
        shown = coverage["staged_not_observed_rows"][:_CLI_LIST_ITEMS]
        for row in shown:
            console.print(
                f"    [dim]- {esc(row['type'])} {esc(row['name'])} (staged, {esc(row['observed'])})[/dim]",
                soft_wrap=True,
            )
        _print_remaining(console, (coverage["staged_not_observed"] or 0) - len(shown))
        activation = coverage.get("activation")
        if activation:
            console.print(f"    [dim]Observed activation (advisory): {esc(activation['summary'])}[/dim]")
    plugin_load = view.get("plugin_load")
    if plugin_load:
        console.print(f"  [bold]Plugin loading:[/bold] requested {esc(plugin_load['requested'])}")
        for row in plugin_load["agents"]:
            census = f"; load census: {row['census_summary']}" if row["census"] else ""
            console.print(
                f"    [dim]- {esc(row['agent'])}: {esc(row['mode'])} ({esc(row['adapter'])}); "
                f"native: {esc(', '.join(row['native']) or 'none')}{esc(census)}[/dim]"
            )
    integration = view.get("integration")
    modes = view.get("lift_modes")
    if modes:
        fallback = " [yellow](fell back)[/yellow]" if modes["fallback"] else ""
        console.print(
            f"  [bold]Lift mode:[/bold] requested {esc(modes['requested'])} · effective {esc(modes['effective'])}{fallback}"
        )
    if integration:
        if integration["measured"]:
            point = (
                f" [dim](point estimate: {esc(integration['point_verdict_label'])})[/dim]"
                if integration["point_verdict_label"]
                else ""
            )
            console.print(
                f"  [bold]Integration:[/bold] {esc(integration['verdict_label']).upper()} "
                f"(lift {integration['integration_lift']}){point} [dim](advisory)[/dim]"
            )
            ci = integration["ci"]
            if ci:
                console.print(
                    f"    [dim]{esc(ci['confidence'])} {esc(ci['interval'])}, precision {esc(ci['precision'])}[/dim]"
                )
            if integration["reason"]:
                console.print(f"    [dim]{esc(integration['reason'])}[/dim]", soft_wrap=True)
            if integration["components"]:
                more = f" (+{integration['components_omitted']} more)" if integration["components_omitted"] else ""
                console.print(f"    [dim]components: {esc(', '.join(integration['components']))}{more}[/dim]")
            if integration["interpretation"]:
                console.print(f"    [dim]{esc(integration['interpretation'])}[/dim]", soft_wrap=True)
        else:
            console.print(
                f"  [bold]Integration:[/bold] [yellow]INCONCLUSIVE[/yellow] — {esc(integration['reason'])} "
                "[dim](advisory)[/dim]",
                soft_wrap=True,
            )
    statistics = view.get("statistics")
    if statistics:
        for scope in statistics["scopes"]:
            _print_plugin_statistics_scope(scope, console, show_label=len(statistics["scopes"]) > 1)
    signals = view.get("signals")
    if signals:
        console.print(
            "  [bold]Plugin signals[/bold] [dim](advisory, report-only; never changes a score or verdict)[/dim]"
        )
        for entry in signals["entries"]:
            missing = f"; {entry['n_missing_trajectory']} without a trajectory" if entry["n_missing_trajectory"] else ""
            console.print(
                f"    [bold]{esc(entry['scope'])} · {esc(entry['arm_label'])}[/bold] ({entry['n_trials']} trials{missing})"
            )
            details = []
            tool = entry["tool_selection"]
            if tool:
                details.append(
                    f"tool selection P/R/F1 {tool['precision']} / {tool['recall']} / {tool['f1']}, "
                    f"{tool['decoy_calls']} decoy call(s)"
                    if tool["applicable"]
                    else f"tool selection {NOT_CONFIGURED}"
                )
            arguments = entry["arguments"]
            if arguments:
                details.append(
                    f"arguments {arguments['pass_rate']} ({arguments['passed']}/{arguments['checked']})"
                    if arguments["applicable"]
                    else f"arguments {NOT_CONFIGURED}"
                )
            mcp = entry["mcp_calls"]
            if mcp:
                details.append(f"MCP calls {mcp['success_rate']} succeeded ({mcp['succeeded']}/{mcp['total']})")
            details.extend(f"{check['name'].lower()} {check['label']}" for check in entry["checks"])
            activation = entry["activation"]
            if activation:
                details.append(f"activation {len(activation['exercised'])}/{len(activation['declared'])} exercised")
            for detail in details:
                console.print(f"      [dim]- {esc(detail)}[/dim]", soft_wrap=True)
    print_plugin_runtime_evidence(view, console)
    console.print()


def print_plugin_runtime_evidence(view: dict, console: Console) -> None:
    """Print the canary, hook census, and MCP proof blocks of a Tier 3 plugin view."""
    esc = escape_markup
    canary = view.get("canary")
    if canary:
        for entry in canary["entries"]:
            style = _STATUS_STYLES.get(entry["verdict_class"], "yellow")
            console.print(
                f"  [bold]Canary exfiltration ({esc(entry['scope'])}):[/bold] [{style}]{esc(entry['verdict'])}[/{style}]"
            )
            for row in entry["rows"]:
                console.print(
                    f"    [dim]{esc(row['arm_label'])}: {row['leaked']} of {esc(str(row['trials']))} trial(s) leaked,"
                    f" decoy planted in {esc(str(row['planted']))} (sinks: {esc(row['sinks'])})[/dim]"
                )
    hook_census = view.get("hook_census")
    if hook_census:
        console.print("  [bold]Hook census[/bold] [dim](advisory)[/dim]")
        for entry in hook_census["entries"]:
            console.print(
                f"    [bold]{esc(entry['scope'])} · {esc(entry['arm_label'])}[/bold]: {esc(entry['summary'])}"
            )
            for row in entry["rows"][:10]:
                console.print(
                    f"      [dim]- {esc(row['hook_id'])} ({esc(row['event'])}): {row['runs']} run(s), "
                    f"{row['failures']} failure(s)[/dim]"
                )
    mcp_proof = view.get("mcp_proof")
    if mcp_proof:
        console.print(f"  [bold]MCP proof:[/bold] {esc(mcp_proof['headline'])} [dim](advisory)[/dim]")
        for row in mcp_proof["rows"]:
            console.print(
                f"    [dim]- {esc(row['server'])}: {esc(row['status_label'])} — {esc(row['detail'])}[/dim]",
                soft_wrap=True,
            )


def _print_plugin_statistics_scope(scope: dict, console: Console, *, show_label: bool) -> None:
    esc = escape_markup
    label = f" — {esc(scope['label'])}" if show_label else ""
    if scope["lift_ci"]:
        table = Table(title=f"Lift Uncertainty (advisory){label}", border_style="cyan", show_header=True)
        table.add_column("Lift", style="bold")
        table.add_column("Estimate", justify="right")
        table.add_column("Interval")
        table.add_column("Precision")
        table.add_column("Cases", justify="right")
        for row in scope["lift_ci"]:
            warning = " [yellow](CI includes zero)[/yellow]" if row["ci_includes_zero"] else ""
            table.add_row(
                esc(row["label"]),
                row["estimate"],
                f"{esc(row['interval'])} {esc(row['confidence'])}{warning}",
                esc(row["precision"]),
                row["n_cases"],
            )
        console.print(table)
    if scope["arms"] and (scope["has_reliability"] or scope["has_tokens"] or scope["has_efficiency"]):
        table = Table(title=f"Reliability and Cost (advisory){label}", border_style="cyan", show_header=True)
        table.add_column("Arm", style="bold")
        if scope["has_reliability"]:
            table.add_column("pass@k", justify="right")
            table.add_column("pass^k", justify="right")
            table.add_column("k", justify="right")
        if scope["has_tokens"]:
            table.add_column("Tokens/success", justify="right")
        if scope["show_usd"]:
            table.add_column("USD/success", justify="right")
        if scope["has_efficiency"]:
            table.add_column("Token eff.", justify="right")
        for arm in scope["arms"]:
            row = [esc(arm["label"])]
            if scope["has_reliability"]:
                row.extend([arm["pass_at_k"], arm["pass_hat_k"], arm["k"]])
            if scope["has_tokens"]:
                row.append(arm["tokens_per_success"])
            if scope["show_usd"]:
                row.append(arm["usd_per_success"])
            if scope["has_efficiency"]:
                row.append(arm["token_efficiency"])
            table.add_row(*row)
        console.print(table)
        if scope["usd_note"]:
            console.print(f"  [dim]{esc(scope['usd_note'])}[/dim]", soft_wrap=True)
    measured = scope["context_measured"]
    if measured:
        if measured["measured"]:
            console.print(
                f"  [bold]Measured context delta:[/bold] {esc(measured['delta'])} per first turn "
                f"({measured['n_pairs']} pairs)"
            )
        else:
            reason = f": {esc(measured['reason'])}" if measured["reason"] else ""
            console.print(f"  [bold]Measured context delta:[/bold] not measured ({esc(measured['status'])}){reason}")
    completeness = scope["completeness"]
    if completeness and completeness["issues"]:
        parts = []
        if completeness["missing_cases"]:
            parts.append(f"missing cases: {', '.join(completeness['missing_cases'])}")
        if completeness["failed_arms"]:
            parts.append(f"failed arms: {', '.join(completeness['failed_arms'])}")
        if completeness["attempt_shortfall"]:
            parts.append(f"{len(completeness['attempt_shortfall'])} case(s) with attempt shortfall")
        detail = "; ".join(parts) or "the compared arms did not score the same cases"
        console.print(f"  [yellow]Integration completeness issues:[/yellow] {esc(detail)}", soft_wrap=True)


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
        # Print summary table
        self._print_summary_table(results, console)
        console.print()
        plugin_block = self._plugin_block_from_results(results)
        static_risk = static_risk_view(plugin_block) if plugin_block is not None else None
        if static_risk is not None:
            print_plugin_tier1_static(static_risk, console)

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

        # A passing AGENT_EVAL is advisory, so it has no details above. A plugin
        # run still prints its compact Tier 3 block: the verdict can be FAIL
        # while the result passes, and loading, canary, census and Integration
        # are only shown here.
        detailed = {id(result) for result in [*failed, *non_blocking]}
        for result in results:
            if id(result) in detailed:
                continue
            agent_eval = result.metadata.get("agent_eval") if result.metadata else None
            plugin_view = tier3_plugin_view(agent_eval) if isinstance(agent_eval, dict) else None
            if plugin_view is None:
                continue
            console.print(f"\n[bold]{escape_markup(f'[{result.validator_name}]')}[/bold] Tier 3 plugin evaluation")
            self._print_agent_eval_verdict(agent_eval, console)
            print_plugin_tier3(plugin_view, console)

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
        console.print(f"\n[bold]{escape_markup(f'[{result.validator_name}]')}[/bold]")
        if result.validator_description:
            console.print(f"[dim]{escape_markup(str(result.validator_description))}[/dim]")

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
            console.print(f"[yellow][INCOMPLETE] {escape_markup(tools)} did not complete[/yellow]\n")
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
        avg_grade_text = escape_markup(str(avg_grade))

        if multi:
            console.print(f"\n  [{gc}]Average: {avg_score:.1f}/100 (Grade: {avg_grade_text})[/{gc}]")
            skill_count = escape_markup(str(qs.get("skill_count", len(skills_list))))
            console.print(f"  [dim]Skills analyzed:[/dim] {skill_count}\n")
        else:
            stype = escape_markup(str(skills_list[0].get("skill_type", "unknown")))
            console.print(f"\n  [{gc}]Overall: {avg_score:.1f}/100 (Grade: {avg_grade_text})[/{gc}]")
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
                escape_markup(str(check.get("id", "?")).replace("_", " ").title()),
                f"[{cc}]{cs}/10[/{cc}]",
                passed,
                escape_markup(str(check.get("notes") or "")),
            )

        console.print(table)
        console.print()

    @staticmethod
    def _print_agent_eval_tables(agent_eval: dict, console: Console) -> None:
        """Print Tier 3 agent evaluation results."""
        CLIReporter._print_agent_eval_verdict(agent_eval, console)

        evaluators = agent_eval.get("evaluators", {})
        if evaluators:
            CLIReporter._print_evaluator_table(evaluators, console)

        plugin_view = tier3_plugin_view(agent_eval)
        if plugin_view is not None:
            print_plugin_tier3(plugin_view, console)

        CLIReporter._print_agent_eval_insights(agent_eval, console)

    @staticmethod
    def _print_agent_eval_verdict(agent_eval: dict, console: Console) -> None:
        """Print the Tier 3 verdict line, runtime, and Harbor links."""
        verdict = agent_eval.get("verdict", "unknown")
        composite = agent_eval.get("composite_lift")
        runtime = agent_eval.get("runtime_seconds", 0.0)

        vc = "green" if verdict == "pass" else ("red" if verdict == "fail" else "yellow")
        composite_text = f"{composite:+.2f}" if isinstance(composite, int | float) else "N/A"
        verdict_text = escape_markup(str(verdict).upper())
        console.print(f"\n  [{vc}]Verdict: {verdict_text} (composite lift = {composite_text})[/{vc}]")
        if runtime:
            console.print(f"  [dim]Runtime: {runtime:.1f}s[/dim]")
        harbor_viewer = normalize_harbor_viewer_for_display(agent_eval)
        if harbor_viewer.get("job_url") or harbor_viewer.get("analysis_url"):
            console.print("  [dim]Harbor artifacts:[/dim]")
            if harbor_viewer.get("job_url"):
                console.print(
                    f"    [dim]Harbor logs:[/dim] [cyan]{escape_markup(harbor_viewer['job_url'])}[/cyan]",
                    soft_wrap=True,
                )
            if harbor_viewer.get("analysis_url"):
                console.print(
                    f"    [dim]Harbor analysis:[/dim] [cyan]{escape_markup(harbor_viewer['analysis_url'])}[/cyan]",
                    soft_wrap=True,
                )
        console.print()

    @staticmethod
    def _print_evaluator_table(evaluators: dict, console: Console) -> None:
        """Print the per-evaluator with-skill, baseline, and lift table."""
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
                scores = scores if isinstance(scores, dict) else {}
                lift = number(scores.get("lift"))
                # A missing lift (no baseline score) is neutral, not a regression or a gain.
                lc = "dim" if lift is None else ("green" if lift > 0.01 else ("red" if lift < -0.01 else "dim"))
                table.add_row(
                    escape_markup(str(name).replace("_", " ").title()),
                    format_score(scores.get("with_skill"), ".2f"),
                    format_score(scores.get("baseline"), ".2f"),
                    f"[{lc}]{format_score(lift, '+.2f')}[/{lc}]",
                )

            console.print(table)
            console.print()

    @staticmethod
    def _print_agent_eval_insights(agent_eval: dict, console: Console) -> None:
        """Print Tier 3 recommendations and LLM-as-Judge insights."""
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
                        console.print(f"    [dim]{link_text}:[/dim] [cyan]{escape_markup(url)}[/cyan]", soft_wrap=True)
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

        if errors:
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
                details = escape_markup(str(provenance.get("message") or "Live evaluation did not run"))
            elif result.is_incomplete:
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
                catalog_summary = plugin_catalog_similarity_summary(result)
                if catalog_summary:
                    details = escape_markup(catalog_summary)
                elif result.metadata.get("skipped"):
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

            table.add_row(escape_markup(str(result.validator_name)), status, details)

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
