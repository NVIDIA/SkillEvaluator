# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Final operator-facing Tier 3 score and failure summaries."""

from __future__ import annotations

import io
import logging
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rich.box import SIMPLE
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from skillevaluator.tier3.harbor.metrics import (
    CUSTOM_ONLY_METRIC_SET,
    DEFAULT_METRIC_SET,
    DEFAULT_METRICS,
    DIMENSION_DISPLAY,
    METRIC_DISPLAY,
)
from skillevaluator.tier3.harbor.progress import redact_progress_detail, secret_values_from_environment
from skillevaluator.tier3.harbor.runner import format_harbor_view_command


def _number(value: object, *, signed: bool = False) -> str:
    numeric = _finite_number(value)
    if numeric is None:
        return "-"
    return f"{numeric:+.3f}" if signed else f"{numeric:.3f}"


def _finite_number(value: object) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _score_style(score: float) -> str:
    if score >= 0.8:
        return "green"
    if score >= 0.6:
        return "yellow"
    return "red"


def _score_bar(score: float) -> str:
    clamped = max(0.0, min(1.0, score))
    filled = round(clamped * 10)
    return "█" * filled + "░" * (10 - filled)


def _score_cell(value: object, *, unavailable: str = "NO SCORE") -> tuple[Text, Text]:
    numeric = _finite_number(value)
    if numeric is None:
        return Text(unavailable, style="dim"), Text("")
    style = _score_style(numeric)
    return Text(f"{numeric:.2f}", style=f"bold {style}"), Text(_score_bar(numeric), style=style)


def _delta_cell(value: object) -> Text:
    numeric = _finite_number(value)
    if numeric is None:
        return Text("NO SCORE", style="dim")
    style = "green" if numeric > 0 else "red" if numeric < 0 else "dim"
    return Text(f"{numeric:+.2f}", style=f"bold {style}")


def _rate(data: object) -> str:
    return _number(data.get("rate") if isinstance(data, Mapping) else None)


def _condition_status(data: Mapping[str, Any], variant: str) -> str:
    conditions = data.get("conditions")
    condition = conditions.get(variant) if isinstance(conditions, Mapping) else None
    if not isinstance(condition, Mapping):
        return ""
    return str(condition.get("execution_status") or condition.get("status") or "")


def _condition_usable(data: Mapping[str, Any], variant: str) -> bool:
    status = _condition_status(data, variant)
    if status:
        return status in {"succeeded", "complete"}
    return data.get("execution_status") == "succeeded"


def _default_with_skill_overall(data: Mapping[str, Any]) -> float | None:
    scores = data.get("with_skill")
    if not isinstance(scores, Mapping):
        return None
    values: list[float] = []
    for metric in DEFAULT_METRICS:
        value = _finite_number(scores.get(metric))
        if value is None:
            return None
        values.append(value)
    return round(sum(values) / len(values), 4)


def _custom_only_with_skill_overall(data: Mapping[str, Any]) -> float | None:
    pass_at_k = data.get("pass_at_k")
    with_skill = pass_at_k.get("with_skill") if isinstance(pass_at_k, Mapping) else None
    cases = with_skill.get("cases") if isinstance(with_skill, Mapping) else None
    if not isinstance(cases, Mapping) or not cases:
        return None

    scores: list[float] = []
    for case in cases.values():
        attempts = case.get("attempts") if isinstance(case, Mapping) else None
        if not isinstance(attempts, list) or not attempts:
            return None
        for attempt in attempts:
            score = _finite_number(attempt.get("score")) if isinstance(attempt, Mapping) else None
            if score is None:
                return None
            scores.append(score)

    attempts_used = with_skill.get("attempts_used")
    if not isinstance(attempts_used, int) or isinstance(attempts_used, bool) or attempts_used != len(scores):
        return None
    return round(sum(scores) / len(scores), 4)


def _with_skill_overall(data: Mapping[str, Any], metric_set: object) -> float | None:
    if data.get("execution_status") != "succeeded":
        return None

    if _condition_status(data, "without_skill") == "skipped":
        if metric_set == DEFAULT_METRIC_SET:
            return _default_with_skill_overall(data)
        if metric_set == CUSTOM_ONLY_METRIC_SET:
            return _custom_only_with_skill_overall(data)
        return None

    lift = data.get("lift")
    overall = lift.get("overall") if isinstance(lift, Mapping) else None
    persisted = overall.get("with_skill") if isinstance(overall, Mapping) else None
    return _finite_number(persisted)


def _unavailable_or_skipped(value: object, *, baseline_skipped: bool) -> str:
    return "skipped" if baseline_skipped else _number(value)


def _condition_failures(agent: str, data: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    conditions = data.get("conditions")
    if not isinstance(conditions, Mapping):
        return failures
    for variant, raw in conditions.items():
        if not isinstance(raw, Mapping):
            continue
        status = str(raw.get("execution_status") or raw.get("status") or "unknown")
        if status in {"succeeded", "skipped", "complete"}:
            continue
        details = raw.get("execution_errors") or raw.get("detail") or "condition did not complete"
        if isinstance(details, list):
            details = "; ".join(str(item) for item in details if str(item).strip())
        label = str(variant).replace("_", "-")
        failures.append(f"{agent} {label} ({status}): {details}")
    return failures


def _trial_failures(agent: str, data: Mapping[str, Any]) -> list[str]:
    rendered: list[str] = []
    conditions = data.get("trial_failures")
    if not isinstance(conditions, Mapping):
        return rendered
    for variant, failures in conditions.items():
        if not isinstance(failures, list):
            continue
        label = str(variant).replace("_", "-")
        for failure in failures:
            if not isinstance(failure, Mapping):
                continue
            trial = str(failure.get("trial") or "unknown trial")
            reason = str(failure.get("reason") or "trial did not complete")
            rendered.append(f"{agent} {label} {trial}: {reason}")
    return rendered


def _display_metrics(result: Mapping[str, Any], agents: Mapping[str, Any]) -> list[str]:
    configured = result.get("metrics")
    metrics: list[str] = []
    if isinstance(configured, list | tuple):
        metrics.extend(str(metric) for metric in configured if str(metric).strip())
    if not metrics:
        present: set[str] = set()
        for raw in agents.values():
            if not isinstance(raw, Mapping):
                continue
            for key in ("with_skill", "without_skill"):
                values = raw.get(key)
                if isinstance(values, Mapping):
                    present.update(str(metric) for metric in values)
        metrics.extend(metric for metric in DEFAULT_METRICS if metric in present)
        metrics.extend(sorted(present - set(metrics)))
    return list(dict.fromkeys(metrics))


def _resolved_agent_model(result: Mapping[str, Any], agent: str, data: Mapping[str, Any]) -> str:
    run_config = result.get("run_config")
    configured_agents = run_config.get("agents") if isinstance(run_config, Mapping) else None
    configured = configured_agents.get(agent) if isinstance(configured_agents, Mapping) else None
    configured_model = configured.get("model") if isinstance(configured, Mapping) else None
    model = configured_model or data.get("model")
    return str(model).strip() if model else ""


def _resolved_skill_name(result: Mapping[str, Any]) -> str:
    explicit = str(result.get("skill_name") or "").strip()
    if explicit:
        return explicit
    run_dir_value = result.get("run_dir") or result.get("output_dir")
    if run_dir_value:
        run_dir = Path(str(run_dir_value))
        if run_dir.name and run_dir.parent.name:
            return run_dir.parent.name
    return "Skill Evaluation"


def _render_agent_scores(
    *,
    console: Console,
    result: Mapping[str, Any],
    agent: str,
    data: Mapping[str, Any],
    metrics: list[str],
    safe: Any,
) -> None:
    with_usable = _condition_usable(data, "with_skill")
    baseline_status = _condition_status(data, "without_skill")
    baseline_skipped = baseline_status == "skipped"
    baseline_usable = _condition_usable(data, "without_skill") and not baseline_skipped
    with_scores = data.get("with_skill") if isinstance(data.get("with_skill"), Mapping) else {}
    baseline_scores = data.get("without_skill") if isinstance(data.get("without_skill"), Mapping) else {}
    lift = data.get("lift") if isinstance(data.get("lift"), Mapping) else {}
    custom_with = data.get("custom_with_skill") if isinstance(data.get("custom_with_skill"), Mapping) else {}
    custom_without = data.get("custom_without_skill") if isinstance(data.get("custom_without_skill"), Mapping) else {}
    custom_lift = data.get("custom_lift") if isinstance(data.get("custom_lift"), Mapping) else {}
    show_baseline = not baseline_skipped and bool(baseline_status or baseline_scores or lift)

    table = Table(
        show_header=True,
        header_style="bold dim",
        box=SIMPLE,
        padding=(0, 1),
        show_edge=False,
        expand=True,
    )
    table.add_column("Evaluator", style="white", min_width=14, ratio=2)
    table.add_column("With Skill", justify="right", no_wrap=True, width=10)
    table.add_column("", no_wrap=True, width=10)
    if show_baseline:
        table.add_column("No Skill", justify="right", no_wrap=True, width=9)
        table.add_column("", no_wrap=True, width=10)
        table.add_column("Lift", justify="right", no_wrap=True, width=8)

    for metric in metrics:
        with_value = with_scores.get(metric) if with_usable else None
        baseline_value = baseline_scores.get(metric) if baseline_usable else None
        label = Text(METRIC_DISPLAY.get(metric, metric.replace("_", " ").title()), style="bold")
        with_score, with_bar = _score_cell(with_value)
        row: list[Text] = [label, with_score, with_bar]
        if show_baseline:
            persisted = lift.get(metric) if isinstance(lift.get(metric), Mapping) else {}
            delta = (
                persisted.get("delta")
                if _finite_number(with_value) is not None and _finite_number(baseline_value) is not None
                else None
            )
            baseline_score, baseline_bar = _score_cell(baseline_value)
            row.extend([baseline_score, baseline_bar, _delta_cell(delta)])
        table.add_row(*row)

    if custom_with or custom_without:
        if metrics:
            table.add_section()
        custom_metrics = sorted({str(name) for name in custom_with} | {str(name) for name in custom_without})
        for metric in custom_metrics:
            with_value = custom_with.get(metric) if with_usable else None
            baseline_value = custom_without.get(metric) if baseline_usable else None
            with_score, with_bar = _score_cell(with_value)
            row = [Text(f"custom: {safe(metric)}", style="cyan"), with_score, with_bar]
            if show_baseline:
                persisted = custom_lift.get(metric) if isinstance(custom_lift.get(metric), Mapping) else {}
                delta = (
                    persisted.get("delta")
                    if _finite_number(with_value) is not None and _finite_number(baseline_value) is not None
                    else None
                )
                baseline_score, baseline_bar = _score_cell(baseline_value)
                row.extend([baseline_score, baseline_bar, _delta_cell(delta)])
            table.add_row(*row)

    if not metrics and not custom_with and not custom_without and not show_baseline:
        with_overall = _with_skill_overall(data, result.get("metric_set"))
        score, bar = _score_cell(with_overall)
        table.add_row(Text("Overall", style="bold"), score, bar)

    overall = lift.get("overall") if isinstance(lift.get("overall"), Mapping) else {}
    with_overall = overall.get("with_skill") if with_usable else None
    baseline_overall = overall.get("without_skill") if baseline_usable else None
    if show_baseline and (_finite_number(with_overall) is not None or _finite_number(baseline_overall) is not None):
        table.add_section()
        with_score, with_bar = _score_cell(with_overall)
        baseline_score, baseline_bar = _score_cell(baseline_overall)
        delta = (
            overall.get("delta")
            if _finite_number(with_overall) is not None and _finite_number(baseline_overall) is not None
            else None
        )
        table.add_row(
            Text("Skill Lift", style="bold"),
            with_score,
            with_bar,
            baseline_score,
            baseline_bar,
            _delta_cell(delta),
        )

    if not table.rows:
        with_score, with_bar = _score_cell(None)
        row = [Text("Overall", style="bold"), with_score, with_bar]
        if show_baseline:
            baseline_score, baseline_bar = _score_cell(None)
            row.extend([baseline_score, baseline_bar, _delta_cell(None)])
        table.add_row(*row)

    model = _resolved_agent_model(result, agent, data)
    skill = _resolved_skill_name(result)
    title = Text("Results by Evaluator", style="bold cyan")
    subtitle = Text()
    subtitle.append(" / ".join(part for part in (safe(skill), safe(agent)) if part), style="dim")
    trials = data.get("num_trials_with")
    if isinstance(trials, int) and not isinstance(trials, bool):
        subtitle.append("\n")
        subtitle.append(f"{trials} trial(s)", style="dim")
    run_config = result.get("run_config")
    configured_agents = run_config.get("agents") if isinstance(run_config, Mapping) else None
    configured = configured_agents.get(agent) if isinstance(configured_agents, Mapping) else None
    model_source = configured.get("source") if isinstance(configured, Mapping) else data.get("model_source")
    if model:
        if subtitle.plain:
            subtitle.append("\n")
        subtitle.append(f"Model: {safe(model)}", style="dim")
        if model_source:
            subtitle.append(f" ({safe(model_source)})", style="dim")
    agent_output = data.get("output_dir")
    if not agent_output and result.get("run_dir"):
        agent_output = Path(str(result["run_dir"])) / agent
    if agent_output:
        if subtitle.plain:
            subtitle.append("\n")
        subtitle.append(safe(agent_output), style="dim")
    console.print(
        Panel(
            table,
            title=title,
            subtitle=subtitle if subtitle.plain else None,
            border_style="cyan",
            padding=(1, 1),
        )
    )


def _render_dimensions(
    *,
    console: Console,
    agents: Mapping[str, Any],
    safe: Any,
) -> None:
    table = Table(
        show_header=True,
        header_style="bold dim",
        box=SIMPLE,
        padding=(0, 0),
        show_edge=False,
        expand=True,
    )
    table.add_column("Dimension", style="white", min_width=15, ratio=2, no_wrap=True)
    table.add_column("With Skill", justify="right", no_wrap=True, width=10)
    table.add_column("", no_wrap=True, width=10)
    table.add_column("No Skill", justify="right", no_wrap=True, width=9)
    table.add_column("", no_wrap=True, width=10)
    table.add_column("Lift", justify="right", no_wrap=True, width=8)

    for agent_index, (agent, raw) in enumerate(agents.items()):
        if agent_index:
            table.add_section()
        if len(agents) > 1:
            table.add_row(Text(f"Agent: {safe(agent)}", style="bold cyan"), "", "", "", "", "")
        data = raw if isinstance(raw, Mapping) else {}
        for dimension in DIMENSION_DISPLAY:
            with_dimensions = data.get("dimensions_with_skill")
            baseline_dimensions = data.get("dimensions_without_skill")
            with_dimension = with_dimensions.get(dimension) if isinstance(with_dimensions, Mapping) else None
            baseline_dimension = (
                baseline_dimensions.get(dimension) if isinstance(baseline_dimensions, Mapping) else None
            )
            with_score = (
                with_dimension.get("score")
                if _condition_usable(data, "with_skill") and isinstance(with_dimension, Mapping)
                else None
            )
            baseline_score = (
                baseline_dimension.get("score")
                if _condition_usable(data, "without_skill") and isinstance(baseline_dimension, Mapping)
                else None
            )
            with_numeric = _finite_number(with_score)
            baseline_numeric = _finite_number(baseline_score)
            with_cell, with_bar = _score_cell(with_numeric)
            baseline_skipped = _condition_status(data, "without_skill") == "skipped"
            if baseline_skipped:
                baseline_cell, baseline_bar = Text("skipped", style="dim"), Text("")
            else:
                baseline_cell, baseline_bar = _score_cell(baseline_numeric)
            delta = (
                with_numeric - baseline_numeric if with_numeric is not None and baseline_numeric is not None else None
            )
            row = [
                Text(DIMENSION_DISPLAY.get(dimension, dimension.title()), style="bold"),
                with_cell,
                with_bar,
                baseline_cell,
                baseline_bar,
                _delta_cell(delta),
            ]
            table.add_row(*row)

    caption = (
        "Each row shows with-skill, no-skill, and dimension lift. "
        "Dimensions are report-only rollups; custom metrics do not change them."
    )
    table.caption = caption
    console.print(
        Panel(
            table,
            title="[bold]Results by Dimension[/bold]",
            border_style="cyan",
            padding=(1, 0),
        )
    )


def _insight_message(item: object) -> str:
    if isinstance(item, Mapping):
        title = str(item.get("title") or "").strip()
        message = str(item.get("message") or item.get("suggestion") or "").strip()
        if title and message and title.casefold() not in message.casefold():
            return f"{title}: {message}"
        return message or title
    return str(item).strip()


def _insight_key(value: object) -> str:
    return " ".join(str(value).casefold().split())


def _insight_rows(items: object, *, excluded_messages: set[str] | None = None) -> list[tuple[str, str]]:
    if not isinstance(items, list):
        return []
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    excluded = {_insight_key(message) for message in excluded_messages or set()}
    excluded.discard("")
    for item in items:
        message = _insight_message(item)
        message_key = _insight_key(message)
        candidates = {message_key}
        if isinstance(item, Mapping):
            raw_message = str(item.get("message") or item.get("suggestion") or "").strip()
            if raw_message:
                candidates.add(_insight_key(raw_message))
        if not message_key or message_key in seen or candidates.intersection(excluded):
            continue
        severity = str(item.get("severity") or "") if isinstance(item, Mapping) else ""
        rows.append((severity, message))
        seen.add(message_key)
    return rows


def _render_feedback_and_suggestions(
    *,
    console: Console,
    result: Mapping[str, Any],
    safe: Any,
    excluded_messages: set[str] | None = None,
) -> None:
    agent_eval = result.get("tier3_feedback")
    if not isinstance(agent_eval, Mapping):
        # Backward compatibility for results produced by the first unified
        # reporting implementation.
        agent_eval = result.get("agent_eval")
    if not isinstance(agent_eval, Mapping):
        return

    feedback = _insight_rows(agent_eval.get("conclusions"), excluded_messages=excluded_messages)
    suggestions = _insight_rows(agent_eval.get("recommendations"), excluded_messages=excluded_messages)
    if not suggestions:
        suggestions = _insight_rows(agent_eval.get("suggestions_v2"), excluded_messages=excluded_messages)
    if not suggestions:
        suggestions = _insight_rows(agent_eval.get("suggestions"), excluded_messages=excluded_messages)
    if not feedback and not suggestions:
        return

    body = Text()
    has_html_report = bool(result.get("report_path"))
    if feedback:
        body.append("Feedback\n", style="bold")
        for severity, message in feedback[:3]:
            icon, style = {
                "fail": ("✗", "red"),
                "warn": ("⚠", "yellow"),
                "pass": ("✓", "green"),
            }.get(severity, ("•", "dim"))
            body.append(f"  {icon} ", style=style)
            body.append(f"{safe(message)}\n")
        if len(feedback) > 3:
            location = "in the HTML report" if has_html_report else "not shown"
            body.append(f"  … {len(feedback) - 3} more feedback item(s) {location}.\n", style="dim")

    if suggestions:
        if feedback:
            body.append("\n")
        body.append("Suggestions\n", style="bold")
        for index, (_severity, message) in enumerate(suggestions[:5], start=1):
            body.append(f"  {index}. ", style="cyan")
            body.append(f"{safe(message)}\n")
        if len(suggestions) > 5:
            location = "in the HTML report" if has_html_report else "not shown"
            body.append(f"  … {len(suggestions) - 5} more suggestion(s) {location}.\n", style="dim")

    console.print(
        Panel(
            body,
            title=Text("Feedback & Suggestions", style="bold cyan"),
            border_style="cyan",
            padding=(1, 1),
        )
    )


def render_evaluation_result(result: Mapping[str, Any], *, console: Console) -> None:
    """Render persisted engine truth, aggregating only canonical score components."""
    secret_values = secret_values_from_environment(os.environ)

    def safe(value: object) -> str:
        return redact_progress_detail(value, secret_values=secret_values)

    rendered_feedback_messages: set[str] = set()
    status = str(result.get("execution_status") or "unknown")
    warnings = result.get("warnings") if isinstance(result.get("warnings"), list) else []
    display_status = "degraded" if status == "succeeded" and warnings else status
    duration = result.get("duration_seconds")
    if _finite_number(duration) is not None:
        console.print(f"Time: {float(duration):.1f}s")

    agents = result.get("agents")
    if isinstance(agents, Mapping) and agents:
        metrics = _display_metrics(result, agents)
        for agent, raw in agents.items():
            data = raw if isinstance(raw, Mapping) else {}
            _render_agent_scores(
                console=console,
                result=result,
                agent=str(agent),
                data=data,
                metrics=metrics,
                safe=safe,
            )
        _render_dimensions(console=console, agents=agents, safe=safe)

        # The per-evaluator findings report — evaluator reasonings, evidence
        # pointers, and next-step suggestions — is the feedback surface that
        # follows the score tables.
        run_dir = result.get("run_dir") or result.get("output_dir")
        try:
            from skillevaluator.tier3.harbor.report import display_findings_report

            rendered_feedback_messages = (
                display_findings_report(
                    dict(result),
                    str(result.get("skill_name") or ""),
                    [str(agent) for agent in agents],
                    Path(str(run_dir)) if run_dir else Path(),
                )
                or set()
            )
        except Exception:  # advisory panel: never break the run summary
            logging.getLogger(__name__).debug("Findings report skipped", exc_info=True)

    errors = result.get("execution_errors") or result.get("error") or []
    if isinstance(errors, str):
        errors = [errors]
    rendered_errors = [str(error) for error in errors if str(error).strip()] if isinstance(errors, list) else []
    if isinstance(agents, Mapping):
        for agent, raw in agents.items():
            if isinstance(raw, Mapping):
                agent_errors = [*_condition_failures(str(agent), raw), *_trial_failures(str(agent), raw)]
                rendered_errors.extend(agent_errors)
                if raw.get("execution_status") not in {None, "succeeded", "complete", "skipped"} and not agent_errors:
                    rendered_errors.append(f"{agent} evaluation failed without diagnostic details")
    if rendered_errors or warnings:
        status_style = "bold red" if rendered_errors else "bold yellow"
        console.print(Text(f"Tier 3 Evaluation: {display_status.upper()}", style=status_style))
        findings = Text()
        if rendered_errors:
            findings.append("Failures\n", style="bold red")
            for error in dict.fromkeys(rendered_errors):
                findings.append(f"  - {safe(error)}\n")
        if warnings:
            findings.append("Warnings\n", style="bold yellow")
            for warning in warnings:
                findings.append(f"  - {safe(warning)}\n")
        console.print(
            Panel(
                findings,
                title="[bold]Findings[/bold]",
                border_style="red" if rendered_errors else "yellow",
                padding=(1, 1),
            )
        )

    # Keep payload-only insights while filtering exact items already printed by
    # the detailed per-evaluator report.
    _render_feedback_and_suggestions(
        console=console,
        result=result,
        safe=safe,
        excluded_messages=rendered_feedback_messages,
    )

    artifact_rows: list[tuple[str, str]] = []
    report_path = result.get("report_path")
    if report_path:
        artifact_rows.append(("📊 HTML report", safe(report_path)))
    output_dir = result.get("run_dir") or result.get("output_dir")
    if output_dir:
        artifact_rows.append(("📁 Output", safe(output_dir)))
    jobs_dir = result.get("harbor_jobs_dir")
    if jobs_dir and result.get("harbor_jobs_retained") and Path(str(jobs_dir)).is_dir():
        artifact_rows.append(("🔍 Inspect jobs", safe(format_harbor_view_command(str(jobs_dir)))))
    if artifact_rows:
        artifacts = Table.grid(padding=(0, 1), expand=True)
        artifacts.add_column(style="bold", no_wrap=True)
        artifacts.add_column(ratio=1, overflow="fold", no_wrap=False)
        for label, value in artifact_rows:
            artifacts.add_row(Text(label), Text(value, overflow="fold", no_wrap=False))
        console.print(
            Panel(
                artifacts,
                title="[bold]Artifacts[/bold]",
                border_style="cyan",
                padding=(0, 1),
            )
        )


def render_result(result: Mapping[str, Any]) -> str:
    """Return the stable non-color rendering used by tests and API callers."""
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=False, color_system=None, width=1000)
    render_evaluation_result(result, console=console)
    return stream.getvalue()
