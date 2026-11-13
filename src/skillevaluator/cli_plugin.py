# SPDX-License-Identifier: Apache-2.0
"""Public plugin commands, sharing existing runtime options and report writers."""

from __future__ import annotations

import copy
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import click

from skillevaluator.models.result import ValidationResult

if TYPE_CHECKING:
    from rich.console import Console


def _emit_reports(
    results: list[ValidationResult],
    formats: tuple[str, ...],
    output: Path,
    console: Console,
    *,
    plugin_name: str | None = None,
) -> None:
    from skillevaluator.reporting import BenchmarkReporter, CLIReporter
    from skillevaluator.tier1.commands import REPORTERS

    # Without the plugin's name the card is titled with the placeholder "skill".
    reporters = {**REPORTERS, "benchmark": partial(BenchmarkReporter, skill_name=plugin_name)}
    # Parse and validate everything before the first output file is created.
    selected = [(name, reporters[name]()) for name in dict.fromkeys(formats) if name != "cli"]
    if "cli" in formats:
        CLIReporter(console=console).print_all(results)
    for name, reporter in selected:
        filename = "BENCHMARK.md" if name == "benchmark" else "plugin-report" + reporter.get_file_extension()
        reporter.save(results, output / filename)
        console.print(f"{name} report: {output / filename}", markup=False, highlight=False)


def _report_output_location(source: Path, output: Path, formats: tuple[str, ...]) -> Path:
    from skillevaluator.cli import _resolve_file_report_output_location
    from skillevaluator.cli_core import resolve_plugin_path

    file_formats = tuple("markdown" if name == "benchmark" else name for name in formats)
    return _resolve_file_report_output_location(resolve_plugin_path(source), output, file_formats)


def plugin_result_from_execution(source: Path, engine_result: object, *, env_mode: str) -> ValidationResult | None:
    """Normalize only the returned plugin run, never another execution's latest."""
    from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory

    run_dir_value = engine_result.get("run_dir") if isinstance(engine_result, dict) else None
    run_dir = Path(run_dir_value) if isinstance(run_dir_value, (str, Path)) and run_dir_value else None
    if run_dir is None or not run_dir.is_dir():
        return None
    return agent_eval_result_from_directory(
        source, run_dir, env_mode=env_mode, engine_result=engine_result, use_llm_judge=False
    )


def emit_plugin_run_reports(
    source: Path,
    engine_result: dict,
    *,
    report_formats: tuple[str, ...],
    output_dir: Path,
    results_dir: Path | None,
    env_mode: str,
    console: Console,
) -> None:
    """Normalize existing score artifacts and attach retained plugin observations."""
    from skillevaluator.cli import _attach_plugin_report
    from skillevaluator.cli_core import resolve_plugin_path
    from skillevaluator.publication_identity import publication_target_name

    source = resolve_plugin_path(source)
    # Another execution may publish latest before these requested formats are
    # written. Only the returned run owns this command's canonical score data.
    result = plugin_result_from_execution(source, engine_result, env_mode=env_mode)
    if result is None:
        result = ValidationResult(validator_name="AGENT_EVAL")
        result.mark_scan_incomplete("canonical-score-summary")
        result.add_warning("Canonical score summary unavailable; retained runtime observations follow.")
    _attach_plugin_report(result, engine_result)
    output_dir = _report_output_location(source, output_dir, report_formats)
    _emit_reports(
        [result], report_formats, output_dir, console, plugin_name=publication_target_name(source) or source.name
    )


def build_plugin_commands(evaluate, *, run_options, report_options, console):
    """Build commands from copies of the skill command's existing options."""

    def plugin_reports(callback):
        callback = report_options(callback)
        for param in callback.__click_params__:
            if param.name == "report_formats":
                param.type = click.Choice(["cli", "json", "html", "markdown", "sarif", "benchmark"])
        return callback

    @click.command("evaluate-plugin", context_settings=evaluate.context_settings, cls=type(evaluate))
    @run_options
    @plugin_reports
    @click.option("--repo-root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None)
    @click.option("--policy", "policy_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
    @click.option("--profile", default=None, help="Operator policy profile for plugin checks.")
    @click.pass_context
    def evaluate_plugin(ctx, report_formats, output_dir, **kwargs):
        """Run supported agents against a plugin and retain explicit loading and behavior evidence."""
        from skillevaluator.evaluation.options import PluginOptions
        from skillevaluator.validators.policy import resolve_policy

        output_dir = _report_output_location(kwargs["skill_path"], output_dir, report_formats)
        controls = {
            name: kwargs.pop(name)
            for name in ("plugin_load", "lift_mode", "probe_mcp", "probe_mcp_env", "trial_retries", "repo_root")
        }
        try:
            policy = resolve_policy(profile=kwargs.pop("profile"), policy_path=kwargs.pop("policy_path"))
        except (OSError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        controls.update(
            allowed_private_hosts=policy.mcp_allowed_private_hosts, hook_allowed_urls=policy.hook_allowed_urls
        )
        ctx.invoke(
            evaluate,
            **kwargs,
            plugin_options=PluginOptions(**controls),
            plugin_report_formats=report_formats,
            plugin_output_dir=output_dir,
        )

    shared = [copy.copy(param) for param in evaluate.params]
    for param in shared:
        if param.name == "skill_path":
            param.type = click.Path(exists=True, path_type=Path)
    evaluate_plugin.params = shared + evaluate_plugin.params

    @click.command("report-plugin", context_settings=evaluate.context_settings, cls=type(evaluate))
    @click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
    @plugin_reports
    def report_plugin(run_dir, report_formats, output_dir):
        """Replay retained runtime evidence without the source, providers or execution dependencies."""
        from skillevaluator.reporting.plugin_runtime_io import load_plugin_runtime_report

        try:
            report = load_plugin_runtime_report(run_dir)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        result = ValidationResult(validator_name="Retained runtime evidence")
        result.mark_scan_incomplete("saved-score-summary-unavailable")
        result.add_message("Passive replay: scoring and publication eligibility are not recomputed.")
        payload = report.payload()
        result.metadata["plugin_runtime_report"] = payload
        _emit_reports([result], report_formats, output_dir, console, plugin_name=payload["name"] or None)

    return evaluate_plugin, report_plugin
