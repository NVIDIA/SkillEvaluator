# SPDX-License-Identifier: Apache-2.0
"""Command for replaying retained plugin observations."""

from __future__ import annotations

from pathlib import Path

import click


def build_report_plugin_command(
    *, report_options, console, command_class: type[click.Command] = click.Command, context_settings=None
) -> click.Command:
    """Build the saved report command with the shared report options."""

    def replay_report_options(callback):
        callback = report_options(callback)
        for param in callback.__click_params__:
            if param.name == "report_formats":
                param.type = click.Choice(["cli", "json", "html", "markdown", "sarif", "benchmark"])
                param.help = "Report formats. Separate values with commas or spaces, or repeat the option."
        return callback

    @click.command("report-plugin", context_settings=context_settings, cls=command_class)
    @click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
    @replay_report_options
    def report_plugin(run_dir: Path, report_formats: tuple[str, ...], output_dir: Path) -> None:
        """Render retained plugin runtime evidence from a saved run."""
        from skillevaluator.models.result import ValidationResult
        from skillevaluator.reporting import BenchmarkReporter, CLIReporter, HTMLReporter
        from skillevaluator.reporting.base import _write_report_atomically
        from skillevaluator.reporting.plugin_reports import augment_plugin_report, render_plugin_report
        from skillevaluator.reporting.plugin_runtime_io import load_plugin_runtime_report
        from skillevaluator.tier1.commands import REPORTERS

        try:
            payload = load_plugin_runtime_report(run_dir).payload()
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

        result = ValidationResult(validator_name="Retained runtime evidence")
        result.mark_scan_incomplete("saved-score-summary-unavailable")
        result.add_message("Replay shows saved observations. Canonical scores and publication eligibility are unavailable.")
        result.metadata["plugin_runtime_report"] = payload
        results = [result]

        for name in dict.fromkeys(report_formats):
            if name == "cli":
                reporter = CLIReporter(console=console)
                reporter.print_all(results)
                fragment = augment_plugin_report(reporter, "", results)
                console.print(fragment, markup=False, highlight=False, soft_wrap=True, end="")
                continue
            if name == "benchmark":
                reporter = BenchmarkReporter(skill_name=payload["name"] or None)
                filename = "BENCHMARK.md"
            else:
                reporter = HTMLReporter(content_label="Plugin") if name == "html" else REPORTERS[name]()
                filename = "plugin-report" + reporter.get_file_extension()
            output_path = output_dir / filename
            document = render_plugin_report(reporter, results)
            _write_report_atomically(output_path, document.encode("utf-8", "backslashreplace"))
            console.print(f"{name} report: {output_path}", markup=False, highlight=False)

    return report_plugin
